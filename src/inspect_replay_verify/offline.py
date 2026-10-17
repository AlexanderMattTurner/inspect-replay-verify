"""Verify a finished `.eval` log after the fact, from any sandbox provider."""

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog, EvalSample, read_eval_log, resolve_sample_attachments
from inspect_ai.tool import Tool
from inspect_ai.util import SandboxEnvironmentType

from .adapters import ReplayAdapter, adapter_for
from .replay import (
    ChecksArg,
    OnDivergence,
    ReplayContext,
    ReplayReport,
    replay,
    resolve_checks,
    task_cleanup,
)
from .scorer import as_spec, pristine_state, resolve_tools
from .transcript import DEFAULT_SKIP_TOOLS, Untranscribed, recorded_calls

logger = logging.getLogger(__name__)


def _pristine_sample(logged: EvalSample, dataset: Mapping[Any, Sample]) -> Sample | None:
    """The sample as it was before the run: from the dataset, or rebuilt from the log.

    A log records the *names* of a sample's files, not their contents, so a sample that
    shipped files can be rebuilt only from its dataset.
    """
    if logged.id in dataset:
        return dataset[logged.id]
    if logged.files:
        return None
    return Sample(
        id=logged.id,
        input=logged.input,
        target=logged.target,
        metadata=logged.metadata,
        sandbox=logged.sandbox,
        setup=logged.setup,
    )


async def replay_log_async(
    log: EvalLog | str | Path,
    checks: ChecksArg,
    *,
    dataset: Iterable[Sample] | None = None,
    sandbox: SandboxEnvironmentType | None = None,
    tools: Sequence[Tool] | None = None,
    adapter: ReplayAdapter | None = None,
    skip_tools: Iterable[str] = DEFAULT_SKIP_TOOLS,
    untranscribed: Untranscribed = "inconclusive",
    on_divergence: OnDivergence = "record",
    call_timeout: float | None = None,
    sample_ids: Iterable[str | int] | None = None,
) -> dict[tuple[str | int, int], ReplayReport]:
    """Replay every sample of a finished log; reports keyed by (sample id, epoch).

    Args:
      log: The log, or a path to it.
      checks: As for `replay_scorer`.
      dataset: The task's samples, for their files (a log keeps only file names). A sample
        that shipped files and has no dataset entry is reported inconclusive.
      sandbox: Override the task-level sandbox recorded in the log.
      tools: Tools to re-issue calls with. Defaults to Inspect's standard sandbox tools
        with default parameters; pass the agent's own when they were configured.
      adapter, skip_tools, untranscribed, on_divergence, call_timeout: As for
        `replay_scorer`.
      sample_ids: Only replay these samples.
    """
    eval_log = log if isinstance(log, EvalLog) else read_eval_log(str(log))
    by_id = {s.id: s for s in (dataset or [])}
    wanted = set(sample_ids) if sample_ids is not None else None
    task_name = eval_log.eval.task
    task_sandbox = as_spec(sandbox) or eval_log.eval.sandbox
    replay_tools, _ = resolve_tools(tools, [])
    reports: dict[tuple[str | int, int], ReplayReport] = {}
    used_specs = []

    for logged in eval_log.samples or []:
        if wanted is not None and logged.id not in wanted:
            continue
        logged = resolve_sample_attachments(logged, True)
        calls, source = recorded_calls(
            events=logged.events,
            messages=logged.messages,
            skip_tools=skip_tools,
            untranscribed=untranscribed,
        )
        source_sample = _pristine_sample(logged, by_id)
        pristine = (
            await pristine_state(source_sample, task_sandbox, task_name)
            if source_sample is not None
            else None
        )
        if pristine is None:
            why = (
                "the sample shipped files the log does not keep; pass dataset="
                if source_sample is None
                else "the log records no sandbox for this sample; pass sandbox="
            )
            chosen = adapter or adapter_for(task_sandbox.type if task_sandbox else "")
            reports[(logged.id, logged.epoch)] = ReplayReport(
                verdict="inconclusive",
                explanation=why,
                sandbox_type=task_sandbox.type if task_sandbox else "",
                adapter=chosen.name,
                transcript_source=source,
            )
            continue
        used_specs.append(pristine.sandbox)
        reports[(logged.id, logged.epoch)] = await replay(
            calls,
            pristine,
            checks=resolve_checks(
                checks,
                ReplayContext(sample_id=logged.id, metadata=logged.metadata, target=logged.target),
            ),
            tools=replay_tools,
            task_name=task_name,
            adapter=adapter,
            on_divergence=on_divergence,
            call_timeout=call_timeout,
            transcript_source=source,
        )

    for spec in {(s.type, repr(s.config)): s for s in used_specs}.values():
        try:
            await task_cleanup(task_name, spec)
        except Exception as ex:
            logger.warning("task cleanup for %s failed: %r", spec.type, ex)
    return reports


def replay_log(
    log: EvalLog | str | Path, checks: ChecksArg, **kwargs: Any
) -> dict[tuple[str | int, int], ReplayReport]:
    """Synchronous `replay_log_async`."""

    async def run() -> dict[tuple[str | int, int], ReplayReport]:
        return await replay_log_async(log, checks, **kwargs)

    return anyio.run(run)
