"""Verify a finished `.eval` log after the fact, from any sandbox provider."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import anyio
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog, EvalSample, read_eval_log, resolve_sample_attachments

from .replay import ChecksArg, Recording, ReplayConfig, ReplayContext, verify
from .report import ReplayReport
from .sandbox import release_task


def logged_recording(
    eval_log: EvalLog, logged: EvalSample, dataset: Mapping[Any, Sample]
) -> Recording:
    """One logged sample's recording, with its pristine `Sample` from the dataset or the log.

    A log records the *names* of a sample's files, not their contents, so a sample that
    shipped files can be rebuilt only from its dataset.
    """
    logged = resolve_sample_attachments(logged, True)
    sample = dataset.get(logged.id)
    if sample is None and not logged.files:
        sample = Sample(
            id=logged.id,
            input=logged.input,
            target=logged.target,
            metadata=logged.metadata,
            sandbox=logged.sandbox,
            setup=logged.setup,
        )
    return Recording(
        events=logged.events,
        messages=logged.messages,
        sample=sample,
        missing="" if sample else "the sample shipped files the log does not keep; pass dataset=",
        task_name=eval_log.eval.task,
        task_sandbox=eval_log.eval.sandbox,
        context=ReplayContext(sample_id=logged.id, metadata=logged.metadata, target=logged.target),
    )


async def replay_log_async(
    log: EvalLog | str | Path,
    checks: ChecksArg,
    *,
    dataset: Iterable[Sample] | None = None,
    sample_ids: Iterable[str | int] | None = None,
    **options: Any,
) -> dict[tuple[str | int, int], ReplayReport]:
    """Replay every sample of a finished log; reports keyed by (sample id, epoch).

    Args:
      log: The log, or a path to it.
      checks: As for `replay_scorer`.
      dataset: The task's samples, for their files (a log keeps only file names). A sample
        that shipped files and has no dataset entry is reported inconclusive.
      sample_ids: Only replay these samples.
      **options: `ReplayConfig` fields. With no `tools`, calls are re-issued through
        Inspect's standard sandbox tools with default parameters.
    """
    config = ReplayConfig(**options)
    eval_log = log if isinstance(log, EvalLog) else read_eval_log(str(log))
    by_id = {s.id: s for s in (dataset or [])}
    wanted = set(sample_ids) if sample_ids is not None else None
    reports: dict[tuple[str | int, int], ReplayReport] = {}
    for logged in eval_log.samples or []:
        if wanted is None or logged.id in wanted:
            recording = logged_recording(eval_log, logged, by_id)
            reports[(logged.id, logged.epoch)] = await verify(recording, checks, config)

    await release_task(eval_log.eval.task)
    return reports


def replay_log(
    log: EvalLog | str | Path, checks: ChecksArg, **kwargs: Any
) -> dict[tuple[str | int, int], ReplayReport]:
    """Synchronous `replay_log_async`."""

    async def run() -> dict[tuple[str | int, int], ReplayReport]:
        return await replay_log_async(log, checks, **kwargs)

    return anyio.run(run)
