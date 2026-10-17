"""`replay_scorer()`: grade a sample on an independent replay, not on its own transcript."""

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from inspect_ai.log import read_eval_log_async, transcript
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Score,
    Scorer,
    Target,
    Value,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from inspect_ai.tool import Tool
from inspect_ai.util import SandboxEnvironmentSpec, SandboxEnvironmentType

from . import _inspect_compat as compat
from .adapters import ReplayAdapter
from .metrics import inconclusive_rate, refuted_rate, verified_rate
from .replay import ChecksArg, OnDivergence, Recording, ReplayConfig, ReplayContext, verify
from .report import ReplayReport
from .transcript import DEFAULT_SKIP_TOOLS, Untranscribed

logger = logging.getLogger(__name__)
_task_sandboxes: dict[str, SandboxEnvironmentSpec | None] = {}


async def running_task_sandbox() -> SandboxEnvironmentSpec | None:
    """The running task's resolved sandbox, read back from its in-progress log header.

    Inspect hands scorers no handle on the task, but it writes the resolved task-level
    sandbox (eval-level `--sandbox` override included) into the log header before the
    first sample starts.
    """
    location = compat.active_log_location()
    if location is None:
        return None
    if location not in _task_sandboxes:
        try:
            header = await read_eval_log_async(location, header_only=True)
        except Exception as ex:
            logger.warning("could not read the task sandbox from %s: %r", location, ex)
            return None
        _task_sandboxes[location] = header.eval.sandbox
    return _task_sandboxes[location]


async def live_recording(state: TaskState, target: Target) -> Recording:
    """The running sample's recording: its transcript so far and its pristine `Sample`."""
    events = list(transcript().events)
    count = getattr(getattr(transcript(), "history", None), "event_count", None)
    sample = compat.active_sample()
    missing = ""
    if isinstance(count, int) and count > len(events):
        missing = (
            f"only {len(events)} of {count} transcript events are readable; a replay of "
            "part of the transcript cannot stand for the whole"
        )
    elif sample is None:
        missing = (
            "no running sample to take the pristine state from; to verify a finished "
            "log, use inspect_replay_verify.replay_log()"
        )
    return Recording(
        events=events,
        messages=state.messages,
        sample=sample,
        missing=missing,
        task_name=compat.active_task_name() or "replay",
        task_sandbox=await running_task_sandbox(),
        task_tools=state.tools,
        original_environments={
            k: compat.unproxy(v) for k, v in compat.current_environments().items()
        },
        context=ReplayContext(
            sample_id=state.sample_id, metadata=state.metadata, target=target.target
        ),
    )


def score_from_report(
    report: ReplayReport,
    *,
    inconclusive_value: Value = NOANSWER,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Score:
    value: Value = {"verified": CORRECT, "refuted": INCORRECT}.get(
        report.verdict, inconclusive_value
    )
    return Score(
        value=value,
        explanation=f"[{report.verdict}] {report.explanation}",
        metadata={"replay": report.to_metadata(), **(extra_metadata or {})},
    )


@scorer(metrics=[accuracy(), stderr(), verified_rate(), refuted_rate(), inconclusive_rate()])
def replay_scorer(
    checks: ChecksArg,
    *,
    sandbox: SandboxEnvironmentType | None = None,
    tools: Sequence[Tool] | None = None,
    adapter: ReplayAdapter | None = None,
    skip_tools: Iterable[str] = DEFAULT_SKIP_TOOLS,
    untranscribed: Untranscribed = "inconclusive",
    on_divergence: OnDivergence = "record",
    call_timeout: float | None = None,
    inconclusive_value: Value = NOANSWER,
    max_concurrent: int | None = None,
) -> Scorer:
    """Score a sample by replaying its recorded tool calls in a fresh sandbox.

    The sample's transcript is read, a new instance of its sandbox is booted from the same
    pristine state (sample files and setup script), every recorded tool call is
    re-issued mechanically with no model involved, and `checks` are run against what the
    replay produced. The original run's own claims and its sandbox are never consulted.

    Args:
      checks: Ground-truth checks (see `inspect_replay_verify.checks`), or a function of
        `ReplayContext` returning them per sample.
      inconclusive_value: Score value for an inconclusive replay (never a pass).
      sandbox, tools, adapter, skip_tools, untranscribed, on_divergence, call_timeout,
        max_concurrent: See `ReplayConfig`. The task-level sandbox is read back from the
        running eval's log header unless `sandbox` overrides it.
    """
    config = ReplayConfig(
        sandbox=sandbox,
        tools=tuple(tools or ()),
        adapter=adapter,
        skip_tools=frozenset(skip_tools),
        untranscribed=untranscribed,
        on_divergence=on_divergence,
        call_timeout=call_timeout,
        max_concurrent=max_concurrent,
    )

    async def score(state: TaskState, target: Target) -> Score:
        report = await verify(await live_recording(state, target), checks, config)
        original = {k: v.value for k, v in (state.scores or {}).items()}
        return score_from_report(
            report,
            inconclusive_value=inconclusive_value,
            extra_metadata={"original_scores": original} if original else None,
        )

    return score
