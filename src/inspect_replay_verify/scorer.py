"""`replay_scorer()`: grade a sample on an independent replay, not on its own transcript."""

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from inspect_ai.dataset import Sample
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
from .adapters import ReplayAdapter, adapter_for
from .metrics import inconclusive_rate, refuted_rate, verified_rate
from .replay import (
    ChecksArg,
    OnDivergence,
    PristineState,
    ReplayContext,
    ReplayReport,
    default_tools,
    fresh_tools,
    replay,
    resolve_checks,
)
from .transcript import (
    DEFAULT_SKIP_TOOLS,
    UNTRANSCRIBED_POLICIES,
    Untranscribed,
    recorded_calls,
)


def as_spec(sandbox: SandboxEnvironmentType | None) -> SandboxEnvironmentSpec | None:
    """A `SandboxEnvironmentSpec` from any of Inspect's sandbox shorthands."""
    if sandbox is None or isinstance(sandbox, SandboxEnvironmentSpec):
        return sandbox
    if isinstance(sandbox, str):
        return SandboxEnvironmentSpec(sandbox)
    return SandboxEnvironmentSpec(sandbox[0], sandbox[1])


async def pristine_state(
    sample: Sample, task_sandbox: SandboxEnvironmentSpec | None, task_name: str | None
) -> PristineState | None:
    """The sandbox spec, files and setup a sample started from, resolved as Inspect does."""
    spec = await compat.resolve_sandbox(task_sandbox, sample, task_name)
    if spec is None:
        return None
    files: dict[str, bytes] = {}
    for path, contents in compat.resolve_sample_files(dict(sample.files or {})).items():
        files[path] = await compat.read_sample_file(contents)
    setup: bytes | None = None
    if sample.setup:
        setup = await compat.read_sample_file(sample.setup)
        text = setup.decode("utf-8")
        if not text.strip().startswith("#!"):
            setup = f"#!/usr/bin/env bash\n\n{text}".encode()
    return PristineState(
        sandbox=spec,
        files=files,
        setup=setup,
        metadata=dict(sample.metadata or {}),
        sample_id=sample.id,
    )


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
            _task_sandboxes[location] = header.eval.sandbox
        except Exception as ex:
            logger.warning("could not read the task sandbox from %s: %r", location, ex)
            return None
    return _task_sandboxes[location]


def resolve_tools(
    provided: Sequence[Tool] | None, task_tools: Sequence[Tool]
) -> tuple[dict[str, Tool], dict[str, str]]:
    """Fresh tool instances by name, and where each came from (provided/task/default)."""
    tools: dict[str, Tool] = {}
    sources: dict[str, str] = {}
    for name, tool in default_tools().items():
        tools[name], sources[name] = tool, "default"
    for name, tool in fresh_tools(task_tools).items():
        tools[name], sources[name] = tool, "task"
    for name, tool in fresh_tools(provided or []).items():
        tools[name], sources[name] = tool, "provided"
    return tools, sources


def score_from_report(
    report: ReplayReport,
    *,
    inconclusive_value: Value = NOANSWER,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Score:
    value: Value = {
        "verified": CORRECT,
        "refuted": INCORRECT,
    }.get(report.verdict, inconclusive_value)
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
      sandbox: Override the task-level sandbox. By default it is read back from the
        running eval's log header. A `Sample`'s own sandbox is layered over it exactly
        as Inspect does.
      tools: Tools to re-issue calls with. Defaults to the task's `state.tools`, then to
        Inspect's standard `bash`/`python`/`text_editor`/`bash_session`. Pass the agent's
        own tools when they are configured (timeout, user) and not on `state.tools`, as
        with `react(tools=...)`.
      adapter: Backend adapter; chosen by sandbox type when omitted.
      skip_tools: Tool names never re-issued (no sandbox effect).
      untranscribed: What a sandbox action no tool call made (a solver's own
        `sandbox().exec`) does to the replay: ``"inconclusive"`` refuses to judge,
        ``"replay"`` re-issues it in order, ``"ignore"`` leaves it out.
      on_divergence: ``"record"`` notes calls whose replayed output differs from the
        recording; ``"inconclusive"`` refuses to judge such a replay.
      call_timeout: Wall-clock bound per re-issued call, beyond the tool's own timeout.
      inconclusive_value: Score value for an inconclusive replay (never a pass).
      max_concurrent: Cap on simultaneous replay sandboxes across the eval.
    """

    if untranscribed not in UNTRANSCRIBED_POLICIES:
        raise ValueError(f"untranscribed is one of {UNTRANSCRIBED_POLICIES}, not {untranscribed!r}")

    async def score(state: TaskState, target: Target) -> Score:
        events = list(transcript().events)
        history = getattr(transcript(), "history", None)
        count = getattr(history, "event_count", None)
        calls, source = recorded_calls(
            events=events,
            messages=state.messages,
            skip_tools=skip_tools,
            untranscribed=untranscribed,
        )
        sample = compat.active_sample()
        task_name = compat.active_task_name() or "replay"
        spec = as_spec(sandbox)
        chosen = adapter or adapter_for(spec.type if spec else "")

        def inconclusive(why: str) -> Score:
            report = ReplayReport(
                verdict="inconclusive",
                explanation=why,
                sandbox_type=spec.type if spec else "",
                adapter=chosen.name,
                transcript_source=source,
            )
            return score_from_report(report, inconclusive_value=inconclusive_value)

        if isinstance(count, int) and count > len(events):
            return inconclusive(
                f"only {len(events)} of {count} transcript events are readable; a replay of "
                "part of the transcript cannot stand for the whole"
            )
        if sample is None:
            return inconclusive(
                "no running sample to take the pristine state from; to verify a finished "
                "log, use inspect_replay_verify.replay_log()"
            )
        pristine = await pristine_state(sample, spec or await running_task_sandbox(), task_name)
        if pristine is None:
            return inconclusive(
                "no sandbox is known for this sample (none on the sample, and the task's "
                "could not be read from the eval log); pass replay_scorer(sandbox=...)"
            )
        chosen = adapter or adapter_for(pristine.sandbox.type)
        replay_tools, tool_sources = resolve_tools(tools, state.tools)
        originals = compat.current_environments()
        report = await replay(
            calls,
            pristine,
            checks=resolve_checks(
                checks,
                ReplayContext(
                    sample_id=state.sample_id, metadata=state.metadata, target=target.target
                ),
            ),
            tools=replay_tools,
            task_name=task_name,
            adapter=chosen,
            on_divergence=on_divergence,
            call_timeout=call_timeout,
            original_identifiers=chosen.session_identifiers(
                {k: compat.unproxy(v) for k, v in originals.items()}
            ),
            transcript_source=source,
            max_concurrent=max_concurrent,
        )
        report.notes["tool_sources"] = {
            c.function: tool_sources.get(c.function, "missing") for c in report.calls if c.replayed
        }
        original = {k: v.value for k, v in (state.scores or {}).items()}
        return score_from_report(
            report,
            inconclusive_value=inconclusive_value,
            extra_metadata={"original_scores": original} if original else None,
        )

    return score
