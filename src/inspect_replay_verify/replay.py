"""The replay engine: re-issue a recording's calls in a pristine sandbox, then judge it.

`verify` is the whole pipeline over a `Recording` (what a run left behind: its
transcript and the sample it started from). `replay_scorer` builds one from the running
sample, `replay_log` from each sample of a finished log. `replay` is the core, for a
caller that already has the calls and the pristine state.
"""

import contextlib
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import anyio
from inspect_ai.approval import ApprovalPolicy, auto_approver
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool, execute_tools
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from inspect_ai.tool import Tool, ToolCall
from inspect_ai.util import (
    SandboxEnvironment,
    SandboxEnvironmentSpec,
    SandboxEnvironmentType,
    concurrency,
    span,
)

from .adapters import ReplayAdapter, adapter_for
from .checks import Check, Sandboxes
from .report import CallReplay, Inconclusive, ReplayReport, render_error, stage
from .sandbox import PristineState, as_spec, booted, pristine_state
from .tools import replay_tools
from .transcript import (
    DEFAULT_SKIP_TOOLS,
    REPLAY_SPAN_TYPE,
    RecordedCall,
    Untranscribed,
    recorded_calls,
    validate_untranscribed,
)

OnDivergence = Literal["record", "inconclusive"]

_OUTPUT_PREVIEW = 1000


@dataclass(frozen=True, kw_only=True, slots=True)
class ReplayConfig:
    """How to replay: the same for every sample of a task.

    Args:
      sandbox: Override the task-level sandbox (by default the one the run recorded).
        A `Sample`'s own sandbox is layered over it exactly as Inspect does.
      tools: Tools to re-issue calls with, over the task's own and Inspect's standard
        `bash`/`python`/`text_editor`/`bash_session`. Pass the agent's own when they are
        configured (timeout, user) and not on `state.tools`, as with `react(tools=...)`.
      adapter: Backend adapter; chosen by sandbox type when omitted.
      skip_tools: Tool names never re-issued (no sandbox effect).
      untranscribed: What a sandbox action no tool call made does to the replay:
        ``"inconclusive"`` refuses to judge, ``"replay"`` re-issues it in order,
        ``"ignore"`` leaves it out.
      on_divergence: ``"record"`` notes calls whose replayed output differs from the
        recording; ``"inconclusive"`` refuses to judge such a replay.
      call_timeout: Wall-clock bound per re-issued call, beyond the tool's own timeout.
      max_concurrent: Cap on simultaneous replay sandboxes in this process.
    """

    sandbox: SandboxEnvironmentType | None = None
    tools: Sequence[Tool] = ()
    adapter: ReplayAdapter | None = None
    skip_tools: frozenset[str] = DEFAULT_SKIP_TOOLS
    untranscribed: Untranscribed = "inconclusive"
    on_divergence: OnDivergence = "record"
    call_timeout: float | None = None
    max_concurrent: int | None = None

    def __post_init__(self) -> None:
        # Iterables a caller passes are read once per sample, so pin them down here.
        object.__setattr__(self, "skip_tools", frozenset(self.skip_tools))
        object.__setattr__(self, "tools", tuple(self.tools))
        validate_untranscribed(self.untranscribed)
        if self.on_divergence not in get_args(OnDivergence):
            raise ValueError(f"on_divergence is one of {get_args(OnDivergence)}")


@dataclass(frozen=True, kw_only=True, slots=True)
class ReplayContext:
    """What a per-sample check factory sees."""

    sample_id: str | int | None
    metadata: Mapping[str, Any]
    target: str | list[str] | None


ChecksArg = Sequence[Check] | Callable[[ReplayContext], Sequence[Check]]


@dataclass(frozen=True, kw_only=True, slots=True)
class Recording:
    """What one sample's run left behind, for a replay to work from."""

    events: Sequence[object]
    messages: Sequence[ChatMessage]
    sample: Sample | None
    """The sample as it was before the run, or None when it cannot be recovered."""
    missing: str = ""
    """Why the transcript or the pristine sample cannot be trusted, if either cannot."""
    task_name: str = "replay"
    task_sandbox: SandboxEnvironmentSpec | None = None
    task_tools: Sequence[Tool] = ()
    original_environments: Mapping[str, SandboxEnvironment] = field(default_factory=dict)
    state: TaskState | None = None
    """The sample's state, for checks that grade with an Inspect scorer."""
    context: ReplayContext


async def verify(recording: Recording, checks: ChecksArg, config: ReplayConfig) -> ReplayReport:
    """Replay a recording in a fresh sandbox and judge what the replay did."""
    calls, source = recorded_calls(
        events=recording.events,
        messages=recording.messages,
        skip_tools=config.skip_tools,
        untranscribed=config.untranscribed,
    )
    spec = as_spec(config.sandbox) or recording.task_sandbox
    report = ReplayReport(sandbox_type=spec.type if spec else "", transcript_source=source)
    with report.concluding():
        if recording.missing:
            raise Inconclusive(recording.missing)
        if recording.sample is None:
            raise Inconclusive("the pristine sample is unavailable")
        with stage("could not resolve the sample's pristine state"):
            pristine = await pristine_state(recording.sample, spec, recording.task_name)
        with stage("the check factory raised"):
            resolved = list(checks(recording.context) if callable(checks) else checks)
        target = recording.context.target
        await _replay_into(
            report,
            calls,
            pristine,
            resolved,
            config,
            task_name=recording.task_name,
            task_tools=recording.task_tools,
            original_environments=recording.original_environments,
            state=recording.state,
            target=Target(target) if target is not None else None,
        )
    return report


async def replay(
    calls: Sequence[RecordedCall],
    pristine: PristineState,
    checks: Sequence[Check],
    config: ReplayConfig | None = None,
    *,
    task_name: str = "replay",
    transcript_source: str = "events",
) -> ReplayReport:
    """Re-issue `calls` in a fresh sandbox booted from `pristine` and judge it with `checks`.

    No model is involved. Any condition under which the replay cannot claim to have
    reproduced the original — a sandbox that will not boot, a setup script that fails, a
    call with an unknown effect, a tool that is not available, a check that cannot be
    evaluated — yields ``inconclusive``. Only definite check outcomes on a complete replay
    yield ``verified`` or ``refuted``.
    """
    config = config or ReplayConfig()
    report = ReplayReport(transcript_source=transcript_source)
    with report.concluding():
        await _replay_into(report, calls, pristine, checks, config, task_name=task_name)
    return report


async def _replay_into(
    report: ReplayReport,
    calls: Sequence[RecordedCall],
    pristine: PristineState,
    checks: Sequence[Check],
    config: ReplayConfig,
    *,
    task_name: str,
    task_tools: Sequence[Tool] = (),
    original_environments: Mapping[str, SandboxEnvironment] | None = None,
    state: TaskState | None = None,
    target: Target | None = None,
) -> None:
    adapter = config.adapter or adapter_for(pristine.sandbox.type)
    report.sandbox_type, report.adapter = pristine.sandbox.type, adapter.name
    if not checks:
        raise Inconclusive(
            "no ground-truth checks were given; replay verification needs an objective effect"
        )
    tools, sources = replay_tools(config.tools, task_tools)
    limit = (
        concurrency("inspect_replay_verify", config.max_concurrent)
        if config.max_concurrent is not None
        else contextlib.nullcontext()
    )
    async with (
        limit,
        span("replay_verify", type=REPLAY_SPAN_TYPE),
        booted(pristine, task_name) as environments,
    ):
        with stage(f"adapter '{adapter.name}' could not prepare the replay"):
            await adapter.prepare(environments)
        identifiers = [
            *adapter.session_identifiers(original_environments or {}),
            *adapter.session_identifiers(environments),
        ]
        try:
            for recorded in calls:
                record = CallReplay.of(recorded)
                report.calls.append(record)
                await _reissue(record, recorded, environments, tools, adapter, identifiers, config)
        finally:
            report.notes["tool_sources"] = {
                c.function: sources.get(c.function, "missing") for c in report.calls if c.replayed
            }
        if report.divergences and config.on_divergence == "inconclusive":
            diverged = [c.id for c in report.calls if c.diverged and not c.credential_scoped]
            raise Inconclusive(
                f"{len(diverged)} call output divergence(s) from the recording: "
                + ", ".join(diverged)
            )
        sandboxes = Sandboxes(environments, state=state, target=target)
        report.checks = [await c.run(sandboxes) for c in checks if c.phase == "live"]
        with stage(f"adapter '{adapter.name}' could not freeze the replay sandbox"):
            reader = await adapter.freeze(environments, report.notes)
        report.checks += [await c.run(reader) for c in checks if c.phase == "evidence"]
    report.judge(report.checks)


@dataclass(frozen=True, kw_only=True, slots=True)
class _Outcome:
    status: str | None
    """The outcome class: a tool error type, an `exec`'s non-zero exit, or None for success."""
    output: str
    error: str | None = None


def _exit_status(returncode: int | None) -> str | None:
    return f"exit {returncode}" if returncode else None


def _recorded_status(call: RecordedCall) -> str | None:
    if call.kind == "exec":
        return _exit_status(call.returncode)
    return call.error.type if call.error else None


async def _reissue(
    record: CallReplay,
    recorded: RecordedCall,
    environments: Mapping[str, SandboxEnvironment],
    tools: Mapping[str, Tool],
    adapter: ReplayAdapter,
    identifiers: Sequence[str],
    config: ReplayConfig,
) -> None:
    """Re-issue one recorded call, raising `Inconclusive` where it does not reproduce."""
    label = f"call {recorded.id} ({recorded.function})"
    if recorded.disposition == "skip":
        return
    if recorded.disposition == "unreproducible":
        raise Inconclusive(f"{label}: {recorded.reason}")
    call = adapter.rewrite_call(recorded)
    record.credential_scoped = bool(adapter.credentials_in(call))
    try:
        with anyio.fail_after(config.call_timeout):
            outcome = await _execute(call, environments, tools)
    except Inconclusive as ex:
        record.reason = str(ex)
        raise
    except TimeoutError as ex:
        raise Inconclusive(f"{label} exceeded {config.call_timeout}s") from ex
    except Exception as ex:
        record.replay_error = f"{type(ex).__name__}: {ex}"
        raise Inconclusive(f"{label} raised on replay: {ex}") from ex
    record.replayed = True
    record.replay_error = outcome.error
    record.replay_output = outcome.output[:_OUTPUT_PREVIEW]
    if outcome.status != _recorded_status(recorded):
        # An outcome the original did not have is a replay that did not reproduce it,
        # whatever the cause, so nothing it leaves behind can stand for the original.
        record.diverged = True
        raise Inconclusive(
            f"{label} ended {outcome.status or 'ok'} on replay but "
            f"{_recorded_status(recorded) or 'ok'} in the original"
        )
    record.diverged = recorded.kind == "tool" and adapter.normalize(
        outcome.output, identifiers
    ) != adapter.normalize(recorded.result, identifiers)


async def _execute(
    call: RecordedCall,
    environments: Mapping[str, SandboxEnvironment],
    tools: Mapping[str, Tool],
) -> _Outcome:
    if call.kind == "tool":
        return await _execute_tool(call, tools)
    if len(environments) > 1:
        raise Inconclusive(
            f"call {call.id}: the transcript does not record which of the sample's "
            "sandboxes ran this action"
        )
    return await _execute_sandbox_action(call, next(iter(environments.values())))


async def _execute_tool(call: RecordedCall, tools: Mapping[str, Tool]) -> _Outcome:
    tool = tools.get(call.function)
    if tool is None:
        raise Inconclusive(f"call {call.id}: tool '{call.function}' is unavailable to the replay")
    message = ChatMessageAssistant(
        content="",
        tool_calls=[ToolCall(id=call.id, function=call.function, arguments=call.arguments)],
    )
    # A call an approver rejected is skipped upstream, so every call here already ran
    # once with approval, and no human is asked again.
    approve_all = [ApprovalPolicy(auto_approver(), "*")]
    result = await execute_tools([message], [tool], approval=approve_all)
    replies = [m for m in result.messages if isinstance(m, ChatMessageTool)]
    if len(replies) != 1:
        raise RuntimeError(f"tool '{call.function}' returned {len(replies)} results, not one")
    [reply] = replies
    return _Outcome(
        status=reply.error.type if reply.error else None,
        output=reply.text,
        error=render_error(reply.error),
    )


async def _execute_sandbox_action(call: RecordedCall, env: SandboxEnvironment) -> _Outcome:
    args = call.arguments
    if call.kind == "write_file":
        await env.write_file(str(args["file"]), str(args["contents"] or ""))
        return _Outcome(status=None, output="")
    result = await env.exec(
        shlex.split(str(args["cmd"])),
        input=args.get("input"),
        cwd=args.get("cwd"),
        env=args.get("env") or {},
        user=args.get("user"),
        timeout=args.get("timeout"),
        timeout_retry=args.get("timeout_retry", True),
    )
    return _Outcome(status=_exit_status(result.returncode), output=result.stdout + result.stderr)
