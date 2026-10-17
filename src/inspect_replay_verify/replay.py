"""The replay engine: pristine sandbox, mechanical re-issue, independent checks, verdict."""

import contextlib
import logging
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import anyio
from inspect_ai.approval import ApprovalPolicy, auto_approver
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, execute_tools
from inspect_ai.tool import Tool, ToolCall
from inspect_ai.util import SandboxEnvironment, SandboxEnvironmentSpec, concurrency, span

from . import _inspect_compat as compat
from .adapters import ReplayAdapter, adapter_for
from .checks import Check, CheckOutcome
from .transcript import REPLAY_SPAN_TYPE, RecordedCall

logger = logging.getLogger(__name__)

Verdict = Literal["verified", "refuted", "inconclusive"]
OnDivergence = Literal["record", "inconclusive"]

_OUTPUT_PREVIEW = 1000


@dataclass(frozen=True, kw_only=True, slots=True)
class PristineState:
    """Everything needed to boot a sandbox exactly as the original sample started."""

    sandbox: SandboxEnvironmentSpec
    files: Mapping[str, bytes] = field(default_factory=dict)
    setup: bytes | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sample_id: str | int | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class ReplayContext:
    """What a per-sample check factory sees."""

    sample_id: str | int | None
    metadata: Mapping[str, Any]
    target: str | list[str] | None


ChecksArg = Sequence[Check] | Callable[[ReplayContext], Sequence[Check]]


@dataclass(kw_only=True, slots=True)
class CallReplay:
    """How one recorded call fared on replay."""

    id: str
    function: str
    disposition: str
    reason: str = ""
    replayed: bool = False
    diverged: bool | None = None
    credential_scoped: bool = False
    nondeterministic: bool = False
    recorded_error: str | None = None
    replay_error: str | None = None
    replay_output: str | None = None


@dataclass(kw_only=True)
class ReplayReport:
    """The outcome of one independent replay."""

    verdict: Verdict
    explanation: str
    sandbox_type: str
    adapter: str
    transcript_source: str
    calls: list[CallReplay] = field(default_factory=list)
    checks: list[CheckOutcome] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def divergences(self) -> int:
        """Replayed calls whose output differed from the recording, tolerated ones excluded."""
        return sum(1 for c in self.calls if c.diverged and not c.credential_scoped)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "explanation": self.explanation,
            "sandbox_type": self.sandbox_type,
            "adapter": self.adapter,
            "transcript_source": self.transcript_source,
            "calls_recorded": len(self.calls),
            "calls_replayed": sum(1 for c in self.calls if c.replayed),
            "divergences": self.divergences,
            "calls": [asdict(c) for c in self.calls],
            "checks": [asdict(c) for c in self.checks],
            "notes": self.notes,
        }


def _inconclusive(
    why: str,
    *,
    sandbox_type: str,
    adapter: ReplayAdapter,
    source: str,
    calls: list[CallReplay] | None = None,
) -> ReplayReport:
    return ReplayReport(
        verdict="inconclusive",
        explanation=why,
        sandbox_type=sandbox_type,
        adapter=adapter.name,
        transcript_source=source,
        calls=calls or [],
    )


# (sandbox type, config) pairs whose `task_init` already ran in this process.
_task_initialized: set[tuple[str, str]] = set()


async def _task_init(sandbox_cls: Any, task_name: str, spec: SandboxEnvironmentSpec) -> None:
    key = (spec.type, repr(spec.config))
    if key not in _task_initialized:
        await sandbox_cls.task_init(task_name, spec.config)
        _task_initialized.add(key)


async def task_cleanup(task_name: str, spec: SandboxEnvironmentSpec) -> None:
    """Release task-level resources for a sandbox type this process replayed into."""
    key = (spec.type, repr(spec.config))
    if key in _task_initialized:
        sandbox_cls = compat.registry_find_sandboxenv(spec.type)
        await sandbox_cls.task_cleanup(task_name, spec.config, True)
        _task_initialized.discard(key)


def _tool_name(tool: Tool) -> str:
    from inspect_ai.tool import ToolDef

    return ToolDef(tool).name


def fresh_tools(tools: Sequence[Tool]) -> dict[str, Tool]:
    """Tools keyed by name, each re-created from its registry params where possible.

    A fresh instance matters for stateful tools (`bash_session`, `text_editor`), whose
    state is keyed by instance: reusing the original's instance would reach for the
    original sandbox's session.
    """
    out: dict[str, Tool] = {}
    for tool in tools:
        try:
            info = compat.registry_info(tool)
            fresh = cast(
                Tool, compat.registry_create("tool", info.name, **compat.registry_params(tool))
            )
        except Exception:
            fresh = tool
        out[_tool_name(tool)] = fresh
    return out


def default_tools() -> dict[str, Tool]:
    """Inspect's standard sandbox tools with default parameters, for logs replayed offline."""
    from inspect_ai.tool import bash, bash_session, python, text_editor

    return {
        "bash": bash(),
        "python": python(),
        "text_editor": text_editor(),
        "bash_session": bash_session(),
    }


def _render(error: Any) -> str | None:
    return None if error is None else f"{error.type}: {error.message}"


async def _boot(
    sandbox_cls: Any, task_name: str, pristine: PristineState
) -> dict[str, SandboxEnvironment]:
    metadata = dict(pristine.metadata)
    metadata["__sample_id__"] = pristine.sample_id
    environments = cast(
        dict[str, SandboxEnvironment],
        await sandbox_cls.sample_init(task_name, pristine.sandbox.config, metadata),
    )
    if not environments:
        raise RuntimeError(f"{pristine.sandbox.type}.sample_init returned no environments")
    return environments


async def _pristine(environments: dict[str, SandboxEnvironment], pristine: PristineState) -> None:
    await compat.copy_sandbox_environment_files(dict(pristine.files), environments)
    if pristine.setup:
        await compat.setup_sandbox_environment(pristine.setup, environments)
    for env in environments.values():
        await compat.resolve_root_access(env)


class _Unavailable(Exception):
    """The replay has no way to re-issue this action."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _Replayed:
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


async def _reissue_action(
    call: RecordedCall,
    environments: Mapping[str, SandboxEnvironment],
    tools: Mapping[str, Tool],
    timeout: float | None,
) -> _Replayed:
    if call.kind == "tool":
        tool = tools.get(call.function)
        if tool is None:
            raise _Unavailable(f"tool '{call.function}' is unavailable to the replay")
        reply = await _reissue(call, tool, timeout)
        return _Replayed(
            status=reply.error.type if reply.error else None,
            output=reply.text,
            error=_render(reply.error),
        )
    if len(environments) > 1:
        raise _Unavailable(
            "the transcript does not record which of the sample's sandboxes ran this action"
        )
    env = next(iter(environments.values()))
    args = call.arguments
    with anyio.fail_after(timeout):
        if call.kind == "write_file":
            await env.write_file(str(args["file"]), str(args["contents"] or ""))
            return _Replayed(status=None, output="")
        result = await env.exec(
            shlex.split(str(args["cmd"])),
            input=args.get("input"),
            cwd=args.get("cwd"),
            env=args.get("env") or {},
            user=args.get("user"),
            timeout=args.get("timeout"),
            timeout_retry=args.get("timeout_retry", True),
        )
    return _Replayed(
        status=_exit_status(result.returncode),
        output=result.stdout + result.stderr,
    )


async def _reissue(call: RecordedCall, tool: Tool, timeout: float | None) -> ChatMessageTool:
    message = ChatMessageAssistant(
        content="",
        tool_calls=[ToolCall(id=call.id, function=call.function, arguments=call.arguments)],
    )
    # Approval already happened in the original: a rejected call is skipped upstream, so
    # every call here is one that ran, and no human is asked again.
    approve_all = [ApprovalPolicy(auto_approver(), "*")]
    with anyio.fail_after(timeout):
        result = await execute_tools([message], [tool], approval=approve_all)
    replies = [m for m in result.messages if isinstance(m, ChatMessageTool)]
    if not replies:
        raise RuntimeError(f"tool '{call.function}' produced no result")
    return replies[0]


def resolve_checks(checks: ChecksArg, context: ReplayContext) -> list[Check]:
    return list(checks(context) if callable(checks) else checks)


def _verdict(outcomes: Sequence[CheckOutcome]) -> tuple[Verdict, str]:
    if not outcomes:
        return "inconclusive", "no ground-truth check produced an outcome"
    failed = [o for o in outcomes if o.passed is False]
    unknown = [o for o in outcomes if o.passed is None]
    if failed:
        return "refuted", "replay failed: " + "; ".join(
            f"{o.name} ({o.explanation})" for o in failed
        )
    if unknown:
        return "inconclusive", "could not evaluate: " + "; ".join(
            f"{o.name} ({o.explanation})" for o in unknown
        )
    return "verified", f"all {len(outcomes)} check(s) passed on an independent replay"


async def replay(
    calls: Sequence[RecordedCall],
    pristine: PristineState,
    *,
    checks: Sequence[Check],
    tools: Mapping[str, Tool],
    task_name: str = "replay",
    adapter: ReplayAdapter | None = None,
    on_divergence: OnDivergence = "record",
    call_timeout: float | None = None,
    original_identifiers: Sequence[str] = (),
    transcript_source: str = "events",
    max_concurrent: int | None = None,
) -> ReplayReport:
    """Re-issue `calls` in a fresh sandbox booted from `pristine` and judge it with `checks`.

    No model is involved. Any condition under which the replay cannot claim to have
    reproduced the original — a sandbox that will not boot, a setup script that fails, a
    call with an unknown effect, a tool that is not available, a check that cannot be
    evaluated — yields ``inconclusive``. Only definite check outcomes on a complete replay
    yield ``verified`` or ``refuted``.
    """
    adapter = adapter or adapter_for(pristine.sandbox.type)
    stype = pristine.sandbox.type

    def stop(why: str, calls_so_far: list[CallReplay] | None = None) -> ReplayReport:
        return _inconclusive(
            why, sandbox_type=stype, adapter=adapter, source=transcript_source, calls=calls_so_far
        )

    if not checks:
        return stop(
            "no ground-truth checks were given; replay verification needs an objective effect"
        )

    try:
        sandbox_cls = compat.registry_find_sandboxenv(stype)
    except Exception as ex:
        return stop(f"sandbox type '{stype}' is not installed: {ex}")

    limiter: Any = (
        concurrency("inspect_replay_verify", max_concurrent)
        if max_concurrent is not None
        else contextlib.nullcontext()
    )
    async with limiter, span("replay_verify", type=REPLAY_SPAN_TYPE):
        try:
            await _task_init(sandbox_cls, task_name, pristine.sandbox)
            environments = await _boot(sandbox_cls, task_name, pristine)
        except Exception as ex:
            return stop(f"replay sandbox could not be started: {type(ex).__name__}: {ex}")

        interrupted = False
        try:
            with compat.bound_environments(environments):
                return await _replay_in(
                    calls,
                    pristine,
                    environments,
                    checks=checks,
                    tools=tools,
                    adapter=adapter,
                    on_divergence=on_divergence,
                    call_timeout=call_timeout,
                    original_identifiers=original_identifiers,
                    stop=stop,
                    transcript_source=transcript_source,
                )
        except anyio.get_cancelled_exc_class():
            interrupted = True
            raise
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await sandbox_cls.sample_cleanup(
                        task_name, pristine.sandbox.config, environments, interrupted
                    )
                except Exception as ex:
                    logger.warning("replay sandbox cleanup failed: %r", ex)


async def _replay_in(
    calls: Sequence[RecordedCall],
    pristine: PristineState,
    environments: dict[str, SandboxEnvironment],
    *,
    checks: Sequence[Check],
    tools: Mapping[str, Tool],
    adapter: ReplayAdapter,
    on_divergence: OnDivergence,
    call_timeout: float | None,
    original_identifiers: Sequence[str],
    stop: Callable[..., ReplayReport],
    transcript_source: str,
) -> ReplayReport:
    try:
        await _pristine(environments, pristine)
    except Exception as ex:
        return stop(f"replay sandbox could not reach the pristine state: {ex}")
    try:
        await adapter.prepare(environments)
    except Exception as ex:
        return stop(f"adapter '{adapter.name}' could not prepare the replay: {ex!r}")
    identifiers = [*original_identifiers, *adapter.session_identifiers(environments)]

    results: list[CallReplay] = []
    for recorded in calls:
        record = CallReplay(
            id=recorded.id,
            function=recorded.function,
            disposition=recorded.disposition,
            reason=recorded.reason,
            nondeterministic=recorded.nondeterministic,
            recorded_error=_render(recorded.error),
        )
        results.append(record)
        if recorded.disposition == "skip":
            continue
        if recorded.disposition == "unreproducible":
            return stop(f"call {recorded.id} ({recorded.function}): {recorded.reason}", results)
        call = adapter.rewrite_call(recorded)
        record.credential_scoped = bool(adapter.credentials_in(call))
        try:
            replayed = await _reissue_action(call, environments, tools, call_timeout)
        except _Unavailable as ex:
            record.reason = str(ex)
            return stop(f"call {recorded.id}: {ex}", results)
        except TimeoutError:
            return stop(
                f"call {recorded.id} ({recorded.function}) exceeded {call_timeout}s", results
            )
        except Exception as ex:
            record.replay_error = f"{type(ex).__name__}: {ex}"
            return stop(f"call {recorded.id} ({recorded.function}) raised on replay: {ex}", results)
        record.replayed = True
        record.replay_error = replayed.error
        record.replay_output = replayed.output[:_OUTPUT_PREVIEW]
        if replayed.status != _recorded_status(recorded):
            # An outcome the original did not have is a replay that did not reproduce it,
            # whatever the cause, so nothing it leaves behind can stand for the original.
            record.diverged = True
            return stop(
                f"call {recorded.id} ({recorded.function}) ended "
                f"{replayed.status or 'ok'} on replay but {_recorded_status(recorded) or 'ok'} "
                "in the original",
                results,
            )
        record.diverged = recorded.kind == "tool" and adapter.normalize(
            replayed.output, identifiers
        ) != adapter.normalize(recorded.result, identifiers)

    diverged = [c for c in results if c.diverged and not c.credential_scoped]
    if diverged and on_divergence == "inconclusive":
        return stop(
            f"{len(diverged)} call output divergence(s) from the recording: "
            + ", ".join(c.id for c in diverged),
            results,
        )

    try:
        collected = await adapter.collect(environments, checks)
    except Exception as ex:
        return stop(f"adapter '{adapter.name}' could not collect evidence: {ex!r}", results)
    report = ReplayReport(
        verdict="inconclusive",
        explanation="",
        sandbox_type=pristine.sandbox.type,
        adapter=adapter.name,
        transcript_source=transcript_source,
        calls=results,
        checks=collected.outcomes,
        notes=dict(collected.notes),
    )
    if collected.inconclusive:
        report.explanation = collected.inconclusive
        return report
    report.verdict, report.explanation = _verdict(collected.outcomes)
    if diverged and report.verdict != "inconclusive":
        report.explanation += f" ({len(diverged)} call output divergence(s) recorded)"
    return report
