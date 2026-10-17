"""Turn a sample's recorded transcript into the ordered list of tool calls to re-issue.

The source of truth is the sample's `ToolEvent`s, which Inspect records for every tool
execution — including calls made inside sub-agents, which never reach `state.messages`.
When a transcript carries no tool events (a condensed log, a hand-built state), the
assistant messages' `tool_calls` and their `ChatMessageTool` replies are used instead.

Each call gets a disposition:

- ``replay``: it executed in the original run and is re-issued.
- ``skip``: it never touched the sandbox, or never executed at all (a parse error, an
  approver's rejection, an agent handoff whose own inner calls are listed separately, a
  tool named in ``skip_tools``). Re-issuing it would do something the original did not.
- ``unreproducible``: it may have had a partial or unknown effect (cancelled mid-flight,
  the sandbox vanished, the call never completed). No replay past it can claim to start
  from the original's state, so the replay stops there and the verdict is inconclusive.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from inspect_ai.event import SpanBeginEvent, ToolEvent
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import ToolCallError

Disposition = Literal["replay", "skip", "unreproducible"]

# Tools that, by default, have no effect on sandbox state worth reproducing (or whose
# effect is outside the sandbox and non-deterministic), so a replay leaves them out.
DEFAULT_SKIP_TOOLS: frozenset[str] = frozenset(
    {
        "think",
        "submit",
        "web_search",
        "ask_user",
        "notify_user",
        "update_plan",
        "todo_write",
        "memory",
    }
)

# Span type this package's own replays run under, so a later read of the same transcript
# never mistakes a replayed call for one the agent made.
REPLAY_SPAN_TYPE = "replay_verify"

# Error types meaning the tool body never ran.
_NOT_EXECUTED = frozenset({"parsing", "approval"})
# Error types meaning the tool may have had a partial or unknown effect.
_UNKNOWN_EFFECT = frozenset({"cancelled", "sandbox_unavailable"})


@dataclass(frozen=True)
class RecordedCall:
    """One tool call as the original run recorded it."""

    id: str
    function: str
    arguments: dict[str, Any]
    result: str
    error: ToolCallError | None
    disposition: Disposition
    reason: str = ""
    nondeterministic: bool = False
    """The call timed out in the original, so its partial effect may not repeat exactly."""
    extra: dict[str, Any] = field(default_factory=dict)


def result_text(result: Any) -> str:
    """A tool result (str, number, Content or list of Content) rendered as text."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "".join(result_text(part) for part in result)
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    return str(result)


def _classify(
    function: str,
    error: ToolCallError | None,
    *,
    skip_tools: frozenset[str],
    handoff: bool,
    incomplete: bool,
) -> tuple[Disposition, str]:
    if handoff:
        return "skip", "agent handoff; the agent's own tool calls are replayed individually"
    if incomplete:
        return "unreproducible", "the original call never completed"
    if error is not None and error.type in _NOT_EXECUTED:
        return "skip", f"never executed in the original ({error.type} error)"
    if error is not None and error.type in _UNKNOWN_EFFECT:
        return "unreproducible", f"original call had an unknown effect ({error.type})"
    if function in skip_tools:
        return "skip", f"'{function}' is in skip_tools"
    return "replay", ""


def _outside_agent_spans(events: list[Any]) -> list[ToolEvent]:
    """The tool events the agent made: none from scoring, none from an earlier replay."""
    spans: dict[str, SpanBeginEvent] = {e.id: e for e in events if isinstance(e, SpanBeginEvent)}

    def excluded(span_id: str | None) -> bool:
        seen: set[str] = set()
        while span_id is not None and span_id not in seen:
            seen.add(span_id)
            begin = spans.get(span_id)
            if begin is None:
                return False
            if begin.type in ("scorer", REPLAY_SPAN_TYPE) or begin.name == "scorers":
                return True
            span_id = begin.parent_id
        return False

    return [e for e in events if isinstance(e, ToolEvent) and not excluded(e.span_id)]


def calls_from_events(
    events: Iterable[Any], *, skip_tools: Iterable[str] = DEFAULT_SKIP_TOOLS
) -> list[RecordedCall]:
    """Recorded calls, in execution order, from a sample's transcript events."""
    skip = frozenset(skip_tools)
    calls: list[RecordedCall] = []
    for event in _outside_agent_spans(list(events)):
        handoff = bool(event.agent or event.agent_span_id)
        incomplete = bool(event.pending) or bool(event.failed)
        disposition, reason = _classify(
            event.function, event.error, skip_tools=skip, handoff=handoff, incomplete=incomplete
        )
        if event.failed and not handoff:
            reason = "the original call failed with a hard error"
        calls.append(
            RecordedCall(
                id=event.id,
                function=event.function,
                arguments=dict(event.arguments),
                result=result_text(event.result),
                error=event.error,
                disposition=disposition,
                reason=reason,
                nondeterministic=event.error is not None and event.error.type == "timeout",
            )
        )
    return calls


def calls_from_messages(
    messages: Sequence[ChatMessage], *, skip_tools: Iterable[str] = DEFAULT_SKIP_TOOLS
) -> list[RecordedCall]:
    """Recorded calls from assistant `tool_calls` and their `ChatMessageTool` replies."""
    skip = frozenset(skip_tools)
    replies = {
        m.tool_call_id: m for m in messages if isinstance(m, ChatMessageTool) and m.tool_call_id
    }
    calls: list[RecordedCall] = []
    for message in messages:
        if not isinstance(message, ChatMessageAssistant):
            continue
        for call in message.tool_calls or []:
            reply = replies.get(call.id)
            error = reply.error if reply is not None else None
            if call.parse_error and error is None:
                error = ToolCallError("parsing", call.parse_error)
            disposition, reason = _classify(
                call.function,
                error,
                skip_tools=skip,
                handoff=False,
                incomplete=reply is None,
            )
            calls.append(
                RecordedCall(
                    id=call.id,
                    function=call.function,
                    arguments=dict(call.arguments),
                    result=reply.text if reply is not None else "",
                    error=error,
                    disposition=disposition,
                    reason=reason,
                    nondeterministic=error is not None and error.type == "timeout",
                )
            )
    return calls


def recorded_calls(
    *,
    events: Iterable[Any] | None = None,
    messages: Sequence[ChatMessage] | None = None,
    skip_tools: Iterable[str] = DEFAULT_SKIP_TOOLS,
) -> tuple[list[RecordedCall], Literal["events", "messages"]]:
    """The calls to replay, preferring tool events and falling back to messages."""
    from_events = calls_from_events(events or [], skip_tools=skip_tools)
    if from_events:
        return from_events, "events"
    return calls_from_messages(messages or [], skip_tools=skip_tools), "messages"
