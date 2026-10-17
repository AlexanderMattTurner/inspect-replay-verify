"""Turn a sample's transcript into the ordered list of actions a replay re-issues.

Tool calls come from the sample's `ToolEvent`s, which include calls made inside
sub-agents; a transcript with none falls back to the assistant messages. Sandbox actions
no tool call made (a solver's `sandbox().exec`, an agent running inside the sandbox) come
from its `SandboxEvent`s, and `untranscribed` decides what becomes of them.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, get_args

from inspect_ai.event import ModelEvent, SandboxEvent, SpanBeginEvent, ToolEvent
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import ToolCallError

Disposition = Literal["replay", "skip", "unreproducible"]
"""``replay`` re-issues it; ``skip`` leaves out what never ran or never touched the
sandbox; ``unreproducible`` stops the replay, which is then inconclusive."""

Kind = Literal["tool", "exec", "write_file"]
Untranscribed = Literal["inconclusive", "replay", "ignore"]

# `memory` is backed by the sample's store, not its sandbox.
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
UNTRANSCRIBED_POLICIES: tuple[str, ...] = get_args(Untranscribed)


def validate_untranscribed(policy: str) -> None:
    if policy not in UNTRANSCRIBED_POLICIES:
        raise ValueError(f"untranscribed is one of {UNTRANSCRIBED_POLICIES}, not {policy!r}")


# The span this package's replays run under, so no later read mistakes one for the agent.
REPLAY_SPAN_TYPE = "replay_verify"

# What each `ToolCallError.type` says about the original call's effect. A type missing
# here (a newer Inspect) is unreproducible: an effect nobody classified is unknown.
ERROR_DISPOSITIONS: dict[str, tuple[Disposition, str]] = {
    "parsing": ("skip", "never executed in the original (parsing error)"),
    "approval": ("skip", "never executed in the original (approval error)"),
    "cancelled": ("unreproducible", "original call was cancelled mid-flight"),
    "sandbox_unavailable": ("unreproducible", "original call lost its sandbox"),
    "timeout": ("replay", ""),
    "unicode_decode": ("replay", ""),
    "permission": ("replay", ""),
    "file_not_found": ("replay", ""),
    "is_a_directory": ("replay", ""),
    "limit": ("replay", ""),
    "output_limit": ("replay", ""),
    "unknown": ("replay", ""),
}

ERROR_TYPES: tuple[str, ...] = get_args(ToolCallError.__annotations__["type"])

# How `SandboxEvent` records content (inspect_ai.util._sandbox.events.content_display):
# text is cut to 10,000 characters with no marker, then to 20 lines with one, and
# bytes become a size. A recorded input this long, marked, or sized is not the input.
_RECORDED_CHARS = 100 * 100
_TRUNCATED = "\n\nOutput truncated ("
_BINARY = "binary ("


@dataclass(frozen=True, kw_only=True, slots=True)
class RecordedCall:
    """One action as the original run recorded it."""

    id: str
    function: str
    arguments: dict[str, Any]
    result: str
    error: ToolCallError | None
    disposition: Disposition
    reason: str = ""
    kind: Kind = "tool"
    returncode: int | None = None
    """An `exec`'s recorded exit status."""
    nondeterministic: bool = False
    """The original timed out, so its partial effect may not repeat exactly."""


def result_text(result: object) -> str:
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
    if error is not None:
        disposition, reason = ERROR_DISPOSITIONS.get(
            error.type, ("unreproducible", f"unclassified error type '{error.type}'")
        )
        if disposition != "replay":
            return disposition, reason
    if function in skip_tools:
        return "skip", f"'{function}' is in skip_tools"
    return "replay", ""


_SCORING = frozenset({"scorer", REPLAY_SPAN_TYPE})


class _Spans:
    """Which events the agent caused: none from init, scoring, a tool body or a replay."""

    def __init__(self, events: Sequence[object]) -> None:
        self._spans = {e.id: e for e in events if isinstance(e, SpanBeginEvent)}

    def _ancestry(self, span_id: str | None) -> list[SpanBeginEvent]:
        chain: list[SpanBeginEvent] = []
        while span_id is not None and (begin := self._spans.get(span_id)) is not None:
            if begin in chain:
                break
            chain.append(begin)
            span_id = begin.parent_id
        return chain

    def outside_scoring(self, span_id: str | None) -> bool:
        return not any(b.type in _SCORING or b.name == "scorers" for b in self._ancestry(span_id))

    def untranscribed(self, span_id: str | None) -> bool:
        """Not scoring, not sample init, and not inside a tool body the replay re-runs.

        The nearest tool or agent span decides: an ordinary tool's body is replayed with
        its call, while an agent's own sandbox calls (a handoff or as-tool agent runs
        inside an `agent` span) are not.
        """
        chain = self._ancestry(span_id)
        if not self.outside_scoring(span_id) or any(b.type == "init" for b in chain):
            return False
        return next((b.type for b in chain if b.type in ("tool", "agent")), None) != "tool"


def _tool_call(event: ToolEvent, skip: frozenset[str]) -> RecordedCall:
    handoff = bool(event.agent or event.agent_span_id)
    disposition, reason = _classify(
        event.function,
        event.error,
        skip_tools=skip,
        handoff=handoff,
        incomplete=bool(event.pending) or bool(event.failed),
    )
    if event.failed and not handoff:
        reason = "the original call failed with a hard error"
    return RecordedCall(
        id=event.id,
        function=event.function,
        arguments=dict(event.arguments),
        result=result_text(event.result),
        error=event.error,
        disposition=disposition,
        reason=reason,
        nondeterministic=event.error is not None and event.error.type == "timeout",
    )


def _overlaps_model(event: SandboxEvent, model_starts: Sequence[datetime]) -> bool:
    if event.completed is None:
        return False
    return any(event.timestamp <= t <= event.completed for t in model_starts)


def _sandbox_action(
    event: SandboxEvent, index: int, policy: Untranscribed, model_starts: Sequence[datetime]
) -> RecordedCall | None:
    if event.action == "read_file" or policy == "ignore":
        return None
    what = f"{event.action} {event.cmd or event.file}"
    content = event.input or ""
    disposition: Disposition = "replay"
    reason = ""
    if policy == "inconclusive":
        disposition = "unreproducible"
        reason = (
            f"sandbox action outside any tool call ({what}); pass untranscribed='replay' "
            "to re-issue such actions or 'ignore' to leave them out"
        )
    elif _overlaps_model(event, model_starts):
        disposition = "unreproducible"
        reason = (
            f"a model call ran during {what}: an agent in the sandbox, not a replayable command"
        )
    elif _TRUNCATED in content or content.startswith(_BINARY) or len(content) >= _RECORDED_CHARS:
        disposition, reason = "unreproducible", f"the transcript kept only part of {what}'s input"
    options = dict(event.options or {})
    arguments: dict[str, Any] = (
        {"cmd": event.cmd, "input": event.input, **options}
        if event.action == "exec"
        else {"file": event.file, "contents": event.input}
    )
    return RecordedCall(
        id=f"sandbox-{index}",
        function=f"sandbox.{event.action}",
        arguments=arguments,
        result=event.output or "",
        error=None,
        disposition=disposition,
        reason=reason,
        kind=event.action,
        returncode=event.result,
    )


def calls_from_events(
    events: Iterable[object],
    *,
    skip_tools: Iterable[str] = DEFAULT_SKIP_TOOLS,
    untranscribed: Untranscribed = "inconclusive",
) -> list[RecordedCall]:
    """Recorded actions, in transcript order, from a sample's events."""
    validate_untranscribed(untranscribed)
    events = list(events)
    spans = _Spans(events)
    skip = frozenset(skip_tools)
    model_starts = [e.timestamp for e in events if isinstance(e, ModelEvent)]
    calls: list[RecordedCall] = []
    for index, event in enumerate(events):
        if isinstance(event, ToolEvent) and spans.outside_scoring(event.span_id):
            calls.append(_tool_call(event, skip))
        elif isinstance(event, SandboxEvent) and spans.untranscribed(event.span_id):
            action = _sandbox_action(event, index, untranscribed, model_starts)
            if action is not None:
                calls.append(action)
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
        prefilled = message.source == "input"
        for call in message.tool_calls or []:
            reply = replies.get(call.id)
            error = reply.error if reply is not None else None
            if call.parse_error and error is None:
                error = ToolCallError("parsing", call.parse_error)
            disposition, reason = _classify(
                call.function, error, skip_tools=skip, handoff=False, incomplete=reply is None
            )
            if prefilled:
                disposition, reason = "skip", "prefilled in the sample's input; it never ran"
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
    events: Iterable[object] | None = None,
    messages: Sequence[ChatMessage] | None = None,
    skip_tools: Iterable[str] = DEFAULT_SKIP_TOOLS,
    untranscribed: Untranscribed = "inconclusive",
) -> tuple[list[RecordedCall], Literal["events", "messages"]]:
    """The actions to replay, from events whenever they record the agent's run.

    Messages stand in only for a transcript with no tool or model event outside scoring
    (a condensed log): there, an empty event list is missing data, not an idle agent.
    """
    events = list(events or [])
    spans = _Spans(events)
    ran = any(
        isinstance(e, ToolEvent | ModelEvent) and spans.outside_scoring(e.span_id) for e in events
    )
    if ran:
        return calls_from_events(
            events, skip_tools=skip_tools, untranscribed=untranscribed
        ), "events"
    return calls_from_messages(messages or [], skip_tools=skip_tools), "messages"
