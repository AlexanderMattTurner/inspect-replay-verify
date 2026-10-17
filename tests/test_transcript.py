"""Which recorded calls are replayed, skipped, or make the replay inconclusive."""

from __future__ import annotations

from inspect_ai.event import SpanBeginEvent, ToolEvent
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import ToolCall, ToolCallError

from inspect_replay_verify.transcript import calls_from_events, calls_from_messages


def tool_event(id: str, function: str = "bash", **kwargs: object) -> ToolEvent:
    return ToolEvent(id=id, function=function, arguments={"cmd": id}, **kwargs)  # type: ignore[arg-type]


def test_dispositions_from_events() -> None:
    events = [
        tool_event("ok", result="out"),
        tool_event("parse", error=ToolCallError("parsing", "bad json")),
        tool_event("rejected", error=ToolCallError("approval", "no")),
        tool_event("slow", error=ToolCallError("timeout", "slow")),
        tool_event("handoff", function="transfer_to_x", agent="x"),
        tool_event("thought", function="think"),
        tool_event("lost", error=ToolCallError("sandbox_unavailable", "gone")),
    ]
    calls = {c.id: c for c in calls_from_events(events)}
    assert calls["ok"].disposition == "replay"
    assert calls["ok"].result == "out"
    assert calls["parse"].disposition == "skip"
    assert calls["rejected"].disposition == "skip"
    assert calls["slow"].disposition == "replay"
    assert calls["slow"].nondeterministic
    assert calls["handoff"].disposition == "skip"
    assert calls["thought"].disposition == "skip"
    assert calls["lost"].disposition == "unreproducible"


def test_pending_or_failed_calls_are_unreproducible() -> None:
    events = [tool_event("p", pending=True), tool_event("f", failed=True)]
    assert [c.disposition for c in calls_from_events(events)] == ["unreproducible"] * 2


def test_scorer_and_replay_spans_are_excluded() -> None:
    events = [
        tool_event("agent"),
        SpanBeginEvent(id="s", name="scorers"),
        SpanBeginEvent(id="r", name="replay_verify", type="replay_verify", parent_id="s"),
        tool_event("replayed", span_id="r"),
        SpanBeginEvent(id="x", name="mine", type="scorer"),
        tool_event("scorer_tool", span_id="x"),
    ]
    assert [c.id for c in calls_from_events(events)] == ["agent"]


def test_messages_fallback_marks_unanswered_calls_unreproducible() -> None:
    messages = [
        ChatMessageAssistant(
            content="",
            tool_calls=[
                ToolCall(id="a", function="bash", arguments={"cmd": "true"}),
                ToolCall(id="b", function="bash", arguments={"cmd": "true"}),
            ],
        ),
        ChatMessageTool(content="ok", tool_call_id="a", function="bash"),
    ]
    calls = calls_from_messages(messages)
    assert [(c.id, c.disposition) for c in calls] == [("a", "replay"), ("b", "unreproducible")]
    assert calls[0].result == "ok"
