"""Which recorded actions are replayed, skipped, or make the replay inconclusive."""

from datetime import UTC, datetime, timedelta

import pytest
from inspect_ai.event import ModelEvent, SandboxEvent, SpanBeginEvent, ToolEvent
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolCall, ToolCallError

from inspect_replay_verify.transcript import (
    ERROR_DISPOSITIONS,
    ERROR_TYPES,
    calls_from_events,
    calls_from_messages,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def tool_event(id: str, function: str = "bash", **kwargs: object) -> ToolEvent:
    return ToolEvent(id=id, function=function, arguments={"cmd": id}, **kwargs)  # type: ignore[arg-type]


def sandbox_event(action: str, *, span_id: str | None = None, **kwargs: object) -> SandboxEvent:
    return SandboxEvent(
        action=action,  # type: ignore[arg-type]
        span_id=span_id,
        timestamp=kwargs.pop("timestamp", T0),  # type: ignore[arg-type]
        completed=kwargs.pop("completed", T0 + timedelta(seconds=1)),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def disposition_of(error_type: str) -> str:
    [call] = calls_from_events([tool_event("c", error=ToolCallError(error_type, "x"))])  # type: ignore[arg-type]
    return call.disposition


def test_every_inspect_error_type_is_classified() -> None:
    """Inspect owns the error vocabulary; older releases name fewer of the types we know."""
    assert ERROR_TYPES, "read no error types from ToolCallError"
    assert set(ERROR_TYPES) <= set(ERROR_DISPOSITIONS)


def test_an_unclassified_error_type_is_unreproducible() -> None:
    assert disposition_of("a_type_from_a_newer_inspect") == "unreproducible"


@pytest.mark.parametrize(
    ("error_type", "disposition"),
    [
        ("parsing", "skip"),
        ("approval", "skip"),
        ("cancelled", "unreproducible"),
        ("sandbox_unavailable", "unreproducible"),
        ("timeout", "replay"),
        ("permission", "replay"),
    ],
)
def test_error_dispositions(error_type: str, disposition: str) -> None:
    assert disposition_of(error_type) == disposition


@pytest.mark.parametrize(
    ("event", "disposition"),
    [
        (tool_event("handoff", function="transfer_to_x", agent="x"), "skip"),
        (tool_event("thought", function="think"), "skip"),
        (tool_event("pending", pending=True), "unreproducible"),
        (tool_event("failed", failed=True), "unreproducible"),
        (tool_event("ok", result="out"), "replay"),
    ],
)
def test_tool_event_dispositions(event: ToolEvent, disposition: str) -> None:
    [call] = calls_from_events([event])
    assert call.disposition == disposition


def test_a_timed_out_call_is_marked_nondeterministic() -> None:
    [call] = calls_from_events([tool_event("slow", error=ToolCallError("timeout", "slow"))])
    assert (call.disposition, call.nondeterministic) == ("replay", True)


def test_scoring_and_replay_spans_are_excluded() -> None:
    events = [
        tool_event("agent"),
        SpanBeginEvent(id="s", name="scorers"),
        SpanBeginEvent(id="r", name="replay_verify", type="replay_verify"),
        tool_event("replayed", span_id="r"),
        SpanBeginEvent(id="x", name="mine", type="scorer"),
        tool_event("scorer_tool", span_id="x"),
        sandbox_event("exec", cmd="true", span_id="x"),
    ]
    assert [c.id for c in calls_from_events(events)] == ["agent"]


def test_only_sandbox_actions_outside_tools_and_init_are_untranscribed() -> None:
    events = [
        SpanBeginEvent(id="i", name="init", type="init"),
        sandbox_event("write_file", file="seed.txt", input="seed", span_id="i"),
        SpanBeginEvent(id="t", name="bash", type="tool"),
        sandbox_event("exec", cmd="bash --login -c ls", span_id="t"),
        sandbox_event("read_file", file="anything"),
        sandbox_event("exec", cmd="git apply patch", result=0),
        sandbox_event("write_file", file="notes.txt", input="hi"),
    ]
    calls = calls_from_events(events, untranscribed="replay")
    assert [(c.kind, c.disposition) for c in calls] == [
        ("exec", "replay"),
        ("write_file", "replay"),
    ]
    assert calls[0].arguments["cmd"] == "git apply patch"
    assert [c.disposition for c in calls_from_events(events)] == ["unreproducible"] * 2
    assert calls_from_events(events, untranscribed="ignore") == []


@pytest.mark.parametrize(
    "event",
    [
        sandbox_event("exec", cmd="cat", input="a\n\nOutput truncated (30 additional lines)"),
        sandbox_event("write_file", file="blob", input="binary (2.00 KB)"),
        sandbox_event("exec", cmd="sleep 60", pending=True),
    ],
)
def test_sandbox_actions_the_transcript_cannot_reproduce(event: SandboxEvent) -> None:
    [call] = calls_from_events([event], untranscribed="replay")
    assert call.disposition == "unreproducible"


def test_a_command_a_model_call_ran_inside_is_unreproducible() -> None:
    model = ModelEvent(
        model="m",
        input=[],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=ModelOutput(),
        timestamp=T0 + timedelta(milliseconds=500),
    )
    events = [sandbox_event("exec", cmd="claude -p go"), model]
    [call] = calls_from_events(events, untranscribed="replay")
    assert call.disposition == "unreproducible"
    assert "model call ran during" in call.reason


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
    assert [(c.id, c.disposition, c.result) for c in calls] == [
        ("a", "replay", "ok"),
        ("b", "unreproducible", ""),
    ]
