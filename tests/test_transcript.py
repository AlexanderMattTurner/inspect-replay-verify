"""Which recorded actions are replayed, skipped, or make the replay inconclusive."""

from datetime import UTC, datetime, timedelta

import pytest
from inspect_ai.event import ModelEvent, SandboxEvent, SpanBeginEvent, ToolEvent
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.tool import ToolCall, ToolCallError

from inspect_replay_verify import replay_scorer
from inspect_replay_verify.transcript import (
    ERROR_DISPOSITIONS,
    ERROR_TYPES,
    calls_from_events,
    calls_from_messages,
    recorded_calls,
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
        sandbox_event("exec", cmd="cat", input="binary (3 B)"),
        sandbox_event("write_file", file="long", input="x" * 10_000),
    ],
    ids=["truncated-lines", "binary-write", "binary-stdin", "cut-at-10k-chars"],
)
def test_sandbox_actions_the_transcript_cannot_reproduce(event: SandboxEvent) -> None:
    [call] = calls_from_events([event], untranscribed="replay")
    assert call.disposition == "unreproducible"


def test_a_command_a_model_call_ran_inside_is_unreproducible() -> None:
    events = [
        sandbox_event("exec", cmd="claude -p go"),
        model_event(T0 + timedelta(milliseconds=500)),
    ]
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


def span(id: str, type: str, parent: str | None = None, name: str = "s") -> SpanBeginEvent:
    return SpanBeginEvent(id=id, name=name, type=type, parent_id=parent)


@pytest.mark.parametrize(
    ("spans", "untranscribed"),
    [
        ([span("t", "tool")], False),
        ([span("a", "agent"), span("t", "tool", "a")], False),
        ([span("h", "handoff"), span("t", "tool", "h"), span("a", "agent", "t")], True),
        ([span("t", "tool"), span("a", "agent", "t")], True),
        ([span("x", "solver", name="init")], True),
        ([span("x", "solver", name="tool")], True),
    ],
    ids=[
        "tool-body",
        "tool-in-agent",
        "handoff-agent",
        "as-tool-agent",
        "named-init",
        "named-tool",
    ],
)
def test_which_span_owns_a_sandbox_action(spans: list[SpanBeginEvent], untranscribed: bool) -> None:
    events = [*spans, sandbox_event("exec", cmd="make", span_id=spans[-1].id)]
    assert len(calls_from_events(events, untranscribed="replay")) == int(untranscribed)


def model_event(at: datetime = T0) -> ModelEvent:
    return ModelEvent(
        model="m",
        input=[],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=ModelOutput(),
        timestamp=at,
    )


def bash_turn(source: str | None = None) -> list[ChatMessage]:
    return [
        ChatMessageAssistant(
            content="",
            source=source,  # type: ignore[arg-type]
            tool_calls=[ToolCall(id="a", function="bash", arguments={"cmd": "echo pass > out"})],
        ),
        ChatMessageTool(content="", tool_call_id="a", function="bash"),
    ]


def test_messages_stand_in_only_for_a_transcript_with_no_run() -> None:
    init_only = [span("i", "init"), sandbox_event("write_file", file="s", input="s", span_id="i")]
    calls, source = recorded_calls(events=init_only, messages=bash_turn())
    assert ([c.id for c in calls], source) == (["a"], "messages")


def test_a_run_with_no_tool_calls_never_replays_message_calls() -> None:
    calls, source = recorded_calls(events=[model_event()], messages=bash_turn())
    assert (calls, source) == ([], "events")


def test_an_in_sandbox_agent_ignored_leaves_nothing_to_replay() -> None:
    events = [model_event(), span("a", "agent"), sandbox_event("exec", cmd="make", span_id="a")]
    calls, source = recorded_calls(events=events, messages=bash_turn(), untranscribed="ignore")
    assert (calls, source) == ([], "events")


def test_prefilled_input_calls_never_ran() -> None:
    [call] = calls_from_messages(bash_turn(source="input"))
    assert call.disposition == "skip"
    [call] = calls_from_messages(bash_turn(source="generate"))
    assert call.disposition == "replay"


@pytest.mark.parametrize("policy", ["skip", "Replay", ""])
def test_an_unknown_untranscribed_policy_is_refused(policy: str) -> None:
    with pytest.raises(ValueError, match="untranscribed is one of"):
        calls_from_events([], untranscribed=policy)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="untranscribed is one of"):
        replay_scorer([], untranscribed=policy)  # type: ignore[arg-type]


def test_the_store_backed_memory_tool_is_not_replayed() -> None:
    [call] = calls_from_events([tool_event("m", function="memory")])
    assert call.disposition == "skip"
