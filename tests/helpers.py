"""Scripted evals: a mock model issues fixed tool calls against a real Inspect sandbox."""

import hashlib
from collections.abc import Sequence
from typing import Any

import anyio
from inspect_ai import Task, eval
from inspect_ai.approval import Approval, ApprovalPolicy, Approver, approver
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessage, ModelOutput, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, use_tools
from inspect_ai.tool import Tool, ToolCall, ToolCallView, ToolDef, ToolError, bash, tool
from inspect_ai.util import SandboxEnvironmentType, sandbox

from inspect_replay_verify import CheckOutcome, EvidenceReader

MODEL = "mockllm/model"
# The bash tool's argument was renamed across Inspect releases (`cmd` -> `command`).
BASH_ARG = next(iter(ToolDef(bash()).parameters.properties))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def bash_call(command: str) -> ModelOutput:
    return ModelOutput.for_tool_call(MODEL, "bash", {BASH_ARG: command})


def run(
    commands: Sequence[str | ModelOutput],
    scorers: Scorer | list[Scorer],
    *,
    sample: Sample | None = None,
    before: Sequence[Solver] = (),
    sandbox: SandboxEnvironmentType = "local",
    tools: Sequence[Tool] = (),
    approval: list[ApprovalPolicy] | None = None,
) -> EvalLog:
    """Run one sample whose agent issues each command, then says it is done.

    A string is a `bash` call; a `ModelOutput` is any other scripted turn.
    """
    outputs = [bash_call(c) if isinstance(c, str) else c for c in commands]
    outputs.append(ModelOutput.from_content(MODEL, "done"))
    task = Task(
        dataset=[sample or Sample(input="do the task", target="done")],
        solver=[*before, use_tools(bash(), *tools), generate()],
        scorer=scorers,
        sandbox=sandbox,
        approval=approval,
    )
    [log] = eval(task, model=get_model(MODEL, custom_outputs=outputs), display="none")
    assert log.status == "success", log.error
    return log


def replay_meta(log: EvalLog, name: str = "replay_scorer") -> dict[str, Any]:
    assert log.samples
    score = (log.samples[0].scores or {})[name]
    return dict((score.metadata or {})["replay"])


def value(log: EvalLog, name: str = "replay_scorer") -> object:
    assert log.samples
    return (log.samples[0].scores or {})[name].value


@solver
def write_out_of_band(path: str, contents: str) -> Solver:
    """Writes a file with no tool call: an effect only a `SandboxEvent` records."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        await sandbox().write_file(path, contents)
        return state

    return solve


@solver
def exec_setup(command: str) -> Solver:
    """A harness setup step run with `sandbox().exec`, as SWE-style tasks do."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        result = await sandbox().exec(["bash", "-c", command])
        assert result.success, result.stderr
        return state

    return solve


@solver
def agent_inside_sandbox() -> Solver:
    """A sandbox command that runs while the model is called: an in-sandbox agent's shape."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        async with anyio.create_task_group() as tg:
            tg.start_soon(sandbox().exec, ["bash", "-c", "sleep 1; printf agent > agent.txt"])
            await anyio.sleep(0.2)
            await get_model("mockllm/in-sandbox-agent").generate("hello")
        return state

    return solve


@scorer(metrics=[])
def file_is(path: str, contents: str) -> Scorer:
    """The naive grader: reads the sandbox the agent itself ran in."""

    async def score(state: TaskState, target: Target) -> Score:
        try:
            text = await sandbox().read_file(path)
        except FileNotFoundError:
            return Score(value=INCORRECT)
        return Score(value=CORRECT if text == contents else INCORRECT)

    return score


@tool
def once(marker: str) -> Tool:
    """Succeeds the first time it runs on this host, and raises every time after."""

    async def execute(path: str) -> str:
        """Write `path`, once.

        Args:
          path: File to write.
        """
        probe = await sandbox().exec(["bash", "-c", f"test -e {marker}"])
        if probe.success:
            raise ToolError("already ran")
        await sandbox().exec(["bash", "-c", f"touch {marker}; printf hi > {path}"])
        return "wrote"

    return execute


@approver
def reject_marked() -> Approver:
    """Rejects any call whose arguments mention REJECT, and approves the rest."""

    async def approve(
        message: str, call: ToolCall, view: ToolCallView, history: list[ChatMessage]
    ) -> Approval:
        rejected = "REJECT" in repr(call.arguments)
        return Approval(decision="reject" if rejected else "approve")

    return approve


# A writer the replay leaves running: it appends to ticks.txt every 0.1s until stopped.
TICKER = "nohup sh -c 'while :; do echo t >> ticks.txt; sleep 0.1; done' >/dev/null 2>&1 &"


async def read_ticks_twice(reader: EvidenceReader) -> CheckOutcome:
    """Passes only when ticks.txt holds still across two reads two seconds apart."""
    first = await reader.read_file("ticks.txt")
    await anyio.sleep(2)
    second = await reader.read_file("ticks.txt")
    return CheckOutcome(
        name="ticks stable", passed=first == second, observed={"bytes": [len(first), len(second)]}
    )
