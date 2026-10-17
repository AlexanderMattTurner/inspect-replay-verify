"""Scripted evals on Inspect's `local` sandbox: a mock model issues fixed tool calls."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver, generate, use_tools
from inspect_ai.tool import ToolDef, bash

MODEL = "mockllm/model"
# The bash tool's argument was renamed across Inspect releases (`cmd` -> `command`).
BASH_ARG = next(iter(ToolDef(bash()).parameters.properties))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def bash_call(command: str) -> ModelOutput:
    return ModelOutput.for_tool_call(MODEL, "bash", {BASH_ARG: command})


def run(
    commands: Sequence[str],
    scorers: Scorer | list[Scorer],
    *,
    sample: Sample | None = None,
    before: Sequence[Solver] = (),
    **task_kwargs: Any,
) -> EvalLog:
    """Run one sample whose agent issues `commands` through `bash`, then says it is done."""
    outputs = [bash_call(c) for c in commands] + [ModelOutput.from_content(MODEL, "done")]
    model = get_model(MODEL, custom_outputs=outputs)
    task = Task(
        dataset=[sample or Sample(input="do the task", target="done")],
        solver=[*before, use_tools(bash()), generate()],
        scorer=scorers,
        sandbox="local",
        **task_kwargs,
    )
    [log] = eval(task, model=model, display="none")
    assert log.status == "success", log.error
    return log


def replay_meta(log: EvalLog, scorer: str = "replay_scorer") -> dict[str, Any]:
    assert log.samples
    score = (log.samples[0].scores or {})[scorer]
    return dict((score.metadata or {})["replay"])


def value(log: EvalLog, scorer: str = "replay_scorer") -> Any:
    assert log.samples
    return (log.samples[0].scores or {})[scorer].value
