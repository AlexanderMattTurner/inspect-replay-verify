"""A task file `inspect eval` loads in a fresh process, naming the scorer like any user."""

import hashlib

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import ToolDef, bash

from inspect_replay_verify import file_hash, replay_scorer

ARG = next(iter(ToolDef(bash()).parameters.properties))


@task
def cli_task() -> Task:
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call("mockllm/model", "bash", {ARG: "printf hello > out.txt"}),
            ModelOutput.from_content("mockllm/model", "done"),
        ],
    )
    return Task(
        dataset=[Sample(input="write hello", target="done")],
        solver=[use_tools(bash()), generate()],
        scorer=replay_scorer([file_hash("out.txt", hashlib.sha256(b"hello").hexdigest())]),
        sandbox="local",
        model=model,
    )
