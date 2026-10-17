"""The tools a replay re-issues recorded calls through, each a fresh instance."""

from collections.abc import Sequence
from typing import cast

from inspect_ai.tool import Tool, ToolDef, bash, bash_session, python, text_editor

from . import _inspect_compat as compat


def _fresh(tool: Tool) -> Tool:
    """A new instance from the tool's registry params, so stateful tools (`bash_session`,
    `text_editor`) never reach for the original sandbox's session."""
    try:
        info = compat.registry_info(tool)
        return cast(Tool, compat.registry_create("tool", info.name, **compat.registry_params(tool)))
    except Exception:
        return tool


def replay_tools(
    provided: Sequence[Tool], task_tools: Sequence[Tool]
) -> tuple[dict[str, Tool], dict[str, str]]:
    """Tools by name, and where each came from: ``provided``, ``task``, or ``default``.

    Later sources win: Inspect's standard sandbox tools with default parameters, then
    the task's own (`state.tools`), then the ones passed explicitly.
    """
    tools: dict[str, Tool] = {}
    sources: dict[str, str] = {}
    defaults = [bash(), python(), text_editor(), bash_session()]
    for source, group in (("default", defaults), ("task", task_tools), ("provided", provided)):
        for tool in group:
            fresh = _fresh(tool)
            name = ToolDef(fresh).name
            tools[name], sources[name] = fresh, source
    return tools, sources
