"""Every private inspect_ai symbol this package touches, in one place.

Inspect has no public API for "boot one more instance of a sandbox type, outside the
sample's own lifecycle", for "run these tools against that instance", or for "the pristine
`Sample` the running sample started from". Those are exactly what an independent replay
needs, so this module reaches for the private seams and fails at import, naming the symbol,
if a release moves one. Nothing else in the package imports from `inspect_ai._*` or
`inspect_ai.*._*`.
"""

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast


def _require(module: str, name: str) -> Any:
    try:
        return getattr(importlib.import_module(module), name)
    except (
        ImportError,
        AttributeError,
    ) as ex:  # pragma: no cover - only on an incompatible release
        raise ImportError(
            f"inspect-replay-verify needs `{module}.{name}`, which this inspect_ai release "
            "does not provide. Pin inspect-ai to a release this package supports, or "
            "upgrade inspect-replay-verify."
        ) from ex


registry_find_sandboxenv = _require("inspect_ai.util._sandbox.registry", "registry_find_sandboxenv")
copy_sandbox_environment_files = _require(
    "inspect_ai.util._sandbox.context", "copy_sandbox_environment_files"
)
setup_sandbox_environment = _require(
    "inspect_ai.util._sandbox.context", "setup_sandbox_environment"
)
_environments_var = _require("inspect_ai.util._sandbox.context", "sandbox_environments_context_var")
_with_environments_var = _require(
    "inspect_ai.util._sandbox.context", "sandbox_with_environments_context_var"
)
_default_var = _require("inspect_ai.util._sandbox.context", "sandbox_default_context_var")
registry_info = _require("inspect_ai._util.registry", "registry_info")
registry_params = _require("inspect_ai._util.registry", "registry_params")
registry_create = _require("inspect_ai._util.registry", "registry_create")
registry_unqualified_name = _require("inspect_ai._util.registry", "registry_unqualified_name")
_read_sandboxenv_file = _require("inspect_ai._eval.task.sandbox", "read_sandboxenv_file")
_resolve_sample_files = _require("inspect_ai._eval.task.sandbox", "resolve_sample_files")
_sample_active = _require("inspect_ai.log._samples", "sample_active")


def unproxy(env: Any) -> Any:
    """The provider's own environment behind Inspect's `SandboxEnvironmentProxy`, if any."""
    return getattr(env, "_sandbox", env)


def current_environments() -> dict[str, Any]:
    """The running sample's own sandbox environments, or `{}` outside a sample."""
    return dict(_environments_var.get(None) or {})


@contextmanager
def bound_environments(environments: dict[str, Any]) -> Iterator[None]:
    """Make `sandbox()` resolve to `environments` for the duration of the block.

    The first key is the default, matching Inspect's own `sample_init` contract. The
    previous binding is restored on exit, so the original sample's tools and any later
    scorer see their own sandbox again.
    """
    tokens = (
        _environments_var.set(environments),
        _with_environments_var.set({}),
        _default_var.set(next(iter(environments))),
    )
    try:
        yield
    finally:
        _default_var.reset(tokens[2])
        _with_environments_var.reset(tokens[1])
        _environments_var.reset(tokens[0])


async def resolve_root_access(environment: Any) -> None:
    """Run Inspect's post-setup root-access probe when this release has one."""
    try:
        module = importlib.import_module("inspect_ai.tool._sandbox_tools_utils.sandbox")
    except ImportError:  # pragma: no cover - older releases have no probe
        return
    probe = getattr(module, "resolve_root_access", None)
    if probe is not None:
        await probe(environment)


def active_sample() -> Any | None:
    """The pristine `Sample` the running sample was built from, or None outside an eval."""
    active = _sample_active()
    return getattr(active, "sample", None) if active is not None else None


def active_task_name() -> str | None:
    """The running sample's task name, or None outside an eval."""
    active = _sample_active()
    name = getattr(active, "task", None) if active is not None else None
    return cast(str, registry_unqualified_name(name)) if isinstance(name, str) else None


def active_log_location() -> str | None:
    """Where the running eval is writing its log, or None outside an eval."""
    active = _sample_active()
    location = getattr(active, "log_location", None) if active is not None else None
    return location if isinstance(location, str) and location else None


async def read_sample_file(contents: str) -> bytes:
    """Resolve one `Sample.files` / `Sample.setup` value exactly as Inspect does."""
    return cast(bytes, await _read_sandboxenv_file(contents))


def resolve_sample_files(files: dict[str, str]) -> dict[str, str]:
    """Expand directory entries in `Sample.files` exactly as Inspect does."""
    return cast(dict[str, str], _resolve_sample_files(files))


_resolve_sandbox = _require("inspect_ai._eval.task.sandbox", "resolve_sandbox")


async def resolve_sandbox(task_sandbox: Any, sample: Any, task_name: str | None) -> Any:
    """The sandbox a sample actually ran in, layering `Sample.sandbox` over the task's."""
    return await _resolve_sandbox(task_sandbox, sample, task_name)
