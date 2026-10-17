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
init_sandbox_environments_sample = _require(
    "inspect_ai.util._sandbox.context", "init_sandbox_environments_sample"
)
cleanup_sandbox_environments_sample = _require(
    "inspect_ai.util._sandbox.context", "cleanup_sandbox_environments_sample"
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


def registry_name(obj: Any) -> str | None:
    """An Inspect registry object's unqualified name (a scorer's, a tool's), if it has one."""
    try:
        return cast(str, registry_unqualified_name(registry_info(obj).name))
    except Exception:
        return None


def unproxy(env: Any) -> Any:
    """The provider's own environment behind Inspect's `SandboxEnvironmentProxy`, if any."""
    return getattr(env, "_sandbox", env)


def current_environments() -> dict[str, Any]:
    """The running sample's own sandbox environments, or `{}` outside a sample."""
    return dict(_environments_var.get(None) or {})


@contextmanager
def preserved_sandbox_binding() -> Iterator[None]:
    """Restore what `sandbox()` resolves to after the block, whatever rebinds it inside.

    `init_sandbox_environments_sample` binds the environments it creates for the rest of
    the context; this puts the original sample's own binding back for later scorers.
    """
    variables = (_environments_var, _with_environments_var, _default_var)
    saved = [(var, var.get(None)) for var in variables]
    try:
        yield
    finally:
        for var, value in saved:
            var.set(value)


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
