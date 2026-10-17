"""The replay sandbox's lifecycle: resolve the pristine state, boot it, tear it down."""

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import anyio
from inspect_ai.dataset import Sample
from inspect_ai.util import SandboxEnvironment, SandboxEnvironmentSpec, SandboxEnvironmentType

from . import _inspect_compat as compat
from .report import Inconclusive, stage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True, slots=True)
class PristineState:
    """Everything needed to boot a sandbox exactly as the original sample started."""

    sandbox: SandboxEnvironmentSpec
    files: Mapping[str, bytes] = field(default_factory=dict)
    setup: bytes | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sample_id: str | int | None = None


def as_spec(sandbox: SandboxEnvironmentType | None) -> SandboxEnvironmentSpec | None:
    """A `SandboxEnvironmentSpec` from any of Inspect's sandbox shorthands."""
    if sandbox is None or isinstance(sandbox, SandboxEnvironmentSpec):
        return sandbox
    if isinstance(sandbox, str):
        return SandboxEnvironmentSpec(sandbox)
    return SandboxEnvironmentSpec(sandbox[0], sandbox[1])


async def pristine_state(
    sample: Sample, task_sandbox: SandboxEnvironmentSpec | None, task_name: str | None
) -> PristineState:
    """The sandbox spec, files and setup a sample started from, resolved as Inspect does."""
    spec = await compat.resolve_sandbox(task_sandbox, sample, task_name)
    if spec is None:
        raise Inconclusive(
            "no sandbox is known for this sample (none on the sample, and no task-level "
            "sandbox was found); pass sandbox="
        )
    files = {
        path: await compat.read_sample_file(contents)
        for path, contents in compat.resolve_sample_files(dict(sample.files or {})).items()
    }
    setup: bytes | None = None
    if sample.setup:
        setup = await compat.read_sample_file(sample.setup)
        text = setup.decode("utf-8")
        if not text.strip().startswith("#!"):
            setup = f"#!/usr/bin/env bash\n\n{text}".encode()
    return PristineState(
        sandbox=spec,
        files=files,
        setup=setup,
        metadata=dict(sample.metadata or {}),
        sample_id=sample.id,
    )


# The task-level inits this process ran, by (sandbox type, config): the task that ran
# each, and the spec to release it with.
_task_initialized: dict[tuple[str, str], tuple[str, SandboxEnvironmentSpec]] = {}


async def release_task(task_name: str) -> None:
    """Run `task_cleanup` for every sandbox type this process initialised for `task_name`."""
    for key, (owner, spec) in list(_task_initialized.items()):
        if owner == task_name:
            del _task_initialized[key]
            try:
                sandbox_cls = compat.registry_find_sandboxenv(spec.type)
                await sandbox_cls.task_cleanup(task_name, spec.config, True)
            except Exception as ex:
                logger.warning("task cleanup for %s failed: %r", spec.type, ex)


@asynccontextmanager
async def booted(
    pristine: PristineState, task_name: str
) -> AsyncIterator[dict[str, SandboxEnvironment]]:
    """A fresh instance of the sample's sandbox in its pristine state, bound as `sandbox()`.

    Inspect's own sample lifecycle does the work: `init_sandbox_environments_sample`
    boots it, copies the files and runs the setup script, and wraps each environment
    so its calls are recorded as `SandboxEvent`s; `cleanup_sandbox_environments_sample`
    tears it down however the block ends. The previous `sandbox()` binding is restored.
    """
    spec = pristine.sandbox
    try:
        sandbox_cls = compat.registry_find_sandboxenv(spec.type)
    except Exception as ex:
        raise Inconclusive(f"sandbox type '{spec.type}' is not installed: {ex}") from ex
    with compat.preserved_sandbox_binding():
        with stage("replay sandbox could not be started in its pristine state"):
            key = (spec.type, repr(spec.config))
            if key not in _task_initialized:
                await sandbox_cls.task_init(task_name, spec.config)
                _task_initialized[key] = (task_name, spec)
            environments = await compat.init_sandbox_environments_sample(
                sandboxenv_type=sandbox_cls,
                task_name=task_name,
                config=spec.config,
                files=dict(pristine.files),
                setup=pristine.setup,
                metadata={**pristine.metadata, "__sample_id__": pristine.sample_id},
            )
        interrupted = False
        try:
            yield environments
        except anyio.get_cancelled_exc_class():
            interrupted = True
            raise
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await compat.cleanup_sandbox_environments_sample(
                        spec.type, task_name, spec.config, environments, interrupted
                    )
                except Exception as ex:
                    logger.warning("replay sandbox cleanup failed: %r", ex)
