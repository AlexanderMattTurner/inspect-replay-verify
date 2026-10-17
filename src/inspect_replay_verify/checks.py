"""Ground-truth checks run against the replay sandbox once every recorded call is re-issued.

A check observes an objective effect — an exit code, a file's bytes, a test run — and
answers pass, fail, or "could not tell". "Could not tell" (`passed=None`) is never a pass:
it makes the whole verdict inconclusive.

Checks come in two phases, run in this order:

- ``live`` checks see the running sandboxes (`Sandboxes`): run the tests, run a probe.
- ``evidence`` checks only read state through an `EvidenceReader`, which the adapter may
  hand out only after freezing the sandbox, so nothing the replay left running can write
  to the evidence while it is read.
"""

import hashlib
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, get_args

from inspect_ai.scorer import NOANSWER, Score, Scorer, Target, value_to_float
from inspect_ai.solver import TaskState
from inspect_ai.util import SandboxEnvironment

from . import _inspect_compat as compat

Phase = Literal["live", "evidence"]
PHASES: tuple[str, ...] = get_args(Phase)


@dataclass(frozen=True, kw_only=True, slots=True)
class CheckOutcome:
    """What one check observed in the replay sandbox."""

    name: str
    passed: bool | None
    """True/False on a definite observation; None when the check could not be evaluated."""
    explanation: str = ""
    observed: dict[str, Any] = field(default_factory=dict)


class Sandboxes:
    """The replay's running sandboxes, by name (the first is the default), and the sample
    they replay, for checks that grade like a scorer."""

    def __init__(
        self,
        environments: Mapping[str, SandboxEnvironment],
        *,
        state: TaskState | None = None,
        target: Target | None = None,
    ) -> None:
        self._environments = environments
        self.state = state
        self.target = target

    def environment(self, name: str | None = None) -> SandboxEnvironment:
        if name is None or name == "default":
            return next(iter(self._environments.values()))
        return self._environments[name]


class EvidenceReader(Protocol):
    """Read-only access to a (possibly frozen) replay sandbox."""

    async def read_file(self, path: str, *, sandbox: str | None = None) -> bytes:
        """The file's bytes. Raises `FileNotFoundError` when it does not exist."""
        ...


Observe = Callable[[Any], Awaitable["CheckOutcome | bool"]]
"""Receives `Sandboxes` (phase ``live``) or an `EvidenceReader` (phase ``evidence``)."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Check:
    """A ground-truth observation of the replay sandbox."""

    name: str
    phase: Phase
    observe: Observe

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"a check's phase is one of {PHASES}, not {self.phase!r}")

    async def run(self, target: Any) -> CheckOutcome:
        """The observation; a raise is an inconclusive outcome, never a pass."""
        try:
            result = await self.observe(target)
        except Exception as ex:
            return CheckOutcome(
                name=self.name, passed=None, explanation=f"check raised {type(ex).__name__}: {ex}"
            )
        if isinstance(result, CheckOutcome):
            return result
        return CheckOutcome(name=self.name, passed=bool(result))


def custom_check(fn: Observe, *, phase: Phase = "live", name: str | None = None) -> Check:
    """Wrap an async function as a check. It returns a bool or a `CheckOutcome`."""
    return Check(name=name or str(getattr(fn, "__name__", "custom")), phase=phase, observe=fn)


def exit_code(
    cmd: str | Sequence[str],
    expected: int = 0,
    *,
    name: str | None = None,
    sandbox: str | None = None,
    user: str | None = None,
    cwd: str | None = None,
    timeout: int | None = 600,
    restore_files: Mapping[str, str | bytes] | None = None,
) -> Check:
    """Pass when `cmd` exits with `expected` in the replay sandbox.

    Args:
      cmd: A shell string (run under `bash -c`) or an argv.
      expected: The exit code that counts as a pass.
      name: Label in the score's metadata (defaults to the command).
      sandbox: Which named sandbox to run in (default: the default sandbox).
      user: User to run as.
      cwd: Working directory.
      timeout: Seconds before the check counts as inconclusive.
      restore_files: Pristine copies to write over the replay's files before running, so
        an agent that rewrote the tests cannot pass by having rewritten them.
    """
    argv = ["bash", "-c", cmd] if isinstance(cmd, str) else list(cmd)
    label = name or (cmd if isinstance(cmd, str) else " ".join(cmd))
    restore = dict(restore_files or {})

    async def observe(target: Sandboxes) -> CheckOutcome:
        env = target.environment(sandbox)
        for path, contents in restore.items():
            await env.write_file(path, contents)
        try:
            result = await env.exec(argv, user=user, cwd=cwd, timeout=timeout)
        except TimeoutError:
            return CheckOutcome(name=label, passed=None, explanation=f"timed out after {timeout}s")
        return CheckOutcome(
            name=label,
            passed=result.returncode == expected,
            explanation=f"exit {result.returncode}, expected {expected}",
            observed={
                "returncode": result.returncode,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
            },
        )

    return Check(name=label, phase="live", observe=observe)


def command_succeeds(cmd: str | Sequence[str], **kwargs: Any) -> Check:
    """Pass when `cmd` exits 0. Takes `exit_code`'s keyword arguments."""
    return exit_code(cmd, 0, **kwargs)


def tests_pass(cmd: str | Sequence[str] = "python -m pytest -q", **kwargs: Any) -> Check:
    """Pass when the test command exits 0. Pair with `restore_files` for the test files."""
    kwargs.setdefault("name", "tests")
    return exit_code(cmd, 0, **kwargs)


Judgement = tuple[bool, str, dict[str, Any]]


def _file_check(
    path: str, name: str, sandbox: str | None, judge: Callable[[bytes | None], Judgement]
) -> Check:
    async def observe(reader: EvidenceReader) -> CheckOutcome:
        try:
            data: bytes | None = await reader.read_file(path, sandbox=sandbox)
        except FileNotFoundError:
            data = None
        passed, explanation, observed = judge(data)
        return CheckOutcome(name=name, passed=passed, explanation=explanation, observed=observed)

    return Check(name=name, phase="evidence", observe=observe)


def file_hash(
    path: str, sha256: str, *, name: str | None = None, sandbox: str | None = None
) -> Check:
    """Pass when `path` exists and its SHA-256 equals `sha256`."""
    expected = sha256.lower()

    def judge(data: bytes | None) -> Judgement:
        if data is None:
            return False, f"{path} does not exist", {}
        digest = hashlib.sha256(data).hexdigest()
        return (
            digest == expected,
            f"sha256 {digest}, expected {expected}",
            {"sha256": digest, "size": len(data)},
        )

    return _file_check(path, name or f"sha256({path})", sandbox, judge)


def file_exists(
    path: str, exists: bool = True, *, name: str | None = None, sandbox: str | None = None
) -> Check:
    """Pass when `path` exists (or, with `exists=False`, when it does not)."""

    def judge(data: bytes | None) -> Judgement:
        present = data is not None
        return (
            present == exists,
            f"{path} {'exists' if present else 'is absent'}",
            {"exists": present},
        )

    return _file_check(path, name or f"exists({path})", sandbox, judge)


def file_contains(
    path: str,
    pattern: str,
    *,
    regex: bool = False,
    name: str | None = None,
    sandbox: str | None = None,
) -> Check:
    """Pass when `path` contains `pattern` (a substring, or a regex with `regex=True`)."""

    def judge(data: bytes | None) -> Judgement:
        if data is None:
            return False, f"{path} does not exist", {}
        text = data.decode("utf-8", errors="replace")
        found = re.search(pattern, text) is not None if regex else pattern in text
        return found, f"{'found' if found else 'did not find'} {pattern!r}", {}

    return _file_check(path, name or f"contains({path}, {pattern!r})", sandbox, judge)


def scorer_check(
    scorer: Scorer,
    *,
    name: str | None = None,
    passing: float = 1.0,
) -> Check:
    """Any Inspect `Scorer` as a check: it grades the replay sandbox, not the original.

    The scorer runs in the live phase with `sandbox()` bound to the replay, on the
    sample's `TaskState` and `Target`, so a task's own scorer re-grades what an
    independent replay did. A value of at least `passing` (after Inspect's
    `value_to_float`) passes; `NOANSWER` means the scorer could not tell.
    """
    label = name or compat.registry_name(scorer) or "scorer"
    to_float = value_to_float()

    async def observe(target: Sandboxes) -> CheckOutcome:
        if target.state is None or target.target is None:
            return CheckOutcome(
                name=label, passed=None, explanation="no TaskState to run the scorer with"
            )
        score: Score | None = await scorer(target.state, target.target)
        if score is None or score.value == NOANSWER:
            return CheckOutcome(name=label, passed=None, explanation="the scorer gave no answer")
        value = to_float(score.value)
        return CheckOutcome(
            name=label,
            passed=value >= passing,
            explanation=score.explanation or f"scored {score.value!r}",
            observed={"value": score.value, "answer": score.answer},
        )

    return Check(name=label, phase="live", observe=observe)
