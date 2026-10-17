"""Ground-truth checks run against the replay sandbox once every recorded call is re-issued.

A check observes an objective effect — an exit code, a file's bytes, a test run — and
answers pass, fail, or "could not tell". "Could not tell" (`passed=None`) is never a pass:
it makes the whole verdict inconclusive.

Checks come in two phases, run in this order:

- ``live`` checks need a running sandbox (run the tests, run a probe command). Their
  outcome is captured host-side, from the exec result.
- ``evidence`` checks only read state (hash a file, look for a string). The adapter may
  freeze the sandbox between the two phases so nothing the replay left running can write
  to the evidence while it is read; see `ReplayAdapter.collect`.
"""

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from inspect_ai.util import SandboxEnvironment

Phase = Literal["live", "evidence"]


@dataclass(frozen=True, kw_only=True, slots=True)
class CheckOutcome:
    """What one check observed in the replay sandbox."""

    name: str
    passed: bool | None
    """True/False on a definite observation; None when the check could not be evaluated."""
    explanation: str = ""
    observed: dict[str, Any] = field(default_factory=dict)


class EvidenceReader(Protocol):
    """Read-only access to a (possibly frozen) replay sandbox."""

    async def read_file(self, path: str, *, sandbox: str | None = None) -> bytes:
        """The file's bytes. Raises `FileNotFoundError` when it does not exist."""
        ...


class LiveTarget(Protocol):
    """A running replay sandbox, by name."""

    def environment(self, name: str | None = None) -> SandboxEnvironment: ...


class Check(ABC):
    """A ground-truth observation of the replay sandbox."""

    phase: Phase
    name: str

    @abstractmethod
    async def run(self, target: Any) -> CheckOutcome:
        """Observe the replay. `target` is a `LiveTarget` or an `EvidenceReader` by phase."""


class LiveCheck(Check):
    phase: Phase = "live"

    @abstractmethod
    async def run(self, target: LiveTarget) -> CheckOutcome: ...


class EvidenceCheck(Check):
    phase: Phase = "evidence"

    @abstractmethod
    async def run(self, target: EvidenceReader) -> CheckOutcome: ...


class _CommandCheck(LiveCheck):
    def __init__(
        self,
        cmd: str | Sequence[str],
        *,
        expected_exit: int,
        name: str | None,
        sandbox: str | None,
        user: str | None,
        cwd: str | None,
        timeout: int | None,
        restore_files: Mapping[str, str | bytes] | None,
    ) -> None:
        self.cmd = ["bash", "-c", cmd] if isinstance(cmd, str) else list(cmd)
        self.expected_exit = expected_exit
        self.name = name or (cmd if isinstance(cmd, str) else " ".join(cmd))
        self.sandbox = sandbox
        self.user = user
        self.cwd = cwd
        self.timeout = timeout
        self.restore_files = dict(restore_files or {})

    async def run(self, target: LiveTarget) -> CheckOutcome:
        env = target.environment(self.sandbox)
        for path, contents in self.restore_files.items():
            await env.write_file(path, contents)
        try:
            result = await env.exec(self.cmd, user=self.user, cwd=self.cwd, timeout=self.timeout)
        except TimeoutError:
            return CheckOutcome(
                name=self.name, passed=None, explanation=f"timed out after {self.timeout}s"
            )
        observed = {
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-2000:],
        }
        passed = result.returncode == self.expected_exit
        return CheckOutcome(
            name=self.name,
            passed=passed,
            explanation=f"exit {result.returncode}, expected {self.expected_exit}",
            observed=observed,
        )


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
) -> LiveCheck:
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
    return _CommandCheck(
        cmd,
        expected_exit=expected,
        name=name,
        sandbox=sandbox,
        user=user,
        cwd=cwd,
        timeout=timeout,
        restore_files=restore_files,
    )


def command_succeeds(cmd: str | Sequence[str], **kwargs: Any) -> LiveCheck:
    """Pass when `cmd` exits 0. Takes `exit_code`'s keyword arguments."""
    return exit_code(cmd, 0, **kwargs)


def tests_pass(cmd: str | Sequence[str] = "python -m pytest -q", **kwargs: Any) -> LiveCheck:
    """Pass when the test command exits 0. Pair with `restore_files` for the test files."""
    kwargs.setdefault("name", "tests")
    return exit_code(cmd, 0, **kwargs)


class _FileCheck(EvidenceCheck):
    def __init__(self, path: str, name: str, sandbox: str | None) -> None:
        self.path = path
        self.name = name
        self.sandbox = sandbox

    async def run(self, target: EvidenceReader) -> CheckOutcome:
        try:
            data: bytes | None = await target.read_file(self.path, sandbox=self.sandbox)
        except FileNotFoundError:
            data = None
        return self.judge(data)

    @abstractmethod
    def judge(self, data: bytes | None) -> CheckOutcome: ...


class _FileHash(_FileCheck):
    def __init__(self, path: str, sha256: str, name: str | None, sandbox: str | None) -> None:
        super().__init__(path, name or f"sha256({path})", sandbox)
        self.expected = sha256.lower()

    def judge(self, data: bytes | None) -> CheckOutcome:
        if data is None:
            return CheckOutcome(
                name=self.name, passed=False, explanation=f"{self.path} does not exist"
            )
        digest = hashlib.sha256(data).hexdigest()
        return CheckOutcome(
            name=self.name,
            passed=digest == self.expected,
            explanation=f"sha256 {digest}, expected {self.expected}",
            observed={"sha256": digest, "size": len(data)},
        )


class _FileExists(_FileCheck):
    def __init__(self, path: str, exists: bool, name: str | None, sandbox: str | None) -> None:
        super().__init__(path, name or f"exists({path})", sandbox)
        self.exists = exists

    def judge(self, data: bytes | None) -> CheckOutcome:
        present = data is not None
        return CheckOutcome(
            name=self.name,
            passed=present == self.exists,
            explanation=f"{self.path} {'exists' if present else 'is absent'}",
            observed={"exists": present},
        )


class _FileContains(_FileCheck):
    def __init__(
        self, path: str, pattern: str, regex: bool, name: str | None, sandbox: str | None
    ) -> None:
        super().__init__(path, name or f"contains({path}, {pattern!r})", sandbox)
        self.pattern = pattern
        self.regex = regex

    def judge(self, data: bytes | None) -> CheckOutcome:
        if data is None:
            return CheckOutcome(
                name=self.name, passed=False, explanation=f"{self.path} does not exist"
            )
        text = data.decode("utf-8", errors="replace")
        found = re.search(self.pattern, text) is not None if self.regex else self.pattern in text
        return CheckOutcome(
            name=self.name,
            passed=found,
            explanation=f"{'found' if found else 'did not find'} {self.pattern!r}",
        )


def file_hash(
    path: str, sha256: str, *, name: str | None = None, sandbox: str | None = None
) -> EvidenceCheck:
    """Pass when `path` exists and its SHA-256 equals `sha256`."""
    return _FileHash(path, sha256, name, sandbox)


def file_exists(
    path: str, exists: bool = True, *, name: str | None = None, sandbox: str | None = None
) -> EvidenceCheck:
    """Pass when `path` exists (or, with `exists=False`, when it does not)."""
    return _FileExists(path, exists, name, sandbox)


def file_contains(
    path: str,
    pattern: str,
    *,
    regex: bool = False,
    name: str | None = None,
    sandbox: str | None = None,
) -> EvidenceCheck:
    """Pass when `path` contains `pattern` (a substring, or a regex with `regex=True`)."""
    return _FileContains(path, pattern, regex, name, sandbox)


class _FunctionCheck(Check):
    def __init__(
        self,
        fn: Callable[[Any], Awaitable[bool | CheckOutcome]],
        phase: Phase,
        name: str,
    ) -> None:
        self.fn = fn
        self.phase = phase
        self.name = name

    async def run(self, target: Any) -> CheckOutcome:
        result = await self.fn(target)
        if isinstance(result, CheckOutcome):
            return result
        return CheckOutcome(name=self.name, passed=bool(result))


def custom_check(
    fn: Callable[[Any], Awaitable[bool | CheckOutcome]],
    *,
    phase: Phase = "live",
    name: str | None = None,
) -> Check:
    """Wrap an async function as a check.

    The function receives a `LiveTarget` (phase ``live``) or an `EvidenceReader` (phase
    ``evidence``) and returns a bool or a `CheckOutcome`. Raising makes it inconclusive.
    """
    return _FunctionCheck(fn, phase, name or str(getattr(fn, "__name__", "custom")))
