"""The per-backend seam: how a replay prepares, compares and collects evidence.

The default adapter is right for any sandbox whose instances are independent and whose
files can be read while it runs (Docker, local, Kubernetes). A backend that needs more —
freezing the guest before evidence is read, masking session-scoped values — subclasses
`ReplayAdapter` and registers under the ``inspect_replay_verify`` entry-point group.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from inspect_ai.util import SandboxEnvironment

from ..checks import Check, CheckOutcome
from ..transcript import RecordedCall

# Session-scoped credential formats, so a token minted for the original run and another
# minted for the replay compare equal, and a call that carries one literally is known to
# be session-bound. Patterns follow agent-glovebox's redactor
# (.claude/hooks/vendor/redactor/agent_sanitizer/secrets/data/secret-detectors.json).
DEFAULT_CREDENTIAL_PATTERNS: Mapping[str, str] = {
    "anthropic": r"(?<![A-Za-z0-9_])sk-ant-(?:api03|admin01|oat01)-[A-Za-z0-9_\-]{93}AA(?![A-Za-z0-9])",  # noqa: E501
    "openai": r"(?<![A-Za-z0-9_])(?<!-)sk-[A-Za-z0-9_\-]{0,256}[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}",  # noqa: E501
    "openrouter": r"(?<![A-Za-z0-9_])sk-or-v1-[0-9a-f]{64}",
    "github": r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9_]{36,}",
    "github_pat": r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{82}",
    "gitlab": r"(?<![A-Za-z0-9_])(?:glpat|gldt|glft|glsoat|glrt)-[A-Za-z0-9_\-]{20,50}(?!\w)",
    "google": r"(?<![A-Za-z0-9_])AIza[0-9A-Za-z_\-]{35}(?![A-Za-z0-9])",
    "groq": r"(?<![A-Za-z0-9_])gsk_[A-Za-z0-9]{32,}",
    "xai": r"(?<![A-Za-z0-9_])xai-[A-Za-z0-9]{40,}",
    "aws_access_key": r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])",
}


@dataclass
class Collected:
    """The checks' outcomes, or why the adapter could not collect them trustworthily."""

    outcomes: list[CheckOutcome] = field(default_factory=list)
    inconclusive: str | None = None
    """Set when the evidence cannot be trusted as a whole (e.g. the guest never froze)."""
    notes: dict[str, Any] = field(default_factory=dict)


class _Live:
    def __init__(self, environments: Mapping[str, SandboxEnvironment]) -> None:
        self._environments = environments

    def environment(self, name: str | None = None) -> SandboxEnvironment:
        if name is None or name == "default":
            return next(iter(self._environments.values()))
        return self._environments[name]


class DefaultEvidenceReader:
    """Reads evidence through the sandbox's own `read_file`."""

    def __init__(self, environments: Mapping[str, SandboxEnvironment]) -> None:
        self._live = _Live(environments)

    async def read_file(self, path: str, *, sandbox: str | None = None) -> bytes:
        return await self._live.environment(sandbox).read_file(path, text=False)


async def run_check(check: Check, target: Any) -> CheckOutcome:
    """One check's outcome; a raise is an inconclusive outcome, never a pass."""
    try:
        return await check.run(target)
    except Exception as ex:
        return CheckOutcome(check.name, None, f"check raised {type(ex).__name__}: {ex}")


class ReplayAdapter:
    """Default replay behaviour, correct as-is for Docker, local and similar sandboxes."""

    name: ClassVar[str] = "default"
    sandbox_types: ClassVar[tuple[str, ...]] = ()
    """Sandbox type names this adapter is chosen for when none is given explicitly."""

    def __init__(self, *, credential_patterns: Mapping[str, str] | None = None) -> None:
        patterns = (
            DEFAULT_CREDENTIAL_PATTERNS if credential_patterns is None else credential_patterns
        )
        self._credentials = [(kind, re.compile(p)) for kind, p in patterns.items()]

    # --- comparing the replay with the recording ------------------------------------

    def session_identifiers(self, environments: Mapping[str, SandboxEnvironment]) -> list[str]:
        """Values unique to one sandbox session (a VM name, a hostname) to mask when comparing."""
        return []

    def normalize(self, text: str, identifiers: Sequence[str] = ()) -> str:
        """`text` with session-scoped values masked, so two honest runs compare equal."""
        for kind, pattern in self._credentials:
            text = pattern.sub(f"<credential:{kind}>", text)
        for ident in sorted({i for i in identifiers if i}, key=len, reverse=True):
            text = text.replace(ident, "<session>")
        return text

    def credentials_in(self, call: RecordedCall) -> list[str]:
        """Kinds of session-scoped credential the call's arguments carry literally."""
        blob = repr(call.arguments)
        return [kind for kind, pattern in self._credentials if pattern.search(blob)]

    def rewrite_call(self, call: RecordedCall) -> RecordedCall:
        """The call as it should be re-issued. Identity by default."""
        return call

    # --- lifecycle ------------------------------------------------------------------

    async def prepare(self, environments: Mapping[str, SandboxEnvironment]) -> None:
        """Called after the replay sandbox reaches its pristine state, before any call."""

    async def collect(
        self, environments: Mapping[str, SandboxEnvironment], checks: Sequence[Check]
    ) -> Collected:
        """Run the live checks, then the evidence checks, against the replay sandbox."""
        collected = Collected()
        live = _Live(environments)
        for check in (c for c in checks if c.phase == "live"):
            collected.outcomes.append(await run_check(check, live))
        reader = DefaultEvidenceReader(environments)
        for check in (c for c in checks if c.phase == "evidence"):
            collected.outcomes.append(await run_check(check, reader))
        return collected


def with_arguments(call: RecordedCall, arguments: dict[str, Any]) -> RecordedCall:
    """A copy of `call` re-issued with different arguments."""
    return replace(call, arguments=arguments)
