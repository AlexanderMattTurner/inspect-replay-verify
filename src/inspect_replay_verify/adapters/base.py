"""The per-backend seam: how a replay compares outputs and how it reads evidence.

The default adapter is right for any sandbox whose instances are independent and whose
files can be read while it runs (Docker, local, Kubernetes). A backend that needs more
subclasses `ReplayAdapter` and registers under the ``inspect_replay_verify`` entry-point
group. Its main lever is `freeze`: what it takes to hand out a reader whose evidence
nothing the replay left running can still change.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, ClassVar

from inspect_ai.util import SandboxEnvironment

from ..checks import EvidenceReader, Sandboxes
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

Remint = Callable[[str, str], str | None]
"""(credential kind, recorded value) -> a value valid for the replay session, or None."""


class SandboxReader:
    """Reads evidence through each sandbox's own `read_file`."""

    def __init__(self, environments: Mapping[str, SandboxEnvironment]) -> None:
        self._sandboxes = Sandboxes(environments)

    async def read_file(self, path: str, *, sandbox: str | None = None) -> bytes:
        return await self._sandboxes.environment(sandbox).read_file(path, text=False)


class ReplayAdapter:
    """Default replay behaviour, correct as-is for Docker, local and similar sandboxes.

    Args:
      credential_patterns: Session-scoped credential formats, by kind.
      remint: Substitutes a value valid for the replay session into a call whose
        arguments carry a recorded credential literally.
    """

    name: ClassVar[str] = "default"
    sandbox_types: ClassVar[tuple[str, ...]] = ()
    """Sandbox type names this adapter is chosen for when none is given explicitly."""

    def __init__(
        self,
        *,
        credential_patterns: Mapping[str, str] | None = None,
        remint: Remint | None = None,
    ) -> None:
        patterns = (
            DEFAULT_CREDENTIAL_PATTERNS if credential_patterns is None else credential_patterns
        )
        self._credentials = [(kind, re.compile(p)) for kind, p in patterns.items()]
        self._remint = remint

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
        """The call as re-issued: with `remint`, each recorded credential freshly minted."""
        if self._remint is None or not self.credentials_in(call):
            return call
        remint = self._remint

        def fresh(kind: str) -> Callable[[re.Match[str]], str]:
            return lambda m: remint(kind, m.group(0)) or m.group(0)

        def substitute(value: Any) -> Any:
            if isinstance(value, str):
                for kind, pattern in self._credentials:
                    value = pattern.sub(fresh(kind), value)
                return value
            if isinstance(value, list):
                return [substitute(v) for v in value]
            if isinstance(value, dict):
                return {k: substitute(v) for k, v in value.items()}
            return value

        return replace(call, arguments=substitute(call.arguments))

    async def prepare(self, environments: Mapping[str, SandboxEnvironment]) -> None:
        """Called once the replay sandbox reaches its pristine state, before any call."""

    async def freeze(
        self, environments: Mapping[str, SandboxEnvironment], notes: dict[str, Any]
    ) -> EvidenceReader:
        """A reader for the evidence, called after the live checks and before the evidence ones.

        Raise `Inconclusive` when the evidence cannot be trusted; anything recorded in
        `notes` reaches the score's metadata either way.
        """
        return SandboxReader(environments)
