"""Replay adapter for agent-glovebox's `glovebox` Inspect sandbox (hypervisor-isolated microVMs).

Two things differ from a plain container:

1. **Evidence is read only from a quiescent guest.** A replayed call can leave a process
   running — a delayed writer — that rewrites the very file a check then reads. glovebox's
   breakout CTF already refuses to trust evidence read from a guest it has not stopped
   (`bin/check-breakout-ctf.bash`; the judge downgrades every unbreached surface to
   UNMEASURED when quiescence is unconfirmed). This adapter mirrors that ordering:

   a. live checks (test runs, probes) execute first, while the guest still admits
      commands, and their outcome is captured host-side from the exec result;
   b. the run is quiesced (`DisposableRun.quiesce()`: admissions stop, attached streams
      close, admitted execs drain, guest root kills every process of every user the run
      ever ran a command as, and a census confirms none survived);
   c. evidence checks read files through a trusted root collector, addressed by sandbox
      name, which starts no agent code and which quiescence does not refuse.

   If quiescence is not established — survivors, a drain that timed out, a run whose
   user is root (whose processes cannot be told from the collectors') — the evidence is
   untrusted and the verdict is inconclusive, never a pass.

2. **Credentials are session-scoped.** glovebox never puts a real credential in the guest:
   the guest holds fixed public placeholders and the host's egress proxy swaps in a token
   minted for that session (GitHub App installation tokens live about an hour). A replay
   VM therefore gets its own freshly minted credentials from `sample_init` with nothing to
   do here, and placeholders are byte-identical across sessions. What can differ is any
   real token that surfaced in a tool output (and, worse, was then pasted into a later
   call): outputs are compared with credentials masked, and a call whose arguments carry
   one literally is marked credential-scoped so its divergence is tolerated rather than
   counted. Pass `remint=` to substitute a fresh value into such calls instead.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from inspect_ai.util import SandboxEnvironment

from .._inspect_compat import unproxy
from ..checks import Check
from ..transcript import RecordedCall
from .base import Collected, ReplayAdapter, _Live, run_check, with_arguments

# Exit status the root collector uses for "no such file", distinct from a read failure.
_MISSING = 44
_READ_SCRIPT = '[ -e "$1" ] || exit 44; [ -d "$1" ] && exit 45; exec head -c "$2" -- "$1"'
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_READ_TIMEOUT_S = 120.0

Remint = Callable[[str, str], str | None]
"""(credential kind, recorded value) -> a value valid for the replay session, or None."""


class EvidenceUnavailable(RuntimeError):
    """The evidence cannot be read from a quiescent guest."""


class GloveboxEvidenceReader:
    """Reads files as guest root through the run's trusted collector, after quiescence."""

    def __init__(self, adapter: GloveboxAdapter, environments: Mapping[str, SandboxEnvironment]):
        self._adapter = adapter
        self._live = _Live(environments)

    async def read_file(self, path: str, *, sandbox: str | None = None) -> bytes:
        env = unproxy(self._live.environment(sandbox))
        if getattr(env, "_container", None) is not None:
            raise EvidenceUnavailable(
                "evidence inside a compose service container cannot be read once the "
                "microVM is quiesced; check the VM's own filesystem instead"
            )
        run = getattr(env, "_run", None)
        if run is None:
            raise EvidenceUnavailable("environment exposes no glovebox run")
        return await self._adapter.read_as_root(run, path)


class GloveboxAdapter(ReplayAdapter):
    """Quiesce-before-read evidence collection and session-scoped credential tolerance."""

    name: ClassVar[str] = "glovebox"
    sandbox_types: ClassVar[tuple[str, ...]] = ("glovebox",)

    def __init__(
        self,
        *,
        remint: Remint | None = None,
        credential_patterns: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(credential_patterns=credential_patterns)
        self._remint = remint

    def session_identifiers(self, environments: Mapping[str, SandboxEnvironment]) -> list[str]:
        names: list[str] = []
        for env in environments.values():
            run = getattr(unproxy(env), "_run", None)
            name = getattr(run, "name", None)
            if isinstance(name, str):
                names.append(name)
        return names

    def rewrite_call(self, call: RecordedCall) -> RecordedCall:
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

        return with_arguments(call, substitute(call.arguments))

    # --- the glovebox-driver seams, overridable in tests ------------------------------

    async def quiesce(self, run: Any) -> Any:
        """Stop the run's untrusted writers. Returns the driver's `Quiescence`."""
        return await asyncio.to_thread(run.quiesce)

    async def read_as_root(self, run: Any, path: str) -> bytes:
        """One file's bytes, read by guest root through the trusted collector."""
        from glovebox_driver import guest_exec

        argv = guest_exec.guest_exec_argv(
            run.name,
            ["/bin/sh", "-c", _READ_SCRIPT, "_", path, str(_MAX_EVIDENCE_BYTES + 1)],
            user="root",
        )
        outcome = await asyncio.to_thread(guest_exec.run_argv, argv, None, _READ_TIMEOUT_S)
        if outcome.returncode == _MISSING:
            raise FileNotFoundError(path)
        if outcome.returncode != 0:
            stderr = outcome.stderr.decode("utf-8", errors="replace").strip()
            raise EvidenceUnavailable(f"root read of {path} exited {outcome.returncode}: {stderr}")
        if len(outcome.stdout) > _MAX_EVIDENCE_BYTES:
            raise EvidenceUnavailable(f"{path} exceeds {_MAX_EVIDENCE_BYTES} bytes")
        return bytes(outcome.stdout)

    # --- collection -------------------------------------------------------------------

    async def collect(
        self, environments: Mapping[str, SandboxEnvironment], checks: Sequence[Check]
    ) -> Collected:
        collected = Collected()
        live = _Live(environments)
        # (a) live checks while the guest still admits commands
        for check in (c for c in checks if c.phase == "live"):
            collected.outcomes.append(await run_check(check, live))

        evidence = [c for c in checks if c.phase == "evidence"]
        # (b) quiesce every distinct run, even with no evidence checks: the record of
        # whether the replay left writers behind is itself worth having
        # (compose siblings carry no run of their own: they live inside the same microVM)
        runs: dict[int, Any] = {}
        for env in environments.values():
            run = getattr(unproxy(env), "_run", None)
            if run is not None and callable(getattr(run, "quiesce", None)):
                runs[id(run)] = run
        if not runs:
            collected.inconclusive = (
                "the replay sandbox exposes no glovebox run to quiesce, so its "
                "evidence cannot be read from a stopped guest"
            )
            return collected
        reasons: list[str] = []
        quiescence: list[dict[str, Any]] = []
        for run in runs.values():
            try:
                result = await self.quiesce(run)
            except Exception as ex:
                reasons.append(f"{getattr(run, 'name', '?')}: quiesce raised {ex!r}")
                continue
            quiescence.append(
                {
                    "run": getattr(run, "name", None),
                    "established": bool(getattr(result, "established", False)),
                    "why": getattr(result, "why", ""),
                    "survivors": list(getattr(result, "survivors", ())),
                }
            )
            if not getattr(result, "established", False):
                reasons.append(f"{getattr(run, 'name', '?')}: {getattr(result, 'why', '')}")
        collected.notes["quiescence"] = quiescence
        if reasons:
            collected.inconclusive = (
                "guest quiescence was not confirmed (" + "; ".join(reasons) + ")"
            )
            return collected

        # (c) evidence through the trusted root collector
        reader = GloveboxEvidenceReader(self, environments)
        for check in evidence:
            collected.outcomes.append(await run_check(check, reader))
        return collected
