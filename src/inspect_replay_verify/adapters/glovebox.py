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
   counted. `remint=` (see `ReplayAdapter`) substitutes a fresh value into such calls.
"""

import asyncio
from collections.abc import Mapping
from typing import Any, ClassVar

from inspect_ai.util import SandboxEnvironment

from .._inspect_compat import unproxy
from ..checks import EvidenceReader, Sandboxes
from ..report import Inconclusive
from .base import ReplayAdapter

# Exit status the root collector uses for "no such file", distinct from a read failure.
_MISSING = 44
_READ_SCRIPT = '[ -e "$1" ] || exit 44; [ -d "$1" ] && exit 45; exec head -c "$2" -- "$1"'
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_READ_TIMEOUT_S = 120.0


class EvidenceUnavailable(RuntimeError):
    """The evidence cannot be read from a quiescent guest."""


def _run_of(env: SandboxEnvironment) -> Any | None:
    """The glovebox run behind an environment; compose siblings carry none of their own."""
    run = getattr(unproxy(env), "_run", None)
    return run if callable(getattr(run, "quiesce", None)) else None


class GloveboxEvidenceReader:
    """Reads files as guest root through the run's trusted collector, after quiescence."""

    def __init__(self, adapter: "GloveboxAdapter", environments: Mapping[str, SandboxEnvironment]):
        self._adapter = adapter
        self._sandboxes = Sandboxes(environments)

    async def read_file(self, path: str, *, sandbox: str | None = None) -> bytes:
        env = self._sandboxes.environment(sandbox)
        if getattr(unproxy(env), "_container", None) is not None:
            raise EvidenceUnavailable(
                "evidence inside a compose service container cannot be read once the "
                "microVM is quiesced; check the VM's own filesystem instead"
            )
        run = _run_of(env)
        if run is None:
            raise EvidenceUnavailable("environment exposes no glovebox run")
        return await self._adapter.read_as_root(run, path)


class GloveboxAdapter(ReplayAdapter):
    """Quiesce-before-read evidence and session-scoped credential tolerance."""

    name: ClassVar[str] = "glovebox"
    sandbox_types: ClassVar[tuple[str, ...]] = ("glovebox",)

    def session_identifiers(self, environments: Mapping[str, SandboxEnvironment]) -> list[str]:
        runs = (_run_of(env) for env in environments.values())
        return [name for run in runs if isinstance(name := getattr(run, "name", None), str)]

    async def freeze(
        self, environments: Mapping[str, SandboxEnvironment], notes: dict[str, Any]
    ) -> EvidenceReader:
        runs = {id(run): run for env in environments.values() if (run := _run_of(env))}
        if not runs:
            raise Inconclusive(
                "the replay sandbox exposes no glovebox run to quiesce, so its evidence "
                "cannot be read from a stopped guest"
            )
        reasons: list[str] = []
        notes["quiescence"] = []
        for run in runs.values():
            try:
                result = await self.quiesce(run)
            except Exception as ex:
                reasons.append(f"{getattr(run, 'name', '?')}: quiesce raised {ex!r}")
                continue
            established = bool(getattr(result, "established", False))
            notes["quiescence"].append(
                {
                    "run": getattr(run, "name", None),
                    "established": established,
                    "why": getattr(result, "why", ""),
                    "survivors": list(getattr(result, "survivors", ())),
                }
            )
            if not established:
                reasons.append(f"{getattr(run, 'name', '?')}: {getattr(result, 'why', '')}")
        if reasons:
            raise Inconclusive("guest quiescence was not confirmed (" + "; ".join(reasons) + ")")
        return GloveboxEvidenceReader(self, environments)

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
