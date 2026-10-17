"""The glovebox adapter against a stand-in microVM: a local sandbox with a fake glovebox run.

The real provider needs KVM; what is under test here is the adapter's ordering and its
refusals, which the fake makes observable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pytest
from helpers import BASH_ARG, sha256
from inspect_ai.tool import bash
from inspect_ai.util import SandboxEnvironmentSpec, sandboxenv
from inspect_ai.util._sandbox.local import LocalSandboxEnvironment

from inspect_replay_verify import (
    GloveboxAdapter,
    PristineState,
    RecordedCall,
    adapter_for,
    command_succeeds,
    file_hash,
    replay,
)

GH_OLD = "ghs_" + "a" * 36
GH_NEW = "ghs_" + "b" * 36


@dataclass
class Quiescence:
    established: bool
    why: str = ""
    survivors: tuple[str, ...] = ()


@dataclass
class FakeRun:
    name: str
    journal: list[str]
    quiescence: Quiescence = field(default_factory=lambda: Quiescence(True))

    def quiesce(self) -> Quiescence:
        self.journal.append("quiesce")
        return self.quiescence


class FakeVM(LocalSandboxEnvironment):
    journal: ClassVar[list[str]] = []
    quiescence = Quiescence(True)
    as_container = False

    def __init__(self) -> None:
        super().__init__()
        self._run = FakeRun(f"gb-{id(self):x}", FakeVM.journal, FakeVM.quiescence)
        if FakeVM.as_container:
            self._container = "svc"

    async def exec(self, cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        FakeVM.journal.append("exec:" + " ".join(cmd)[-40:])
        return await super().exec(cmd, *args, **kwargs)


@sandboxenv(name="fakebox")
class FakeBox(LocalSandboxEnvironment):
    @classmethod
    async def sample_init(cls, task_name: str, config: Any, metadata: dict[str, str]) -> Any:
        return {"default": FakeVM()}


class RecordingAdapter(GloveboxAdapter):
    """Reads 'as root' straight from the fake VM's directory, journaling each read."""

    def __init__(self, vm_dirs: dict[str, str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.vm_dirs = vm_dirs

    async def read_as_root(self, run: Any, path: str) -> bytes:
        FakeVM.journal.append(f"root-read:{path}")
        file = Path(self.vm_dirs[run.name]) / path
        if not file.exists():
            raise FileNotFoundError(path)
        return file.read_bytes()

    async def prepare(self, environments: Any) -> None:
        for env in environments.values():
            if hasattr(env, "_run"):
                self.vm_dirs[env._run.name] = env.directory.name


@pytest.fixture(autouse=True)
def _reset() -> None:
    FakeVM.journal = []
    FakeVM.quiescence = Quiescence(True)
    FakeVM.as_container = False


def call(id: str, command: str, result: str = "") -> RecordedCall:
    return RecordedCall(
        id=id,
        function="bash",
        arguments={BASH_ARG: command},
        result=result,
        error=None,
        disposition="replay",
    )


PRISTINE = PristineState(sandbox=SandboxEnvironmentSpec("fakebox"))


async def go(calls: list[RecordedCall], checks: list[Any], **kwargs: Any) -> Any:
    adapter = kwargs.pop("adapter", None) or RecordingAdapter({})
    return await replay(
        calls, PRISTINE, checks=checks, tools={"bash": bash()}, adapter=adapter, **kwargs
    )


def test_glovebox_adapter_is_chosen_for_glovebox() -> None:
    assert isinstance(adapter_for("glovebox"), GloveboxAdapter)
    assert not isinstance(adapter_for("docker"), GloveboxAdapter)


async def test_live_checks_then_quiesce_then_root_reads() -> None:
    report = await go(
        [call("c1", "printf hello > out.txt")],
        [file_hash("out.txt", sha256("hello")), command_succeeds("test -f out.txt")],
    )
    assert report.verdict == "verified", report.explanation
    steps = [e for e in FakeVM.journal if not e.startswith("exec:") or "test -f" in e]
    assert [e.split(":")[0] for e in steps] == ["exec", "quiesce", "root-read"]
    assert "test -f out.txt" in steps[0]
    assert report.notes["quiescence"][0]["established"] is True


async def test_unconfirmed_quiescence_is_inconclusive_even_when_evidence_passes() -> None:
    FakeVM.quiescence = Quiescence(False, "1 process(es) of agent survived the stop", ("42 sh",))
    report = await go(
        [call("c1", "printf hello > out.txt")], [file_hash("out.txt", sha256("hello"))]
    )
    assert report.verdict == "inconclusive"
    assert "quiescence was not confirmed" in report.explanation
    assert not any(e.startswith("root-read") for e in FakeVM.journal)


async def test_compose_container_evidence_is_inconclusive() -> None:
    FakeVM.as_container = True
    report = await go(
        [call("c1", "printf hello > out.txt")], [file_hash("out.txt", sha256("hello"))]
    )
    assert report.verdict == "inconclusive"
    assert "compose service" in report.explanation


async def test_environment_without_a_run_is_inconclusive() -> None:
    report = await replay(
        [call("c1", "true")],
        PristineState(sandbox=SandboxEnvironmentSpec("local")),
        checks=[file_hash("x", sha256(""))],
        tools={"bash": bash()},
        adapter=GloveboxAdapter(),
    )
    assert report.verdict == "inconclusive"
    assert "no glovebox run" in report.explanation


async def test_session_credentials_are_tolerated_not_counted() -> None:
    recorded = call("c1", f"echo token={GH_OLD}", result=f"token={GH_OLD}\n")
    report = await go([recorded], [command_succeeds("true")])
    assert report.verdict == "verified"
    [c] = report.calls
    assert c.credential_scoped is True
    assert report.divergences == 0


async def test_remint_substitutes_a_fresh_credential() -> None:
    adapter = RecordingAdapter({}, remint=lambda kind, old: GH_NEW if kind == "github" else None)
    recorded = call("c1", f"printf {GH_OLD} > tok.txt", result="")
    report = await go([recorded], [file_hash("tok.txt", sha256(GH_NEW))], adapter=adapter)
    assert report.verdict == "verified", report.explanation


def test_normalize_masks_credentials_and_session_names() -> None:
    adapter = GloveboxAdapter()
    text = f"vm gb-1234 token {GH_OLD} key sk-or-v1-{'0' * 64}"
    assert adapter.normalize(text, ["gb-1234"]) == (
        "vm <session> token <credential:github> key <credential:openrouter>"
    )


async def test_unknown_tool_is_inconclusive() -> None:
    weird = RecordedCall(
        id="w", function="custom_tool", arguments={}, result="", error=None, disposition="replay"
    )
    report = await go([call("c1", "true"), weird], [command_succeeds("true")])
    assert report.verdict == "inconclusive"
    assert "custom_tool" in report.explanation
    assert [c.replayed for c in report.calls] == [True, False]


async def test_unreproducible_call_stops_the_replay() -> None:
    lost = RecordedCall(
        id="l",
        function="bash",
        arguments={BASH_ARG: "true"},
        result="",
        error=None,
        disposition="unreproducible",
        reason="the original call never completed",
    )
    report = await go([lost, call("after", "touch x")], [command_succeeds("true")])
    assert report.verdict == "inconclusive"
    assert len(report.calls) == 1


@sandboxenv(name="fakebox-compose")
class FakeComposeBox(LocalSandboxEnvironment):
    @classmethod
    async def sample_init(cls, task_name: str, config: Any, metadata: dict[str, str]) -> Any:
        return {"default": FakeVM(), "sibling": LocalSandboxEnvironment()}


async def test_sibling_without_a_run_does_not_spoil_the_vm_evidence() -> None:
    compose = PristineState(sandbox=SandboxEnvironmentSpec("fakebox-compose"))
    calls = [call("c1", "printf hello > out.txt")]
    good = [file_hash("out.txt", sha256("hello"))]
    report = await replay(
        calls, compose, checks=good, tools={"bash": bash()}, adapter=RecordingAdapter({})
    )
    assert report.verdict == "verified", report.explanation

    on_sibling = [*good, file_hash("out.txt", sha256("hello"), sandbox="sibling")]
    report = await replay(
        calls, compose, checks=on_sibling, tools={"bash": bash()}, adapter=RecordingAdapter({})
    )
    assert report.verdict == "inconclusive"
    assert "no glovebox run" in report.explanation


def test_driver_seams_match_glovebox_driver() -> None:
    """The adapter's calls into glovebox-driver, checked against the installed driver."""
    import inspect

    guest_exec = pytest.importorskip("glovebox_driver.guest_exec")
    run_mod = pytest.importorskip("glovebox_driver.run")
    argv = inspect.signature(guest_exec.guest_exec_argv)
    assert list(argv.parameters)[:2] == ["name", "cmd"]
    assert argv.parameters["user"].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(inspect.signature(guest_exec.run_argv).parameters)[:3] == [
        "argv",
        "input_bytes",
        "timeout",
    ]
    fields = {f.name for f in __import__("dataclasses").fields(guest_exec.ExecOutcome)}
    assert {"returncode", "stdout", "stderr"} <= fields
    assert callable(run_mod.DisposableRun.quiesce)
    assert {"established", "why", "survivors"} <= set(run_mod.Quiescence.__dataclass_fields__)
