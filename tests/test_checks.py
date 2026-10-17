"""Each ground-truth check in both directions, against a real `local` sandbox."""

from collections.abc import Mapping, Sequence

import pytest
from helpers import sha256
from inspect_ai.util import SandboxEnvironment, SandboxEnvironmentSpec

import inspect_replay_verify as irv
from inspect_replay_verify.adapters import Collected

PRISTINE = irv.PristineState(
    sandbox=SandboxEnvironmentSpec("local"),
    files={"out.txt": b"hello\n", "run.sh": b"exit 3\n"},
)


async def verdict(check: irv.Check, adapter: irv.ReplayAdapter | None = None) -> str:
    report = await irv.replay([], PRISTINE, checks=[check], tools={}, adapter=adapter)
    return report.verdict


@pytest.mark.parametrize(
    ("check", "expected"),
    [
        (irv.file_hash("out.txt", sha256("hello\n")), "verified"),
        (irv.file_hash("out.txt", sha256("other\n")), "refuted"),
        (irv.file_hash("absent.txt", sha256("")), "refuted"),
        (irv.file_exists("out.txt"), "verified"),
        (irv.file_exists("absent.txt"), "refuted"),
        (irv.file_exists("absent.txt", exists=False), "verified"),
        (irv.file_exists("out.txt", exists=False), "refuted"),
        (irv.file_contains("out.txt", "ell"), "verified"),
        (irv.file_contains("out.txt", "nope"), "refuted"),
        (irv.file_contains("out.txt", r"^h.l+o$", regex=True), "verified"),
        (irv.file_contains("out.txt", r"^x", regex=True), "refuted"),
        (irv.exit_code("bash run.sh", 3), "verified"),
        (irv.exit_code("bash run.sh", 0), "refuted"),
        (irv.command_succeeds("true"), "verified"),
        (irv.command_succeeds("false"), "refuted"),
        (irv.command_succeeds("sleep 30", timeout=1), "inconclusive"),
    ],
    ids=lambda v: getattr(v, "name", v),
)
async def test_check_directions(check: irv.Check, expected: str) -> None:
    assert await verdict(check) == expected


async def test_restore_files_overwrites_before_running() -> None:
    check = irv.exit_code("bash run.sh", 0, restore_files={"run.sh": "exit 0\n"})
    assert await verdict(check) == "verified"


class SilentAdapter(irv.ReplayAdapter):
    """Returns no outcome for the checks it was handed."""

    async def collect(
        self, environments: Mapping[str, SandboxEnvironment], checks: Sequence[irv.Check]
    ) -> Collected:
        return Collected()


async def test_an_adapter_that_reports_nothing_is_inconclusive() -> None:
    assert await verdict(irv.command_succeeds("true"), adapter=SilentAdapter()) == "inconclusive"
