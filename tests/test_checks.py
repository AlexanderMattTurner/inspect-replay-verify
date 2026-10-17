"""Each ground-truth check in both directions, against a real `local` sandbox."""

from collections.abc import Mapping
from typing import Any

import pytest
from helpers import sha256
from inspect_ai.util import SandboxEnvironment, SandboxEnvironmentSpec

import inspect_replay_verify as irv

PRISTINE = irv.PristineState(
    sandbox=SandboxEnvironmentSpec("local"),
    files={"out.txt": b"hello\n", "run.sh": b"exit 3\n"},
)


async def judged(
    checks: list[irv.Check], adapter: irv.ReplayAdapter | None = None
) -> irv.ReplayReport:
    return await irv.replay([], PRISTINE, checks, irv.ReplayConfig(adapter=adapter))


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
    assert (await judged([check])).verdict == expected


async def test_restore_files_overwrites_before_running() -> None:
    check = irv.exit_code("bash run.sh", 0, restore_files={"run.sh": "exit 0\n"})
    assert (await judged([check])).verdict == "verified"


class Untrusting(irv.ReplayAdapter):
    """Refuses to hand out evidence, after recording why."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def freeze(
        self, environments: Mapping[str, SandboxEnvironment], notes: dict[str, Any]
    ) -> irv.EvidenceReader:
        notes["froze"] = False
        raise self.error


@pytest.mark.parametrize(
    ("error", "explanation"),
    [
        (irv.Inconclusive("the guest kept writing"), "the guest kept writing"),
        (
            RuntimeError("driver gone"),
            "could not freeze the replay sandbox: RuntimeError: driver gone",
        ),
    ],
)
async def test_evidence_an_adapter_will_not_freeze_is_never_read(
    error: Exception, explanation: str
) -> None:
    reads: list[str] = []

    async def spy(reader: irv.EvidenceReader) -> bool:
        reads.append("read")
        return True

    report = await judged(
        [irv.command_succeeds("true"), irv.custom_check(spy, phase="evidence")], Untrusting(error)
    )
    assert (report.verdict, reads, report.notes["froze"]) == ("inconclusive", [], False)
    assert explanation in report.explanation
    assert [(o.name, o.passed) for o in report.checks] == [("true", True)]


def test_a_phase_outside_the_two_is_refused() -> None:
    async def ok(target: object) -> bool:
        return True

    with pytest.raises(ValueError, match="phase is one of"):
        irv.custom_check(ok, phase="Evidence")  # type: ignore[arg-type]


def test_no_outcome_at_all_is_never_a_pass() -> None:
    report = irv.ReplayReport()
    report.judge([])
    assert report.verdict == "inconclusive"
