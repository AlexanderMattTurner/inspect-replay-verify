"""The replay against a real glovebox microVM, through the `glovebox` Inspect provider.

Needs KVM and a provisioned glovebox host: CI's `e2e-glovebox` job builds one with
agent-glovebox's own runner setup, at the commit that job pins.
"""

import pytest
from helpers import file_is, replay_meta, run, sha256, value
from inspect_ai.scorer import CORRECT

import inspect_replay_verify as irv

pytestmark = pytest.mark.glovebox


def test_honest_run_is_verified_from_a_quiescent_guest() -> None:
    log = run(
        ["printf hello > out.txt"],
        irv.replay_scorer(
            [irv.command_succeeds("test -s out.txt"), irv.file_hash("out.txt", sha256("hello"))]
        ),
        sandbox="glovebox",
    )
    meta = replay_meta(log)
    assert (meta["verdict"], meta["adapter"]) == ("verified", "glovebox"), meta["explanation"]
    [quiescence] = meta["notes"]["quiescence"]
    assert (quiescence["established"], quiescence["survivors"]) == (True, [])


def test_a_delayed_writer_the_replay_starts_is_stopped_before_evidence_is_read() -> None:
    """The writer would forge out.txt long after the replay ends; quiescence kills it first."""
    forge_later = "nohup sh -c 'sleep 120; printf forged > out.txt' >/dev/null 2>&1 &"
    log = run(
        ["printf hello > out.txt", forge_later],
        irv.replay_scorer([irv.file_hash("out.txt", sha256("hello"))]),
        sandbox="glovebox",
    )
    meta = replay_meta(log)
    assert meta["verdict"] == "verified", meta["explanation"]
    [quiescence] = meta["notes"]["quiescence"]
    assert quiescence["established"] is True, quiescence["why"]


def test_root_owned_evidence_is_read_after_quiescence() -> None:
    log = run(
        ["printf secret > out.txt && chmod 000 out.txt"],
        irv.replay_scorer([irv.file_hash("out.txt", sha256("secret"))]),
        sandbox="glovebox",
    )
    assert replay_meta(log)["verdict"] == "verified", replay_meta(log)["explanation"]


def test_the_original_microvm_is_untouched_by_the_replay() -> None:
    log = run(
        ["printf orig > out.txt"],
        [irv.replay_scorer([irv.file_exists("out.txt")]), file_is("out.txt", "orig")],
        sandbox="glovebox",
    )
    assert value(log, "file_is") == CORRECT
    assert replay_meta(log)["verdict"] == "verified", replay_meta(log)["explanation"]
