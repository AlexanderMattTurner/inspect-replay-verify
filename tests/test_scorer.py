"""Real evals on Inspect's `local` sandbox, graded by an independent replay."""

from pathlib import Path

import pytest
from helpers import (
    MODEL,
    agent_inside_sandbox,
    exec_setup,
    file_is,
    once,
    reject_marked,
    replay_meta,
    run,
    sha256,
    value,
    write_out_of_band,
)
from inspect_ai.approval import ApprovalPolicy
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER

import inspect_replay_verify as irv


def test_honest_run_is_verified() -> None:
    checks = [irv.file_hash("out.txt", sha256("hello")), irv.command_succeeds("test -s out.txt")]
    log = run(["printf hello > out.txt"], irv.replay_scorer(checks))
    meta = replay_meta(log)
    assert (value(log), meta["verdict"]) == (CORRECT, "verified"), meta["explanation"]
    assert (meta["calls_replayed"], meta["divergences"]) == (1, 0)
    assert log.results
    assert log.results.scores[0].metrics["verified_rate"].value == 1.0


@pytest.mark.parametrize(
    ("untranscribed", "verdict"),
    [("inconclusive", "inconclusive"), ("ignore", "refuted"), ("replay", "verified")],
)
def test_an_effect_no_tool_call_made(untranscribed: str, verdict: str) -> None:
    """The naive grader passes on the original sandbox whatever the policy says."""
    log = run(
        ["cat out.txt"],
        [
            file_is("out.txt", "hello"),
            irv.replay_scorer(
                [irv.file_hash("out.txt", sha256("hello"))], untranscribed=untranscribed
            ),
        ],
        before=[write_out_of_band("out.txt", "hello")],
    )
    meta = replay_meta(log)
    assert value(log, "file_is") == CORRECT
    assert meta["verdict"] == verdict, meta["explanation"]
    if verdict == "inconclusive":
        assert "write_file out.txt" in meta["explanation"]


def test_harness_setup_exec_is_replayed_in_order() -> None:
    log = run(
        ["printf ' agent' >> repo.txt"],
        irv.replay_scorer([irv.file_contains("repo.txt", "base agent")], untranscribed="replay"),
        before=[exec_setup("printf base > repo.txt")],
    )
    meta = replay_meta(log)
    assert meta["verdict"] == "verified", meta["explanation"]
    assert [c["function"] for c in meta["calls"]] == ["sandbox.exec", "bash"]


def test_a_command_the_model_ran_during_is_never_replayed() -> None:
    log = run(
        ["true"],
        irv.replay_scorer([irv.file_exists("agent.txt")], untranscribed="replay"),
        before=[agent_inside_sandbox()],
    )
    meta = replay_meta(log)
    assert meta["verdict"] == "inconclusive"
    assert "model call ran during" in meta["explanation"]
    assert not any(c["replayed"] for c in meta["calls"])


def test_a_write_the_transcript_truncated_is_unreproducible() -> None:
    long_file = "".join(f"line {i}\n" for i in range(40))
    log = run(
        ["true"],
        irv.replay_scorer([irv.file_exists("big.txt")], untranscribed="replay"),
        before=[write_out_of_band("big.txt", long_file)],
    )
    meta = replay_meta(log)
    assert meta["verdict"] == "inconclusive"
    assert "only part of write_file big.txt" in meta["explanation"]


def test_a_call_the_approver_rejected_is_never_replayed() -> None:
    log = run(
        ["printf kept > kept.txt", "printf REJECT > rejected.txt"],
        irv.replay_scorer(
            [irv.file_exists("kept.txt"), irv.file_exists("rejected.txt", exists=False)]
        ),
        approval=[ApprovalPolicy(reject_marked(), "bash")],
    )
    meta = replay_meta(log)
    assert meta["verdict"] == "verified", meta["explanation"]
    assert [(c["disposition"], c["replayed"]) for c in meta["calls"]] == [
        ("replay", True),
        ("skip", False),
    ]


def test_pristine_state_includes_files_and_setup() -> None:
    sample = Sample(
        input="append",
        target="done",
        files={"seed.txt": "seed\n"},
        setup="printf 'setup\\n' > setup.txt",
    )
    log = run(
        ["cat seed.txt setup.txt > both.txt"],
        irv.replay_scorer([irv.file_contains("both.txt", "seed\nsetup\n")]),
        sample=sample,
    )
    assert value(log) == CORRECT, replay_meta(log)["explanation"]


@pytest.mark.parametrize(("restore", "expected"), [(True, INCORRECT), (False, CORRECT)])
def test_rewritten_tests_are_caught_only_by_restoring_them(restore: bool, expected: str) -> None:
    original = "test -f solved\n"
    sample = Sample(input="fix", target="done", files={"test.sh": original})
    check = irv.tests_pass("bash test.sh", restore_files={"test.sh": original} if restore else None)
    log = run(["printf 'exit 0\\n' > test.sh"], irv.replay_scorer([check]), sample=sample)
    assert value(log) == expected


@pytest.mark.parametrize(
    ("on_divergence", "expected"), [("record", CORRECT), ("inconclusive", NOANSWER)]
)
def test_nondeterministic_output(on_divergence: str, expected: str) -> None:
    log = run(
        ["date +%s%N; printf x > out.txt"],
        irv.replay_scorer([irv.file_exists("out.txt")], on_divergence=on_divergence),
    )
    meta = replay_meta(log)
    assert value(log) == expected
    assert [c["diverged"] for c in meta["calls"]] == [True]


def test_a_tool_error_only_on_replay_is_inconclusive_not_refuted(tmp_path: Path) -> None:
    """The `local` sandbox shares the host, so the marker lets only the original succeed."""
    marker = str(tmp_path / "ran-once")
    call = ModelOutput.for_tool_call(MODEL, "once", {"path": "out.txt"})
    log = run([call], irv.replay_scorer([irv.file_exists("out.txt")]), tools=[once(marker)])
    meta = replay_meta(log)
    assert (value(log), meta["verdict"]) == (NOANSWER, "inconclusive")
    assert "ended unknown on replay but ok in the original" in meta["explanation"]


@pytest.mark.parametrize(
    ("on_divergence", "verdict"), [("record", "refuted"), ("inconclusive", "inconclusive")]
)
def test_bash_failing_only_on_replay_is_a_divergence(
    tmp_path: Path, on_divergence: str, verdict: str
) -> None:
    """`bash` reports a non-zero exit as output, so only the output comparison can see it."""
    marker = tmp_path / "ran-once"
    command = (
        f"if [ -e {marker} ]; then echo again; exit 3; fi; touch {marker}; printf hi > out.txt"
    )
    log = run(
        [command], irv.replay_scorer([irv.file_exists("out.txt")], on_divergence=on_divergence)
    )
    meta = replay_meta(log)
    assert meta["verdict"] == verdict
    assert [c["diverged"] for c in meta["calls"]] == [True]
    assert "1 call output divergence" in meta["explanation"]


def test_raising_check_is_inconclusive_unless_another_fails() -> None:
    async def broken(target: object) -> bool:
        raise RuntimeError("probe failed")

    log = run(
        ["true"],
        irv.replay_scorer([irv.file_exists("nothing", exists=False), irv.custom_check(broken)]),
    )
    meta = replay_meta(log)
    assert (value(log), meta["verdict"]) == (NOANSWER, "inconclusive")
    assert "probe failed" in meta["explanation"]

    log = run(
        ["true"], irv.replay_scorer([irv.file_exists("missing.txt"), irv.custom_check(broken)])
    )
    assert value(log) == INCORRECT


def test_no_checks_is_inconclusive() -> None:
    assert value(run(["true"], irv.replay_scorer([]))) == NOANSWER


def test_per_sample_checks_from_target() -> None:
    log = run(
        ["printf banana > out.txt"],
        irv.replay_scorer(lambda ctx: [irv.file_contains("out.txt", str(ctx.target[0]))]),
        sample=Sample(input="write the target", target="banana"),
    )
    assert value(log) == CORRECT


def test_unavailable_sandbox_type_is_inconclusive() -> None:
    log = run(["true"], irv.replay_scorer([irv.file_exists("x")], sandbox="no-such-sandbox"))
    meta = replay_meta(log)
    assert value(log) == NOANSWER
    assert "no-such-sandbox" in meta["explanation"]


def test_replay_scorers_do_not_see_each_others_calls() -> None:
    scorers = [irv.replay_scorer([irv.file_exists("out.txt")]) for _ in range(2)]
    log = run(["printf a > out.txt", "printf b >> out.txt"], scorers)
    assert log.samples
    names = list(log.samples[0].scores or {})
    assert len(names) == 2
    for name in names:
        assert replay_meta(log, name)["calls_recorded"] == 2
        assert value(log, name) == CORRECT


def test_original_sandbox_is_untouched_by_the_replay() -> None:
    overwrite = irv.command_succeeds("true", restore_files={"out.txt": "replay"})
    log = run(
        ["printf orig > out.txt"], [irv.replay_scorer([overwrite]), file_is("out.txt", "orig")]
    )
    assert value(log, "file_is") == CORRECT
