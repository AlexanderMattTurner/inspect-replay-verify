"""End-to-end: real evals on the `local` sandbox, graded by an independent replay."""

from __future__ import annotations

from helpers import replay_meta, run, sha256, value
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, Score, Target, scorer
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import sandbox

from inspect_replay_verify import (
    command_succeeds,
    custom_check,
    file_contains,
    file_exists,
    file_hash,
    replay_scorer,
)
from inspect_replay_verify import (
    tests_pass as tests_pass_check,
)


@scorer(metrics=[])
def trusts_the_sandbox():
    """A naive grader: reads the original sandbox the agent ran in."""

    async def score(state: TaskState, target: Target) -> Score:
        try:
            text = await sandbox().read_file("out.txt")
        except FileNotFoundError:
            return Score(value=INCORRECT)
        return Score(value=CORRECT if text == "hello" else INCORRECT)

    return score


@solver
def out_of_band_write():
    """Writes the answer without any tool call: an effect the transcript does not show."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        await sandbox().write_file("out.txt", "hello")
        return state

    return solve


def test_honest_run_is_verified() -> None:
    log = run(
        ["printf hello > out.txt"],
        replay_scorer([file_hash("out.txt", sha256("hello")), command_succeeds("test -s out.txt")]),
    )
    meta = replay_meta(log)
    assert value(log) == CORRECT, meta["explanation"]
    assert meta["verdict"] == "verified"
    assert meta["calls_replayed"] == 1
    assert meta["divergences"] == 0
    assert log.results and log.results.scores[0].metrics["verified_rate"].value == 1.0


def test_effect_outside_the_transcript_is_refuted() -> None:
    """The original sandbox passes a naive grader; the replay of the transcript does not."""
    log = run(
        ["cat out.txt"],
        [trusts_the_sandbox(), replay_scorer([file_hash("out.txt", sha256("hello"))])],
        before=[out_of_band_write()],
    )
    assert value(log, "trusts_the_sandbox") == CORRECT
    assert value(log) == INCORRECT
    assert replay_meta(log)["verdict"] == "refuted"


def test_pristine_state_includes_files_and_setup() -> None:
    sample = Sample(
        input="append",
        target="done",
        files={"seed.txt": "seed\n"},
        setup="printf 'setup\\n' > setup.txt",
    )
    log = run(
        ["cat seed.txt setup.txt > both.txt"],
        replay_scorer([file_contains("both.txt", "seed\nsetup\n")]),
        sample=sample,
    )
    assert value(log) == CORRECT, replay_meta(log)["explanation"]


def test_rewritten_tests_are_caught_by_restoring_them() -> None:
    sample = Sample(input="fix", target="done", files={"test.sh": "test -f solved\n"})
    tamper = ["printf 'exit 0\\n' > test.sh"]
    restored = replay_scorer(
        [tests_pass_check("bash test.sh", restore_files={"test.sh": "test -f solved\n"})]
    )
    log = run(tamper, restored, sample=sample)
    assert value(log) == INCORRECT

    trusting = replay_scorer([tests_pass_check("bash test.sh")])
    log = run(tamper, trusting, sample=sample)
    assert value(log) == CORRECT


def test_nondeterministic_output_is_recorded_or_refused() -> None:
    commands = ["date +%s%N; printf x > out.txt"]
    checks = [file_exists("out.txt")]
    log = run(commands, replay_scorer(checks))
    meta = replay_meta(log)
    assert value(log) == CORRECT
    assert meta["divergences"] == 1
    assert meta["calls"][0]["diverged"] is True

    log = run(commands, replay_scorer(checks, on_divergence="inconclusive"))
    assert value(log) == NOANSWER
    assert replay_meta(log)["verdict"] == "inconclusive"


def test_a_check_that_raises_is_inconclusive_not_a_pass() -> None:
    async def broken(target: object) -> bool:
        raise RuntimeError("probe failed")

    log = run(["true"], replay_scorer([file_exists("nothing", exists=False), custom_check(broken)]))
    meta = replay_meta(log)
    assert value(log) == NOANSWER
    assert meta["verdict"] == "inconclusive"
    assert "probe failed" in meta["explanation"]


def test_a_definite_failure_outranks_an_unknown() -> None:
    async def broken(target: object) -> bool:
        raise RuntimeError("probe failed")

    log = run(["true"], replay_scorer([file_exists("missing.txt"), custom_check(broken)]))
    assert value(log) == INCORRECT


def test_no_checks_is_inconclusive() -> None:
    log = run(["true"], replay_scorer([]))
    assert value(log) == NOANSWER


def test_per_sample_checks_from_target() -> None:
    sample = Sample(input="write the target", target="banana")
    log = run(
        ["printf banana > out.txt"],
        replay_scorer(lambda ctx: [file_contains("out.txt", str(ctx.target[0]))]),
        sample=sample,
    )
    assert value(log) == CORRECT


def test_unavailable_sandbox_type_is_inconclusive() -> None:
    log = run(["true"], replay_scorer([file_exists("x")], sandbox="no-such-sandbox"))
    meta = replay_meta(log)
    assert value(log) == NOANSWER
    assert "no-such-sandbox" in meta["explanation"]


def test_two_replay_scorers_do_not_see_each_others_calls() -> None:
    scorers = [replay_scorer([file_exists("out.txt")]), replay_scorer([file_exists("out.txt")])]
    log = run(["printf a > out.txt", "printf b >> out.txt"], scorers)
    assert log.samples
    names = list(log.samples[0].scores or {})
    assert len(names) == 2
    for name in names:
        assert replay_meta(log, name)["calls_recorded"] == 2
        assert value(log, name) == CORRECT


def test_original_sandbox_is_untouched_by_the_replay() -> None:
    @scorer(metrics=[])
    def original_still_there():
        async def score(state: TaskState, target: Target) -> Score:
            text = await sandbox().read_file("out.txt")
            return Score(value=CORRECT if text == "orig" else INCORRECT)

        return score

    log = run(
        ["printf orig > out.txt"],
        [replay_scorer([file_exists("out.txt")]), original_still_there()],
    )
    assert value(log, "original_still_there") == CORRECT
