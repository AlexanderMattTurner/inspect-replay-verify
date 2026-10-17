"""Verifying a finished `.eval` log after the fact."""

from helpers import exec_setup, file_is, run, sha256, write_out_of_band
from inspect_ai.dataset import Sample

import inspect_replay_verify as irv

HELLO = [irv.file_hash("out.txt", sha256("hello"))]


def test_replay_a_finished_log() -> None:
    log = run(["printf hello > out.txt"], file_is("out.txt", "hello"))
    [report] = irv.replay_log(log.location, HELLO).values()
    assert (report.verdict, report.transcript_source) == ("verified", "events"), report.explanation


def test_an_out_of_band_effect_in_a_finished_log() -> None:
    log = run(
        ["cat out.txt"], file_is("out.txt", "hello"), before=[write_out_of_band("out.txt", "hello")]
    )
    verdicts = {
        policy: next(
            iter(irv.replay_log(log, HELLO, config=irv.ReplayConfig(untranscribed=policy)).values())
        ).verdict
        for policy in ("inconclusive", "ignore", "replay")
    }
    assert verdicts == {"inconclusive": "inconclusive", "ignore": "refuted", "replay": "verified"}


def test_harness_setup_in_a_finished_log() -> None:
    log = run(
        ["printf ' agent' >> repo.txt"],
        file_is("repo.txt", "base agent"),
        before=[exec_setup("printf base > repo.txt")],
    )
    checks = [irv.file_contains("repo.txt", "base agent")]
    [report] = irv.replay_log(log, checks, config=irv.ReplayConfig(untranscribed="replay")).values()
    assert report.verdict == "verified", report.explanation


def test_sample_files_need_the_dataset() -> None:
    sample = Sample(id="s1", input="x", target="done", files={"seed.txt": "seed"})
    log = run(["cp seed.txt out.txt"], file_is("out.txt", "seed"), sample=sample)
    checks = [irv.file_contains("out.txt", "seed")]

    [report] = irv.replay_log(log, checks).values()
    assert report.verdict == "inconclusive"
    assert "dataset" in report.explanation

    [report] = irv.replay_log(log, checks, dataset=[sample]).values()
    assert report.verdict == "verified", report.explanation


def test_replay_falls_back_to_messages_without_events() -> None:
    log = run(["printf hello > out.txt"], file_is("out.txt", "hello"))
    assert log.samples
    log.samples[0].events = []
    [report] = irv.replay_log(log, HELLO).values()
    assert (report.verdict, report.transcript_source) == ("verified", "messages"), (
        report.explanation
    )


def test_the_tasks_own_scorer_regrades_a_finished_log() -> None:
    log = run(["printf hello > out.txt"], file_is("out.txt", "hello"))
    checks = [irv.scorer_check(file_is("out.txt", "hello"))]
    [report] = irv.replay_log(log, checks).values()
    assert report.verdict == "verified", report.explanation
