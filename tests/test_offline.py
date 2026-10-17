"""Verifying a finished `.eval` log after the fact."""

from __future__ import annotations

from helpers import run, sha256
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, Score, Target, scorer
from inspect_ai.solver import TaskState
from test_scorer import out_of_band_write

from inspect_replay_verify import file_contains, file_hash, replay_log


@scorer(metrics=[])
def always_pass():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=CORRECT)

    return score


def test_replay_a_finished_log() -> None:
    log = run(["printf hello > out.txt"], always_pass())
    [report] = replay_log(log.location, [file_hash("out.txt", sha256("hello"))]).values()
    assert report.verdict == "verified", report.explanation
    assert report.transcript_source == "events"


def test_replay_a_finished_log_refutes_an_out_of_band_effect() -> None:
    log = run(["cat out.txt"], always_pass(), before=[out_of_band_write()])
    [report] = replay_log(log, [file_hash("out.txt", sha256("hello"))]).values()
    assert report.verdict == "refuted"


def test_sample_files_need_the_dataset() -> None:
    sample = Sample(id="s1", input="x", target="done", files={"seed.txt": "seed"})
    log = run(["cp seed.txt out.txt"], always_pass(), sample=sample)
    checks = [file_contains("out.txt", "seed")]

    [report] = replay_log(log, checks).values()
    assert report.verdict == "inconclusive"
    assert "dataset" in report.explanation

    [report] = replay_log(log, checks, dataset=[sample]).values()
    assert report.verdict == "verified", report.explanation


def test_replay_falls_back_to_messages_without_events() -> None:
    log = run(["printf hello > out.txt"], always_pass())
    assert log.samples
    log.samples[0].events = []
    [report] = replay_log(log, [file_hash("out.txt", sha256("hello"))]).values()
    assert report.transcript_source == "messages"
    assert report.verdict == "verified", report.explanation
