"""The replay against a real Docker sandbox: Inspect's own `docker` provider, a real container."""

from pathlib import Path

import pytest
from helpers import exec_setup, file_is, replay_meta, run, sha256, value, write_out_of_band
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT

import inspect_replay_verify as irv

pytestmark = pytest.mark.docker

DOCKER = ("docker", str(Path(__file__).with_name("compose.yaml")))


def test_honest_run_is_verified_in_a_fresh_container() -> None:
    log = run(
        ["printf hello > out.txt", "python -c 'print(open(\"out.txt\").read())'"],
        irv.replay_scorer([irv.file_hash("out.txt", sha256("hello"))]),
        sandbox=DOCKER,
    )
    meta = replay_meta(log)
    assert (meta["verdict"], meta["sandbox_type"], meta["calls_replayed"]) == (
        "verified",
        "docker",
        2,
    )
    assert meta["divergences"] == 0, meta["calls"]


@pytest.mark.parametrize(
    ("untranscribed", "verdict"),
    [("inconclusive", "inconclusive"), ("ignore", "refuted"), ("replay", "verified")],
)
def test_an_effect_no_tool_call_made(untranscribed: str, verdict: str) -> None:
    log = run(
        ["cat out.txt"],
        [
            file_is("out.txt", "hello"),
            irv.replay_scorer(
                [irv.file_hash("out.txt", sha256("hello"))], untranscribed=untranscribed
            ),
        ],
        before=[write_out_of_band("out.txt", "hello")],
        sandbox=DOCKER,
    )
    assert value(log, "file_is") == CORRECT
    assert replay_meta(log)["verdict"] == verdict


def test_a_file_only_the_original_container_holds_is_absent_from_the_replay() -> None:
    """The replay boots its own container: what the agent's container held does not leak in."""
    log = run(
        ["true"],
        [file_is("leak.txt", "x"), irv.replay_scorer([irv.file_exists("leak.txt", exists=False)])],
        before=[exec_setup("printf x > leak.txt")],
        sandbox=DOCKER,
    )
    assert value(log, "file_is") == CORRECT
    meta = replay_meta(log)
    assert meta["verdict"] == "inconclusive"
    assert "exec bash -c 'printf x > leak.txt'" in meta["explanation"]


def test_rewritten_tests_are_caught_by_restoring_them() -> None:
    original = "import os\nassert os.path.exists('solved')\n"
    sample = Sample(input="fix", target="done", files={"test_it.py": original})
    check = irv.tests_pass("python test_it.py", restore_files={"test_it.py": original})
    log = run(
        ["printf 'pass\\n' > test_it.py"], irv.replay_scorer([check]), sample=sample, sandbox=DOCKER
    )
    assert value(log) == INCORRECT


def test_replay_a_finished_docker_log() -> None:
    log = run(["printf hello > out.txt"], file_is("out.txt", "hello"), sandbox=DOCKER)
    [report] = irv.replay_log(log, [irv.file_hash("out.txt", sha256("hello"))]).values()
    assert (report.verdict, report.sandbox_type) == ("verified", "docker"), report.explanation
