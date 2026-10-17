"""`inspect eval` in a fresh process: the package as Inspect itself loads it."""

import subprocess
import sys
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

TASK = Path(__file__).with_name("fixtures") / "cli_task.py"


def test_inspect_eval_cli_runs_the_replay_scorer(tmp_path: Path) -> None:
    argv = [sys.executable, "-m", "inspect_ai", "eval", TASK.name, "--display", "none"]
    proc = subprocess.run(
        [*argv, "--log-dir", str(tmp_path)],
        cwd=TASK.parent,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2500:]
    [info] = list_eval_logs(str(tmp_path))
    log = read_eval_log(info)
    assert log.status == "success", log.error
    assert log.samples and log.results
    [score] = (log.samples[0].scores or {}).values()
    assert (score.value, (score.metadata or {})["replay"]["verdict"]) == ("C", "verified")
    assert [s.name for s in log.results.scores] == ["replay_scorer"]
