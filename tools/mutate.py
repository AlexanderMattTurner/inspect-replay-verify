"""Targeted mutation testing: each decision the verdict rests on is broken, and a test must go red.

Every mutant names the exact text it replaces, and a mutant whose anchor no longer occurs
exactly once fails the run instead of silently testing unmodified code. Run from the repo
root: `python tools/mutate.py`. Exits non-zero when any mutant survives.
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "inspect_replay_verify"


@dataclass(frozen=True, kw_only=True, slots=True)
class Mutant:
    file: str
    old: str
    new: str
    breaks: str


MUTANTS = [
    Mutant(
        file="transcript.py",
        old='"approval": ("skip",',
        new='"approval": ("replay",',
        breaks="a call an approver rejected is re-issued",
    ),
    Mutant(
        file="transcript.py",
        old='"cancelled": ("unreproducible",',
        new='"cancelled": ("replay",',
        breaks="a cancelled call is treated as reproducible",
    ),
    Mutant(
        file="transcript.py",
        old='error.type, ("unreproducible", f"unclassified',
        new='error.type, ("replay", f"unclassified',
        breaks="an unclassified error type fails open",
    ),
    Mutant(
        file="transcript.py",
        old='_SCORING = frozenset({"scorer", REPLAY_SPAN_TYPE})',
        new='_SCORING = frozenset({"scorer"})',
        breaks="a replay's own calls are read as the agent's",
    ),
    Mutant(
        file="transcript.py",
        old='return next((b.type for b in chain if b.type in ("tool", "agent")), None) != "tool"',
        new="return True",
        breaks="a tool's own sandbox calls are counted as untranscribed",
    ),
    Mutant(
        file="transcript.py",
        old='    if policy == "inconclusive":\n        disposition = "unreproducible"',
        new='    if policy == "inconclusive":\n        disposition = "replay"',
        breaks="the default policy re-issues untranscribed actions",
    ),
    Mutant(
        file="transcript.py",
        old="return any(event.timestamp <= t <= event.completed for t in model_starts)",
        new="return False",
        breaks="a command a model call ran inside is replayed",
    ),
    Mutant(
        file="transcript.py",
        old="elif _TRUNCATED in content or",
        new="elif False and _TRUNCATED in content or",
        breaks="a truncated recorded input is re-issued",
    ),
    Mutant(
        file="transcript.py",
        old="or len(content) >= _RECORDED_CHARS:",
        new=":",
        breaks="input Inspect cut at 10,000 characters is re-issued as if whole",
    ),
    Mutant(
        file="transcript.py",
        old="    if ran:\n",
        new="    if not ran:\n",
        breaks="message calls stand in for a run the events recorded",
    ),
    Mutant(
        file="transcript.py",
        old="            if prefilled:\n",
        new="            if False:\n",
        breaks="tool calls prefilled in the sample's input are replayed",
    ),
    Mutant(
        file="replay.py",
        old='    if recorded.disposition == "skip":\n        return',
        new='    if recorded.disposition == "skip":\n        pass',
        breaks="skipped calls are re-issued",
    ),
    Mutant(
        file="replay.py",
        old="    if outcome.status != _recorded_status(recorded):",
        new="    if False:",
        breaks="an outcome only the replay had is not treated as non-reproduction",
    ),
    Mutant(
        file="replay.py",
        old='        if report.divergences and config.on_divergence == "inconclusive":',
        new="        if False:",
        breaks="on_divergence='inconclusive' is ignored",
    ),
    Mutant(
        file="report.py",
        old='        if failed:\n            self.verdict = "refuted"',
        new='        if False:\n            self.verdict = "refuted"',
        breaks="a failed check does not refute",
    ),
    Mutant(
        file="report.py",
        old="        elif unknown:",
        new="        elif False:",
        breaks="an unevaluable check counts as a pass",
    ),
    Mutant(
        file="checks.py",
        old="            digest == expected,",
        new="            True,",
        breaks="file_hash passes whatever the bytes",
    ),
    Mutant(
        file="checks.py",
        old="            await env.write_file(path, contents)",
        new="            pass",
        breaks="restore_files restores nothing",
    ),
    Mutant(
        file="checks.py",
        old="passed=result.returncode == expected,",
        new="passed=True,",
        breaks="exit_code passes whatever the exit status",
    ),
    Mutant(
        file="adapters/base.py",
        old='            text = pattern.sub(f"<credential:{kind}>", text)',
        new="            pass",
        breaks="credentials are compared verbatim",
    ),
    Mutant(
        file="adapters/glovebox.py",
        old="            if not established:",
        new="            if False:",
        breaks="evidence is read from a guest that never went quiescent",
    ),
    Mutant(
        file="replay.py",
        old='        report.checks = [await c.run(sandboxes) for c in checks if c.phase == "live"]',
        new="        report.checks = []",
        breaks="live checks never run",
    ),
    Mutant(
        file="replay.py",
        old="            reader = await adapter.freeze(environments, report.notes)",
        new="            reader = await ReplayAdapter.freeze(adapter, environments, report.notes)",
        breaks="the adapter's freeze is bypassed, so evidence is read from a live guest",
    ),
    Mutant(
        file="report.py",
        old='        raise Inconclusive(f"{what}: {type(ex).__name__}: {ex}") from ex',
        new="        raise",
        breaks="a failing stage escapes as an error instead of an inconclusive verdict",
    ),
    Mutant(
        file="report.py",
        old='        if not outcomes:\n            self.verdict = "inconclusive"',
        new='        if False:\n            self.verdict = "inconclusive"',
        breaks="no outcome at all is a pass",
    ),
    Mutant(
        file="checks.py",
        old="        if self.phase not in PHASES:",
        new="        if False:",
        breaks="a misspelt phase silently drops its check",
    ),
    Mutant(
        file="checks.py",
        old="            passed=value >= passing,",
        new="            passed=True,",
        breaks="a scorer check passes whatever the scorer said",
    ),
    Mutant(
        file="checks.py",
        old="        if score is None or score.value == NOANSWER:",
        new="        if False:",
        breaks="a scorer that could not tell counts as a verdict",
    ),
    Mutant(
        file="scorer.py",
        old='{"verified": CORRECT, "refuted": INCORRECT}',
        new='{"verified": CORRECT, "refuted": CORRECT}',
        breaks="a refuted replay scores as a pass",
    ),
]


def run_suite() -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-p", "no:warnings"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def main() -> None:
    assert MUTANTS, "no mutants defined"
    if not run_suite():
        raise SystemExit("the suite is red before any mutation; fix it first")
    survivors: list[Mutant] = []
    for mutant in MUTANTS:
        path = SRC / mutant.file
        original = path.read_text()
        count = original.count(mutant.old)
        if count != 1:
            raise SystemExit(
                f"{mutant.file}: anchor occurs {count} times, not once: {mutant.old!r}"
            )
        path.write_text(original.replace(mutant.old, mutant.new))
        started = time.monotonic()
        try:
            survived = run_suite()
        finally:
            path.write_text(original)
        verdict = "SURVIVED" if survived else "killed"
        print(f"{verdict:8} {time.monotonic() - started:5.1f}s  {mutant.file}: {mutant.breaks}")
        if survived:
            survivors.append(mutant)
    if survivors:
        raise SystemExit(f"{len(survivors)} of {len(MUTANTS)} mutants survived")
    print(f"all {len(MUTANTS)} mutants killed")


if __name__ == "__main__":
    main()
