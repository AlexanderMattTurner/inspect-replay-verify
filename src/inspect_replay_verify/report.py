"""What a replay concluded, and the one way any step of it declares it cannot conclude."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .checks import CheckOutcome
from .transcript import RecordedCall

Verdict = Literal["verified", "refuted", "inconclusive"]


class Inconclusive(Exception):
    """Raised by any step that cannot stand for the original run; the message says why.

    The replay turns it into an ``inconclusive`` verdict at exactly one place, so no step
    can let a replay it could not complete fall through to a pass.
    """


@contextmanager
def stage(what: str) -> Iterator[None]:
    """Report any failure inside the block as `Inconclusive`, prefixed with `what`."""
    try:
        yield
    except Inconclusive:
        raise
    except Exception as ex:
        raise Inconclusive(f"{what}: {type(ex).__name__}: {ex}") from ex


@dataclass(kw_only=True, slots=True)
class CallReplay:
    """How one recorded call fared on replay."""

    id: str
    function: str
    disposition: str
    reason: str = ""
    replayed: bool = False
    diverged: bool | None = None
    credential_scoped: bool = False
    nondeterministic: bool = False
    recorded_error: str | None = None
    replay_error: str | None = None
    replay_output: str | None = None

    @classmethod
    def of(cls, call: RecordedCall) -> "CallReplay":
        return cls(
            id=call.id,
            function=call.function,
            disposition=call.disposition,
            reason=call.reason,
            nondeterministic=call.nondeterministic,
            recorded_error=render_error(call.error),
        )


def render_error(error: Any) -> str | None:
    return None if error is None else f"{error.type}: {error.message}"


@dataclass(kw_only=True)
class ReplayReport:
    """The outcome of one independent replay, filled in as the replay proceeds."""

    verdict: Verdict = "inconclusive"
    explanation: str = ""
    sandbox_type: str = ""
    adapter: str = ""
    transcript_source: str = ""
    calls: list[CallReplay] = field(default_factory=list)
    checks: list[CheckOutcome] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def divergences(self) -> int:
        """Replayed calls whose output differed from the recording, tolerated ones excluded."""
        return sum(1 for c in self.calls if c.diverged and not c.credential_scoped)

    def judge(self, outcomes: Sequence[CheckOutcome]) -> None:
        """Set the verdict from the checks of a complete replay."""
        self.checks = list(outcomes)
        failed = [o for o in outcomes if o.passed is False]
        unknown = [o for o in outcomes if o.passed is None]
        if failed:
            self.verdict = "refuted"
            self.explanation = "replay failed: " + _describe(failed)
        elif unknown:
            self.verdict = "inconclusive"
            self.explanation = "could not evaluate: " + _describe(unknown)
        else:
            self.verdict = "verified"
            self.explanation = f"all {len(outcomes)} check(s) passed on an independent replay"
        if self.divergences and self.verdict != "inconclusive":
            self.explanation += f" ({self.divergences} call output divergence(s) recorded)"

    @contextmanager
    def concluding(self) -> Iterator[None]:
        """Turn an `Inconclusive` raised inside the block into this report's verdict."""
        try:
            yield
        except Inconclusive as ex:
            self.verdict = "inconclusive"
            self.explanation = str(ex)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "explanation": self.explanation,
            "sandbox_type": self.sandbox_type,
            "adapter": self.adapter,
            "transcript_source": self.transcript_source,
            "calls_recorded": len(self.calls),
            "calls_replayed": sum(1 for c in self.calls if c.replayed),
            "divergences": self.divergences,
            "calls": [asdict(c) for c in self.calls],
            "checks": [asdict(c) for c in self.checks],
            "notes": self.notes,
        }


def _describe(outcomes: Sequence[CheckOutcome]) -> str:
    return "; ".join(f"{o.name} ({o.explanation})" for o in outcomes)
