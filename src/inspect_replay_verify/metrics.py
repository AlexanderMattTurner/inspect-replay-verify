"""Verdict-rate metrics over replay scores."""

from __future__ import annotations

from inspect_ai.scorer import Metric, SampleScore, metric


def _rate(scores: list[SampleScore], verdict: str) -> float:
    if not scores:
        return 0.0
    hits = sum(
        1
        for s in scores
        if ((s.score.metadata or {}).get("replay") or {}).get("verdict") == verdict
    )
    return hits / len(scores)


@metric
def verified_rate() -> Metric:
    """Fraction of samples whose effect an independent replay reproduced and every check passed."""

    def compute(scores: list[SampleScore]) -> float:
        return _rate(scores, "verified")

    return compute


@metric
def refuted_rate() -> Metric:
    """Fraction of samples where a complete replay definitely failed a check."""

    def compute(scores: list[SampleScore]) -> float:
        return _rate(scores, "refuted")

    return compute


@metric
def inconclusive_rate() -> Metric:
    """Fraction of samples the replay could not reproduce or could not judge."""

    def compute(scores: list[SampleScore]) -> float:
        return _rate(scores, "inconclusive")

    return compute
