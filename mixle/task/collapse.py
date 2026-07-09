"""``collapse_monitor`` -- the shared collapse-detection utility for self-improvement loops.

Every self-improvement round claims to be getting better. This shared check evaluates that claim
without each loop reimplementing collapse detection: across rounds, the held-out verified score must
be non-decreasing, and the proposal diversity must not be shrinking. A loop that improves its score by
collapsing onto a few candidates is overfitting to the verifier, not genuinely improving.

    verdict = collapse_monitor(history)
    verdict.ok                 # True iff both checks hold across every round
    verdict.reason             # None, or "score_decreased" / "diversity_shrunk" (which check failed)

``history`` is one entry per round: a dict with the round's held-out verified score and its pool of
candidates, or a precomputed diversity number.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def distinct_count_diversity(candidates: Sequence[Any]) -> float:
    """Diversity proxy: the number of distinct candidates (by ``str`` identity) in the round's pool."""
    return float(len({str(c) for c in candidates}))


def entropy_diversity(candidates: Sequence[Any]) -> float:
    """Diversity proxy: Shannon entropy (nats) of the candidate-frequency distribution in the round's pool."""
    counts: dict[str, int] = {}
    for c in candidates:
        key = str(c)
        counts[key] = counts.get(key, 0) + 1
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return float(-sum((k / n) * math.log(k / n) for k in counts.values()))


@dataclass
class CollapseVerdict:
    """The result of :func:`collapse_monitor`: ``ok`` plus which check failed, and the raw series."""

    ok: bool
    reason: str | None  # None if ok; else "score_decreased" or "diversity_shrunk"
    scores: list[float] = field(default_factory=list)
    diversities: list[float] = field(default_factory=list)
    failed_round: int | None = None  # index of the first round where the failing check tripped


def collapse_monitor(
    history: Sequence[Mapping[str, Any]],
    *,
    score_key: str = "score",
    candidates_key: str = "candidates",
    diversity_fn: Callable[[Sequence[Any]], float] = distinct_count_diversity,
    score_tol: float = 0.0,
    diversity_tol: float = 0.0,
) -> CollapseVerdict:
    """Check a self-improvement round history for collapse: score non-decreasing and diversity not shrinking.

    Each entry of ``history`` supplies the round's held-out verified score under ``score_key`` and either
    its candidate pool under ``candidates_key`` (diversity computed via ``diversity_fn``) or, when
    ``candidates_key`` is absent, a precomputed diversity number directly under ``"diversity"``.
    ``score_tol``/``diversity_tol`` allow a small, explicitly-named amount of round-to-round noise before
    a decrease/shrink counts as a real regression (0.0 = strict non-decreasing). The first round to violate
    either check ends the scan -- ``reason`` names which check failed, ``failed_round`` where.
    """
    scores: list[float] = []
    diversities: list[float] = []
    reason: str | None = None
    failed_round: int | None = None

    for i, round_ in enumerate(history):
        score = float(round_[score_key])
        if candidates_key in round_:
            diversity = float(diversity_fn(list(round_[candidates_key])))
        else:
            diversity = float(round_["diversity"])
        scores.append(score)
        diversities.append(diversity)

        if reason is None and i > 0:
            if scores[i] < scores[i - 1] - score_tol:
                reason, failed_round = "score_decreased", i
            elif diversities[i] < diversities[i - 1] - diversity_tol:
                reason, failed_round = "diversity_shrunk", i

    return CollapseVerdict(
        ok=reason is None, reason=reason, scores=scores, diversities=diversities, failed_round=failed_round
    )


__all__ = ["CollapseVerdict", "collapse_monitor", "distinct_count_diversity", "entropy_diversity"]
