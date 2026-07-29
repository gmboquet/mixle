"""``collapse_monitor`` -- the shared collapse-detection utility for self-improvement loops.

Every self-improvement round claims to be getting better. This shared check evaluates that claim
without each loop reimplementing collapse detection: across rounds, the held-out verified score must
be non-decreasing, and the proposal diversity must not be shrinking. A loop that improves its score by
collapsing onto a few candidates is overfitting to the verifier, not genuinely improving.

    verdict = collapse_monitor(history)
    verdict.ok                 # True iff enough evidence exists and both checks hold
    verdict.status             # "ok", "collapse_detected", or "insufficient_evidence"
    verdict.reason             # None, or the specific failure/insufficiency reason

``history`` is one entry per round: a dict with the round's held-out verified score and its pool of
candidates, or a precomputed diversity number.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np


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
    reason: str | None  # None if ok; otherwise a collapse or insufficiency reason
    status: str = "ok"
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
    a decrease/shrink counts as a real regression (0.0 = strict non-decreasing). At least two rounds
    are required to make a comparison. Shorter histories return ``status="insufficient_evidence"`` and
    ``ok=False`` rather than certifying success. Non-finite evidence is rejected.
    """
    if isinstance(history, (str, bytes)):
        raise TypeError("history must be a sequence of round mappings")
    try:
        rounds = list(history)
    except TypeError as exc:
        raise TypeError("history must be a sequence of round mappings") from exc
    for key, name in ((score_key, "score_key"), (candidates_key, "candidates_key")):
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} must be a non-empty string")
    if not callable(diversity_fn):
        raise TypeError("diversity_fn must be callable")

    def tolerance(value: Any, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    score_tol = tolerance(score_tol, "score_tol")
    diversity_tol = tolerance(diversity_tol, "diversity_tol")
    scores: list[float] = []
    diversities: list[float] = []
    reason: str | None = None
    failed_round: int | None = None

    for i, round_ in enumerate(rounds):
        if not isinstance(round_, Mapping):
            raise TypeError(f"history round {i} must be a mapping")
        if score_key not in round_:
            raise ValueError(f"history round {i} is missing {score_key!r}")
        try:
            score = float(round_[score_key])
        except (TypeError, ValueError) as exc:
            raise TypeError(f"history round {i} score must be a real scalar") from exc
        if not math.isfinite(score):
            raise ValueError(f"history round {i} score must be finite")
        if candidates_key in round_:
            candidates = round_[candidates_key]
            if isinstance(candidates, (str, bytes)):
                raise TypeError(f"history round {i} candidates must be a sequence, not a string")
            try:
                candidate_list = list(candidates)
            except TypeError as exc:
                raise TypeError(f"history round {i} candidates must be a sequence") from exc
            diversity_value = diversity_fn(candidate_list)
        else:
            if "diversity" not in round_:
                raise ValueError(f"history round {i} must provide {candidates_key!r} or 'diversity'")
            diversity_value = round_["diversity"]
        try:
            diversity = float(diversity_value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"history round {i} diversity must be a real scalar") from exc
        if not math.isfinite(diversity) or diversity < 0.0:
            raise ValueError(f"history round {i} diversity must be finite and non-negative")
        scores.append(score)
        diversities.append(diversity)

        if reason is None and i > 0:
            if scores[i] < scores[i - 1] - score_tol:
                reason, failed_round = "score_decreased", i
            elif diversities[i] < diversities[i - 1] - diversity_tol:
                reason, failed_round = "diversity_shrunk", i

    if len(rounds) < 2:
        return CollapseVerdict(
            ok=False,
            reason="insufficient_evidence",
            status="insufficient_evidence",
            scores=scores,
            diversities=diversities,
        )
    return CollapseVerdict(
        ok=reason is None,
        reason=reason,
        status="ok" if reason is None else "collapse_detected",
        scores=scores,
        diversities=diversities,
        failed_round=failed_round,
    )


__all__ = ["CollapseVerdict", "collapse_monitor", "distinct_count_diversity", "entropy_diversity"]
