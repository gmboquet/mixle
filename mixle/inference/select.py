"""Verifier-based selection — the generic test-time-compute selector.

Pure orchestration over a user ``score`` function: score every candidate, return the best (the
largest score, or the smallest when ``lower_is_better``). This is the "best-of-N" / verifier pattern
that test-time-compute stacks lean on — generate several candidates, score each with a verifier, keep
the winner — with no assumption about what a candidate *is* (a string, a model, a plan, a sample).

When ``heuristic_alpha`` is given, the result also carries a ``confident`` flag: whether the winner's
lead over the runner-up clears a normal score-spread heuristic. This is a ranking diagnostic, not a
conformal or bootstrap coverage guarantee. The former ``conformal_alpha`` spelling remains as a
deprecated compatibility alias and produces the same explicitly heuristic result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["SelectionResult", "select_best"]


@dataclass
class SelectionResult:
    """Result of a :func:`select_best` call.

    Attributes:
        best: the winning candidate (the actual object, not its index).
        best_index: the position of the winner in the input ``candidates``.
        scores: per-candidate scores in input order (a numpy float array).
        confident: whether the winner's lead clears the optional score-spread heuristic — ``None``
            when neither ``heuristic_alpha`` nor its deprecated alias was supplied.
        margin: the winner's lead over the runner-up (in score units, always ``>= 0``); ``None`` when
            there is a single candidate.
        band: the heuristic band the margin was compared against — ``None`` when no threshold was
            requested or there is a single candidate.
        confidence_method: the executed diagnostic, ``"normal_score_spread_heuristic"``, or ``None``.
        coverage_guarantee: always ``False``; one verifier score per selected candidate cannot
            establish held-out conformal or bootstrap coverage.
    """

    best: Any
    best_index: int
    scores: np.ndarray
    confident: bool | None = None
    margin: float | None = None
    band: float | None = None
    confidence_method: str | None = None
    coverage_guarantee: bool = False
    _extras: dict[str, Any] = field(default_factory=dict, repr=False)

    def __getitem__(self, key: str) -> Any:
        """Dict-style access (``result["best"]``) for callers that prefer a mapping."""
        return getattr(self, key)


def select_best(
    candidates: Any,
    *,
    score: Callable[[Any], float],
    lower_is_better: bool = False,
    conformal_alpha: float | None = None,
    heuristic_alpha: float | None = None,
) -> SelectionResult:
    """Score each candidate and return the best, the verifier-based test-time-compute selector.

    Args:
        candidates: an iterable of candidate objects (anything ``score`` accepts).
        score: a verifier ``score(candidate) -> float``; the winner maximizes it (or minimizes it
            when ``lower_is_better``).
        lower_is_better: if ``True``, the winner is the candidate with the *smallest* score.
        conformal_alpha: deprecated compatibility alias for ``heuristic_alpha``. It does not provide
            conformal coverage because candidate scores are neither held-out conformity scores nor
            exchangeable repeated measurements.
        heuristic_alpha: optional normal score-spread threshold level in ``(0, 1)``. When given,
            ``confident`` is a heuristic ranking flag only; ``coverage_guarantee`` remains ``False``.

    Returns:
        A :class:`SelectionResult`. It is also subscriptable (``result["best"]``), so callers may treat
        it as a small dict with keys ``best``, ``best_index``, ``scores``, ``confident``.

    Raises:
        ValueError: if ``candidates`` is empty, or ``conformal_alpha`` is outside ``(0, 1)``.
    """
    candidates = list(candidates)
    if not candidates:
        raise ValueError("select_best needs at least one candidate.")
    if not callable(score):
        raise TypeError("score must be callable")
    if not isinstance(lower_is_better, (bool, np.bool_)):
        raise TypeError("lower_is_better must be a boolean")
    if conformal_alpha is not None and heuristic_alpha is not None:
        raise ValueError("pass only heuristic_alpha; conformal_alpha is its deprecated alias")
    alpha = heuristic_alpha if heuristic_alpha is not None else conformal_alpha
    if alpha is not None and not (0.0 < alpha < 1.0):
        raise ValueError("heuristic_alpha must be in the open interval (0, 1).")
    if conformal_alpha is not None:
        import warnings

        warnings.warn(
            "conformal_alpha is a deprecated name for a score-spread heuristic and carries no "
            "conformal coverage guarantee; use heuristic_alpha",
            DeprecationWarning,
            stacklevel=2,
        )

    scores = np.asarray([float(score(c)) for c in candidates], dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("score must return one finite numeric value per candidate")
    # rank for "best": argmax, or argmin when lower is better. Use the sign-flipped score so the rest
    # of the logic (lead over runner-up) is written once for a maximization.
    oriented = -scores if lower_is_better else scores
    best_index = int(np.argmax(oriented))

    result = SelectionResult(
        best=candidates[best_index],
        best_index=best_index,
        scores=scores,
    )
    if len(candidates) == 1 or alpha is None:
        return result

    # lead of the winner over the runner-up (in the oriented, larger-is-better orientation)
    ordered = np.sort(oriented)[::-1]
    margin = float(ordered[0] - ordered[1])
    result.margin = margin

    # A z-scaled score-spread heuristic. One score per selected candidate cannot establish a sampling
    # distribution, exchangeability, or held-out calibration, so this intentionally carries no coverage
    # claim and is not labeled conformal/bootstrap in the result.
    from scipy.stats import norm

    spread = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    z = float(norm.ppf(1.0 - alpha / 2.0))
    band = z * spread
    result.band = band
    result.confident = bool(margin > band)
    result.confidence_method = "normal_score_spread_heuristic"
    return result
