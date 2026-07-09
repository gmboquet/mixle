"""Verifier-based selection — the generic test-time-compute selector.

Pure orchestration over a user ``score`` function: score every candidate, return the best (the
largest score, or the smallest when ``lower_is_better``). This is the "best-of-N" / verifier pattern
that test-time-compute stacks lean on — generate several candidates, score each with a verifier, keep
the winner — with no assumption about what a candidate *is* (a string, a model, a plan, a sample).

When ``conformal_alpha`` is given, the result also carries a ``confident`` flag: whether the winner's
lead over the runner-up clears a calibration band derived from the spread of the candidate scores
(a simple conformal/bootstrap SE band). A clear winner is ``confident=True``; a near-tie is ``False``.
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
        confident: whether the winner's lead clears the conformal band — ``None`` when no
            ``conformal_alpha`` was supplied.
        margin: the winner's lead over the runner-up (in score units, always ``>= 0``); ``None`` when
            there is a single candidate.
        band: the conformal/bootstrap band the margin was compared against — ``None`` when no
            ``conformal_alpha`` was supplied or there is a single candidate.
    """

    best: Any
    best_index: int
    scores: np.ndarray
    confident: bool | None = None
    margin: float | None = None
    band: float | None = None
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
) -> SelectionResult:
    """Score each candidate and return the best, the verifier-based test-time-compute selector.

    Args:
        candidates: an iterable of candidate objects (anything ``score`` accepts).
        score: a verifier ``score(candidate) -> float``; the winner maximizes it (or minimizes it
            when ``lower_is_better``).
        lower_is_better: if ``True``, the winner is the candidate with the *smallest* score.
        conformal_alpha: optional miscoverage level in ``(0, 1)``. When given, the result's
            ``confident`` flag reports whether the winner's lead over the runner-up exceeds a
            conformal/bootstrap band at confidence ``1 - conformal_alpha`` (a ``z``-scaled estimate
            of the score spread). ``None`` (default) leaves ``confident`` unset.

    Returns:
        A :class:`SelectionResult`. It is also subscriptable (``result["best"]``), so callers may treat
        it as a small dict with keys ``best``, ``best_index``, ``scores``, ``confident``.

    Raises:
        ValueError: if ``candidates`` is empty, or ``conformal_alpha`` is outside ``(0, 1)``.
    """
    candidates = list(candidates)
    if not candidates:
        raise ValueError("select_best needs at least one candidate.")
    if conformal_alpha is not None and not (0.0 < conformal_alpha < 1.0):
        raise ValueError("conformal_alpha must be in the open interval (0, 1).")

    scores = np.asarray([float(score(c)) for c in candidates], dtype=float)
    # rank for "best": argmax, or argmin when lower is better. Use the sign-flipped score so the rest
    # of the logic (lead over runner-up) is written once for a maximization.
    oriented = -scores if lower_is_better else scores
    best_index = int(np.argmax(oriented))

    result = SelectionResult(
        best=candidates[best_index],
        best_index=best_index,
        scores=scores,
    )
    if len(candidates) == 1 or conformal_alpha is None:
        return result

    # lead of the winner over the runner-up (in the oriented, larger-is-better orientation)
    ordered = np.sort(oriented)[::-1]
    margin = float(ordered[0] - ordered[1])
    result.margin = margin

    # A simple conformal / bootstrap band: a z-scaled estimate of the score spread. With only a handful
    # of scalar scores, the distribution-free spread estimate is the sample standard deviation
    # of the scores (the conformity scores); the winner is "confident" when its lead clears z * that
    # spread, z = Phi^{-1}(1 - alpha/2). This reuses no per-item resampling because each candidate
    # contributes exactly one score -- bootstrap over a length-N score vector reduces to the same SE.
    from scipy.stats import norm

    spread = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    z = float(norm.ppf(1.0 - conformal_alpha / 2.0))
    band = z * spread
    result.band = band
    result.confident = bool(margin > band)
    return result
