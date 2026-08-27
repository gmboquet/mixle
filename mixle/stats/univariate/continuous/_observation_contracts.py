"""Shared fail-closed contracts for continuous observations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class UnscorableObservation(ValueError):
    """A record a scorer refuses because it lies outside the law's admissible observation space.

    Distinct from an ordinary ``ValueError`` so a caller can tell "this record is not scorable" --
    a fact about the data, which serving reports as an unscorable record -- from "this call is
    malformed", which is a bug. It subclasses ``ValueError`` so existing handlers keep working.
    """


def finite_observations(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    """Return an owned one-dimensional finite observation array within optional bounds."""
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a one-dimensional finite real array") from exc
    if result.ndim != 1 or np.any(~np.isfinite(result)):
        raise ValueError(f"{label} must be a one-dimensional finite real array")
    if minimum is not None and np.any(result < minimum):
        raise ValueError(f"{label} must be greater than or equal to {minimum!r}")
    if maximum is not None and np.any(result > maximum):
        raise ValueError(f"{label} must be less than or equal to {maximum!r}")
    return np.array(result, dtype=np.float64, copy=True)


def finite_observation(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return one finite scalar observation within optional bounds."""
    result = finite_observations([value], label=label, minimum=minimum, maximum=maximum)
    return float(result[0])


def scored_observation(value: Any, *, label: str, allow_infinite: bool = False) -> float:
    """Return one scalar observation admitted by a scalar scorer's input policy.

    A scalar scorer and its encoder must admit the same observations, otherwise a
    caller sees a plausible score where a batch of the same data is refused. Every
    continuous encoder rejects NaN, so NaN is rejected here as well: it is malformed
    evidence rather than a point carrying zero density.

    ``allow_infinite`` selects between the two per-law encoder policies. Families whose
    encoder is :func:`finite_observations` reject infinities too and leave it ``False``;
    families whose encoder documents "finite or infinite real-valued observations"
    (Exponential, Gumbel, Laplace, Logistic, Uniform) pass ``True`` so an infinity keeps
    scoring as the zero-density limit it already scores as through the encoded path.

    The float coercion itself is intentionally permissive, matching the ``np.asarray``
    coercion the encoders apply; only the finiteness policy is enforced here.
    """

    result = float(value)
    if math.isnan(result):
        raise UnscorableObservation(f"{label} rejects NaN observations.")
    if not allow_infinite and math.isinf(result):
        raise UnscorableObservation(f"{label} rejects infinite observations.")
    return result


def consistent_anchored_triple(suff_stat: Any, sum_x: float, count: float) -> tuple[float, float, float] | None:
    """Return the ``(anchor, a_sum, a_sum2)`` payload of ``suff_stat`` when it is usable, else ``None``.

    Shared by every scalar family whose shift-anchored moment track is a single first/second-moment
    pair riding on the raw ``(sum, sum2, count)`` sufficient statistic -- currently the Gaussian and
    Logistic families (the higher-order families, GeneralizedGaussian and GeneralizedExtremeValue,
    carry more moments and are not this shape). Extracted after the duplicate-body scanner caught
    the two copies drifting apart risk: this is exactly the sibling-bug class D-0200/D-0202 spent
    three release waves closing, and a shared implementation means the next family that needs this
    payload gets the fix for free instead of a third copy to keep in sync.

    ``None`` falls back to the raw reduced-moment M-step, so a payload is only trusted when it is
    finite and agrees with the raw first moment it claims to describe -- a hand-built SuffStat whose
    payload contradicts its tuple must not silently change the estimate the tuple alone would have
    produced.
    """
    anchored = getattr(suff_stat, "anchored", None)
    if anchored is None or count <= 0.0:
        return None
    anchor, a_sum, a_sum2 = anchored
    if not (np.isfinite(anchor) and np.isfinite(a_sum) and np.isfinite(a_sum2)) or a_sum2 < 0.0:
        return None
    implied_sum = a_sum + count * anchor
    tolerance = 1.0e-6 * max(abs(sum_x), abs(count * anchor), 1.0)
    if abs(implied_sum - sum_x) > tolerance:
        return None
    return float(anchor), float(a_sum), float(a_sum2)


def scale_anchored_triple(
    anchor: float | None, a_sum: float, a_sum2: float, c: float
) -> tuple[float | None, float, float]:
    """Scale an ``(anchor, a_sum, a_sum2)`` track by ``c``, the way uniform weight scaling requires.

    Shared by every scalar family's ``scale()`` override for the reason
    :func:`consistent_anchored_triple` gives. Uniform weight scaling is exactly linear in both
    anchored moments and leaves the anchor -- a data value, not a statistic -- alone, so the track
    scales as the raw moments do; an unset anchor (``None``) passes through unchanged.
    """
    if anchor is None:
        return anchor, a_sum, a_sum2
    return anchor, a_sum * c, a_sum2 * c
