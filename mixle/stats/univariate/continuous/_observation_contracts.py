"""Shared fail-closed contracts for continuous observations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


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
        raise ValueError(f"{label} rejects NaN observations.")
    if not allow_infinite and math.isinf(result):
        raise ValueError(f"{label} rejects infinite observations.")
    return result
