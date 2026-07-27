"""Shared fail-closed contracts for continuous observations."""

from __future__ import annotations

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
