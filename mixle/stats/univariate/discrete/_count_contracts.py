"""Shared validation for exact integer observations and non-negative weights."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np

_INT64_MIN = np.iinfo(np.int64).min
_INT64_MAX = np.iinfo(np.int64).max


def exact_integer_observations(
    values: Sequence[Any] | np.ndarray,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> np.ndarray:
    """Return an exact int64 array after validating every observation."""
    raw = np.asarray(values, dtype=object)
    encoded = np.empty(raw.shape, dtype=np.int64)
    for index, value in np.ndenumerate(raw):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{label} must be exact integers, not booleans.")
        if isinstance(value, numbers.Integral):
            integer = int(value)
        elif isinstance(value, numbers.Real):
            numeric = float(value)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"{label} must be finite exact integers.")
            integer = int(numeric)
        else:
            raise ValueError(f"{label} must be finite exact integers.")
        if integer < _INT64_MIN or integer > _INT64_MAX:
            raise ValueError(f"{label} must fit in signed 64-bit integer storage.")
        if minimum is not None and integer < minimum:
            raise ValueError(f"{label} must be at least {minimum}.")
        if maximum is not None and integer > maximum:
            raise ValueError(f"{label} must be at most {maximum}.")
        encoded[index] = integer
    return encoded


def nonnegative_weights(values: Any, *, shape: tuple[int, ...], label: str = "weights") -> np.ndarray:
    """Return finite, non-negative float64 weights aligned to an observation array."""
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != shape:
        raise ValueError(f"{label} must have the same shape as the encoded observations.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{label} must be finite and non-negative.")
    return weights
