"""Shared fail-closed contracts for sparse multinomial observations."""

from __future__ import annotations

from operator import index
from typing import Any, TypeVar

import numpy as np
from scipy.special import gammaln

T = TypeVar("T")


def exact_integer(value: Any, *, label: str, nonnegative: bool = False) -> int:
    if isinstance(value, (bool, np.bool_, str, bytes)) or np.ndim(value) != 0:
        raise TypeError("%s must be an integer" % label)
    try:
        result = index(value)
    except TypeError:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("%s must be an integer" % label) from exc
        if not np.isfinite(numeric) or numeric != np.floor(numeric):
            raise ValueError("%s must be an integer" % label)
        result = int(numeric)
    if nonnegative and result < 0:
        raise ValueError("%s must be non-negative" % label)
    return result


def finite_weight(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError("%s must be a real scalar" % label)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be a real scalar" % label) from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("%s must be finite and non-negative" % label)
    return result


def observation_weights(value: Any, rows: int, *, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError("%s must be real-valued, not boolean" % label)
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be numeric" % label) from exc
    if result.shape != (rows,):
        raise ValueError("%s must have exact shape (%d,)" % (label, rows))
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("%s must be finite and non-negative" % label)
    return result


def simplex(value: Any, *, label: str) -> tuple[np.ndarray, float]:
    try:
        probabilities = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be a finite nonempty simplex vector" % label) from exc
    if (
        probabilities.ndim != 1
        or probabilities.size == 0
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
    ):
        raise ValueError("%s must be a finite nonempty simplex vector" % label)
    total = float(probabilities.sum())
    if not np.isclose(total, 1.0, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("%s must sum to one" % label)
    normalized = probabilities.copy() / total
    normalized.setflags(write=False)
    return normalized, total


def canonical_integer_bag(
    value: Any,
    *,
    min_val: int | None = None,
    max_val: int | None = None,
    reject_outside: bool = False,
) -> tuple[list[tuple[int, int]], int, bool]:
    if isinstance(value, (str, bytes)):
        raise TypeError("integer multinomial observation must be a sequence of (category, count) pairs")
    try:
        entries = list(value)
    except TypeError as exc:
        raise TypeError("integer multinomial observation must be a sequence of (category, count) pairs") from exc
    combined: dict[int, int] = {}
    outside = False
    for item in entries:
        if isinstance(item, (str, bytes)):
            raise ValueError("integer multinomial entries must be (category, count) pairs")
        try:
            pair = list(item)
        except TypeError as exc:
            raise ValueError("integer multinomial entries must be (category, count) pairs") from exc
        if len(pair) != 2:
            raise ValueError("integer multinomial entries must be (category, count) pairs")
        category = exact_integer(pair[0], label="integer multinomial category")
        count = exact_integer(
            pair[1],
            label="integer multinomial count",
            nonnegative=True,
        )
        if count == 0:
            continue
        # Each bound is checked on its own: a pinned floor with a learned ceiling passes max_val=None
        # and still has to reject categories below the floor.
        is_outside = (min_val is not None and category < min_val) or (max_val is not None and category > max_val)
        if is_outside:
            outside = True
            if reject_outside:
                raise ValueError("integer multinomial category is outside the configured support")
        combined[category] = combined.get(category, 0) + count
    pairs = sorted(combined.items())
    return pairs, sum(combined.values()), outside


def canonical_bag(value: Any) -> tuple[list[tuple[Any, int]], int]:
    if isinstance(value, (str, bytes)):
        raise TypeError("multinomial observation must be a sequence of (value, count) pairs")
    try:
        entries = list(value)
    except TypeError as exc:
        raise TypeError("multinomial observation must be a sequence of (value, count) pairs") from exc
    combined: dict[Any, int] = {}
    order: list[Any] = []
    for item in entries:
        if isinstance(item, (str, bytes)):
            raise ValueError("multinomial entries must be (value, count) pairs")
        try:
            pair = list(item)
        except TypeError as exc:
            raise ValueError("multinomial entries must be (value, count) pairs") from exc
        if len(pair) != 2:
            raise ValueError("multinomial entries must be (value, count) pairs")
        category = pair[0]
        try:
            hash(category)
        except TypeError as exc:
            raise TypeError("multinomial category values must be hashable") from exc
        count = exact_integer(
            pair[1],
            label="multinomial count",
            nonnegative=True,
        )
        if count == 0:
            continue
        if category not in combined:
            combined[category] = 0
            order.append(category)
        combined[category] += count
    pairs = [(category, combined[category]) for category in order]
    return pairs, sum(combined.values())


def log_coefficient(counts: Any) -> float:
    array = np.asarray(counts, dtype=np.float64)
    total = float(array.sum())
    return float(gammaln(total + 1.0) - np.sum(gammaln(array + 1.0)))
