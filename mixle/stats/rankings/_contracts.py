"""Shared validation contracts for ranking distributions, encoders, and accumulators."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mixle.stats.rankings._permutation_kernels import (
    _validate_orderings,
    _validate_permutation,
)


def exact_integer(value: Any, *, label: str) -> int:
    """Return one scalar exact integer, rejecting booleans and lossy coercions."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an exact integer.")
    raw = np.asarray(value)
    if raw.ndim != 0 or np.iscomplexobj(raw):
        raise TypeError(f"{label} must be an exact integer.")
    try:
        result = int(raw.item())
        numeric = float(raw.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be an exact integer.") from exc
    if not math.isfinite(numeric) or numeric != result:
        raise ValueError(f"{label} must be an exact integer.")
    return result


def positive_integer(value: Any, *, label: str, minimum: int = 1) -> int:
    """Return one exact integer greater than or equal to ``minimum``."""
    result = exact_integer(value, label=label)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return result


def nonnegative_integer(value: Any, *, label: str) -> int:
    """Return one exact nonnegative integer."""
    result = exact_integer(value, label=label)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative.")
    return result


def finite_nonnegative(value: Any, *, label: str) -> float:
    """Return one finite nonnegative scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a finite nonnegative number.")
    raw = np.asarray(value)
    if raw.ndim != 0 or np.iscomplexobj(raw):
        raise TypeError(f"{label} must be a finite nonnegative number.")
    try:
        result = float(raw.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a finite nonnegative number.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number.")
    return result


def finite_positive(value: Any, *, label: str) -> float:
    """Return one finite strictly positive scalar."""
    result = finite_nonnegative(value, label=label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return result


def sample_size(size: Any | None) -> int | None:
    """Validate the common ``sample(size=...)`` contract."""
    return None if size is None else nonnegative_integer(size, label="sample size")


def permutation(value: Any, dim: int, *, label: str = "ordering") -> np.ndarray:
    """Return one exact permutation of the declared model support."""
    return _validate_permutation(value, label=label, expected_dim=dim)


def permutation_batch(value: Any, dim: int, *, label: str = "orderings", allow_empty: bool = True) -> np.ndarray:
    """Return an exact two-dimensional permutation batch of declared width."""
    rows = _validate_orderings(value, label=label, expected_dim=dim)
    if not allow_empty and rows.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one ordering.")
    return rows


def pair(value: Any, dim: int, *, label: str = "comparison") -> tuple[int, int]:
    """Return one exact in-support ordered pair with distinct items."""
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a two-item sequence.")
    try:
        if len(value) != 2:
            raise TypeError(f"{label} must be a two-item sequence.")
    except TypeError as exc:
        raise TypeError(f"{label} must be a two-item sequence.") from exc
    first = exact_integer(value[0], label=f"{label} first item")
    second = exact_integer(value[1], label=f"{label} second item")
    if first < 0 or first >= dim or second < 0 or second >= dim:
        raise ValueError(f"{label} item identifiers must be in [0, {dim}).")
    if first == second:
        raise ValueError(f"{label} must compare two distinct items.")
    return first, second


def pair_batch(value: Any, dim: int, *, label: str = "comparisons", allow_empty: bool = True) -> np.ndarray:
    """Return an exact ``(N, 2)`` comparison batch."""
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1:] != (2,):
        raise ValueError(f"{label} must have shape (N, 2).")
    if not allow_empty and raw.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one comparison.")
    rows = [pair(row, dim, label=f"{label} row {index}") for index, row in enumerate(raw)]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def tie_comparison(value: Any, dim: int, *, label: str = "tie comparison") -> tuple[int, int, int]:
    """Return one canonical ``(lo, hi, outcome)`` comparison with outcome in ``{0,1,2}``."""
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a three-item sequence.")
    try:
        if len(value) != 3:
            raise TypeError(f"{label} must be a three-item sequence.")
    except TypeError as exc:
        raise TypeError(f"{label} must be a three-item sequence.") from exc
    first, second = pair(value[:2], dim, label=label)
    outcome = exact_integer(value[2], label=f"{label} outcome")
    if outcome < 0 or outcome > 2:
        raise ValueError(f"{label} outcome must be in {{0, 1, 2}}.")
    if first > second:
        first, second = second, first
        if outcome != 2:
            outcome = 1 - outcome
    return first, second, outcome


def tie_batch(value: Any, dim: int, *, label: str = "tie comparisons", allow_empty: bool = True) -> np.ndarray:
    """Return an exact canonical ``(N, 3)`` tie-comparison batch."""
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1:] != (3,):
        raise ValueError(f"{label} must have shape (N, 3).")
    if not allow_empty and raw.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one comparison.")
    rows = [tie_comparison(row, dim, label=f"{label} row {index}") for index, row in enumerate(raw)]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def weights(value: Any, row_count: int) -> np.ndarray:
    """Return one finite nonnegative weight per encoded row."""
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or len(result) != row_count:
        raise ValueError("weights must be one-dimensional with one value per observation.")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("weights must be finite and nonnegative.")
    return result


def count_matrix_statistics(
    value: Any,
    dim: int,
    *,
    label: str,
    entries_per_observation: float,
) -> tuple[float, np.ndarray]:
    """Validate a weighted total and fixed-shape nonnegative count matrix."""
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a two-item statistic tuple.")
    try:
        if len(value) != 2:
            raise TypeError(f"{label} must be a two-item statistic tuple.")
    except TypeError as exc:
        raise TypeError(f"{label} must be a two-item statistic tuple.") from exc
    total = finite_nonnegative(value[0], label=f"{label} total weight")
    matrix = np.asarray(value[1], dtype=np.float64)
    if matrix.shape != (dim, dim):
        raise ValueError(f"{label} count matrix must have shape ({dim}, {dim}).")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{label} count matrix must be finite and nonnegative.")
    expected_total = total * finite_nonnegative(
        entries_per_observation,
        label=f"{label} entries_per_observation",
    )
    if not math.isclose(float(matrix.sum()), expected_total, rel_tol=1.0e-10, abs_tol=1.0e-10):
        raise ValueError(f"{label} total weight must equal the count-matrix total.")
    return total, matrix.copy()


def matrix_statistics(value: Any, dim: int, *, label: str) -> tuple[float, np.ndarray]:
    """Validate a one-count-per-observation matrix statistic."""
    return count_matrix_statistics(
        value,
        dim,
        label=label,
        entries_per_observation=1.0,
    )


def bounded_sum_statistics(
    value: Any,
    *,
    label: str,
    minimum_per_observation: float,
    maximum_per_observation: float,
) -> tuple[float, float]:
    """Validate ``(weighted_sum, total_weight)`` with known per-observation bounds."""
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a two-item statistic tuple.")
    try:
        if len(value) != 2:
            raise TypeError(f"{label} must be a two-item statistic tuple.")
    except TypeError as exc:
        raise TypeError(f"{label} must be a two-item statistic tuple.") from exc
    weighted_sum = finite_nonnegative(value[0], label=f"{label} weighted sum")
    total = finite_nonnegative(value[1], label=f"{label} total weight")
    lower = finite_nonnegative(minimum_per_observation, label=f"{label} lower bound") * total
    upper = finite_nonnegative(maximum_per_observation, label=f"{label} upper bound") * total
    tolerance = 1.0e-10 * max(1.0, upper)
    if weighted_sum < lower - tolerance or weighted_sum > upper + tolerance:
        raise ValueError(f"{label} weighted sum is incompatible with total weight.")
    return weighted_sum, total


def tie_statistics(value: Any, dim: int, *, label: str) -> tuple[float, np.ndarray, np.ndarray]:
    """Validate win/tie count matrices and their total observation weight."""
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a three-item statistic tuple.")
    try:
        if len(value) != 3:
            raise TypeError(f"{label} must be a three-item statistic tuple.")
    except TypeError as exc:
        raise TypeError(f"{label} must be a three-item statistic tuple.") from exc
    total = finite_nonnegative(value[0], label=f"{label} total weight")
    matrices = []
    for name, raw in (("win", value[1]), ("tie", value[2])):
        matrix = np.asarray(raw, dtype=np.float64)
        if matrix.shape != (dim, dim):
            raise ValueError(f"{label} {name} counts must have shape ({dim}, {dim}).")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
            raise ValueError(f"{label} {name} counts must be finite and nonnegative.")
        matrices.append(matrix.copy())
    if not math.isclose(
        float(matrices[0].sum() + matrices[1].sum()),
        total,
        rel_tol=1.0e-10,
        abs_tol=1.0e-10,
    ):
        raise ValueError(f"{label} total weight must equal win-plus-tie count totals.")
    return total, matrices[0], matrices[1]
