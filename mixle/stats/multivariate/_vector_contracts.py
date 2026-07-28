"""Shared fail-closed contracts for finite vector-valued probability families."""

from __future__ import annotations

from operator import index
from typing import Any

import numpy as np

from mixle.utils.aliasing import broadcast_pseudo_count


def dimension(value: Any, *, label: str, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be a positive integer" % label)
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError("%s must be a positive integer" % label) from exc
    if result <= 0:
        raise ValueError("%s must be a positive integer" % label)
    return result


def finite_scalar(value: Any, *, label: str, nonnegative: bool = False, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError("%s must be a real scalar" % label)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be a real scalar" % label) from exc
    if not np.isfinite(result):
        raise ValueError("%s must be finite" % label)
    if positive and not result > 0.0:
        raise ValueError("%s must be positive" % label)
    if nonnegative and result < 0.0:
        raise ValueError("%s must be non-negative" % label)
    return result


def vector(value: Any, *, label: str, dim: int | None = None, positive: bool = False) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be a finite numeric vector" % label) from exc
    expected = None if dim is None else (dim,)
    if result.ndim != 1 or result.size == 0 or (expected is not None and result.shape != expected):
        shape_text = "a nonempty one-dimensional vector" if dim is None else "exact shape (%d,)" % dim
        raise ValueError("%s must have %s" % (label, shape_text))
    if np.any(~np.isfinite(result)):
        raise ValueError("%s must contain only finite values" % label)
    if positive and np.any(result <= 0.0):
        raise ValueError("%s must contain only positive values" % label)
    return result.copy()


def matrix(
    value: Any,
    *,
    label: str,
    dim: int,
    symmetric: bool = False,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be a finite numeric matrix" % label) from exc
    if result.shape != (dim, dim):
        raise ValueError("%s must have exact shape (%d, %d)" % (label, dim, dim))
    if np.any(~np.isfinite(result)):
        raise ValueError("%s must contain only finite values" % label)
    if symmetric and not np.allclose(result, result.T):
        raise ValueError("%s must be symmetric" % label)
    return result.copy()


def event(value: Any, dim: int, *, label: str) -> np.ndarray:
    return vector(value, label=label, dim=dim)


def batch(value: Any, dim: int, *, label: str, allow_empty: bool = True) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be a finite numeric matrix" % label) from exc
    if allow_empty and result.shape == (0,):
        return np.empty((0, dim), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != dim:
        raise ValueError("%s must have exact shape (N, %d)" % (label, dim))
    if not allow_empty and len(result) == 0:
        raise ValueError("%s must contain at least one row" % label)
    if np.any(~np.isfinite(result)):
        raise ValueError("%s must contain only finite values" % label)
    return result


def weight(value: Any, *, label: str = "observation weight") -> float:
    return finite_scalar(value, label=label, nonnegative=True)


def weights(value: Any, rows: int, *, label: str = "observation weights") -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s must be numeric" % label) from exc
    if result.shape != (rows,):
        raise ValueError("%s must have exact shape (%d,)" % (label, rows))
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("%s must be finite and non-negative" % label)
    return result


def pseudo_counts(value: Any, *, label: str) -> tuple[float | None, float | None]:
    raw = broadcast_pseudo_count(value, 2)
    if raw is None:
        raw = (None, None)
    if not isinstance(raw, (tuple, list)) or len(raw) != 2:
        raise ValueError("%s must be a scalar or a two-item sequence" % label)
    checked = tuple(None if item is None else finite_scalar(item, label=label, nonnegative=True) for item in raw)
    return checked[0], checked[1]


def gaussian_prior_statistics(
    value: Any,
    dim: int | None,
    *,
    diagonal: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("Gaussian prior sufficient statistic must be a (mean, covariance) tuple")
    checked_dim = dimension(dim, label="Gaussian dimension", allow_none=True)
    prior_mean = None
    prior_covar = None
    if value[0] is not None:
        prior_mean = vector(value[0], label="Gaussian prior mean", dim=checked_dim)
        if checked_dim is None:
            checked_dim = len(prior_mean)
    if value[1] is not None:
        if diagonal:
            prior_covar = vector(
                value[1],
                label="diagonal Gaussian prior covariance",
                dim=checked_dim,
                positive=True,
            )
            if checked_dim is None:
                checked_dim = len(prior_covar)
        else:
            try:
                raw = np.asarray(value[1], dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Gaussian prior covariance must be a finite numeric matrix") from exc
            if checked_dim is None:
                if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[0] != raw.shape[1]:
                    raise ValueError("Gaussian prior covariance must be a nonempty square matrix")
                checked_dim = raw.shape[0]
            prior_covar = matrix(
                raw,
                label="Gaussian prior covariance",
                dim=checked_dim,
                symmetric=True,
            )
            try:
                np.linalg.cholesky(prior_covar)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Gaussian prior covariance must be positive definite") from exc
    for parameter in (prior_mean, prior_covar):
        if parameter is not None:
            parameter.setflags(write=False)
    return prior_mean, prior_covar, checked_dim


def require_pseudo_moments(
    counts: tuple[float | None, float | None],
    prior_mean: np.ndarray | None,
    prior_covar: np.ndarray | None,
) -> None:
    if counts[0] not in (None, 0.0) and prior_mean is None:
        raise ValueError("a positive Gaussian mean pseudo-count requires a finite prior mean")
    if counts[1] not in (None, 0.0) and (prior_mean is None or prior_covar is None):
        raise ValueError(
            "a positive Gaussian covariance pseudo-count requires a finite prior mean and valid prior covariance"
        )


def pooled_gaussian_covariance(
    sum_x: np.ndarray,
    sum_xx: np.ndarray,
    count: float,
    mean: np.ndarray,
    pseudo_count: float | None,
    prior_mean: np.ndarray | None,
    prior_covar: np.ndarray | None,
    *,
    diagonal: bool,
) -> np.ndarray:
    if diagonal:
        observed_scatter = sum_xx - 2.0 * mean * sum_x + count * mean * mean
        scale = np.maximum.reduce(
            (
                np.abs(sum_xx),
                np.abs(2.0 * mean * sum_x),
                np.abs(count * mean * mean),
                np.ones_like(mean),
            )
        )
        if np.any(observed_scatter < -1.0e-6 * scale):
            raise ValueError("Gaussian sufficient statistics imply a negative centered second moment")
        observed_scatter = np.maximum(observed_scatter, 0.0)
        if pseudo_count not in (None, 0.0):
            prior_scatter = pseudo_count * (prior_covar + (prior_mean - mean) ** 2)
            return (observed_scatter + prior_scatter) / (count + pseudo_count)
    else:
        observed_scatter = sum_xx - np.outer(mean, sum_x) - np.outer(sum_x, mean) + count * np.outer(mean, mean)
        observed_scatter = 0.5 * (observed_scatter + observed_scatter.T)
        eigenvalues = np.linalg.eigvalsh(observed_scatter)
        scale = max(
            float(np.linalg.norm(sum_xx, ord=2)),
            float(np.linalg.norm(np.outer(mean, sum_x), ord=2)),
            float(np.linalg.norm(count * np.outer(mean, mean), ord=2)),
            1.0,
        )
        if eigenvalues[0] < -1.0e-6 * scale:
            raise ValueError("Gaussian sufficient statistics imply a non-positive-semidefinite scatter")
        if pseudo_count not in (None, 0.0):
            delta = prior_mean - mean
            prior_scatter = pseudo_count * (prior_covar + np.outer(delta, delta))
            return (observed_scatter + prior_scatter) / (count + pseudo_count)
    if count == 0.0:
        return np.zeros_like(sum_xx)
    return observed_scatter / count


def marginal_indices(value: Any, dim: int) -> np.ndarray:
    try:
        raw = list(value)
    except TypeError as exc:
        raise TypeError("kept indices must be an iterable of integers") from exc
    if not raw:
        raise ValueError("keep at least one dimension")
    checked: list[int] = []
    for item in raw:
        if isinstance(item, (bool, np.bool_)):
            raise TypeError("kept indices must be integers")
        try:
            coordinate = index(item)
        except TypeError as exc:
            raise TypeError("kept indices must be integers") from exc
        if not 0 <= coordinate < dim:
            raise ValueError("kept indices must be in [0, dim)")
        checked.append(coordinate)
    if len(set(checked)) != len(checked):
        raise ValueError("kept indices must be unique; use an explicit duplication transform to repeat coordinates")
    return np.asarray(checked, dtype=np.int64)


def gaussian_moments(
    value: Any,
    dim: int | None,
    *,
    diagonal: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, float, int | None]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("Gaussian sufficient statistic must be a (sum, second_moment, count) tuple")
    count = weight(value[2], label="Gaussian sufficient-statistic count")
    if value[0] is None or value[1] is None:
        if value[0] is not None or value[1] is not None or count != 0.0:
            raise ValueError("empty Gaussian sufficient statistics require both moments to be None and count zero")
        return None, None, count, dim
    sum_x = vector(value[0], label="Gaussian first-moment statistic", dim=dim)
    inferred_dim = len(sum_x) if dim is None else dim
    if diagonal:
        sum_xx = vector(value[1], label="Gaussian diagonal second-moment statistic", dim=inferred_dim)
        if np.any(sum_xx < 0.0):
            raise ValueError("Gaussian diagonal second-moment statistic must be non-negative")
    else:
        sum_xx = matrix(
            value[1],
            label="Gaussian second-moment statistic",
            dim=inferred_dim,
        )
        scale = max(float(np.linalg.norm(sum_xx, ord=2)), 1.0)
        # sum_i w_i x_i x_i^T is symmetric by construction, so any asymmetry here is float
        # accumulation noise -- and float32/GPU EM produces exactly that. Requiring exact symmetry
        # rejected those legitimate sufficient statistics outright. Refuse asymmetry too large to be
        # noise (same relative tolerance as the PSD check below, which already concedes the point),
        # then symmetrize the remainder rather than failing on it.
        if np.max(np.abs(sum_xx - sum_xx.T), initial=0.0) > 1.0e-6 * scale:
            raise ValueError("Gaussian second-moment statistic must be symmetric")
        sum_xx = 0.5 * (sum_xx + sum_xx.T)
        if np.linalg.eigvalsh(sum_xx)[0] < -1.0e-6 * scale:
            raise ValueError("Gaussian second-moment statistic must be positive semidefinite")
    if count == 0.0 and (np.any(sum_x != 0.0) or np.any(sum_xx != 0.0)):
        raise ValueError("zero-count Gaussian sufficient statistics require zero moments")
    return sum_x, sum_xx, count, inferred_dim


def student_t_moments(
    value: Any,
    dim: int | None,
) -> tuple[float, float, np.ndarray | None, np.ndarray | None, int | None]:
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError("Student-t sufficient statistic must be a (count, latent_weight, sum, second_moment) tuple")
    count = weight(value[0], label="Student-t sufficient-statistic count")
    sum_u = weight(value[1], label="Student-t latent-weight total")
    if value[2] is None or value[3] is None:
        if value[2] is not None or value[3] is not None or count != 0.0 or sum_u != 0.0:
            raise ValueError(
                "empty Student-t sufficient statistics require both moments to be None and both totals zero"
            )
        return count, sum_u, None, None, dim
    sum_ux = vector(value[2], label="Student-t first-moment statistic", dim=dim)
    inferred_dim = len(sum_ux) if dim is None else dim
    sum_uxx = matrix(
        value[3],
        label="Student-t second-moment statistic",
        dim=inferred_dim,
        symmetric=True,
    )
    scale = max(float(np.linalg.norm(sum_uxx, ord=2)), 1.0)
    if np.linalg.eigvalsh(sum_uxx)[0] < -1.0e-6 * scale:
        raise ValueError("Student-t second-moment statistic must be positive semidefinite")
    if (count == 0.0) != (sum_u == 0.0):
        raise ValueError("Student-t count and latent-weight total must both be zero or both be positive")
    if sum_u == 0.0 and (np.any(sum_ux != 0.0) or np.any(sum_uxx != 0.0)):
        raise ValueError("zero Student-t latent weight requires zero weighted moments")
    return count, sum_u, sum_ux, sum_uxx, inferred_dim
