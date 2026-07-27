"""Fail-closed contracts shared by scalar Gaussian-family estimators."""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.utils.aliasing import broadcast_pseudo_count


def _finite_scalar(
    value: Any,
    *,
    label: str,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError("%s must be a real scalar" % label)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("%s must be a real scalar" % label) from exc
    if not np.isfinite(result):
        raise ValueError("%s must be finite" % label)
    if positive and result <= 0.0:
        raise ValueError("%s must be positive" % label)
    if nonnegative and result < 0.0:
        raise ValueError("%s must be non-negative" % label)
    return result


def scalar_estimator_configuration(
    pseudo_count: Any,
    suff_stat: Any,
    min_covar: Any,
) -> tuple[tuple[float | None, float | None], tuple[float | None, float | None], float]:
    """Validate pseudo-observation moments and the variance floor."""
    raw_counts = broadcast_pseudo_count(pseudo_count, 2)
    if raw_counts is None:
        raw_counts = (None, None)
    if not isinstance(raw_counts, (tuple, list)) or len(raw_counts) != 2:
        raise ValueError("Gaussian pseudo-count must be a scalar or a two-item sequence")
    counts = tuple(
        None
        if value is None
        else _finite_scalar(value, label="Gaussian pseudo-count", nonnegative=True)
        for value in raw_counts
    )
    if not isinstance(suff_stat, tuple) or len(suff_stat) != 2:
        raise ValueError("Gaussian prior sufficient statistic must be a (mean, variance) tuple")
    prior_mean = (
        None
        if suff_stat[0] is None
        else _finite_scalar(suff_stat[0], label="Gaussian prior mean")
    )
    prior_variance = (
        None
        if suff_stat[1] is None
        else _finite_scalar(suff_stat[1], label="Gaussian prior variance", positive=True)
    )
    if counts[0] not in (None, 0.0) and prior_mean is None:
        raise ValueError("a positive Gaussian mean pseudo-count requires a finite prior mean")
    if counts[1] not in (None, 0.0) and (prior_mean is None or prior_variance is None):
        raise ValueError(
            "a positive Gaussian variance pseudo-count requires a finite prior mean and positive prior variance"
        )
    floor = _finite_scalar(min_covar, label="Gaussian min_covar", positive=True)
    return (counts[0], counts[1]), (prior_mean, prior_variance), floor


def scalar_gaussian_moments(value: Any) -> tuple[float, float, float]:
    """Validate ``(sum, sum_squares, count, count2)`` reduced moments."""
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError(
            "Gaussian sufficient statistic must be a (sum, second_moment, count, count2) tuple"
        )
    sum_x = _finite_scalar(value[0], label="Gaussian first-moment statistic")
    sum_xx = _finite_scalar(
        value[1],
        label="Gaussian second-moment statistic",
        nonnegative=True,
    )
    count = _finite_scalar(
        value[2],
        label="Gaussian sufficient-statistic count",
        nonnegative=True,
    )
    count2 = _finite_scalar(
        value[3],
        label="Gaussian second-moment count",
        nonnegative=True,
    )
    if count != count2:
        raise ValueError("Gaussian first- and second-moment counts must match")
    if count == 0.0 and (sum_x != 0.0 or sum_xx != 0.0):
        raise ValueError("zero-count Gaussian sufficient statistics require zero moments")
    return sum_x, sum_xx, count


def pooled_scalar_variance(
    sum_x: float,
    sum_xx: float,
    count: float,
    mean: float,
    pseudo_count: float | None,
    prior_mean: float | None,
    prior_variance: float | None,
) -> float:
    """Return the variance about ``mean`` for observed and virtual prior data."""
    observed_scatter = sum_xx - 2.0 * mean * sum_x + count * mean * mean
    scale = max(abs(sum_xx), abs(2.0 * mean * sum_x), abs(count * mean * mean), 1.0)
    if observed_scatter < -1.0e-12 * scale:
        raise ValueError("Gaussian sufficient statistics imply a negative centered second moment")
    observed_scatter = max(observed_scatter, 0.0)
    if pseudo_count not in (None, 0.0):
        prior_scatter = pseudo_count * (prior_variance + (prior_mean - mean) ** 2)
        return (observed_scatter + prior_scatter) / (count + pseudo_count)
    if count == 0.0:
        return 0.0
    return observed_scatter / count
