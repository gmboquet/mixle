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
    """Validate pseudo-observation moments and the variance floor.

    A pseudo-count is only *usable* for the moment whose prior statistic was also supplied.
    ``pseudo_count=(1.0, 1.0)`` with the default ``suff_stat=(None, None)`` is a long-standing,
    legitimate spelling that means "no pseudo-observations": the M-step falls back to the plain
    maximum-likelihood moment. That pairing is resolved where the moments are pooled
    (:func:`pooled_scalar_variance` and the estimators' ``estimate``), not here, so the counts the
    caller passed round-trip unchanged through the ``pseudo_count`` attribute.
    """
    raw_counts = broadcast_pseudo_count(pseudo_count, 2)
    if raw_counts is None:
        raw_counts = (None, None)
    if not isinstance(raw_counts, (tuple, list)) or len(raw_counts) != 2:
        raise ValueError("Gaussian pseudo-count must be a scalar or a two-item sequence")
    counts = tuple(
        None if value is None else _finite_scalar(value, label="Gaussian pseudo-count", nonnegative=True)
        for value in raw_counts
    )
    if not isinstance(suff_stat, tuple) or len(suff_stat) != 2:
        raise ValueError("Gaussian prior sufficient statistic must be a (mean, variance) tuple")
    prior_mean = None if suff_stat[0] is None else _finite_scalar(suff_stat[0], label="Gaussian prior mean")
    prior_variance = (
        None if suff_stat[1] is None else _finite_scalar(suff_stat[1], label="Gaussian prior variance", positive=True)
    )
    floor = _finite_scalar(min_covar, label="Gaussian min_covar", positive=True)
    return (counts[0], counts[1]), (prior_mean, prior_variance), floor


def scalar_gaussian_moments(value: Any) -> tuple[float, float, float]:
    """Validate ``(sum, sum_squares, count, count2)`` reduced moments."""
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError("Gaussian sufficient statistic must be a (sum, second_moment, count, count2) tuple")
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
    # A zero-count component is a starved one, which is a normal EM state and the whole reason the
    # mixture weight floor exists. Its moments are never read: the estimator sees count == 0 and
    # returns the floor defaults whatever they hold, so rejecting them fails closed on dead data and
    # turns a component the floor was built to revive into a hard crash. Finiteness and
    # non-negativity were both still checked above, unconditionally. Same reasoning as the
    # multivariate sibling in stats/multivariate/_vector_contracts.py.
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
    """Return the variance about ``mean`` for observed and virtual prior data.

    ``pseudo_count`` only contributes when ``prior_variance`` was supplied; an unpaired count
    carries no second-moment information, so the result is the plain observed variance. When a
    prior variance is supplied without a prior mean, the virtual observations are taken to be
    centered on ``mean`` and contribute no mean-offset scatter.
    """
    observed_scatter = sum_xx - 2.0 * mean * sum_x + count * mean * mean
    scale = max(abs(sum_xx), abs(2.0 * mean * sum_x), abs(count * mean * mean), 1.0)
    if observed_scatter < -1.0e-12 * scale:
        raise ValueError("Gaussian sufficient statistics imply a negative centered second moment")
    if observed_scatter < 1.0e-12 * scale:
        # Below the cancellation-noise bound of the terms just subtracted, a positive residue is
        # arithmetic, not data: a single-observation component lands at exactly 0.0 on one code
        # path and at +O(eps * scale) on an algebraically equivalent one, and the scale-relative
        # variance floor then treats the residue as a genuine (absurdly small) spread -- two
        # equivalent fits disagreed by ~9 nats per row (campaign wave). The tolerance mirrors the
        # negative-scatter check above: what would be forgiven as rounding below zero is equally
        # rounding above it.
        observed_scatter = 0.0
    observed_scatter = max(observed_scatter, 0.0)
    if pseudo_count not in (None, 0.0) and prior_variance is not None:
        offset = 0.0 if prior_mean is None else (prior_mean - mean) ** 2
        prior_scatter = pseudo_count * (prior_variance + offset)
        return (observed_scatter + prior_scatter) / (count + pseudo_count)
    if count == 0.0:
        return 0.0
    return observed_scatter / count
