"""Automatic detector for positive Weibull lifetime or strength data.

The detector fits a two-parameter Weibull with fixed zero location, rejects
near-exponential shapes, and supplies scoring and factory hooks for automatic
distribution selection.
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register


def _applies(arr: np.ndarray) -> bool:
    if arr.size == 0:
        return False
    return bool(np.all(np.isfinite(arr)) and np.all(arr > 0.0))


def _fit(arr: np.ndarray):
    """Two-parameter Weibull MLE (location fixed at 0). Returns (shape, scale) or None."""
    from scipy import stats

    try:
        shape, loc, scale = stats.weibull_min.fit(arr, floc=0.0)
    except Exception:  # noqa: BLE001
        return None
    if not (np.isfinite(shape) and np.isfinite(scale) and shape > 0.0 and scale > 0.0):
        return None
    return float(shape), float(scale)


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from scipy import stats

    from mixle.utils.automatic.profiling import _bic_penalty_bits

    fit = _fit(arr)
    if fit is None:
        return None
    shape, scale = fit

    # Weibull(shape == 1) is just an Exponential. When the data has no real Weibull
    # signature (shape indistinguishable from 1), refuse to compete so we never steal
    # exponential/gamma-shaped data with a degenerate fit.
    if abs(shape - 1.0) < 0.15:
        return None

    logpdf = stats.weibull_min.logpdf(arr, shape, loc=0.0, scale=scale)
    if not np.all(np.isfinite(logpdf)):
        return None
    nll_nats_per_obs = -float(np.mean(logpdf))
    return nll_nats_per_obs / math.log(2.0) + _bic_penalty_bits(2, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import WeibullDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _fit(_value_array_from_vdict(vdict))
    shape, scale = fit if fit is not None else (1.0, 1.0)
    return WeibullDistribution(shape, scale).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    fit = _fit(arr)
    if fit is None:
        return None
    shape, scale = fit
    return stats.weibull_min.cdf(arr, shape, loc=0.0, scale=scale)


register(
    Detector(name="weibull", kind="continuous", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=2)
)
