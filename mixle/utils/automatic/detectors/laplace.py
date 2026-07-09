"""Automatic detector for symmetric Laplace continuous data.

The detector fits the median and mean absolute deviation, scores the
double-exponential likelihood with a BIC-style penalty, and returns the matching
Mixle estimator factory.
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register


def _applies(arr: np.ndarray) -> bool:
    return arr.size > 0  # any real-valued data


def _fit(arr: np.ndarray) -> tuple[float, float] | None:
    """Return the Laplace MLE ``(loc, b)`` (median, mean absolute deviation), or None if degenerate."""
    loc = float(np.median(arr))
    b = float(np.mean(np.abs(arr - loc)))
    if not (b > 0.0):
        return None
    return loc, b


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from mixle.utils.automatic.profiling import _bic_penalty_bits

    fit = _fit(arr)
    if fit is None:
        return None
    _loc, b = fit
    nll_nats_per_obs = math.log(2.0 * b) + 1.0  # Laplace NLL at its MLE (scale = mean abs deviation)
    return nll_nats_per_obs / math.log(2.0) + _bic_penalty_bits(2, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import LaplaceDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _fit(_value_array_from_vdict(vdict))
    loc, b = fit if fit is not None else (0.0, 1.0)
    return LaplaceDistribution(loc, b).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    fit = _fit(arr)
    if fit is None:
        return None
    loc, b = fit
    return stats.laplace.cdf(arr, loc=loc, scale=b)


register(
    Detector(name="laplace", kind="continuous", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=2)
)
