"""Generalized Extreme Value (GEV) continuous candidate -- the limit law of block maxima.

GEV is the Fisher-Tippett-Gnedenko limit for normalized block maxima (floods, wind speeds, record
losses). It is a flexible 3-parameter family (location, scale, shape) whose Gumbel sub-case (shape 0)
spans all reals, so the support gate is any real-valued data; the 3-parameter BIC penalty keeps it
from stealing the simpler symmetric Gaussian / monotone Exponential when those are the appropriate fit.
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register


def _applies(arr: np.ndarray) -> bool:
    return arr.size > 0  # any real-valued data (block maxima); Gumbel sub-case spans all reals


def _fit_params(arr: np.ndarray):
    """Fit GEV by MLE (scipy ``genextreme``); return (c, loc, scale) or None if degenerate."""
    from scipy import stats

    if arr.size < 3 or float(np.std(arr)) <= 0.0:
        return None
    try:
        c, loc, scale = stats.genextreme.fit(arr)
    except Exception:
        return None
    if not (np.isfinite(c) and np.isfinite(loc) and np.isfinite(scale)) or scale <= 0.0:
        return None
    return float(c), float(loc), float(scale)


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from scipy import stats

    from mixle.utils.automatic.profiling import _bic_penalty_bits

    fit = _fit_params(arr)
    if fit is None:
        return None
    c, loc, scale = fit
    logpdf = stats.genextreme.logpdf(arr, c, loc=loc, scale=scale)
    if not np.all(np.isfinite(logpdf)):
        return None
    nll_nats_per_obs = float(-np.mean(logpdf))
    return nll_nats_per_obs / math.log(2.0) + _bic_penalty_bits(3, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import GeneralizedExtremeValueDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _fit_params(_value_array_from_vdict(vdict))
    c, loc, scale = fit if fit is not None else (0.1, 0.0, 1.0)
    return GeneralizedExtremeValueDistribution(loc, scale, c).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    fit = _fit_params(arr)
    if fit is None:
        return None
    c, loc, scale = fit
    return stats.genextreme.cdf(arr, c, loc=loc, scale=scale)


register(
    Detector(
        name="generalized_extreme_value",
        kind="continuous",
        applies=_applies,
        score=_score,
        factory=_factory,
        cdf=_cdf,
        n_params=3,
    )
)
