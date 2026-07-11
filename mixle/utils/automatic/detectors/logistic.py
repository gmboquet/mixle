"""Automatic detector for the logistic continuous family.

The logistic candidate is a symmetric real-line alternative with heavier tails than a Gaussian and a less
singular peak than a Laplace distribution. Its support gate accepts any finite real-valued sample; the
likelihood score and two-parameter BIC penalty decide whether its tail shape is justified.
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register


def _applies(arr: np.ndarray) -> bool:
    if arr.size == 0:
        return False
    return bool(np.all(np.isfinite(arr)))


def _fit(arr: np.ndarray):
    from scipy import stats

    try:
        loc, scale = stats.logistic.fit(arr)
    except Exception:  # noqa: BLE001
        return None
    if not (scale > 0.0) or not math.isfinite(scale) or not math.isfinite(loc):
        return None
    return float(loc), float(scale)


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from scipy import stats

    from mixle.utils.automatic.profiling import _bic_penalty_bits

    fit = _fit(arr)
    if fit is None:
        return None
    loc, scale = fit
    logpdf = stats.logistic.logpdf(arr, loc=loc, scale=scale)
    if not np.all(np.isfinite(logpdf)):
        return None
    nll_nats_per_obs = -float(np.mean(logpdf))
    return nll_nats_per_obs / math.log(2.0) + _bic_penalty_bits(2, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import LogisticDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _fit(_value_array_from_vdict(vdict))
    loc, scale = fit if fit is not None else (0.0, 1.0)
    return LogisticDistribution(loc, scale).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    fit = _fit(arr)
    if fit is None:
        return None
    loc, scale = fit
    return stats.logistic.cdf(arr, loc=loc, scale=scale)


register(
    Detector(name="logistic", kind="continuous", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=2)
)
