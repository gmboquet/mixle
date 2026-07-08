"""Rayleigh continuous candidate -- non-negative data with a mode AWAY from zero (magnitude of a 2-D Gaussian)."""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register


def _fit(arr: np.ndarray) -> float | None:
    if arr.size == 0 or np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        return None
    sigma = math.sqrt(float(np.mean(arr * arr)) / 2.0)  # MLE for Rayleigh(sigma): sigma^2 = mean(x^2)/2
    return sigma if sigma > 0.0 and math.isfinite(sigma) else None


def _applies(arr: np.ndarray) -> bool:
    # non-negative data whose mode sits away from 0 (unlike the half-normal); the Rayleigh rises then falls.
    if arr.size == 0 or np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        return False
    mean = float(arr.mean())
    return mean > 0.0 and float(np.min(arr)) >= 0.0


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from scipy import stats

    from mixle.utils.automatic.profiling import _bic_penalty_bits

    sigma = _fit(arr)
    if sigma is None:
        return None
    nll_nats = -float(np.mean(stats.rayleigh.logpdf(arr, scale=sigma)))
    if not math.isfinite(nll_nats):
        return None
    return nll_nats / math.log(2.0) + _bic_penalty_bits(1, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import RayleighDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    sigma = _fit(_value_array_from_vdict(vdict))
    return RayleighDistribution(sigma if sigma is not None else 1.0).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    sigma = _fit(arr)
    return None if sigma is None else stats.rayleigh.cdf(arr, scale=sigma)


register(
    Detector(name="rayleigh", kind="continuous", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=1)
)
