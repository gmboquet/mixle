"""Laplace (double-exponential) continuous candidate -- symmetric, heavier peak/tails than Gaussian."""

import math

import numpy as np

from pysp.utils.automatic.detectors import Detector, register


def _applies(arr: np.ndarray) -> bool:
    return arr.size > 0  # any real-valued data


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from pysp.utils.automatic.profiling import _bic_penalty_bits

    loc = float(np.median(arr))
    b = float(np.mean(np.abs(arr - loc)))
    if not (b > 0.0):
        return None
    nll_nats_per_obs = math.log(2.0 * b) + 1.0  # Laplace NLL at its MLE (scale = mean abs deviation)
    return nll_nats_per_obs / math.log(2.0) + _bic_penalty_bits(2, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from pysp.stats import LaplaceDistribution

    return LaplaceDistribution(0.0, 1.0).estimator()


def _cdf(arr: np.ndarray):
    from scipy import stats

    loc = float(np.median(arr))
    b = float(np.mean(np.abs(arr - loc)))
    if not (b > 0.0):
        return None
    return stats.laplace.cdf(arr, loc=loc, scale=b)


register(
    Detector(name="laplace", kind="continuous", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=2)
)
