"""Geometric discrete candidate -- the number of trials to the first success, on {1, 2, 3, ...} (memoryless)."""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register


def _fit(arr: np.ndarray) -> float | None:
    if arr.size == 0 or np.any(arr < 1) or np.any(arr != np.floor(arr)):
        return None
    mean = float(arr.mean())
    p = 1.0 / mean  # MLE on support {1,2,...}: E[X] = 1/p
    return p if 0.0 < p < 1.0 else None


def _applies(arr: np.ndarray) -> bool:
    # positive integer counts starting at 1 (mixle's geometric support); the mode is at 1 and decays.
    if arr.size == 0 or np.any(arr != np.floor(arr)) or np.any(arr < 1):
        return False
    return float(arr.mean()) > 1.0


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from scipy import stats

    from mixle.utils.automatic.profiling import _bic_penalty_bits

    p = _fit(arr)
    if p is None:
        return None
    nll_nats = -float(np.mean(stats.geom.logpmf(arr.astype(np.int64), p)))
    if not math.isfinite(nll_nats):
        return None
    return nll_nats / math.log(2.0) + _bic_penalty_bits(1, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import GeometricDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    p = _fit(_value_array_from_vdict(vdict))
    return GeometricDistribution(p if p is not None else 0.5).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    p = _fit(arr)
    return None if p is None else stats.geom.cdf(arr.astype(np.int64), p)


register(
    Detector(name="geometric", kind="discrete", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=1)
)
