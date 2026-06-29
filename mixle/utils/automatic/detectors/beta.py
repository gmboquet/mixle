"""Beta continuous candidate -- unit-interval support (0, 1), flexible shapes via two positive params."""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register


def _applies(arr: np.ndarray) -> bool:
    # Support gate: Beta lives on the open unit interval, so every observation must be strictly in
    # (0, 1). This keeps the family out of the candidate set for signed (Gaussian) or unbounded
    # positive (Exponential/Gamma) data, which is the whole no-steal guarantee.
    if arr.size == 0:
        return False
    return bool(np.all(np.isfinite(arr)) and np.all(arr > 0.0) and np.all(arr < 1.0))


def _fit(arr: np.ndarray) -> tuple[float, float] | None:
    """Return MLE ``(a, b)`` with loc/scale pinned to the unit interval, or None if degenerate."""
    from scipy import stats

    if arr.size < 2:
        return None
    try:
        a, b, _loc, _scale = stats.beta.fit(arr, floc=0.0, fscale=1.0)
    except Exception:
        return None
    if not (math.isfinite(a) and math.isfinite(b) and a > 0.0 and b > 0.0):
        return None
    return float(a), float(b)


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from mixle.utils.automatic.profiling import _bic_penalty_bits

    params = _fit(arr)
    if params is None:
        return None
    a, b = params
    from scipy import stats

    logpdf = stats.beta.logpdf(arr, a, b)
    if not np.all(np.isfinite(logpdf)):
        return None
    nll_nats_per_obs = -float(np.mean(logpdf))
    return nll_nats_per_obs / math.log(2.0) + _bic_penalty_bits(2, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import BetaDistribution

    return BetaDistribution(1.0, 1.0).estimator()


def _cdf(arr: np.ndarray):
    from scipy import stats

    params = _fit(arr)
    if params is None:
        return None
    a, b = params
    return stats.beta.cdf(arr, a, b)


register(
    Detector(name="beta", kind="continuous", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=2)
)
