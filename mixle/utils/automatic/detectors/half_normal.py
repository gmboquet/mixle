"""Half-normal continuous candidate -- non-negative data with its mode AT zero (a folded Gaussian)."""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register


def _fit(arr: np.ndarray) -> float | None:
    if arr.size == 0 or np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        return None
    sigma = math.sqrt(float(np.mean(arr * arr)))  # MLE for HalfNormal(sigma): sigma^2 = mean(x^2)
    return sigma if sigma > 0.0 and math.isfinite(sigma) else None


def _applies(arr: np.ndarray) -> bool:
    # non-negative data whose density piles up at 0 (mode at the boundary) -- the half-normal signature.
    if arr.size == 0 or np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        return False
    mean = float(arr.mean())
    if mean <= 0.0 or float(np.min(arr)) >= 0.25 * mean:  # needs mass near the 0 boundary (mode at 0)
        return False
    # a continuous family has P(X=0)=0: a spike of EXACT zeros signals a zero-atom model (Tweedie /
    # zero-inflated), not a smooth half-normal, so decline when exact zeros are more than a rounding-level share.
    return float(np.mean(arr == 0.0)) < 0.01


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from scipy import stats

    from mixle.utils.automatic.profiling import _bic_penalty_bits

    sigma = _fit(arr)
    if sigma is None:
        return None
    nll_nats = -float(np.mean(stats.halfnorm.logpdf(arr, scale=sigma)))
    if not math.isfinite(nll_nats):
        return None
    return nll_nats / math.log(2.0) + _bic_penalty_bits(1, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import HalfNormalDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    sigma = _fit(_value_array_from_vdict(vdict))
    return HalfNormalDistribution(sigma if sigma is not None else 1.0).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    sigma = _fit(arr)
    return None if sigma is None else stats.halfnorm.cdf(arr, scale=sigma)


register(
    Detector(
        name="half_normal", kind="continuous", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=1
    )
)
