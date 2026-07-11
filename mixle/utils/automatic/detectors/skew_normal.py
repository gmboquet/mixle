"""Skew-normal continuous candidate -- a Gaussian tilted by an asymmetry (shape) parameter.

The skew-normal only earns its third parameter when the data carry a genuine, in-family asymmetry: the
gate requires a non-trivial sample skewness whose magnitude stays inside the family's reach (the
skew-normal can only attain ``|skewness| < ~0.995``), so symmetric Gaussian data and heavily-skewed
exponential data both fall outside the gate and are left to the builtins.
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register

# largest attainable |skewness| for the skew-normal (delta -> +/-1).
_B = math.sqrt(2.0 / math.pi)
_MAX_SKEW = ((4.0 - math.pi) / 2.0) * _B**3 / (1.0 - _B * _B) ** 1.5

# the gate window: enough skew that the third parameter is real, but inside the family's range.
_MIN_ABS_SKEW = 0.20
_MAX_ABS_SKEW = 0.95 * _MAX_SKEW


def _sample_skewness(arr: np.ndarray) -> float | None:
    mean = float(arr.mean())
    var = float(arr.var())
    if not (var > 0.0) or not math.isfinite(var):
        return None
    m3 = float(np.mean((arr - mean) ** 3))
    return m3 / (var**1.5)


def _applies(arr: np.ndarray) -> bool:
    if arr.size < 8 or not np.all(np.isfinite(arr)):
        return False
    g1 = _sample_skewness(arr)
    if g1 is None:
        return False
    return _MIN_ABS_SKEW <= abs(g1) <= _MAX_ABS_SKEW


def _fit(arr: np.ndarray):
    from scipy import stats

    try:
        shape, loc, scale = stats.skewnorm.fit(arr)
    except Exception:  # noqa: BLE001
        return None
    if not (scale > 0.0) or not (np.isfinite(shape) and np.isfinite(loc) and np.isfinite(scale)):
        return None
    return float(shape), float(loc), float(scale)


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from scipy import stats

    from mixle.utils.automatic.profiling import _bic_penalty_bits

    fitted = _fit(arr)
    if fitted is None:
        return None
    shape, loc, scale = fitted
    logpdf = stats.skewnorm.logpdf(arr, shape, loc=loc, scale=scale)
    if not np.all(np.isfinite(logpdf)):
        return None
    nll_nats_per_obs = -float(np.mean(logpdf))
    return nll_nats_per_obs / math.log(2.0) + _bic_penalty_bits(3, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import SkewNormalDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _fit(_value_array_from_vdict(vdict))
    shape, loc, scale = fit if fit is not None else (1.0, 0.0, 1.0)
    return SkewNormalDistribution(loc, scale, shape).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    fitted = _fit(arr)
    if fitted is None:
        return None
    shape, loc, scale = fitted
    return stats.skewnorm.cdf(arr, shape, loc=loc, scale=scale)


register(
    Detector(
        name="skew_normal", kind="continuous", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=3
    )
)
