"""Inverse-gamma continuous candidate -- strictly-positive data with a heavy right tail (1/Gamma)."""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register


def _fit(arr: np.ndarray) -> tuple[float, float] | None:
    if arr.size == 0 or np.any(arr <= 0.0) or not np.all(np.isfinite(arr)):
        return None
    try:
        from scipy import stats

        a, _loc, scale = stats.invgamma.fit(arr, floc=0.0)  # fix loc at 0: a positive-support family
    except Exception:  # noqa: BLE001
        return None
    if not (a > 0.0 and scale > 0.0 and math.isfinite(a) and math.isfinite(scale)):
        return None
    return float(a), float(scale)


def _applies(arr: np.ndarray) -> bool:
    return arr.size > 0 and bool(np.all(arr > 0.0)) and bool(np.all(np.isfinite(arr)))


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from scipy import stats

    from mixle.utils.automatic.profiling import _bic_penalty_bits

    fit = _fit(arr)
    if fit is None:
        return None
    a, scale = fit
    nll_nats = -float(np.mean(stats.invgamma.logpdf(arr, a, scale=scale)))
    if not math.isfinite(nll_nats):
        return None
    return nll_nats / math.log(2.0) + _bic_penalty_bits(2, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import InverseGammaDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _fit(_value_array_from_vdict(vdict))
    a, scale = fit if fit is not None else (2.0, 1.0)
    return InverseGammaDistribution(a, scale).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    fit = _fit(arr)
    if fit is None:
        return None
    a, scale = fit
    return stats.invgamma.cdf(arr, a, scale=scale)


register(
    Detector(
        name="inverse_gamma", kind="continuous", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=2
    )
)
