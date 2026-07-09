"""Automatic detector for beta-distributed proportions and rates.

The beta candidate is reserved for finite samples whose observations all lie strictly inside the open unit
interval. That support rule distinguishes rates, probabilities, and normalized scores from signed
real-valued data or unbounded positive measurements. Parameter estimation pins ``loc`` and ``scale`` to
``0`` and ``1`` so the score compares only shape flexibility against the BIC penalty.
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register


def _applies(arr: np.ndarray) -> bool:
    # Support gate: Beta lives on the open unit interval, so every observation must be strictly in
    # (0, 1). This keeps the family out of the candidate set for signed or unbounded data.
    if arr.size == 0:
        return False
    return bool(np.all(np.isfinite(arr)) and np.all(arr > 0.0) and np.all(arr < 1.0))


def _fit(arr: np.ndarray) -> tuple[float, float] | None:
    """Return the MLE ``(a, b)`` with loc/scale pinned to the unit interval, or ``None``."""
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
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _fit(_value_array_from_vdict(vdict))
    a, b = fit if fit is not None else (1.0, 1.0)
    return BetaDistribution(a, b).estimator(pseudo_count=pseudo_count)


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
