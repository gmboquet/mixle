"""Automatic detector for the Gumbel maximum extreme-value family.

The Gumbel candidate is intended for real-valued samples with a moderate, stable right skew that is
consistent with maxima or upper-tail exceedance summaries. The detector keeps the support gate broad
because the family is defined on the full real line, then relies on the skewness window, finite-fit checks,
and BIC penalty to prevent it from displacing Gaussian, exponential, or heavier-tailed alternatives.
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register

# Gumbel has a fixed right skewness of 12*sqrt(6)*zeta(3)/pi**3 ~= 1.1395. The gate accepts a
# deliberately broad neighbourhood around that value, enough for finite-sample variation but narrow
# enough to avoid symmetric data and strongly skewed exponential-like data.
_GUMBEL_SKEW = 1.1395470994046486
_SKEW_LO = 0.4
_SKEW_HI = 1.9


def _sample_skew(arr: np.ndarray) -> float | None:
    m = float(arr.mean())
    d = arr - m
    var = float(np.mean(d * d))
    if not (var > 0.0):
        return None
    return float(np.mean(d**3)) / (var**1.5)


def _applies(arr: np.ndarray) -> bool:
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return False
    skew = _sample_skew(arr)
    if skew is None:
        return False
    return _SKEW_LO <= skew <= _SKEW_HI


def _fit(arr: np.ndarray) -> tuple[float, float] | None:
    try:
        from scipy import stats

        loc, scale = stats.gumbel_r.fit(arr)
    except Exception:
        return None
    if not (scale > 0.0 and math.isfinite(scale) and math.isfinite(loc)):
        return None
    return float(loc), float(scale)


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from mixle.utils.automatic.profiling import _bic_penalty_bits

    if arr.size == 0:
        return None
    params = _fit(arr)
    if params is None:
        return None
    loc, scale = params
    z = (arr - loc) / scale
    # per-row NLL in nats: -[-log(scale) - z - exp(-z)]
    nll_nats = -np.mean(-math.log(scale) - z - np.exp(-z))
    if not math.isfinite(nll_nats):
        return None
    return float(nll_nats) / math.log(2.0) + _bic_penalty_bits(2, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import GumbelDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _fit(_value_array_from_vdict(vdict))
    loc, scale = fit if fit is not None else (0.0, 1.0)
    return GumbelDistribution(loc, scale).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    params = _fit(arr)
    if params is None:
        return None
    loc, scale = params
    return stats.gumbel_r.cdf(arr, loc=loc, scale=scale)


register(
    Detector(name="gumbel", kind="continuous", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=2)
)
