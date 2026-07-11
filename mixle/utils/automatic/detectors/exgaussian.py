"""Exponentially-modified Gaussian (ex-Gaussian) candidate -- a Gaussian convolved with an exponential.

The ex-Gaussian ``X = N(mu, sigma^2) + Exp(rate=lam)`` is a flexible 3-parameter right-skewed family that
DEGENERATES into a Gaussian (as ``lam -> inf``, the exponential component vanishes) and into a pure
exponential (as ``sigma -> 0``). That overlap is exactly what makes it dangerous in automatic selection: a
full 3-parameter fit can shadow the simpler builtins on their own data. So the support gate is deliberately
TIGHT -- it fires only on data whose sample skewness sits strictly inside the ex-Gaussian's own range
``(0, 2)``, away from the symmetric (Gaussian, skew ~ 0) and pure-exponential (skew ~ 2) boundaries -- and the
BIC carries the full 3-parameter penalty so it does not win on a marginal difference.
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register

# The ex-Gaussian skewness is 2 / (1 + (lam*sigma)^2)^1.5, which lives strictly in (0, 2). Fire only on a
# clearly-skewed-but-not-exponential interior band: this keeps the family off symmetric Gaussian data
# (skew ~ 0) and off the pure-exponential boundary (skew ~ 2), where a simpler builtin is the appropriate answer.
_SKEW_LO = 0.35
_SKEW_HI = 1.75


def _sample_skew(arr: np.ndarray) -> float | None:
    m = float(arr.mean())
    d = arr - m
    var = float(np.mean(d * d))
    if not (var > 0.0):
        return None
    return float(np.mean(d**3)) / (var**1.5)


def _applies(arr: np.ndarray) -> bool:
    if arr.size < 8 or not np.all(np.isfinite(arr)):
        return False
    skew = _sample_skew(arr)
    if skew is None:
        return False
    return _SKEW_LO <= skew <= _SKEW_HI


def _fit(arr: np.ndarray) -> tuple[float, float, float] | None:
    """Return ex-Gaussian ``(mu, sigma2, lam)`` from a scipy ``exponnorm`` MLE, or None.

    scipy parameterizes ``exponnorm`` by ``K = 1/(lam*sigma)``, ``loc = mu``, ``scale = sigma``.
    """
    try:
        from scipy import stats

        k, loc, scale = stats.exponnorm.fit(arr)
    except Exception:  # noqa: BLE001
        return None
    if not (k > 0.0 and scale > 0.0 and math.isfinite(k) and math.isfinite(scale) and math.isfinite(loc)):
        return None
    sigma = float(scale)
    lam = 1.0 / (k * sigma)
    if not (lam > 0.0 and math.isfinite(lam)):
        return None
    return float(loc), sigma * sigma, lam


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from mixle.utils.automatic.profiling import _bic_penalty_bits

    if arr.size == 0:
        return None
    params = _fit(arr)
    if params is None:
        return None
    mu, sigma2, lam = params
    try:
        from scipy import stats

        sigma = math.sqrt(sigma2)
        k = 1.0 / (lam * sigma)
        nll_nats = -float(np.mean(stats.exponnorm.logpdf(arr, k, loc=mu, scale=sigma)))
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(nll_nats):
        return None
    return nll_nats / math.log(2.0) + _bic_penalty_bits(3, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import ExponentiallyModifiedGaussianDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _fit(_value_array_from_vdict(vdict))
    mu, sigma2, lam = fit if fit is not None else (0.0, 1.0, 1.0)
    return ExponentiallyModifiedGaussianDistribution(mu, sigma2, lam).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    params = _fit(arr)
    if params is None:
        return None
    mu, sigma2, lam = params
    sigma = math.sqrt(sigma2)
    return stats.exponnorm.cdf(arr, 1.0 / (lam * sigma), loc=mu, scale=sigma)


register(
    Detector(
        name="exgaussian", kind="continuous", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=3
    )
)
