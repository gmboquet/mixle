"""Automatic detector for positive inverse-Gaussian continuous data.

The detector fits the Wald first-passage-time law, scores it with a BIC-style
penalty, and exposes the estimator factory used by automatic distribution
selection.
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register


def _applies(arr: np.ndarray) -> bool:
    # Positive support only: the inverse Gaussian is defined on x > 0.
    return arr.size > 0 and bool(np.all(np.isfinite(arr))) and bool(np.all(arr > 0.0))


def _mle(arr: np.ndarray) -> tuple[float, float] | None:
    """Closed-form inverse Gaussian MLE (mu = mean, lam = n / sum(1/x - 1/mu)), or None."""
    if arr.size == 0 or not np.all(np.isfinite(arr)) or not np.all(arr > 0.0):
        return None
    mu = float(arr.mean())
    if not (mu > 0.0):
        return None
    inv_gap = float(np.mean(1.0 / arr)) - 1.0 / mu
    if not (inv_gap > 0.0):
        return None
    lam = 1.0 / inv_gap
    if not (lam > 0.0) or not math.isfinite(lam):
        return None
    return mu, lam


def _nll_nats_per_obs(arr: np.ndarray, mu: float, lam: float) -> float:
    """Per-observation negative log-likelihood (nats) of arr under InverseGaussian(mu, lam)."""
    logs = np.log(arr)
    # log f = 0.5*(log lam - log 2pi - 3 log x) - lam*(x - mu)^2 / (2 mu^2 x)
    mean_term = float(np.mean((arr - mu) ** 2 / arr))
    ll = 0.5 * (math.log(lam) - math.log(2.0 * math.pi)) - 1.5 * float(logs.mean()) - lam * mean_term / (2.0 * mu * mu)
    return -ll


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from mixle.utils.automatic.profiling import _bic_penalty_bits

    params = _mle(arr)
    if params is None:
        return None
    mu, lam = params
    nll = _nll_nats_per_obs(arr, mu, lam)
    if not math.isfinite(nll):
        return None
    return nll / math.log(2.0) + _bic_penalty_bits(2, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import InverseGaussianDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _mle(_value_array_from_vdict(vdict))
    mu, lam = fit if fit is not None else (1.0, 1.0)
    return InverseGaussianDistribution(mu, lam).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    params = _mle(arr)
    if params is None:
        return None
    mu, lam = params
    # scipy invgauss is parameterized by mu_scipy = mu/lam with scale = lam.
    return stats.invgauss.cdf(arr, mu / lam, scale=lam)


register(
    Detector(
        name="inverse_gaussian",
        kind="continuous",
        applies=_applies,
        score=_score,
        factory=_factory,
        cdf=_cdf,
        n_params=2,
    )
)
