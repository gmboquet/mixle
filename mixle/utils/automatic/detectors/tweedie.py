"""Tweedie (compound Poisson-Gamma, 1 < p < 2) continuous candidate.

The Tweedie signature is unmistakable: non-negative data with a genuine point mass at exactly zero
*plus* a strictly-positive continuous part (``E[Y] = mu``, ``Var[Y] = phi*mu**p``). That zero-inflated
non-negative support is the tight gate -- it excludes Gaussian (has negatives) and the exponential /
gamma family (no exact-zero atom), so this detector only competes on data that genuinely looks like a
compound Poisson-Gamma.
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register

_P = 1.5  # fixed Tweedie power (the mixle family fixes p; 1.5 is the canonical midpoint of (1, 2)).


def _applies(arr: np.ndarray) -> bool:
    # Tight support gate: non-negative data with BOTH an exact-zero atom and a positive continuous part.
    if arr.size == 0 or not bool(np.all(np.isfinite(arr))):
        return False
    if bool(np.any(arr < 0.0)):
        return False  # any negative value rules out the non-negative Tweedie support (excludes Gaussian).
    n_zero = int(np.count_nonzero(arr == 0.0))
    n_pos = int(np.count_nonzero(arr > 0.0))
    if n_zero == 0 or n_pos == 0:
        return False  # need a genuine zero mass AND a positive part (excludes exponential / gamma).
    # Require the zero atom to be a real fraction of the data, not a stray rounding artifact.
    return (n_zero / arr.size) >= 0.01


def _moments(arr: np.ndarray) -> tuple[float, float] | None:
    """Method-of-moments ``(mu, phi)`` at fixed power ``p`` (exact: ``E[Y]=mu``, ``Var[Y]=phi*mu**p``)."""
    mu = float(arr.mean())
    if not (mu > 0.0) or not math.isfinite(mu):
        return None
    var = float(arr.var())
    if not (var > 0.0) or not math.isfinite(var):
        return None
    phi = var / mu**_P
    if not (phi > 0.0) or not math.isfinite(phi):
        return None
    return mu, phi


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from mixle.stats import TweedieDistribution
    from mixle.utils.automatic.profiling import _bic_penalty_bits

    params = _moments(arr)
    if params is None:
        return None
    mu, phi = params
    try:
        dist = TweedieDistribution(mu, phi, _P)
        ll = dist.seq_log_density(np.asarray(arr, dtype=np.float64))
    except (ValueError, FloatingPointError, OverflowError):
        return None
    if ll.size == 0 or not np.all(np.isfinite(ll)):
        return None
    nll_nats_per_obs = -float(ll.mean())
    if not math.isfinite(nll_nats_per_obs):
        return None
    # p is fixed, so the free parameters are mu and phi (2); add a third for the implicit power choice.
    return nll_nats_per_obs / math.log(2.0) + _bic_penalty_bits(3, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import TweedieDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _moments(_value_array_from_vdict(vdict))
    mu, phi = fit if fit is not None else (1.0, 1.0)
    return TweedieDistribution(mu, phi, _P).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    # No low-cost closed-form Tweedie CDF (it is a compound Poisson-Gamma series); skip GoF here.
    return None


register(
    Detector(name="tweedie", kind="continuous", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=3)
)
