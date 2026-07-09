"""Automatic detector for the Pareto type-I heavy-tailed family.

The Pareto candidate targets strictly positive samples with an empirical lower bound near the observed
bulk and a power-law right tail. The support and lower-bound gates keep it focused on type-I Pareto data
instead of broad positive families whose density rises away from zero, while the score charges only the
free tail-index parameter because the lower bound is fixed by the sample minimum.
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register

# A Pareto type-I lives on [xm, inf) with xm > 0: its density is largest at the lower bound xm and
# decays as a power law. The smallest observation therefore sits near the data bulk (min/median near
# one), unlike exponential, gamma, or lognormal samples whose observed minima tend to move toward zero.
_MIN_OVER_MEDIAN_GATE = 0.25


def _fit(arr: np.ndarray) -> tuple[float, float] | None:
    """Return the Pareto type-I MLE ``(xm, alpha)`` for positive data, or ``None``."""
    if arr.size < 2 or not np.all(np.isfinite(arr)) or not np.all(arr > 0.0):
        return None
    xm = float(arr.min())
    if not (xm > 0.0):
        return None
    s = float(np.sum(np.log(arr / xm)))
    if not (s > 0.0):
        return None
    alpha = arr.size / s
    if not (alpha > 0.0) or not math.isfinite(alpha):
        return None
    return xm, alpha


def _applies(arr: np.ndarray) -> bool:
    if arr.size < 2 or not np.all(np.isfinite(arr)) or not np.all(arr > 0.0):
        return False
    median = float(np.median(arr))
    if not (median > 0.0):
        return False
    # Pareto type-I signature: support starts at xm = min near the data bulk, not at zero.
    return float(arr.min()) / median >= _MIN_OVER_MEDIAN_GATE


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from mixle.utils.automatic.profiling import _bic_penalty_bits

    fit = _fit(arr)
    if fit is None:
        return None
    xm, alpha = fit
    # Per-observation Pareto NLL in nats: -[log alpha + alpha*log xm - (alpha+1)*log x].
    log_x = np.log(arr)
    nll_nats_per_obs = -(math.log(alpha) + alpha * math.log(xm) - (alpha + 1.0) * float(log_x.mean()))
    if not math.isfinite(nll_nats_per_obs):
        return None
    # The scale xm is pinned to the data minimum (a support bound, not a likelihood-estimated
    # parameter -- cf. StatisticSpec("min_val", kind="support_bound")), so only alpha is free: the
    # BIC charges one parameter. This is also what correctly distinguishes a true Pareto from the
    # unconstrained generalized-Pareto superset, which fits two free parameters above the same
    # threshold and so must out-fit Pareto by more than a one-parameter penalty to win.
    return nll_nats_per_obs / math.log(2.0) + _bic_penalty_bits(1, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import ParetoDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _fit(_value_array_from_vdict(vdict))
    xm, alpha = fit if fit is not None else (1.0, 1.0)
    return ParetoDistribution(xm, alpha).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    fit = _fit(arr)
    if fit is None:
        return None
    xm, alpha = fit
    return stats.pareto.cdf(arr, b=alpha, scale=xm)


register(
    Detector(name="pareto", kind="continuous", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=1)
)
