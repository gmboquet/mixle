"""Automatic detector for the Pareto type-I heavy-tailed family.

The Pareto candidate targets strictly positive samples with an empirical lower bound near the observed
bulk and a power-law right tail. The support and lower-bound gates keep it focused on type-I Pareto data
instead of broad positive families whose density rises away from zero. The score charges both the
tail index and the lower bound selected from the sample.
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register

# A Pareto type-I lives on [xm, inf) with xm > 0: its density is largest at the lower bound xm and
# decays as a power law. The smallest observation therefore sits near the data bulk (min/median near
# one), unlike exponential, gamma, or lognormal samples whose observed minima tend to move toward zero.
_MIN_OVER_MEDIAN_GATE = 0.25


def _log_excess(arr: np.ndarray, xm: float) -> np.ndarray:
    """``log(x / xm)`` for every observation, accurate when the sample sits far above zero.

    ``log(arr / xm)`` evaluates the log of a ratio that is 1 + a tiny number, and both the division
    and the log lose the tiny number to rounding: on a sample at 1e15 whose spread is a few units,
    every ratio rounds to 1.0 or to 1 + one ulp, so the log-excesses become quantization noise and
    the fitted tail index is an artifact of the float grid rather than a measurement of the data.
    ``x - xm`` is exact for values this close together (Sterbenz), and ``log1p`` is accurate for
    small arguments, so this form keeps the full precision the sample actually carries.
    """
    return np.log1p((arr - xm) / xm)


def _fit(arr: np.ndarray) -> tuple[float, float] | None:
    """Return the Pareto type-I MLE ``(xm, alpha)`` for positive data, or ``None``."""
    if arr.size < 2 or not np.all(np.isfinite(arr)) or not np.all(arr > 0.0):
        return None
    xm = float(arr.min())
    if not (xm > 0.0):
        return None
    s = float(np.sum(_log_excess(arr, xm)))
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
    # Per-observation Pareto NLL in nats: -[log alpha + alpha*log xm - (alpha+1)*log x], regrouped
    # around log(x/xm) so the two O(alpha*log xm) terms cancel algebraically instead of numerically.
    # Evaluated as written, a sample at 1e15 subtracted alpha*log(xm) ~ 3e18 from (alpha+1)*E[log x]
    # ~ 3e18 and kept the rounding noise, which came out NEGATIVE and handed the Pareto a code
    # length of -2.87 bits/obs -- a fictitious win over the Gaussian's 1.97 on plain N(1e15, 1) data.
    nll_nats_per_obs = -math.log(alpha) + math.log(xm) + (alpha + 1.0) * float(_log_excess(arr, xm).mean())
    if not math.isfinite(nll_nats_per_obs):
        return None
    # xm is selected from these observations (its boundary MLE is their minimum),
    # so it is a fitted support parameter and must be charged alongside alpha.
    return nll_nats_per_obs / math.log(2.0) + _bic_penalty_bits(2, nobs)


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
    Detector(
        name="pareto",
        kind="continuous",
        applies=_applies,
        score=_score,
        factory=_factory,
        cdf=_cdf,
        n_params=2,
        # ``_log_excess`` and the regrouped NLL keep every term O(spread/xm), so the code length is
        # the same at any offset -- measured stable to 1e-6 bits/obs from 1e2 through 1e15 on the
        # same shifted-exponential sample. The Pareto is therefore not dropped by the profiler's
        # conditioning gate, and shifted-exponential data far from the origin still selects it.
        offset_stable=True,
    )
)
