"""Generalized Pareto (peaks-over-threshold) continuous candidate -- heavy-tailed exceedances.

By the Pickands-Balkema-de Haan theorem the distribution of exceedances over a high threshold
converges to a generalized Pareto distribution (GPD). Its signature is a strictly-positive,
monotone-decreasing density with a *heavy* (Pareto) upper tail -- shape ``xi`` clearly above 0.

The ``xi -> 0`` limit of the GPD is the exponential, and the gamma family already covers that
(and the exponential itself); to avoid stealing exponential / gamma data the gate fires only when
a moment estimate of the tail index ``xi`` is unmistakably positive (a genuinely heavy tail). With
that gate, exponential and Gaussian data never reach this candidate.
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register

# Minimum moment-estimated tail index for the GPD to even be considered. The exponential limit is
# xi = 0; requiring a clearly-positive xi keeps the candidate off exponential / gamma / Gaussian data.
_MIN_XI = 0.12


def _fit(arr: np.ndarray):
    """Return ``(loc, scale, shape)`` from a fixed-threshold method-of-moments GPD fit, or None.

    The threshold ``loc`` is the data minimum (the peaks-over-threshold setup); ``scale`` and
    ``shape`` follow from the exceedance mean ``m`` and variance ``v`` in closed form
    (``xi = (1 - m^2/v)/2``, ``sigma = m (1 - xi)``), valid for ``xi < 1/2``.
    """
    loc = float(np.min(arr))
    y = arr - loc
    m = float(np.mean(y))
    v = float(np.var(y))
    if not (m > 0.0) or not (v > 0.0) or not math.isfinite(v):
        return None
    xi = 0.5 * (1.0 - (m * m) / v)
    scale = m * (1.0 - xi)
    if not (scale > 0.0) or not math.isfinite(scale):
        return None
    return loc, scale, xi


def _applies(arr: np.ndarray) -> bool:
    # Positive support (threshold exceedances / tails); exclude non-positive data outright.
    if arr.size < 16 or not np.all(np.isfinite(arr)) or not np.all(arr > 0.0):
        return False
    fit = _fit(arr)
    if fit is None:
        return False
    _, _, xi = fit
    # Only fire on an unmistakably heavy (Pareto) tail. Exponential / gamma / Gaussian samples
    # produce xi near 0 (or negative) and are screened out here, so the candidate cannot steal them.
    return xi >= _MIN_XI


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from scipy import stats

    from mixle.utils.automatic.profiling import _bic_penalty_bits

    try:
        # Fit the GPD by MLE with the threshold pinned to the data minimum (floc), matching the
        # peaks-over-threshold convention; this fits shape + scale (2 free parameters).
        floc = float(np.min(arr))
        shape, loc, scale = stats.genpareto.fit(arr, floc=floc)
        if not (scale > 0.0) or not math.isfinite(scale) or not math.isfinite(shape):
            return None
        nll_nats = -float(np.mean(stats.genpareto.logpdf(arr, shape, loc=loc, scale=scale)))
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(nll_nats):
        return None
    return nll_nats / math.log(2.0) + _bic_penalty_bits(2, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import GeneralizedParetoDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _fit(_value_array_from_vdict(vdict))
    if fit is not None:
        loc, scale, xi = fit
    else:
        keys = [float(k) for k in vdict.keys() if isinstance(k, (int, float, np.integer, np.floating))]
        loc, scale, xi = (min(keys) if keys else 0.0), 1.0, 0.1
    return GeneralizedParetoDistribution(scale=scale, shape=xi, loc=loc).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    try:
        floc = float(np.min(arr))
        shape, loc, scale = stats.genpareto.fit(arr, floc=floc)
        if not (scale > 0.0) or not math.isfinite(scale) or not math.isfinite(shape):
            return None
        return stats.genpareto.cdf(arr, shape, loc=loc, scale=scale)
    except Exception:  # noqa: BLE001
        return None


register(
    Detector(
        name="generalized_pareto",
        kind="continuous",
        applies=_applies,
        score=_score,
        factory=_factory,
        cdf=_cdf,
        n_params=2,
    )
)
