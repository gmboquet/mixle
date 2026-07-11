"""Generalized-Gaussian (exponential-power) continuous candidate.

Symmetric family with a tunable shape ``beta``: ``beta=2`` is Gaussian, ``beta=1`` is Laplace,
``beta<1`` is peakier/heavier-tailed and ``beta>1`` flatter-topped. The candidate only fires on
plausibly *symmetric* real-valued data (a strong skew is a clear out-of-family signature -- e.g. an
exponential or other one-sided law -- which the symmetric exponential-power model would otherwise
"steal" with a poorly-justified sub-Gaussian shape).
"""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register

# A genuine generalized-Gaussian is symmetric, so its sample skew is ~0; a clearly one-sided law
# (exponential, log-normal, ...) has large |skew|. Gate the candidate on near-symmetry so it does
# not steal asymmetric data that no symmetric family should explain.
_MAX_ABS_SKEW = 0.6


def _applies(arr: np.ndarray) -> bool:
    if arr.size < 8 or not bool(np.all(np.isfinite(arr))):
        return False
    sd = float(arr.std())
    if not (sd > 0.0):
        return False
    z = (arr - float(arr.mean())) / sd
    skew = float(np.mean(z**3))
    return math.isfinite(skew) and abs(skew) <= _MAX_ABS_SKEW


def _fit(arr: np.ndarray):
    from scipy import stats

    try:
        beta, loc, scale = stats.gennorm.fit(arr)
    except Exception:  # noqa: BLE001
        return None
    if not (scale > 0.0 and beta > 0.0):
        return None
    if not (math.isfinite(beta) and math.isfinite(loc) and math.isfinite(scale)):
        return None
    return float(beta), float(loc), float(scale)


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from scipy import stats

    from mixle.utils.automatic.profiling import _bic_penalty_bits

    fit = _fit(arr)
    if fit is None:
        return None
    beta, loc, scale = fit
    logpdf = stats.gennorm.logpdf(arr, beta, loc=loc, scale=scale)
    if not np.all(np.isfinite(logpdf)):
        return None
    nll_nats_per_obs = -float(np.mean(logpdf))
    return nll_nats_per_obs / math.log(2.0) + _bic_penalty_bits(3, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import GeneralizedGaussianDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _fit(_value_array_from_vdict(vdict))
    beta, loc, scale = fit if fit is not None else (2.0, 0.0, 1.0)
    return GeneralizedGaussianDistribution(loc, scale, beta).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    fit = _fit(arr)
    if fit is None:
        return None
    beta, loc, scale = fit
    return stats.gennorm.cdf(arr, beta, loc=loc, scale=scale)


register(
    Detector(
        name="generalized_gaussian",
        kind="continuous",
        applies=_applies,
        score=_score,
        factory=_factory,
        cdf=_cdf,
        n_params=3,
    )
)
