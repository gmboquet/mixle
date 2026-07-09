"""Automatic detector for overdispersed negative-binomial count data.

The detector accepts non-negative integer samples whose variance clearly exceeds
the mean, estimates moment-based parameters, and exposes the distribution
factory for automatic model selection.
"""

import numpy as np

from mixle.utils.automatic.detectors import Detector, register


def _params(arr):
    mean = float(arr.mean())
    var = float(arr.var())
    if not (mean > 0.0 and var > mean):  # needs genuine overdispersion (Poisson has var == mean)
        return None
    p = mean / var  # in (0,1); mixle NegativeBinomial mean = r(1-p)/p, var = mean/p
    r = mean * mean / (var - mean)
    if not (0.0 < p < 1.0 and r > 0.0 and np.isfinite(r)):
        return None
    return r, p


def _applies(arr: np.ndarray) -> bool:
    # non-negative integer counts that are overdispersed by a clear margin (else Poisson is the right model)
    if arr.size == 0 or np.any(arr < 0) or np.any(arr != np.floor(arr)):
        return False
    mean = float(arr.mean())
    return mean > 0.0 and float(arr.var()) > 1.05 * mean


def _score(arr: np.ndarray, nobs: int) -> float | None:
    import math

    from scipy import stats

    from mixle.utils.automatic.profiling import _bic_penalty_bits

    pr = _params(arr)
    if pr is None:
        return None
    r, p = pr
    nll_nats = -float(np.mean(stats.nbinom.logpmf(arr.astype(np.int64), r, p)))
    if not np.isfinite(nll_nats):
        return None
    return nll_nats / math.log(2.0) + _bic_penalty_bits(2, nobs)


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import NegativeBinomialDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    fit = _params(_value_array_from_vdict(vdict))
    r, p = fit if fit is not None else (1.0, 0.5)
    return NegativeBinomialDistribution(r, p).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    pr = _params(arr)
    if pr is None:
        return None
    r, p = pr
    return stats.nbinom.cdf(arr.astype(np.int64), r, p)


register(
    Detector(
        name="negative_binomial",
        kind="discrete",
        applies=_applies,
        score=_score,
        factory=_factory,
        cdf=_cdf,
        n_params=2,
    )
)
