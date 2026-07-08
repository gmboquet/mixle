"""Beta-binomial discrete candidate -- bounded counts, OVER-dispersed vs a plain binomial (n trials, random p)."""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register

_MAX_N = 100000
_MIN_RHO = 0.01  # minimum intra-class correlation to claim genuine overdispersion (not sampling noise)


def _params(arr: np.ndarray) -> tuple[int, float, float] | None:
    if arr.size == 0 or np.any(arr < 0) or np.any(arr != np.floor(arr)):
        return None
    n = int(np.max(arr))
    # a genuine count distribution has many observations per outcome; an arithmetic index range (n ~ sample
    # size, every value once) is not beta-binomial counts even though BetaBinom(n,1,1) equals a discrete uniform.
    if n < 2 or n > _MAX_N or arr.size < 3 * (n + 1):
        return None
    mean = float(arr.mean())
    var = float(arr.var())
    p = mean / n
    binom_var = n * p * (1.0 - p)
    if not (0.0 < p < 1.0 and binom_var > 0.0 and var > binom_var):  # must be OVER-dispersed vs binomial
        return None
    # method of moments via the intra-class correlation rho: var = binom_var * (1 + (n-1) rho), s = a+b = 1/rho - 1.
    # Require rho above a floor: a beta-binomial nests the binomial as rho -> 0, so on (near-)binomial data the two
    # are all-but-tied and the extra parameter would win on sampling noise alone. Only claim the family when the
    # overdispersion is genuine, so plain binomial data stays binomial.
    rho = (var / binom_var - 1.0) / (n - 1)
    if not (_MIN_RHO < rho < 1.0):
        return None
    s = 1.0 / rho - 1.0
    a, b = p * s, (1.0 - p) * s
    return (n, a, b) if a > 0.0 and b > 0.0 and math.isfinite(a) and math.isfinite(b) else None


def _applies(arr: np.ndarray) -> bool:
    if arr.size == 0 or np.any(arr < 0) or np.any(arr != np.floor(arr)):
        return False
    return _params(arr) is not None


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from scipy import stats

    from mixle.utils.automatic.profiling import _bic_penalty_bits

    pr = _params(arr)
    if pr is None:
        return None
    n, a, b = pr
    nll_nats = -float(np.mean(stats.betabinom.logpmf(arr.astype(np.int64), n, a, b)))
    if not math.isfinite(nll_nats):
        return None
    return nll_nats / math.log(2.0) + _bic_penalty_bits(2, nobs)  # a, b free; n fixed from the support


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import BetaBinomialDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    pr = _params(_value_array_from_vdict(vdict))
    n, a, b = pr if pr is not None else (2, 1.0, 1.0)
    return BetaBinomialDistribution(n, a, b).estimator()


def _cdf(arr: np.ndarray):
    from scipy import stats

    pr = _params(arr)
    if pr is None:
        return None
    n, a, b = pr
    return stats.betabinom.cdf(arr.astype(np.int64), n, a, b)


register(
    Detector(
        name="beta_binomial", kind="discrete", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=2
    )
)
