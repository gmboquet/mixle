"""Binomial discrete candidate -- bounded counts of successes in n trials (UNDER-dispersed vs Poisson)."""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register

_MAX_N = 100000  # a huge inferred n means the "trials" reading is implausible -- decline rather than guess


def _params(arr: np.ndarray) -> tuple[int, float] | None:
    if arr.size == 0 or np.any(arr < 0) or np.any(arr != np.floor(arr)):
        return None
    n = int(np.max(arr))  # n is unknown; the observed maximum is the standard estimate of the trial count
    mean = float(arr.mean())
    if n < 1 or n > _MAX_N:
        return None
    p = mean / n
    return (n, p) if 0.0 < p < 1.0 else None


def _applies(arr: np.ndarray) -> bool:
    # non-negative integers that are UNDER-dispersed (var < mean): the binomial signature, opposite Poisson.
    if arr.size == 0 or np.any(arr < 0) or np.any(arr != np.floor(arr)):
        return False
    mean = float(arr.mean())
    n = int(np.max(arr))
    # a genuine count distribution has many observations per possible outcome; a bare arithmetic range (every
    # integer once, n ~ sample size) is index-like, not binomial counts -- require real replication.
    if not (mean > 0.0 and 1 <= n <= _MAX_N and arr.size >= 3 * (n + 1)):
        return False
    return float(arr.var()) < 0.95 * mean  # UNDER-dispersed (var < mean): the binomial signature


def _score(arr: np.ndarray, nobs: int) -> float | None:
    from scipy import stats

    from mixle.utils.automatic.profiling import _bic_penalty_bits

    pr = _params(arr)
    if pr is None:
        return None
    n, p = pr
    nll_nats = -float(np.mean(stats.binom.logpmf(arr.astype(np.int64), n, p)))
    if not math.isfinite(nll_nats):
        return None
    return nll_nats / math.log(2.0) + _bic_penalty_bits(1, nobs)  # n is fixed from the support, p is the free param


def _factory(vdict, pseudo_count, emp_suff_stat, use_bstats):
    from mixle.stats import BinomialDistribution
    from mixle.utils.automatic.profiling import _value_array_from_vdict

    pr = _params(_value_array_from_vdict(vdict))
    n, p = pr if pr is not None else (1, 0.5)
    return BinomialDistribution(p, n).estimator(pseudo_count=pseudo_count)


def _cdf(arr: np.ndarray):
    from scipy import stats

    pr = _params(arr)
    if pr is None:
        return None
    n, p = pr
    return stats.binom.cdf(arr.astype(np.int64), n, p)


register(
    Detector(name="binomial", kind="discrete", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=1)
)
