"""Binomial discrete candidate -- bounded counts of successes in n trials (UNDER-dispersed vs Poisson)."""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register

_MAX_N = 100000  # a huge inferred n means the "trials" reading is implausible -- decline rather than guess


def _params(arr: np.ndarray) -> tuple[int, float] | None:
    """Profile the joint binomial likelihood over integer ``n`` and ``p``."""
    from scipy import optimize, special

    if arr.size == 0 or np.any(arr < 0) or np.any(arr != np.floor(arr)):
        return None
    values, counts = np.unique(arr.astype(np.int64), return_counts=True)
    lower = int(values[-1])
    mean = float(arr.mean())
    if lower < 1 or lower > _MAX_N:
        return None

    total = int(counts.sum())
    successes = float(np.dot(counts, values))

    def log_likelihood(n: float) -> float:
        p = mean / n
        if not 0.0 < p < 1.0:
            return -math.inf
        log_combinations = special.gammaln(n + 1.0) - special.gammaln(values + 1.0) - special.gammaln(
            n - values + 1.0
        )
        return float(
            np.dot(counts, log_combinations)
            + successes * math.log(p)
            + (total * n - successes) * math.log1p(-p)
        )

    candidates = {lower, _MAX_N}
    if lower < _MAX_N:
        result = optimize.minimize_scalar(
            lambda candidate: -log_likelihood(candidate),
            bounds=(lower, _MAX_N),
            method="bounded",
            options={"xatol": 0.25, "maxiter": 96},
        )
        if result.success and math.isfinite(result.x):
            center = int(round(result.x))
            candidates.update(range(max(lower, center - 3), min(_MAX_N, center + 3) + 1))
    n = max(candidates, key=log_likelihood)
    p = mean / n
    return (n, p) if math.isfinite(log_likelihood(n)) else None


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
    return nll_nats / math.log(2.0) + _bic_penalty_bits(2, nobs)


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
    Detector(name="binomial", kind="discrete", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=2)
)
