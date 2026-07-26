"""Beta-binomial discrete candidate -- bounded counts, OVER-dispersed vs a plain binomial (n trials, random p)."""

import math

import numpy as np

from mixle.utils.automatic.detectors import Detector, register

_MAX_N = 100000
_MIN_RHO = 0.01  # minimum intra-class correlation to claim genuine overdispersion (not sampling noise)


def _moment_params(arr: np.ndarray, n: float) -> tuple[float, float] | None:
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
    return (a, b) if a > 0.0 and b > 0.0 and math.isfinite(a) and math.isfinite(b) else None


def _params(arr: np.ndarray) -> tuple[int, float, float] | None:
    """Numerically profile the beta-binomial likelihood over ``n, a, b``."""
    from scipy import optimize, special

    if arr.size == 0 or np.any(arr < 0) or np.any(arr != np.floor(arr)):
        return None
    values, counts = np.unique(arr.astype(np.int64), return_counts=True)
    lower = int(values[-1])
    # A genuine count distribution has many observations per outcome; an arithmetic index range (n ~ sample
    # size, every value once) is not beta-binomial counts even though BetaBinom(n,1,1) equals a discrete uniform.
    if lower < 2 or lower > _MAX_N or arr.size < 3 * (lower + 1):
        return None
    initial = _moment_params(arr, lower)
    if initial is None:
        return None

    def log_likelihood(n: float, a: float, b: float) -> float:
        if n < lower or a <= 0.0 or b <= 0.0:
            return -math.inf
        log_pmf = (
            special.gammaln(n + 1.0)
            - special.gammaln(values + 1.0)
            - special.gammaln(n - values + 1.0)
            + special.betaln(values + a, n - values + b)
            - special.betaln(a, b)
        )
        return float(np.dot(counts, log_pmf))

    max_q = math.log(_MAX_N - lower + 1.0)

    def objective(theta):
        n = lower + math.exp(float(theta[0])) - 1.0
        return -log_likelihood(n, math.exp(float(theta[1])), math.exp(float(theta[2])))

    starts = [lower, min(_MAX_N, 2 * lower), min(_MAX_N, lower + max(16, int(math.sqrt(arr.size))))]
    relaxed: list[tuple[float, float, float, float]] = []
    for start_n in dict.fromkeys(starts):
        start_ab = _moment_params(arr, float(start_n)) or initial
        result = optimize.minimize(
            objective,
            np.asarray(
                [
                    math.log(start_n - lower + 1.0),
                    math.log(start_ab[0]),
                    math.log(start_ab[1]),
                ]
            ),
            method="L-BFGS-B",
            bounds=((0.0, max_q), (-18.0, 18.0), (-18.0, 18.0)),
            options={"maxiter": 160, "ftol": 1.0e-10},
        )
        if math.isfinite(float(result.fun)):
            n_relaxed = lower + math.exp(float(result.x[0])) - 1.0
            relaxed.append((float(result.fun), n_relaxed, math.exp(float(result.x[1])), math.exp(float(result.x[2]))))
    if not relaxed:
        return None

    _, best_n, best_a, best_b = min(relaxed)
    candidate_ns = {lower, _MAX_N}
    center = int(round(best_n))
    candidate_ns.update(range(max(lower, center - 3), min(_MAX_N, center + 3) + 1))

    best: tuple[float, int, float, float] | None = None
    for n in candidate_ns:
        result = optimize.minimize(
            lambda log_ab, n=n: -log_likelihood(n, math.exp(float(log_ab[0])), math.exp(float(log_ab[1]))),
            np.log([best_a, best_b]),
            method="L-BFGS-B",
            bounds=((-18.0, 18.0), (-18.0, 18.0)),
            options={"maxiter": 120, "ftol": 1.0e-11},
        )
        candidate = (float(result.fun), n, math.exp(float(result.x[0])), math.exp(float(result.x[1])))
        if math.isfinite(candidate[0]) and (best is None or candidate[0] < best[0]):
            best = candidate
    if best is None:
        return None
    _, n, a, b = best
    return n, a, b


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
    return nll_nats / math.log(2.0) + _bic_penalty_bits(3, nobs)


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
        name="beta_binomial", kind="discrete", applies=_applies, score=_score, factory=_factory, cdf=_cdf, n_params=3
    )
)
