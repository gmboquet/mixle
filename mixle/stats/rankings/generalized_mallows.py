"""Generalized Mallows distribution over permutations under a choice of distance metric.

The Mallows family concentrates probability around a central permutation ``sigma0`` with dispersion
``theta >= 0``:

    p(sigma) = exp(-theta * d(sigma, sigma0)) / Z(theta, n),

generalizing :class:`~mixle.stats.rankings.mallows.MallowsDistribution` (Kendall-only) to any of the six
metrics in :mod:`mixle.stats.rankings._permutation_kernels`. The per-datum distance is the numba kernel;
this module supplies the metric-specific normalizer ``Z``. Three metrics have a closed-form (fast-DP)
normalizer and moment ``E_theta[d]``:

    kendall   Z = prod_{i=1}^{n-1} (1 - phi^{i+1}) / (1 - phi)          (phi = e^{-theta})
    cayley    Z = prod_{i=1}^{n-1} (1 + i phi)
    hamming   Z = sum_{m=0}^{n} C(n, m) D_m phi^m                       (D_m = subfactorial)

The other three are #P-hard and use an exact small-``n`` normalizer with a numba Monte-Carlo fallback
beyond a size cap (the fallback is an *approximation*, controlled by ``n_mc`` / ``seed``):

    footrule  Z = perm(phi^{|i-j|})    exact Ryser permanent for n <= max_exact (16), else MC
    spearman  Z = perm(phi^{(i-j)^2})  exact Ryser permanent for n <= max_exact (16), else MC
    ulam      Z = sum over the LIS-distance histogram, exact for n <= max_enum (9), else MC

Data type: ``List[int]`` -- a full ordering, a permutation of ``0..n-1`` with ``x[r]`` the item at rank
``r`` (best first).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.rankings._permutation_kernels import (
    METRICS,
    cayley_perm,
    footrule_perm,
    hamming_perm,
    kendall_perm,
    metric_id,
    ryser_log_permanent,
    seq_distance_to_center,
    spearman_perm,
    ulam_perm,
)
from mixle.utils.optional_deps import numba

_MAX_THETA = 700.0
_CLOSED_FORM = ("kendall", "cayley", "hamming")


# --- metric-specific normalizer and expected distance --------------------------------------------
def _log_subfactorials(n: int) -> np.ndarray:
    """``log D_m`` for ``m = 0..n`` (rencontres / derangement counts; ``D_1 = 0`` -> ``-inf``)."""
    prev2, prev1 = 1.0, 0.0  # D_0, D_1
    logs = [0.0, -np.inf]
    for m in range(2, n + 1):
        cur = (m - 1) * (prev1 + prev2)
        logs.append(math.log(cur))
        prev2, prev1 = prev1, cur
    return np.asarray(logs[: n + 1], dtype=float)


def log_normalizer(metric: str, theta: float, n: int) -> float:
    """Return ``log Z(theta, n)`` for a closed-form metric."""
    if n <= 1:
        return 0.0
    if theta <= 0.0:
        return float(math.lgamma(n + 1))  # uniform: Z = n!
    phi = math.exp(-theta)
    if metric == "kendall":
        if phi <= 0.0:
            return 0.0
        log1m = math.log1p(-phi)
        return float(sum(math.log1p(-(phi ** (i + 1))) - log1m for i in range(1, n)))
    if metric == "cayley":
        return float(sum(math.log1p(i * phi) for i in range(1, n)))
    if metric == "hamming":
        m = np.arange(n + 1)
        log_binom = math.lgamma(n + 1) - np.array([math.lgamma(k + 1) + math.lgamma(n - k + 1) for k in m])
        terms = log_binom + _log_subfactorials(n) + m * math.log(phi)
        return float(logsumexp(terms))
    raise ValueError(f"log_normalizer: metric {metric!r} has no closed form (use the approximate path).")


def expected_distance(metric: str, theta: float, n: int) -> float:
    """Return ``E_theta[d]`` for a closed-form metric (uniform value at ``theta = 0``)."""
    if n <= 1:
        return 0.0
    if theta <= 1e-7:  # uniform (the phi->1 closed forms cancel catastrophically this close to 0)
        if metric == "kendall":
            return n * (n - 1) / 4.0
        if metric == "cayley":
            return float(sum(i / (1.0 + i) for i in range(1, n)))
        if metric == "hamming":
            return float(n - 1)  # E[fixed points] = 1 under the uniform permutation
    phi = math.exp(-theta)
    if metric == "kendall":
        return float(sum(phi / (1.0 - phi) - (i + 1) * phi ** (i + 1) / (1.0 - phi ** (i + 1)) for i in range(1, n)))
    if metric == "cayley":
        return float(sum(i * phi / (1.0 + i * phi) for i in range(1, n)))
    if metric == "hamming":
        m = np.arange(n + 1)
        log_binom = math.lgamma(n + 1) - np.array([math.lgamma(k + 1) + math.lgamma(n - k + 1) for k in m])
        terms = log_binom + _log_subfactorials(n) + m * math.log(phi)
        return float(np.sum(m * np.exp(terms - logsumexp(terms))))
    raise ValueError(f"expected_distance: metric {metric!r} has no closed form.")


def solve_theta(metric: str, mean_distance: float, n: int) -> float:
    """Return the ``theta`` whose ``E_theta[d]`` matches ``mean_distance`` (monotone bisection)."""
    uniform_mean = expected_distance(metric, 0.0, n)
    if n <= 1 or mean_distance >= uniform_mean:
        return 0.0
    if mean_distance <= 0.0:
        return _MAX_THETA
    lo, hi = 0.0, 1.0
    while expected_distance(metric, hi, n) > mean_distance and hi < _MAX_THETA:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if expected_distance(metric, mid, n) > mean_distance:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- numba Metropolis sampler (universal across metrics) -----------------------------------------
@numba.njit("int64(int64[:], int64)", cache=True)
def _perm_distance(r, mid):
    """Distance of the relative-rank permutation r from the identity (dispatch by metric id)."""
    if mid == 0:
        return kendall_perm(r)
    elif mid == 1:
        return cayley_perm(r)
    elif mid == 2:
        return hamming_perm(r)
    elif mid == 3:
        return footrule_perm(r)
    elif mid == 4:
        return spearman_perm(r)
    return ulam_perm(r)


@numba.njit("int64[:, :](int64[:], float64, int64, int64, int64, int64, int64)", cache=True)
def _mh_sample(rank_center, theta, mid, n_samples, burn, thin, seed):
    """Metropolis sampler on orderings: propose a random rank transposition, accept by exp(-theta dd)."""
    np.random.seed(seed)
    n = rank_center.shape[0]
    sigma = np.arange(n)  # current ordering (items at ranks); start at identity
    np.random.shuffle(sigma)
    r = np.empty(n, dtype=np.int64)
    for i in range(n):
        r[i] = rank_center[sigma[i]]
    cur = _perm_distance(r, mid)
    out = np.empty((n_samples, n), dtype=np.int64)
    total = burn + n_samples * thin
    taken = 0
    for step in range(total):
        a = np.random.randint(0, n)
        b = np.random.randint(0, n)
        if a != b:
            sigma[a], sigma[b] = sigma[b], sigma[a]
            r[a], r[b] = rank_center[sigma[a]], rank_center[sigma[b]]
            prop = _perm_distance(r, mid)
            if prop <= cur or np.random.random() < math.exp(-theta * (prop - cur)):
                cur = prop
            else:  # reject -> undo
                sigma[a], sigma[b] = sigma[b], sigma[a]
                r[a], r[b] = rank_center[sigma[a]], rank_center[sigma[b]]
        if step >= burn and (step - burn) % thin == 0 and taken < n_samples:
            out[taken, :] = sigma
            taken += 1
    return out


# --- #P-hard normalizers (footrule / spearman / ulam): exact small-n + numba Monte-Carlo ----------
@numba.njit("int64[:](int64, int64, int64, int64)", cache=True)
def _uniform_distances(n, mid, n_mc, seed):
    """Distances-from-identity of n_mc uniform random permutations (Fisher-Yates), for MC normalizers."""
    np.random.seed(seed)
    out = np.empty(n_mc, dtype=np.int64)
    p = np.arange(n).astype(np.int64)
    for k in range(n_mc):
        for i in range(n - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            tmp = p[i]
            p[i] = p[j]
            p[j] = tmp
        out[k] = _perm_distance(p, mid)
    return out


_SAMPLE_CACHE: dict[tuple, np.ndarray] = {}
_HIST_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _uniform_sample(metric: str, n: int, n_mc: int, seed: int) -> np.ndarray:
    key = (metric, n, n_mc, seed)
    if key not in _SAMPLE_CACHE:
        _SAMPLE_CACHE[key] = np.asarray(_uniform_distances(n, metric_id(metric), n_mc, seed), dtype=float)
    return _SAMPLE_CACHE[key]


def _exact_histogram(metric: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Full enumeration of S_n: distinct distance values and their counts (small n only)."""
    key = (metric, n)
    if key not in _HIST_CACHE:
        import itertools

        perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64)
        ds = seq_distance_to_center(perms, np.arange(n, dtype=np.int64), metric)
        vals, counts = np.unique(ds, return_counts=True)
        _HIST_CACHE[key] = (vals.astype(float), counts.astype(float))
    return _HIST_CACHE[key]


def metric_log_normalizer(
    metric: str, theta: float, n: int, *, n_mc: int = 20000, seed: int = 0, max_exact: int = 16, max_enum: int = 9
) -> float:
    """``log Z`` for any metric: closed form / exact permanent / exact enumeration / Monte-Carlo."""
    if metric in _CLOSED_FORM:
        return log_normalizer(metric, theta, n)
    if n <= 1:
        return 0.0
    if theta <= 0.0:
        return float(math.lgamma(n + 1))
    phi = math.exp(-theta)
    if metric in ("footrule", "spearman") and n <= max_exact:  # exact: Z = perm(phi^{d_ij})
        i = np.arange(n)
        d = np.abs(i[:, None] - i[None, :]) if metric == "footrule" else (i[:, None] - i[None, :]) ** 2
        m = np.ascontiguousarray(phi ** d.astype(float), dtype=float)
        return float(ryser_log_permanent(m))
    if metric == "ulam" and n <= max_enum:  # exact: enumerate the LIS-distance histogram
        vals, counts = _exact_histogram(metric, n)
        return float(logsumexp(np.log(counts) - theta * vals))
    du = _uniform_sample(metric, n, n_mc, seed)  # Monte-Carlo: Z ~ n! * E_uniform[phi^d]
    return float(math.lgamma(n + 1) + logsumexp(-theta * du) - math.log(du.size))


def metric_solve_theta(
    metric: str, mean_distance: float, n: int, *, n_mc: int = 20000, seed: int = 0, max_enum: int = 9
) -> float:
    """Fit ``theta`` to a target mean distance for any metric (closed form, else IS over a distance pool)."""
    if metric in _CLOSED_FORM:
        return solve_theta(metric, mean_distance, n)
    if n <= max_enum:  # exact distance population from full enumeration
        d_pop, w_pop = _exact_histogram(metric, n)
    else:  # uniform Monte-Carlo population (self-normalized importance sampling for E_theta[d])
        d_pop, w_pop = _uniform_sample(metric, n, n_mc, seed), None
    log_w = np.log(w_pop) if w_pop is not None else np.zeros(d_pop.size)
    uniform_mean = float(np.sum(d_pop * np.exp(log_w - logsumexp(log_w))))
    if n <= 1 or mean_distance >= uniform_mean:
        return 0.0
    if mean_distance <= float(d_pop.min()):
        return _MAX_THETA

    def e_dist(theta: float) -> float:
        lw = -theta * d_pop + log_w
        return float(np.sum(d_pop * np.exp(lw - logsumexp(lw))))

    lo, hi = 0.0, 1.0
    while e_dist(hi) > mean_distance and hi < _MAX_THETA:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if e_dist(mid) > mean_distance:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class GeneralizedMallowsDistribution(SequenceEncodableProbabilityDistribution):
    """Mallows distribution under a configurable distance ``metric`` (closed-form normalizer metrics)."""

    @classmethod
    def compute_capabilities(cls):
        """Declare the NumPy and numba execution path used by Mallows kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="Permutation distances run through dedicated numba kernels, not a compute engine.",
        )

    def __init__(
        self,
        sigma0: Sequence[int] | np.ndarray,
        theta: float = 1.0,
        metric: str = "kendall",
        name: str | None = None,
        keys: str | None = None,
        n_mc: int = 20000,
        seed: int = 0,
        max_exact: int = 16,
        max_enum: int = 9,
    ) -> None:
        s0 = np.asarray(sigma0, dtype=int)
        n = len(s0)
        if n < 2 or not np.array_equal(np.sort(s0), np.arange(n)):
            raise ValueError("sigma0 must be a permutation of 0,...,n-1 with n >= 2.")
        if theta < 0.0 or not np.isfinite(theta):
            raise ValueError("GeneralizedMallowsDistribution requires theta >= 0.")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}.")
        self.sigma0 = s0
        self.theta = float(theta)
        self.metric = metric
        self.dim = n
        self.rank0 = np.empty(n, dtype=np.int64)
        self.rank0[s0] = np.arange(n)
        self.n_mc = int(n_mc)
        self.seed = int(seed)
        self.max_exact = int(max_exact)
        self.max_enum = int(max_enum)
        # exact for kendall/cayley/hamming; permanent (footrule/spearman) or enumeration (ulam) up to the
        # caps; Monte-Carlo beyond -- an approximation (documented) so very large n stays tractable.
        self.log_z = metric_log_normalizer(
            metric, self.theta, n, n_mc=self.n_mc, seed=self.seed, max_exact=self.max_exact, max_enum=self.max_enum
        )
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "GeneralizedMallowsDistribution(%s, theta=%s, metric=%s, name=%s, keys=%s)" % (
            repr([int(v) for v in self.sigma0]),
            repr(self.theta),
            repr(self.metric),
            repr(self.name),
            repr(self.keys),
        )

    def distance(self, x: Sequence[int]) -> int:
        """Distance between ordering ``x`` and the central permutation under this metric."""
        return int(seq_distance_to_center(np.asarray(x, dtype=np.int64)[None, :], self.rank0, self.metric)[0])

    def density(self, x: Sequence[int]) -> float:
        """Return the probability of one ordering under the Mallows model."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int]) -> float:
        """Return the log-probability of one ordering."""
        return -self.theta * self.distance(x) - self.log_z

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for encoded orderings."""
        dist = seq_distance_to_center(x, self.rank0, self.metric)
        return -self.theta * dist - self.log_z

    def sampler(self, seed: int | None = None) -> GeneralizedMallowsSampler:
        """Return a sampler for this Mallows distribution."""
        return GeneralizedMallowsSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> GeneralizedMallowsEstimator:
        """Return a Mallows estimator with this metric and item count."""
        return GeneralizedMallowsEstimator(
            dim=self.dim,
            metric=self.metric,
            name=self.name,
            keys=self.keys,
            n_mc=self.n_mc,
            seed=self.seed,
            max_exact=self.max_exact,
            max_enum=self.max_enum,
        )

    def dist_to_encoder(self) -> GeneralizedMallowsDataEncoder:
        """Return the full-ranking encoder used by vectorized methods."""
        return GeneralizedMallowsDataEncoder(dim=self.dim)


class GeneralizedMallowsSampler(DistributionSampler):
    """Draw orderings via the exact RIM (Kendall) or a numba Metropolis sampler (other metrics)."""

    def __init__(
        self, dist: GeneralizedMallowsDistribution, seed: int | None = None, burn: int = 1000, thin: int = 10
    ) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.burn = burn
        self.thin = thin

    def _sample_kendall(self, size: int) -> list[list[int]]:
        n, phi, theta = self.dist.dim, math.exp(-self.dist.theta), self.dist.theta
        out = []
        for _ in range(size):
            perm: list[int] = []
            for i in range(n):
                if theta <= 0.0:
                    j = self.rng.randint(0, i + 1)
                else:
                    cdf = np.cumsum(phi ** np.arange(i + 1))
                    j = int(np.searchsorted(cdf, self.rng.rand() * cdf[-1]))
                perm.insert(i - j, int(self.dist.sigma0[i]))
            out.append(perm)
        return out

    def sample(self, size: int | None = None) -> list[int] | list[list[int]]:
        """Draw one ordering or ``size`` iid orderings."""
        k = 1 if size is None else size
        if self.dist.metric == "kendall":
            draws = self._sample_kendall(k)
        else:
            seed = int(self.rng.randint(0, 2**31 - 1))
            arr = _mh_sample(
                self.dist.rank0, self.dist.theta, metric_id(self.dist.metric), k, self.burn, self.thin, seed
            )
            draws = [[int(v) for v in row] for row in arr]
        return draws[0] if size is None else draws


class GeneralizedMallowsAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the rank-count matrix, the precede matrix, and a bounded reservoir of orderings.

    ``rank_count[item, rank]`` and ``precede[a, b]`` give the consensus (Borda / Copeland); the reservoir
    supplies the empirical mean metric-distance to the fitted center (exact when the data fit in it).
    """

    def __init__(self, dim: int, reservoir: int = 10000, keys: str | None = None) -> None:
        self.dim = dim
        self.rank_count = np.zeros((dim, dim))
        self.precede = np.zeros((dim, dim))
        self.count = 0.0
        self.reservoir = reservoir
        self._res_x: list[np.ndarray] = []
        self._res_w: list[float] = []
        self.keys = keys

    def _push(self, row: np.ndarray, w: float) -> None:
        if len(self._res_x) < self.reservoir:
            self._res_x.append(np.asarray(row, dtype=np.int64))
            self._res_w.append(float(w))

    def update(self, x: Sequence[int], weight: float, estimate: Any) -> None:
        """Update rank, precedence, and reservoir statistics from one ordering."""
        self.seq_update(np.asarray([x], dtype=int), np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize statistics from one weighted ordering."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Update rank, precedence, and reservoir statistics from encoded orderings."""
        n = self.dim
        r_idx, rp_idx = np.triu_indices(n, 1)
        ranks = np.arange(n)
        for row, w in zip(x, weights):
            np.add.at(self.rank_count, (row, ranks), w)  # item row[r] got rank r
            np.add.at(self.precede, (row[r_idx], row[rp_idx]), w)
            self._push(row, w)
        self.count += float(np.sum(weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize statistics from encoded orderings."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat) -> GeneralizedMallowsAccumulator:
        """Merge consensus statistics and reservoir samples from another accumulator."""
        count, rank_count, precede, res_x, res_w = suff_stat
        self.count += count
        self.rank_count += rank_count
        self.precede += precede
        for row, w in zip(res_x, res_w):
            if len(self._res_x) < self.reservoir:
                self._res_x.append(np.asarray(row, dtype=np.int64))
                self._res_w.append(float(w))
        return self

    def value(self):
        """Return count, consensus matrices, and bounded reservoir contents."""
        return (
            self.count,
            self.rank_count,
            self.precede,
            [np.asarray(r) for r in self._res_x],
            list(self._res_w),
        )

    def from_value(self, x) -> GeneralizedMallowsAccumulator:
        """Restore accumulator state from ``value`` output."""
        self.count, self.rank_count, self.precede, res_x, res_w = x
        self.dim = self.rank_count.shape[0]
        self._res_x = [np.asarray(r, dtype=np.int64) for r in res_x]
        self._res_w = [float(w) for w in res_w]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into ``stats_dict`` under its configured key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's state from keyed statistics when present."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> GeneralizedMallowsDataEncoder:
        """Return the ranking encoder compatible with these sufficient statistics."""
        return GeneralizedMallowsDataEncoder(dim=self.dim)


class GeneralizedMallowsAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for generalized Mallows consensus statistics."""

    def __init__(self, dim: int, reservoir: int = 10000, keys: str | None = None) -> None:
        self.dim = dim
        self.reservoir = reservoir
        self.keys = keys

    def make(self) -> GeneralizedMallowsAccumulator:
        """Create an empty generalized Mallows accumulator."""
        return GeneralizedMallowsAccumulator(dim=self.dim, reservoir=self.reservoir, keys=self.keys)


class GeneralizedMallowsEstimator(ParameterEstimator):
    """Estimate the central permutation (Copeland/Borda) and dispersion theta (moment match)."""

    def __init__(
        self,
        dim: int,
        metric: str = "kendall",
        theta: float | None = None,
        reservoir: int = 10000,
        name: str | None = None,
        keys: str | None = None,
        n_mc: int = 20000,
        seed: int = 0,
        max_exact: int = 16,
        max_enum: int = 9,
    ) -> None:
        if dim is None or dim < 2:
            raise ValueError("GeneralizedMallowsEstimator requires dim >= 2.")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}.")
        self.dim = int(dim)
        self.metric = metric
        self.theta = theta
        self.reservoir = reservoir
        self.name = name
        self.keys = keys
        self.n_mc = int(n_mc)
        self.seed = int(seed)
        self.max_exact = int(max_exact)
        self.max_enum = int(max_enum)

    def accumulator_factory(self) -> GeneralizedMallowsAccumulatorFactory:
        """Return a factory for Mallows sufficient-statistic accumulators."""
        return GeneralizedMallowsAccumulatorFactory(dim=self.dim, reservoir=self.reservoir, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat) -> GeneralizedMallowsDistribution:
        """Estimate the central ordering and dispersion from accumulated rankings."""
        count, rank_count, precede, res_x, res_w = suff_stat
        n = self.dim
        kw = dict(
            name=self.name,
            keys=self.keys,
            n_mc=self.n_mc,
            seed=self.seed,
            max_exact=self.max_exact,
            max_enum=self.max_enum,
        )
        if count <= 0.0:
            return GeneralizedMallowsDistribution(np.arange(n), 0.0, self.metric, **kw)
        if self.metric == "kendall":  # Copeland consensus
            scores = precede.sum(axis=1) - precede.sum(axis=0)
        else:  # Borda consensus: order items by ascending mean rank
            mean_rank = (rank_count * np.arange(n)[None, :]).sum(axis=1) / count
            scores = -mean_rank
        sigma0 = np.argsort(-scores, kind="stable")
        rank0 = np.empty(n, dtype=np.int64)
        rank0[sigma0] = np.arange(n)

        if self.theta is not None:
            theta = self.theta
        else:
            x = np.asarray(res_x, dtype=np.int64)
            w = np.asarray(res_w, dtype=float)
            dist = seq_distance_to_center(x, rank0, self.metric)
            mean_distance = float(np.sum(dist * w) / np.sum(w))
            theta = metric_solve_theta(
                self.metric, mean_distance, n, n_mc=self.n_mc, seed=self.seed, max_enum=self.max_enum
            )
        return GeneralizedMallowsDistribution(sigma0, theta, self.metric, **kw)


class GeneralizedMallowsDataEncoder(DataSequenceEncoder):
    """Encode a sequence of orderings (permutations of 0,...,n-1) into an (N, n) integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim

    def __str__(self) -> str:
        return "GeneralizedMallowsDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GeneralizedMallowsDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        """Validate and encode full orderings as a dense integer matrix."""
        rv = np.asarray([list(row) for row in x], dtype=np.int64)
        if rv.ndim != 2 or rv.shape[0] == 0:
            raise ValueError("GeneralizedMallowsDistribution requires a non-empty sequence of orderings.")
        expected = np.arange(rv.shape[1])
        for row in rv:
            if not np.array_equal(np.sort(row), expected):
                raise ValueError("orderings must be permutations of 0,...,n-1.")
        return rv


__all__ = [
    "GeneralizedMallowsDistribution",
    "GeneralizedMallowsSampler",
    "GeneralizedMallowsAccumulator",
    "GeneralizedMallowsAccumulatorFactory",
    "GeneralizedMallowsEstimator",
    "GeneralizedMallowsDataEncoder",
    "METRICS",
]
