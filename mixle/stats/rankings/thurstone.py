"""Thurstone (Thurstonian) ranking model -- the Gaussian random-utility model over permutations.

Each item has a latent utility ``U_i ~ Normal(mu_i, 1)`` (Case V, equal variance) and an ordering is the
descending sort of the utilities, so

    p(sigma) = P( U_sigma[0] > U_sigma[1] > ... > U_sigma[n-1] ).

This is the Gaussian counterpart of :class:`PlackettLuceDistribution` (which is the same construction
with Gumbel noise). The probability is the Gaussian-orthant probability of the consecutive-difference
cone ``D_r = U_sigma[r] - U_sigma[r+1] > 0``; ``D`` has mean ``mu_sigma[r] - mu_sigma[r+1]`` and a fixed
tridiagonal covariance (``2`` on the diagonal, ``-1`` off it), independent of ``sigma``. There is no
closed form for ``n > 3``, so the likelihood is a numba **Genz separation-of-variables** Monte-Carlo
estimate of the orthant (low variance, always positive) seeded deterministically. Sampling is exact
(draw utilities, sort); ``mu`` is fit in closed form from the pairwise-preference marginals (the
Thurstone-Mosteller Case V estimator), identified up to an additive constant (stored mean-zero).

Data type: ``List[int]`` -- a full ordering, a permutation of ``0..n-1`` with ``x[r]`` the item at rank
``r`` (best first).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.utils.optional_deps import numba

_SQRT2 = math.sqrt(2.0)


@numba.njit("float64(float64)", cache=True)
def _phi(x):
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


@numba.njit("float64(float64)", cache=True)
def _phinv(p):
    """Standard-normal quantile via Acklam's rational approximation (~1e-9, pure arithmetic)."""
    if p <= 0.0:
        return -1e10
    if p >= 1.0:
        return 1e10
    a0, a1, a2, a3, a4, a5 = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b1, b2, b3, b4, b5 = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c0, c1, c2, c3, c4, c5 = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d1, d2, d3, d4 = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / (
            (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5)
            * q
            / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)


@numba.njit("float64[:](float64[:], int64[:, :], float64[:], float64[:], int64, int64)", cache=True)
def _seq_thurstone_logp(mu, orderings, ldiag, lsub, n_mc, seed):
    """Genz SOV Monte-Carlo log-orthant probability of each ordering's consecutive-difference cone."""
    np.random.seed(seed)
    big_n, n = orderings.shape
    d = n - 1
    out = np.empty(big_n, dtype=np.float64)
    lower = np.empty(d, dtype=np.float64)
    for t in range(big_n):
        sig = orderings[t]
        for r in range(d):
            lower[r] = -(mu[sig[r]] - mu[sig[r + 1]])  # constraint D_r > 0  <=>  (D_r - m_r) > -m_r
        acc = 0.0
        for _ in range(n_mc):
            p_prod = 1.0
            zprev = 0.0
            for i in range(d):
                s = lsub[i] * zprev if i > 0 else 0.0
                a = (lower[i] - s) / ldiag[i]
                lo = _phi(a)
                f = 1.0 - lo
                p_prod *= f
                if i < d - 1:
                    zprev = _phinv(lo + np.random.random() * f)
            acc += p_prod
        mean = acc / n_mc
        out[t] = math.log(mean) if mean > 0.0 else -math.inf
    return out


def _cholesky_tridiag(d: int) -> tuple[np.ndarray, np.ndarray]:
    """Cholesky of the fixed difference covariance tridiag(-1, 2, -1): returns (diag, sub-diagonal)."""
    ldiag = np.empty(d)
    lsub = np.zeros(d)
    ldiag[0] = math.sqrt(2.0)
    for i in range(1, d):
        lsub[i] = -1.0 / ldiag[i - 1]
        ldiag[i] = math.sqrt(2.0 - lsub[i] * lsub[i])
    return ldiag, lsub


class ThurstoneDistribution(SequenceEncodableProbabilityDistribution):
    """Thurstone Case V Gaussian random-utility ranking model with mean utilities ``mu``."""

    @classmethod
    def compute_capabilities(cls):
        """Declare the NumPy and numba execution path used by Thurstone kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="The Gaussian-orthant likelihood runs through a dedicated numba Genz kernel.",
        )

    def __init__(
        self,
        mu: Sequence[float] | np.ndarray,
        name: str | None = None,
        keys: str | None = None,
        n_mc: int = 4000,
        seed: int = 0,
    ) -> None:
        m = np.asarray(mu, dtype=float)
        if m.ndim != 1 or m.size < 2 or not np.all(np.isfinite(m)):
            raise ValueError("mu must be a finite length-K vector with K >= 2.")
        self.mu = m - m.mean()  # identified up to a global location
        self.dim = m.size
        self.n_mc = int(n_mc)
        self.seed = int(seed)
        self._ldiag, self._lsub = _cholesky_tridiag(self.dim - 1)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "ThurstoneDistribution(%s, name=%s, keys=%s)" % (
            repr([float(v) for v in self.mu]),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Sequence[int]) -> float:
        """Return the probability of a full ordering."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int]) -> float:
        """Return the log-probability of one full ordering."""
        return float(self.seq_log_density(np.asarray(x, dtype=np.int64)[None, :])[0])

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for encoded full orderings."""
        x = np.ascontiguousarray(np.asarray(x, dtype=np.int64))
        return _seq_thurstone_logp(self.mu, x, self._ldiag, self._lsub, self.n_mc, self.seed)

    def sampler(self, seed: int | None = None) -> ThurstoneSampler:
        """Return an exact random-utility sampler for this distribution."""
        return ThurstoneSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> ThurstoneEstimator:
        """Return a Thurstone-Mosteller estimator with this distribution's dimension."""
        return ThurstoneEstimator(dim=self.dim, n_mc=self.n_mc, seed=self.seed, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> ThurstoneDataEncoder:
        """Return the dense full-ranking encoder used by vectorized methods."""
        return ThurstoneDataEncoder(dim=self.dim)


class ThurstoneSampler(DistributionSampler):
    """Exact Thurstone draws: sample utilities ``U ~ Normal(mu, 1)`` and sort descending."""

    def __init__(self, dist: ThurstoneDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def _sample_one(self) -> list[int]:
        u = self.dist.mu + self.rng.standard_normal(self.dist.dim)
        return [int(i) for i in np.argsort(-u)]

    def sample(self, size: int | None = None) -> list[int] | list[list[int]]:
        """Draw one ordering or ``size`` iid orderings."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(size)]


class ThurstoneAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the pairwise-precedence matrix ``precede[i, j]`` = weighted count of ``i`` ranked before ``j``."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.precede = np.zeros((dim, dim))
        self.count = 0.0
        self.keys = keys

    def update(self, x: Sequence[int], weight: float, estimate: Any) -> None:
        """Update pairwise-precedence counts from one full ordering."""
        self.seq_update(np.asarray([x], dtype=np.int64), np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize precedence counts from one ordering."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Update pairwise-precedence counts from encoded orderings."""
        n = self.dim
        r_idx, rp_idx = np.triu_indices(n, 1)
        for row, w in zip(x, weights):
            np.add.at(self.precede, (row[r_idx], row[rp_idx]), w)
        self.count += float(np.sum(weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize precedence counts from a batch of encoded orderings."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat) -> ThurstoneAccumulator:
        """Merge count and pairwise-precedence statistics."""
        self.count += suff_stat[0]
        self.precede += suff_stat[1]
        return self

    def value(self):
        """Return accumulated observation weight and pairwise-precedence matrix."""
        return self.count, self.precede

    def from_value(self, x) -> ThurstoneAccumulator:
        """Restore accumulator state from ``value`` output."""
        self.count, self.precede = x[0], np.asarray(x[1])
        self.dim = self.precede.shape[0]
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

    def acc_to_encoder(self) -> ThurstoneDataEncoder:
        """Return the ranking encoder compatible with this accumulator."""
        return ThurstoneDataEncoder(dim=self.dim)


class ThurstoneAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for Thurstone pairwise-precedence statistics."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.keys = keys

    def make(self) -> ThurstoneAccumulator:
        """Create an empty Thurstone accumulator."""
        return ThurstoneAccumulator(dim=self.dim, keys=self.keys)


class ThurstoneEstimator(ParameterEstimator):
    """Thurstone-Mosteller Case V estimator: ``mu_i - mu_j = sqrt(2) * Phi^{-1}(P(i before j))``."""

    def __init__(
        self, dim: int, n_mc: int = 4000, seed: int = 0, name: str | None = None, keys: str | None = None
    ) -> None:
        if dim is None or dim < 2:
            raise ValueError("ThurstoneEstimator requires dim >= 2.")
        self.dim = int(dim)
        self.n_mc = int(n_mc)
        self.seed = int(seed)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> ThurstoneAccumulatorFactory:
        """Return a factory for Thurstone sufficient-statistic accumulators."""
        return ThurstoneAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat) -> ThurstoneDistribution:
        """Estimate centered latent utilities from pairwise-precedence statistics."""
        count, precede = suff_stat
        n = self.dim
        kw = dict(n_mc=self.n_mc, seed=self.seed, name=self.name, keys=self.keys)
        if count <= 0.0:
            return ThurstoneDistribution(np.zeros(n), **kw)
        tot = precede + precede.T
        with np.errstate(invalid="ignore"):
            p = np.where(tot > 0, precede / np.maximum(tot, 1e-12), 0.5)
        np.fill_diagonal(p, 0.5)
        p = np.clip(p, 1e-4, 1.0 - 1e-4)
        from scipy.special import ndtri

        d = _SQRT2 * ndtri(p)  # pairwise utility-difference estimates
        mu = d.mean(axis=1)  # least-squares solution of mu_i - mu_j = d_ij under sum(mu)=0
        return ThurstoneDistribution(mu - mu.mean(), **kw)


class ThurstoneDataEncoder(DataSequenceEncoder):
    """Encode a sequence of orderings (permutations of 0,...,n-1) into an (N, n) integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim

    def __str__(self) -> str:
        return "ThurstoneDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ThurstoneDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        """Validate and encode full orderings as a dense integer matrix."""
        rv = np.asarray([list(row) for row in x], dtype=np.int64)
        if rv.ndim != 2 or rv.shape[0] == 0:
            raise ValueError("ThurstoneDistribution requires a non-empty sequence of orderings.")
        expected = np.arange(rv.shape[1])
        for row in rv:
            if not np.array_equal(np.sort(row), expected):
                raise ValueError("orderings must be permutations of 0,...,n-1.")
        return rv


__all__ = [
    "ThurstoneDistribution",
    "ThurstoneSampler",
    "ThurstoneAccumulator",
    "ThurstoneAccumulatorFactory",
    "ThurstoneEstimator",
    "ThurstoneDataEncoder",
]
