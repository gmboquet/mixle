"""Low-rank assignment (first-order Fourier) model over permutations.

A permutation Gibbs model that scores an ordering by a low-rank item-by-rank score matrix:

    p(sigma) = exp( sum_r S[sigma[r], r] ) / Z,      S = U V^T   (rank r << n),   Z = permanent(exp(S)).

This is the maximum-entropy distribution matching first-order (item-at-rank) marginals -- the
first-order term of the Fourier / coset expansion on the symmetric group -- and the low-rank
factorization ``S = U V^T`` is the structured, ``O(n r)``-parameter version of the full
:class:`MatchingDistribution`. The normalizer is a permanent (#P-hard), so it is computed exactly by a
numba Ryser kernel for ``n <= max_exact`` and approximated by a numba Sinkhorn / Bethe permanent beyond
that (documented approximation). Fitting is a Sinkhorn-marginal gradient ascent on ``U, V`` toward the
empirical item-by-rank marginals.

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
from mixle.stats.rankings._permutation_kernels import ryser_log_permanent, sinkhorn_bethe
from mixle.utils.optional_deps import numba


@numba.njit("int64[:, :](float64[:, :], int64, int64, int64, int64)", cache=True)
def _mh_assignment(s, n_samples, burn, thin, seed):
    """Metropolis sampler for the assignment Gibbs model: propose a rank swap, accept by exp(delta)."""
    np.random.seed(seed)
    n = s.shape[0]
    sigma = np.arange(n)
    np.random.shuffle(sigma)
    out = np.empty((n_samples, n), dtype=np.int64)
    total = burn + n_samples * thin
    taken = 0
    for step in range(total):
        a = np.random.randint(0, n)
        b = np.random.randint(0, n)
        if a != b:
            ia, ib = sigma[a], sigma[b]
            delta = s[ia, b] + s[ib, a] - s[ia, a] - s[ib, b]  # swap items at ranks a, b
            if delta >= 0.0 or np.random.random() < math.exp(delta):
                sigma[a], sigma[b] = ib, ia
        if step >= burn and (step - burn) % thin == 0 and taken < n_samples:
            out[taken, :] = sigma
            taken += 1
    return out


def _log_normalizer(s: np.ndarray, max_exact: int, sinkhorn_iter: int) -> float:
    n = s.shape[0]
    if n <= max_exact:
        return float(ryser_log_permanent(np.ascontiguousarray(np.exp(s))))
    _, logz = sinkhorn_bethe(np.ascontiguousarray(s), sinkhorn_iter)
    return float(logz)


class LowRankPermutationDistribution(SequenceEncodableProbabilityDistribution):
    """Permutation Gibbs model with a low-rank item-by-rank score matrix ``S = U V^T``."""

    @classmethod
    def compute_capabilities(cls):
        """Declare the NumPy and numba execution path used by low-rank permutation kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="Permanent normalizer / Sinkhorn marginals run through dedicated numba kernels.",
        )

    def __init__(
        self,
        u: np.ndarray,
        v: np.ndarray,
        name: str | None = None,
        keys: str | None = None,
        max_exact: int = 12,
        sinkhorn_iter: int = 200,
    ) -> None:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        if u.ndim != 2 or v.ndim != 2 or u.shape != v.shape or u.shape[0] < 2:
            raise ValueError("u and v must be equal-shape (n, rank) matrices with n >= 2.")
        self.u = u
        self.v = v
        self.dim = u.shape[0]
        self.rank = u.shape[1]
        self.s = u @ v.T
        self.max_exact = int(max_exact)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.log_z = _log_normalizer(self.s, self.max_exact, self.sinkhorn_iter)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "LowRankPermutationDistribution(rank=%d, dim=%d, name=%s, keys=%s)" % (
            self.rank,
            self.dim,
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Sequence[int]) -> float:
        """Return the probability of one ordering."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int]) -> float:
        """Return the log-probability of one ordering."""
        x = np.asarray(x, dtype=np.int64)
        return float(self.s[x, np.arange(self.dim)].sum() - self.log_z)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for encoded orderings."""
        ranks = np.arange(self.dim)
        return self.s[x, ranks[None, :]].sum(axis=1) - self.log_z

    def marginals(self) -> np.ndarray:
        """Model first-order marginals ``P[item, rank]`` (Sinkhorn doubly-stochastic transport plan)."""
        p, _ = sinkhorn_bethe(np.ascontiguousarray(self.s), self.sinkhorn_iter)
        return p

    def sampler(self, seed: int | None = None) -> LowRankPermutationSampler:
        """Return a Metropolis sampler for this low-rank assignment model."""
        return LowRankPermutationSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> LowRankPermutationEstimator:
        """Return a Sinkhorn-marginal estimator with this dimension and rank."""
        return LowRankPermutationEstimator(
            dim=self.dim,
            rank=self.rank,
            max_exact=self.max_exact,
            sinkhorn_iter=self.sinkhorn_iter,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> LowRankPermutationDataEncoder:
        """Return the full-ranking encoder used by vectorized methods."""
        return LowRankPermutationDataEncoder(dim=self.dim)


class LowRankPermutationSampler(DistributionSampler):
    """Draw orderings via a numba Metropolis sampler on the assignment scores."""

    def __init__(
        self, dist: LowRankPermutationDistribution, seed: int | None = None, burn: int = 2000, thin: int = 20
    ) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.burn = burn
        self.thin = thin

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[int] | list[list[int]]:
        """Draw one ordering or ``size`` approximate iid orderings."""
        k = 1 if size is None else size
        seed = int(self.rng.randint(0, 2**31 - 1))
        arr = _mh_assignment(np.ascontiguousarray(self.dist.s), k, self.burn, self.thin, seed)
        draws = [[int(v) for v in row] for row in arr]
        return draws[0] if size is None else draws


class LowRankPermutationAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the item-by-rank count matrix ``C[item, rank]`` -- the first-order sufficient statistic."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.counts = np.zeros((dim, dim))
        self.count = 0.0
        self.keys = keys

    def update(self, x: Sequence[int], weight: float, estimate: Any) -> None:
        """Update item-by-rank counts from one weighted ordering."""
        self.seq_update(np.asarray([x], dtype=np.int64), np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize item-by-rank counts from one weighted ordering."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Update item-by-rank counts from encoded orderings."""
        ranks = np.arange(self.dim)
        for row, w in zip(x, weights):
            self.counts[row, ranks] += w
        self.count += float(np.sum(weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize item-by-rank counts from encoded orderings."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat) -> LowRankPermutationAccumulator:
        """Merge observation weight and item-by-rank count statistics."""
        self.count += suff_stat[0]
        self.counts += suff_stat[1]
        return self

    def value(self):
        """Return accumulated observation weight and item-by-rank counts."""
        return self.count, self.counts

    def from_value(self, x) -> LowRankPermutationAccumulator:
        """Restore accumulator state from ``value`` output."""
        self.count, self.counts = x[0], np.asarray(x[1])
        self.dim = self.counts.shape[0]
        return self

    def acc_to_encoder(self) -> LowRankPermutationDataEncoder:
        """Return the encoder compatible with item-by-rank sufficient statistics."""
        return LowRankPermutationDataEncoder(dim=self.dim)


class LowRankPermutationAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for low-rank permutation sufficient statistics."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.keys = keys

    def make(self) -> LowRankPermutationAccumulator:
        """Create an empty low-rank permutation accumulator."""
        return LowRankPermutationAccumulator(dim=self.dim, keys=self.keys)


class LowRankPermutationEstimator(ParameterEstimator):
    """Fit ``U, V`` by Sinkhorn-marginal gradient ascent toward the empirical item-by-rank marginals."""

    def __init__(
        self,
        dim: int,
        rank: int = 2,
        max_exact: int = 12,
        sinkhorn_iter: int = 200,
        max_iter: int = 300,
        lr: float = 0.5,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if dim is None or dim < 2:
            raise ValueError("LowRankPermutationEstimator requires dim >= 2.")
        self.dim = int(dim)
        self.rank = int(rank)
        self.max_exact = int(max_exact)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> LowRankPermutationAccumulatorFactory:
        """Return a factory for low-rank permutation sufficient-statistic accumulators."""
        return LowRankPermutationAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat) -> LowRankPermutationDistribution:
        """Estimate low-rank score factors from item-by-rank marginal counts."""
        count, counts = suff_stat
        n, r = self.dim, self.rank
        kw = dict(max_exact=self.max_exact, sinkhorn_iter=self.sinkhorn_iter, name=self.name, keys=self.keys)
        if count <= 0.0:
            return LowRankPermutationDistribution(np.zeros((n, r)), np.zeros((n, r)), **kw)
        m = counts / count  # empirical item-by-rank marginals (doubly stochastic)
        # low-rank init from the centered log-marginals
        log_m = np.log(np.clip(m, 1e-6, None))
        log_m -= log_m.mean(axis=1, keepdims=True) + log_m.mean(axis=0, keepdims=True) - log_m.mean()
        uu, ss, vt = np.linalg.svd(log_m)
        u = uu[:, :r] * np.sqrt(ss[:r])[None, :]
        v = (vt[:r, :].T) * np.sqrt(ss[:r])[None, :]
        for _ in range(self.max_iter):  # ascend the (approximate) likelihood: grad_S = M - model marginals
            p, _ = sinkhorn_bethe(np.ascontiguousarray(u @ v.T), self.sinkhorn_iter)
            grad = m - p
            u = u + self.lr * grad @ v
            v = v + self.lr * grad.T @ u
        return LowRankPermutationDistribution(u, v, **kw)


class LowRankPermutationDataEncoder(DataSequenceEncoder):
    """Encode a sequence of orderings (permutations of 0,...,n-1) into an (N, n) integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim

    def __str__(self) -> str:
        return "LowRankPermutationDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LowRankPermutationDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        """Validate and encode full orderings as a dense integer matrix."""
        rv = np.asarray([list(row) for row in x], dtype=np.int64)
        if rv.ndim != 2 or rv.shape[0] == 0:
            raise ValueError("LowRankPermutationDistribution requires a non-empty sequence of orderings.")
        expected = np.arange(rv.shape[1])
        for row in rv:
            if not np.array_equal(np.sort(row), expected):
                raise ValueError("orderings must be permutations of 0,...,n-1.")
        return rv


__all__ = [
    "LowRankPermutationDistribution",
    "LowRankPermutationSampler",
    "LowRankPermutationAccumulator",
    "LowRankPermutationAccumulatorFactory",
    "LowRankPermutationEstimator",
    "LowRankPermutationDataEncoder",
]
