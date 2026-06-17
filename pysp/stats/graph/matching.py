"""Create, estimate, sample, and enumerate a weighted bipartite perfect-matching distribution.

Defines the MatchingDistribution, MatchingEnumerator, MatchingSampler, MatchingAccumulatorFactory,
MatchingAccumulator, MatchingEstimator, and the MatchingDataEncoder classes for use with pysparkplug.

Data type: a perfect matching of the complete bipartite graph K_{n,n} given as a permutation ``x`` of
0,...,n-1, where left node i is matched to right node ``x[i]``.

Each left-right edge (i, j) has a positive weight ``w[i, j]``. A matching (permutation) sigma has

    p(sigma) = prod_i w[i, sigma(i)] / Z,    Z = sum over permutations of prod_i w[i, sigma(i)] = perm(W),

the matrix permanent. Unlike the Plackett-Luce model (a single worth vector over items), this scores a
full edge-weight matrix, so it is the natural assignment / matching law. The permanent is computed
exactly with Ryser's formula, which is exponential in n, so the family targets small-to-moderate n
(default cap ``max_nodes = 12``). Sampling draws each match in turn from the exact conditional
distribution (via permanents of the remaining submatrix); enumeration scans the finite permutation
support and sorts by exact fitted probability. Estimation matches the empirical or symmetrically
smoothed assignment frequencies to the model edge marginals by projected gradient ascent on the
log-weights.
"""

from collections.abc import Sequence
import itertools
from itertools import combinations
from typing import Any

import numpy as np
from numpy.random import RandomState

from pysp.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_MIN_LOG_WEIGHT = -30.0
_MAX_LOG_WEIGHT = 30.0
_DEFAULT_MAX_NODES = 12


def _permanent(a: np.ndarray) -> float:
    """Return the permanent of a square matrix via Ryser's inclusion-exclusion formula."""
    n = a.shape[0]
    if n == 0:
        return 1.0
    total = 0.0
    for k in range(1, n + 1):
        sign = 1.0 if (n - k) % 2 == 0 else -1.0
        sub = 0.0
        for cols in combinations(range(n), k):
            sub += float(np.prod(a[:, cols].sum(axis=1)))
        total += sign * sub
    return total


def _edge_marginals(weights: np.ndarray) -> np.ndarray:
    """Return P(sigma(i) = j) = w[i,j] * perm(minor_ij) / perm(W) for every edge."""
    n = weights.shape[0]
    z = _permanent(weights)
    marg = np.zeros((n, n))
    if z <= 0.0:
        return marg
    rows = np.arange(n)
    for i in range(n):
        for j in range(n):
            minor = weights[np.ix_(rows[rows != i], rows[rows != j])]
            marg[i, j] = weights[i, j] * _permanent(minor) / z
    return marg


class MatchingDistribution(SequenceEncodableProbabilityDistribution):
    """Weighted bipartite perfect-matching distribution over n left/right nodes (permanent-normalized).

    Data type: a permutation x of 0,...,n-1 (left node i matched to right node x[i]).
    """

    @classmethod
    def compute_capabilities(cls):
        from pysp.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="permanent normalizer (Ryser) over matchings is numpy-native.",
        )

    def __init__(
        self,
        weights: Sequence[Sequence[float]] | np.ndarray,
        max_nodes: int = _DEFAULT_MAX_NODES,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """MatchingDistribution object.

        Args:
            weights (Union[Sequence[Sequence[float]], np.ndarray]): n-by-n matrix of positive edge
                weights; ``weights[i, j]`` is the worth of matching left node i to right node j.
            max_nodes (int): Guard on n for the exponential-time permanent (raises above this).
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            weights (np.ndarray): Edge-weight matrix.
            dim (int): Number of nodes n.
            log_weights (np.ndarray): Elementwise log weights.
            log_z (float): log permanent normalizer.

        """
        w = np.asarray(weights, dtype=float).copy()
        n = w.shape[0]
        if w.ndim != 2 or w.shape != (n, n) or n < 1:
            raise ValueError("MatchingDistribution requires a square n-by-n weight matrix with n >= 1.")
        if n > max_nodes:
            raise ValueError(
                "MatchingDistribution n=%d exceeds max_nodes=%d (permanent is exponential)." % (n, max_nodes)
            )
        if np.any(w <= 0.0) or not np.all(np.isfinite(w)):
            raise ValueError("MatchingDistribution requires finite positive edge weights.")
        self.weights = w
        self.dim = n
        self.max_nodes = max_nodes
        self.log_weights = np.log(w)
        self.log_z = float(np.log(_permanent(w)))
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return string representation of MatchingDistribution object."""
        return "MatchingDistribution(%s, max_nodes=%s, name=%s, keys=%s)" % (
            repr([[float(v) for v in row] for row in self.weights]),
            repr(self.max_nodes),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Sequence[int]) -> float:
        """Return the probability of a matching x (a permutation)."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int]) -> float:
        """Return the log-probability of a matching x (left i matched to right x[i])."""
        sigma = np.asarray(x, dtype=int)
        return float(np.sum(self.log_weights[np.arange(self.dim), sigma])) - self.log_z

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for an (N, n) array of matchings."""
        rows = np.arange(self.dim)
        return self.log_weights[rows[None, :], x].sum(axis=1) - self.log_z

    def sampler(self, seed: int | None = None) -> "MatchingSampler":
        """Return a sampler for drawing matchings from this distribution."""
        return MatchingSampler(self, seed)

    def enumerator(self) -> "MatchingEnumerator":
        """Return an exact finite enumerator over all matchings in decreasing probability order."""
        return MatchingEnumerator(self)

    def estimator(self, pseudo_count: float | None = 1.0) -> "MatchingEstimator":
        """Return an estimator that keeps the node count fixed at this distribution's n."""
        return MatchingEstimator(
            dim=self.dim, max_nodes=self.max_nodes, pseudo_count=pseudo_count, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "MatchingDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return MatchingDataEncoder(dim=self.dim)


class MatchingEnumerator(DistributionEnumerator):
    """Enumerate all finite perfect matchings in descending probability order."""

    def __init__(self, dist: MatchingDistribution) -> None:
        super().__init__(dist)
        with np.errstate(divide="ignore"):
            entries = [(list(p), float(dist.log_density(p))) for p in itertools.permutations(range(dist.dim))]
        entries = [(v, lp) for v, lp in entries if lp > -np.inf]
        entries.sort(key=lambda u: -u[1])
        self._entries = entries
        self._pos = 0

    def __next__(self) -> tuple[list[int], float]:
        if self._pos >= len(self._entries):
            raise StopIteration
        item = self._entries[self._pos]
        self._pos += 1
        return item


class MatchingSampler(DistributionSampler):
    """Draw iid matchings by sampling each left node's match from the exact conditional permanent."""

    def __init__(self, dist: MatchingDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _sample_one(self) -> list[int]:
        n = self.dist.dim
        available = list(range(n))  # remaining right nodes
        sigma = [0] * n
        for i in range(n):
            sub_rows = list(range(i + 1, n))
            probs = np.empty(len(available))
            for t, j in enumerate(available):
                rest = [c for c in available if c != j]
                minor = self.dist.weights[np.ix_(sub_rows, rest)] if sub_rows else np.zeros((0, 0))
                probs[t] = self.dist.weights[i, j] * _permanent(minor)
            probs /= probs.sum()
            choice = int(self.rng.choice(len(available), p=probs))
            sigma[i] = available.pop(choice)
        return sigma

    def sample(self, size: int | None = None) -> list[int] | list[list[int]]:
        """Draw matchings (each a permutation); a single matching when size is None."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(size)]


class MatchingAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted assignment-frequency matrix (the sufficient statistic for the weights)."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.assign_counts = np.zeros((dim, dim))
        self.count = 0.0
        self.key = keys

    def update(self, x: Sequence[int], weight: float, estimate: MatchingDistribution | None) -> None:
        sigma = np.asarray(x, dtype=int)
        self.assign_counts[np.arange(self.dim), sigma] += weight
        self.count += weight

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: MatchingDistribution | None) -> None:
        rows = np.arange(self.dim)
        for sigma, w in zip(x, weights):
            self.assign_counts[rows, sigma] += w
        self.count += float(np.sum(weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray]) -> "MatchingAccumulator":
        self.count += suff_stat[0]
        self.assign_counts += suff_stat[1]
        return self

    def value(self) -> tuple[float, np.ndarray]:
        return self.count, self.assign_counts

    def from_value(self, x: tuple[float, np.ndarray]) -> "MatchingAccumulator":
        self.count, self.assign_counts = x[0], np.asarray(x[1])
        self.dim = self.assign_counts.shape[0]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> "MatchingDataEncoder":
        return MatchingDataEncoder(dim=self.dim)


class MatchingAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for MatchingAccumulator."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.keys = keys

    def make(self) -> MatchingAccumulator:
        return MatchingAccumulator(dim=self.dim, keys=self.keys)


class MatchingEstimator(ParameterEstimator):
    """Maximum-likelihood estimator for the edge weights (matches empirical and model edge marginals)."""

    def __init__(
        self,
        dim: int,
        max_nodes: int = _DEFAULT_MAX_NODES,
        pseudo_count: float | None = 1.0,
        max_steps: int = 500,
        learning_rate: float = 1.0,
        tol: float = 1.0e-7,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if dim is None or dim < 1:
            raise ValueError("MatchingEstimator requires the number of nodes dim >= 1.")
        if pseudo_count is not None and pseudo_count < 0.0:
            raise ValueError("MatchingEstimator requires a non-negative pseudo_count.")
        self.dim = int(dim)
        self.max_nodes = max_nodes
        self.pseudo_count = pseudo_count
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.tol = tol
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> MatchingAccumulatorFactory:
        return MatchingAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray]) -> MatchingDistribution:
        count, assign_counts = suff_stat
        n = self.dim
        if count <= 0.0:
            return MatchingDistribution(np.ones((n, n)), max_nodes=self.max_nodes, name=self.name, keys=self.keys)

        # Symmetric smoothing preserves row and column sums of the assignment-marginal target.
        if self.pseudo_count is None:
            target = assign_counts / count
        else:
            target = (assign_counts + self.pseudo_count) / (count + n * self.pseudo_count)
        log_w = np.zeros((n, n))
        weights = np.ones((n, n))
        for _ in range(self.max_steps):
            marginals = _edge_marginals(weights)
            grad = target - marginals
            if np.max(np.abs(grad)) < self.tol:
                break
            log_w = np.clip(log_w + self.learning_rate * grad, _MIN_LOG_WEIGHT, _MAX_LOG_WEIGHT)
            # Fix the row/column scale gauge (p(sigma) is invariant to scaling any row or column).
            log_w = log_w - log_w.mean(axis=1, keepdims=True)
            log_w = log_w - log_w.mean(axis=0, keepdims=True)
            weights = np.exp(log_w)

        return MatchingDistribution(weights, max_nodes=self.max_nodes, name=self.name, keys=self.keys)


class MatchingDataEncoder(DataSequenceEncoder):
    """Encode a sequence of matchings (permutations) into an (N, n) integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim

    def __str__(self) -> str:
        return "MatchingDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MatchingDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        rv = np.asarray([list(row) for row in x], dtype=int)
        if rv.ndim != 2 or rv.shape[0] == 0:
            raise ValueError("MatchingDistribution requires a non-empty sequence of matchings.")
        expected = np.arange(rv.shape[1])
        for row in rv:
            if not np.array_equal(np.sort(row), expected):
                raise ValueError("MatchingDistribution matchings must be permutations of 0,...,n-1.")
        return rv
