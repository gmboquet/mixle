"""Weighted bipartite perfect-matching distributions.

Data type: a perfect matching of the complete bipartite graph K_{n,n} given as a permutation ``x`` of
0,...,n-1, where left node i is matched to right node ``x[i]``.

Each left-right edge (i, j) has a positive weight ``w[i, j]``. A matching (permutation) sigma has

    p(sigma) = prod_i w[i, sigma(i)] / Z,    Z = sum over permutations of prod_i w[i, sigma(i)] = perm(W),

the matrix permanent. Unlike the Plackett-Luce model (a single worth vector over items), this scores a
full edge-weight matrix, so it is the natural assignment / matching law. The permanent is computed
exactly with Ryser's formula, which is exponential in n, so the family targets small-to-moderate n
(default cap ``max_nodes = 12``). Sampling draws each match in turn from the exact conditional
distribution (via permanents of the remaining submatrix); enumeration is lazy and streams matchings in
decreasing probability via Murty's k-best assignment (no n! materialization). Estimation matches the empirical or symmetrically
smoothed assignment frequencies to the model edge marginals by projected gradient ascent on the
log-weights.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.random import RandomState

from mixle.enumeration.assignment import k_best_assignments
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.rankings._contracts import (
    count_matrix_statistics,
    finite_nonnegative,
    finite_positive,
    permutation,
    permutation_batch,
    positive_integer,
    sample_size,
)
from mixle.stats.rankings._contracts import (
    weights as validate_weights,
)
from mixle.stats.rankings._permutation_kernels import log_matrix_permanent

_MIN_LOG_WEIGHT = -30.0
_MAX_LOG_WEIGHT = 30.0
_DEFAULT_MAX_NODES = 12


def _permanent(a: np.ndarray) -> float:
    """Compatibility helper returning the permanent when it is representable as float64."""
    weights = np.asarray(a, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("permanent input must be a square matrix.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("permanent input must be finite and nonnegative.")
    with np.errstate(divide="ignore"):
        log_value = log_matrix_permanent(np.log(weights))
    try:
        value = math.exp(log_value)
    except OverflowError as exc:
        raise FloatingPointError("permanent is not representable as float64; use a log-domain API.") from exc
    if not math.isfinite(value):
        raise FloatingPointError("permanent is not representable as float64; use a log-domain API.")
    return value


def _edge_marginals_from_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Return exact edge marginals using log-domain permanent ratios."""
    n = log_weights.shape[0]
    log_z = log_matrix_permanent(log_weights)
    if log_z == -np.inf:
        raise ValueError("matching weights do not support a perfect matching.")
    marg = np.zeros((n, n))
    rows = np.arange(n)
    for i in range(n):
        for j in range(n):
            minor = log_weights[np.ix_(rows[rows != i], rows[rows != j])]
            log_marginal = log_weights[i, j] + log_matrix_permanent(minor) - log_z
            marg[i, j] = math.exp(log_marginal) if log_marginal > -np.inf else 0.0
    return marg


def _edge_marginals(weights: np.ndarray) -> np.ndarray:
    """Return exact edge marginals for a finite nonnegative weight matrix."""
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matching weights must be a square matrix.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("matching weights must be finite and nonnegative.")
    with np.errstate(divide="ignore"):
        marg = _edge_marginals_from_log_weights(np.log(matrix))
    if not np.all(np.isfinite(marg)):
        raise FloatingPointError("matching marginal evaluation produced non-finite values.")
    return marg


def _matching_statistics(value, dim: int) -> tuple[float, np.ndarray]:
    """Validate weighted assignment counts against the perfect-matching polytope."""
    count, counts = count_matrix_statistics(
        value,
        dim,
        label="matching statistics",
        entries_per_observation=float(dim),
    )
    tolerance = 1.0e-10 * max(1.0, count)
    if not np.allclose(counts.sum(axis=1), count, rtol=1.0e-10, atol=tolerance):
        raise ValueError("matching statistic row sums must equal total observation weight.")
    if not np.allclose(counts.sum(axis=0), count, rtol=1.0e-10, atol=tolerance):
        raise ValueError("matching statistic column sums must equal total observation weight.")
    return count, counts


@dataclass(frozen=True)
class MatchingFitDiagnostics:
    """Convergence evidence attached to an estimated matching model."""

    converged: bool
    iterations: int
    max_marginal_error: float
    regularized: bool
    pseudo_count: float


class MatchingDistribution(SequenceEncodableProbabilityDistribution):
    """Weighted bipartite perfect-matching distribution over n left/right nodes (permanent-normalized).

    Data type: a permutation x of 0,...,n-1 (left node i matched to right node x[i]).
    """

    @classmethod
    def compute_capabilities(cls):
        """Return compute-backend metadata for the matching distribution."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

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
        fit_diagnostics: MatchingFitDiagnostics | None = None,
    ) -> None:
        """Create a distribution over bipartite perfect matchings.

        Args:
            weights (Union[Sequence[Sequence[float]], np.ndarray]): n-by-n matrix of positive edge
                weights; ``weights[i, j]`` is the worth of matching left node i to right node j.
            max_nodes (int): Guard on n for the exponential-time permanent (raises above this).
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            weights (np.ndarray): Edge-weight matrix.
            dim (int): Number of nodes n.
            log_weights (np.ndarray): Elementwise log weights.
            log_z (float): log permanent normalizer.

        """
        w = np.asarray(weights, dtype=float)
        if w.ndim != 2:
            raise ValueError("MatchingDistribution requires a square n-by-n weight matrix with n >= 1.")
        n = w.shape[0]
        if w.ndim != 2 or w.shape != (n, n) or n < 1:
            raise ValueError("MatchingDistribution requires a square n-by-n weight matrix with n >= 1.")
        checked_max_nodes = positive_integer(max_nodes, label="max_nodes")
        if n > checked_max_nodes:
            raise ValueError(
                "MatchingDistribution n=%d exceeds max_nodes=%d (permanent is exponential)." % (n, checked_max_nodes)
            )
        if n > 22:
            raise ValueError("MatchingDistribution exact permanent evaluation is limited to 22 nodes.")
        if np.any(w <= 0.0) or not np.all(np.isfinite(w)):
            raise ValueError("MatchingDistribution requires finite positive edge weights.")
        self.weights = w.copy()
        self.weights.setflags(write=False)
        self.dim = n
        self.max_nodes = checked_max_nodes
        self.log_weights = np.log(self.weights)
        self.log_weights.setflags(write=False)
        self.log_z = log_matrix_permanent(self.log_weights)
        self.name = name
        self.keys = keys
        if fit_diagnostics is not None and not isinstance(fit_diagnostics, MatchingFitDiagnostics):
            raise TypeError("fit_diagnostics must be a MatchingFitDiagnostics record.")
        self.fit_diagnostics = fit_diagnostics

    def __str__(self) -> str:
        """Return a constructor-style representation of the matching distribution."""
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
        sigma = permutation(x, self.dim, label="matching")
        return float(np.sum(self.log_weights[np.arange(self.dim), sigma])) - self.log_z

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for an (N, n) array of matchings."""
        checked = permutation_batch(x, self.dim, label="matchings")
        rows = np.arange(self.dim)
        return self.log_weights[rows[None, :], checked].sum(axis=1) - self.log_z

    def sampler(self, seed: int | None = None) -> "MatchingSampler":
        """Return a sampler for drawing matchings from this distribution."""
        return MatchingSampler(self, seed)

    def enumerator(self) -> "MatchingEnumerator":
        """Return an exact finite enumerator over all matchings in decreasing probability order."""
        return MatchingEnumerator(self)

    def support_size(self) -> int:
        """Return the number of perfect matchings."""
        return math.factorial(self.dim)

    def estimator(self, pseudo_count: float | None = 1.0) -> "MatchingEstimator":
        """Return an estimator that keeps the node count fixed at this distribution's n."""
        return MatchingEstimator(
            dim=self.dim, max_nodes=self.max_nodes, pseudo_count=pseudo_count, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "MatchingDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return MatchingDataEncoder(dim=self.dim)


class MatchingEnumerator(DistributionEnumerator):
    """Enumerate finite-probability perfect matchings in descending probability order.

    Lazily, via Murty's k-best assignment on the edge-cost matrix ``-log(weights)``: decreasing probability is
    increasing assignment cost, and zero-weight edges become +inf costs (forbidden), so only positive-probability
    matchings are yielded. This streams the top matchings without materializing the n! permutation support.
    """

    def __init__(self, dist: MatchingDistribution) -> None:
        super().__init__(dist)
        with np.errstate(divide="ignore"):
            cost = -np.log(dist.weights)  # +inf where weight == 0 -> forbidden edge
        self._gen = k_best_assignments(cost)

    def __next__(self) -> tuple[list[int], float]:
        total_cost, rows, cols = next(self._gen)  # StopIteration propagates at the end of the support
        sigma = [0] * self.dist.dim
        for r, c in zip(rows, cols):
            sigma[int(r)] = int(c)
        return sigma, float(-total_cost - self.dist.log_z)


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
            log_probs = np.empty(len(available))
            for t, j in enumerate(available):
                rest = [c for c in available if c != j]
                minor = self.dist.log_weights[np.ix_(sub_rows, rest)]
                log_probs[t] = self.dist.log_weights[i, j] + log_matrix_permanent(minor)
            shift = float(np.max(log_probs))
            probs = np.exp(log_probs - shift)
            probs /= probs.sum()
            choice = int(self.rng.choice(len(available), p=probs))
            sigma[i] = available.pop(choice)
        return sigma

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[int] | list[list[int]]:
        """Draw matchings (each a permutation); a single matching when size is None."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(sample_size(size))]


class MatchingAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted assignment-frequency matrix (the sufficient statistic for the weights)."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim")
        self.assign_counts = np.zeros((self.dim, self.dim))
        self.count = 0.0
        self.keys = keys

    def update(self, x: Sequence[int], weight: float, estimate: MatchingDistribution | None) -> None:
        """Accumulate weighted assignment counts for one matching."""
        sigma = permutation(x, self.dim, label="matching")
        checked_weight = finite_nonnegative(weight, label="weight")
        self.assign_counts[np.arange(self.dim), sigma] += checked_weight
        self.count += checked_weight

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted matching."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: MatchingDistribution | None) -> None:
        """Accumulate weighted assignment counts for encoded matchings."""
        checked = permutation_batch(x, self.dim, label="matchings")
        checked_weights = validate_weights(weights, len(checked))
        rows = np.arange(self.dim)
        for sigma, weight in zip(checked, checked_weights):
            self.assign_counts[rows, sigma] += weight
        self.count += float(np.sum(checked_weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded matchings."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray]) -> "MatchingAccumulator":
        """Merge serialized assignment-count statistics into this accumulator."""
        count, assign_counts = _matching_statistics(suff_stat, self.dim)
        self.count += count
        self.assign_counts += assign_counts
        return self

    def value(self) -> tuple[float, np.ndarray]:
        """Return the accumulated weight and assignment-count matrix."""
        return self.count, self.assign_counts.copy()

    def from_value(self, x: tuple[float, np.ndarray]) -> "MatchingAccumulator":
        """Restore the accumulator from serialized assignment-count statistics."""
        self.count, self.assign_counts = _matching_statistics(x, self.dim)
        return self

    def acc_to_encoder(self) -> "MatchingDataEncoder":
        """Return an encoder compatible with the accumulated matching dimension."""
        return MatchingDataEncoder(dim=self.dim)


class MatchingAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for MatchingAccumulator."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim")
        self.keys = keys

    def make(self) -> MatchingAccumulator:
        """Create an empty matching accumulator."""
        return MatchingAccumulator(dim=self.dim, keys=self.keys)


class MatchingEstimator(ParameterEstimator):
    """Maximum-likelihood estimator for the edge weights (matches empirical and model edge marginals)."""

    def __init__(
        self,
        dim: int,
        max_nodes: int = _DEFAULT_MAX_NODES,
        pseudo_count: float | None = 1.0,
        # 5000, not 500. Measured on this estimator's OWN samples (400 draws from a seeded
        # MatchingDistribution, fit back at tol=1e-7): n=3 converges in 164 iterations, n=4 in 869,
        # n=5 in 1015, n=8 in 1172. The old default of 500 therefore FAILED the round trip for every
        # dimension above three -- and because non-convergence here raises rather than returning a
        # flagged fit, the default configuration was a guaranteed RuntimeError on ordinary data. The
        # loop breaks the moment it converges, so the larger ceiling costs nothing when fewer steps
        # suffice (~1s worst case measured at the max_nodes=12 permanent cap).
        max_steps: int = 5000,
        learning_rate: float = 1.0,
        tol: float = 1.0e-7,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dim = positive_integer(dim, label="dim")
        self.max_nodes = positive_integer(max_nodes, label="max_nodes")
        if self.dim > self.max_nodes:
            raise ValueError("dim must not exceed max_nodes.")
        if self.dim > 22:
            raise ValueError("MatchingEstimator exact permanent evaluation is limited to 22 nodes.")
        self.pseudo_count = None if pseudo_count is None else finite_nonnegative(pseudo_count, label="pseudo_count")
        self.max_steps = positive_integer(max_steps, label="max_steps")
        self.learning_rate = finite_positive(learning_rate, label="learning_rate")
        self.tol = finite_positive(tol, label="tol")
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> MatchingAccumulatorFactory:
        """Return a factory for matching sufficient-statistic accumulators."""
        return MatchingAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray]) -> MatchingDistribution:
        """Estimate a permanent-normalized matching distribution from assignment counts."""
        if nobs is not None:
            finite_nonnegative(nobs, label="nobs")
        count, assign_counts = _matching_statistics(suff_stat, self.dim)
        n = self.dim
        if count <= 0.0:
            return MatchingDistribution(np.ones((n, n)), max_nodes=self.max_nodes, name=self.name, keys=self.keys)

        # Symmetric smoothing preserves row and column sums of the assignment-marginal target.
        if self.pseudo_count is None:
            target = assign_counts / count
        else:
            target = (assign_counts + self.pseudo_count) / (count + n * self.pseudo_count)
        log_w = np.zeros((n, n))
        converged = False
        max_error = math.inf
        iterations = 0
        for step in range(self.max_steps):
            marginals = _edge_marginals_from_log_weights(log_w)
            grad = target - marginals
            max_error = float(np.max(np.abs(grad)))
            iterations = step + 1
            if max_error < self.tol:
                converged = True
                break
            log_w = np.clip(log_w + self.learning_rate * grad, _MIN_LOG_WEIGHT, _MAX_LOG_WEIGHT)
            # Fix the row/column scale gauge (p(sigma) is invariant to scaling any row or column).
            log_w = log_w - log_w.mean(axis=1, keepdims=True)
            log_w = log_w - log_w.mean(axis=0, keepdims=True)
        if not converged:
            raise RuntimeError(
                "Matching estimation did not converge in "
                f"{iterations} steps (maximum marginal error {max_error:.6g}, tolerance {self.tol:.6g})."
            )
        weights = np.exp(log_w)
        diagnostics = MatchingFitDiagnostics(
            converged=True,
            iterations=iterations,
            max_marginal_error=max_error,
            regularized=self.pseudo_count is not None and self.pseudo_count > 0.0,
            pseudo_count=0.0 if self.pseudo_count is None else self.pseudo_count,
        )
        return MatchingDistribution(
            weights,
            max_nodes=self.max_nodes,
            name=self.name,
            keys=self.keys,
            fit_diagnostics=diagnostics,
        )


class MatchingDataEncoder(DataSequenceEncoder):
    """Encode a sequence of matchings (permutations) into an (N, n) integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = None if dim is None else positive_integer(dim, label="dim")

    def __str__(self) -> str:
        return "MatchingDataEncoder(dim=%s)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MatchingDataEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        """Validate and encode matchings as a two-dimensional integer array."""
        raw = np.asarray([list(row) for row in x])
        if self.dim is None:
            if raw.ndim != 2 or raw.shape[0] == 0:
                raise ValueError("MatchingDistribution requires a non-empty sequence of matchings.")
            return permutation_batch(raw, raw.shape[1], label="matchings", allow_empty=False)
        return permutation_batch(raw, self.dim, label="matchings", allow_empty=False)


__all__ = [
    "MatchingDistribution",
    "MatchingEnumerator",
    "MatchingSampler",
    "MatchingAccumulator",
    "MatchingAccumulatorFactory",
    "MatchingEstimator",
    "MatchingDataEncoder",
    "MatchingFitDiagnostics",
]
