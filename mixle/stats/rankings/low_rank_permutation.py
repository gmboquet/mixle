"""Low-rank assignment (first-order Fourier) model over permutations.

A permutation Gibbs model that scores an ordering by a low-rank item-by-rank score matrix:

    p(sigma) = exp( sum_r S[sigma[r], r] ) / Z,      S = U V^T   (rank r << n),   Z = permanent(exp(S)).

This is the maximum-entropy distribution matching first-order (item-at-rank) marginals -- the
first-order term of the Fourier / coset expansion on the symmetric group -- and the low-rank
factorization ``S = U V^T`` is the structured, ``O(n r)``-parameter version of the full
:class:`MatchingDistribution`. The normalizer is a permanent (#P-hard), so probability-distribution
construction is limited by an explicit exact-computation cap. Sinkhorn is exposed separately as a
transport relaxation; it is never substituted for the model normalizer or marginals. Fitting uses
exact model marginals and records optimization diagnostics.

Data type: ``List[int]`` -- a full ordering, a permutation of ``0..n-1`` with ``x[r]`` the item at rank
``r`` (best first).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
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
from mixle.stats.rankings._contracts import (
    assignment_statistics,
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
from mixle.stats.rankings._permutation_kernels import log_matrix_permanent, sinkhorn_bethe
from mixle.utils.exact import require_exact_bool


def _log_normalizer(s: np.ndarray, max_exact: int, sinkhorn_iter: int) -> float:
    """Return the exact log normalizer within the declared exact-computation budget."""
    n = s.shape[0]
    if n > max_exact:
        raise ValueError(
            f"dimension {n} exceeds max_exact={max_exact}; a Bethe relaxation cannot normalize a "
            "probability distribution. Raise max_exact within the supported limit or use "
            "sinkhorn_relaxation() as an explicitly approximate transport result."
        )
    return log_matrix_permanent(s)


def _exact_marginals(s: np.ndarray, log_z: float | None = None) -> np.ndarray:
    """Return exact item-by-rank marginals from log-permanent ratios."""
    n = s.shape[0]
    normalizer = log_matrix_permanent(s) if log_z is None else log_z
    result = np.zeros((n, n), dtype=np.float64)
    indices = np.arange(n)
    for item in range(n):
        for rank in range(n):
            minor = s[np.ix_(indices[indices != item], indices[indices != rank])]
            result[item, rank] = math.exp(s[item, rank] + log_matrix_permanent(minor) - normalizer)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("exact low-rank permutation marginals are non-finite.")
    return result


@dataclass(frozen=True)
class LowRankPermutationFitDiagnostics:
    """Optimization evidence attached to an estimated low-rank permutation law."""

    converged: bool
    iterations: int
    gradient_norm: float
    regularized: bool
    pseudo_count: float
    marginal_algorithm: str = "exact_log_permanent"


@dataclass(frozen=True)
class LowRankPermutationComputationDiagnostics:
    """Exactness and algorithm provenance for the represented probability law."""

    normalizer_algorithm: str = "exact_log_permanent"
    normalizer_exact: bool = True
    marginal_algorithm: str = "exact_log_permanent_ratios"
    marginals_exact: bool = True
    sampler_algorithm: str = "exact_conditional_permanent"
    sampler_exact: bool = True
    optional_relaxation: str = "sinkhorn_bethe_transport"


class LowRankPermutationDistribution(SequenceEncodableProbabilityDistribution):
    """Permutation Gibbs model with a low-rank item-by-rank score matrix ``S = U V^T``."""

    @classmethod
    def compute_capabilities(cls):
        """Declare the NumPy and numba execution path used by low-rank permutation kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="Exact permanent normalization and marginals run through dedicated numba kernels.",
        )

    def __init__(
        self,
        u: np.ndarray,
        v: np.ndarray,
        name: str | None = None,
        keys: str | None = None,
        max_exact: int = 12,
        sinkhorn_iter: int = 200,
        fit_diagnostics: LowRankPermutationFitDiagnostics | None = None,
    ) -> None:
        checked_u = np.asarray(u, dtype=float)
        checked_v = np.asarray(v, dtype=float)
        if checked_u.ndim != 2 or checked_v.ndim != 2 or checked_u.shape != checked_v.shape or checked_u.shape[0] < 2:
            raise ValueError("u and v must be equal-shape (n, rank) matrices with n >= 2.")
        if not np.all(np.isfinite(checked_u)) or not np.all(np.isfinite(checked_v)):
            raise ValueError("u and v must contain only finite values.")
        self.dim = checked_u.shape[0]
        self.rank = positive_integer(checked_u.shape[1], label="rank")
        if self.rank > self.dim:
            raise ValueError("rank must not exceed the permutation dimension.")
        self.max_exact = positive_integer(max_exact, label="max_exact", minimum=2)
        if self.max_exact > 22:
            raise ValueError("max_exact must not exceed the exact permanent implementation limit of 22.")
        self.sinkhorn_iter = positive_integer(sinkhorn_iter, label="sinkhorn_iter")
        self.u = checked_u.copy()
        self.v = checked_v.copy()
        self.u.setflags(write=False)
        self.v.setflags(write=False)
        with np.errstate(over="ignore", invalid="ignore"):
            score = self.u @ self.v.T
        if not np.all(np.isfinite(score)):
            raise ValueError("u @ v.T must be finite throughout the score matrix.")
        self.s = score
        self.s.setflags(write=False)
        self.log_z = _log_normalizer(self.s, self.max_exact, self.sinkhorn_iter)
        self.computation_diagnostics = LowRankPermutationComputationDiagnostics()
        self.name = name
        self.keys = keys
        if fit_diagnostics is not None and not isinstance(fit_diagnostics, LowRankPermutationFitDiagnostics):
            raise TypeError("fit_diagnostics must be a LowRankPermutationFitDiagnostics record.")
        self.fit_diagnostics = fit_diagnostics

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
        checked = permutation(x, self.dim, label="low-rank permutation ordering")
        return float(self.s[checked, np.arange(self.dim)].sum() - self.log_z)

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for encoded orderings."""
        checked = permutation_batch(x, self.dim, label="low-rank permutation orderings")
        ranks = np.arange(self.dim)
        return self.s[checked, ranks[None, :]].sum(axis=1) - self.log_z

    def marginals(self) -> np.ndarray:
        """Return exact model marginals ``P[item, rank]``."""
        return _exact_marginals(self.s, self.log_z)

    def sinkhorn_relaxation(self) -> np.ndarray:
        """Return an explicitly approximate doubly-stochastic transport relaxation."""
        plan, _ = sinkhorn_bethe(np.array(self.s, order="C", copy=True), self.sinkhorn_iter)
        return plan

    def sampler(self, seed: int | None = None) -> LowRankPermutationSampler:
        """Return an exact conditional-permanent sampler."""
        return LowRankPermutationSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> LowRankPermutationEstimator:
        """Return an exact-marginal estimator with this dimension and rank."""
        return LowRankPermutationEstimator(
            dim=self.dim,
            rank=self.rank,
            max_exact=self.max_exact,
            sinkhorn_iter=self.sinkhorn_iter,
            pseudo_count=pseudo_count,
            prior_marginals=self.marginals(),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> LowRankPermutationDataEncoder:
        """Return the full-ranking encoder used by vectorized methods."""
        return LowRankPermutationDataEncoder(dim=self.dim)


class LowRankPermutationSampler(DistributionSampler):
    """Draw iid orderings from exact conditional permanent ratios."""

    def __init__(self, dist: LowRankPermutationDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def _sample_one(self) -> list[int]:
        available = list(range(self.dist.dim))
        ordering: list[int] = []
        for rank in range(self.dist.dim):
            remaining_ranks = list(range(rank + 1, self.dist.dim))
            log_probabilities = np.empty(len(available), dtype=np.float64)
            for index, item in enumerate(available):
                remaining_items = [candidate for candidate in available if candidate != item]
                minor = self.dist.s[np.ix_(remaining_items, remaining_ranks)]
                log_probabilities[index] = self.dist.s[item, rank] + log_matrix_permanent(minor)
            shift = float(np.max(log_probabilities))
            probabilities = np.exp(log_probabilities - shift)
            probabilities /= probabilities.sum()
            selected = int(self.rng.choice(len(available), p=probabilities))
            ordering.append(available.pop(selected))
        return ordering

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[int] | list[list[int]]:
        """Draw one ordering or ``size`` exact iid orderings."""
        checked_size = sample_size(size)
        if checked_size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(checked_size)]


class LowRankPermutationAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the item-by-rank count matrix ``C[item, rank]`` -- the first-order sufficient statistic."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.counts = np.zeros((self.dim, self.dim))
        self.count = 0.0
        self.keys = keys

    def update(self, x: Sequence[int], weight: float, estimate: Any) -> None:
        """Update item-by-rank counts from one weighted ordering."""
        checked = permutation(x, self.dim, label="low-rank permutation ordering")
        self.seq_update(checked[None, :], np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize item-by-rank counts from one weighted ordering."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Update item-by-rank counts from encoded orderings."""
        checked = permutation_batch(x, self.dim, label="low-rank permutation orderings")
        checked_weights = validate_weights(weights, len(checked))
        ranks = np.arange(self.dim)
        for row, weight in zip(checked, checked_weights):
            self.counts[row, ranks] += weight
        self.count += float(np.sum(checked_weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize item-by-rank counts from encoded orderings."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat) -> LowRankPermutationAccumulator:
        """Merge observation weight and item-by-rank count statistics."""
        count, counts = assignment_statistics(suff_stat, self.dim, label="low-rank permutation statistics")
        self.count += count
        self.counts += counts
        return self

    def value(self):
        """Return accumulated observation weight and item-by-rank counts."""
        return self.count, self.counts.copy()

    def from_value(self, x) -> LowRankPermutationAccumulator:
        """Restore accumulator state from ``value`` output."""
        self.count, self.counts = assignment_statistics(x, self.dim, label="low-rank permutation statistics")
        return self

    def acc_to_encoder(self) -> LowRankPermutationDataEncoder:
        """Return the encoder compatible with item-by-rank sufficient statistics."""
        return LowRankPermutationDataEncoder(dim=self.dim)


class LowRankPermutationAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for low-rank permutation sufficient statistics."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
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
        tol: float = 1.0e-6,
        pseudo_count: float | None = None,
        prior_marginals: np.ndarray | None = None,
        require_convergence: bool = True,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.rank = positive_integer(rank, label="rank")
        if self.rank > self.dim:
            raise ValueError("rank must not exceed dim.")
        self.max_exact = positive_integer(max_exact, label="max_exact", minimum=2)
        if self.max_exact > 22:
            raise ValueError("max_exact must not exceed the exact permanent implementation limit of 22.")
        if self.dim > self.max_exact:
            raise ValueError("dim must not exceed max_exact for normalized low-rank permutation fitting.")
        self.sinkhorn_iter = positive_integer(sinkhorn_iter, label="sinkhorn_iter")
        self.max_iter = positive_integer(max_iter, label="max_iter")
        self.lr = finite_positive(lr, label="lr")
        self.tol = finite_positive(tol, label="tol")
        self.pseudo_count = None if pseudo_count is None else finite_nonnegative(pseudo_count, label="pseudo_count")
        if prior_marginals is None:
            prior = np.full((self.dim, self.dim), 1.0 / self.dim)
        else:
            prior = np.asarray(prior_marginals, dtype=np.float64)
            if (
                prior.shape != (self.dim, self.dim)
                or not np.all(np.isfinite(prior))
                or np.any(prior < 0.0)
                or not np.allclose(prior.sum(axis=0), 1.0)
                or not np.allclose(prior.sum(axis=1), 1.0)
            ):
                raise ValueError("prior_marginals must be a finite nonnegative doubly-stochastic matrix.")
            prior = prior.copy()
        self.prior_marginals = prior
        if not isinstance(require_convergence, (bool, np.bool_)):
            raise TypeError("require_convergence must be a Boolean.")
        self.require_convergence = require_exact_bool(require_convergence, "require_convergence")
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> LowRankPermutationAccumulatorFactory:
        """Return a factory for low-rank permutation sufficient-statistic accumulators."""
        return LowRankPermutationAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat) -> LowRankPermutationDistribution:
        """Estimate low-rank score factors from item-by-rank marginal counts."""
        if nobs is not None:
            finite_nonnegative(nobs, label="nobs")
        count, counts = assignment_statistics(suff_stat, self.dim, label="low-rank permutation statistics")
        n, r = self.dim, self.rank
        kw = dict(max_exact=self.max_exact, sinkhorn_iter=self.sinkhorn_iter, name=self.name, keys=self.keys)
        pseudo_count = 0.0 if self.pseudo_count is None else self.pseudo_count
        if pseudo_count > 0.0:
            counts += pseudo_count * self.prior_marginals
            count += pseudo_count
        if count <= 0.0:
            diagnostics = LowRankPermutationFitDiagnostics(
                converged=True,
                iterations=0,
                gradient_norm=0.0,
                regularized=False,
                pseudo_count=0.0,
            )
            return LowRankPermutationDistribution(
                np.zeros((n, r)),
                np.zeros((n, r)),
                fit_diagnostics=diagnostics,
                **kw,
            )
        m = counts / count  # empirical item-by-rank marginals (doubly stochastic)
        # low-rank init from the centered log-marginals
        log_m = np.log(np.clip(m, 1e-6, None))
        log_m -= log_m.mean(axis=1, keepdims=True) + log_m.mean(axis=0, keepdims=True) - log_m.mean()
        uu, ss, vt = np.linalg.svd(log_m)
        u = uu[:, :r] * np.sqrt(ss[:r])[None, :]
        v = (vt[:r, :].T) * np.sqrt(ss[:r])[None, :]
        converged = False
        gradient_norm = math.inf
        iterations = 0
        for iteration in range(self.max_iter):
            p = _exact_marginals(u @ v.T)
            score_gradient = m - p
            gradient_u = score_gradient @ v
            gradient_v = score_gradient.T @ u
            gradient_norm = float(max(np.max(np.abs(gradient_u)), np.max(np.abs(gradient_v))))
            iterations = iteration + 1
            if gradient_norm < self.tol:
                converged = True
                break
            old_u, old_v = u, v
            u = old_u + self.lr * gradient_u
            v = old_v + self.lr * gradient_v
            if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
                raise FloatingPointError("low-rank permutation fitting produced non-finite factors.")
        if self.require_convergence and not converged:
            raise RuntimeError(
                "Low-rank permutation fitting did not converge in "
                f"{iterations} iterations (gradient norm {gradient_norm:.6g}, tolerance {self.tol:.6g})."
            )
        diagnostics = LowRankPermutationFitDiagnostics(
            converged=converged,
            iterations=iterations,
            gradient_norm=gradient_norm,
            regularized=pseudo_count > 0.0,
            pseudo_count=pseudo_count,
        )
        return LowRankPermutationDistribution(u, v, fit_diagnostics=diagnostics, **kw)


class LowRankPermutationDataEncoder(DataSequenceEncoder):
    """Encode a sequence of orderings (permutations of 0,...,n-1) into an (N, n) integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = None if dim is None else positive_integer(dim, label="dim", minimum=2)

    def __str__(self) -> str:
        return "LowRankPermutationDataEncoder(dim=%s)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LowRankPermutationDataEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        """Validate and encode full orderings as a dense integer matrix."""
        raw = np.asarray([list(row) for row in x])
        if self.dim is None:
            if raw.ndim != 2 or raw.shape[0] == 0:
                raise ValueError("LowRankPermutationDistribution requires a non-empty sequence of orderings.")
            return permutation_batch(raw, raw.shape[1], label="low-rank permutation orderings", allow_empty=False)
        return permutation_batch(
            raw,
            self.dim,
            label="low-rank permutation orderings",
            allow_empty=False,
        )


__all__ = [
    "LowRankPermutationDistribution",
    "LowRankPermutationSampler",
    "LowRankPermutationAccumulator",
    "LowRankPermutationAccumulatorFactory",
    "LowRankPermutationEstimator",
    "LowRankPermutationDataEncoder",
    "LowRankPermutationFitDiagnostics",
    "LowRankPermutationComputationDiagnostics",
]
