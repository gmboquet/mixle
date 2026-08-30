"""Spearman-rho distribution over full item orderings.

Observations use the ranking convention shared by the other ranking families:
``x[rank] = item``. The location parameter is a (possibly fractional) mean
rank vector, ``sigma[item] = expected rank``. Use :class:`ItemOrdering` and
:class:`RankVector` when a boundary needs to make that distinction explicit.

For an ordering ``x``, let ``r`` be its inverse rank vector. Then

``p(x) = exp(-rho * sum_item (r[item] - sigma[item])**2) / Z``.

The normalizer is the permanent of the rank-by-item log-weight matrix. It is
computed by exact subset dynamic programming under an explicit dimension
budget; no factorial permutation array is materialized or globally cached.
Sampling uses exact conditional permanents and enumeration uses lazy k-best
assignment search.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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
    finite_nonnegative,
    finite_positive,
    permutation,
    permutation_batch,
    positive_integer,
    sample_size,
)
from mixle.stats.rankings._contracts import weights as validate_weights
from mixle.stats.rankings._permutation_kernels import log_matrix_permanent
from mixle.stats.rankings.representations import ItemOrdering, RankVector
from mixle.utils.vector import owned_backend_parameter

_DEFAULT_MAX_DIM = 10
_HARD_EXACT_DIM = 22


@dataclass(frozen=True)
class SpearmanRankingFitDiagnostics:
    """Concentration-fit and regularization evidence attached to an estimate."""

    concentration_algorithm: str
    converged: bool
    iterations: int
    boundary_solution: bool
    regularized: bool
    pseudo_count: float

    # Estimation provenance rides on fitted distributions as a constructor argument, so it cannot
    # be re-derived from parameters and must round-trip through the closed JSON registry. Not a
    # distribution or estimator, so it opts in explicitly -- the same mechanism as other
    # serializable value classes discovered by the mixle.stats registry walk. Unannotated on
    # purpose: an annotated name would become a dataclass field.
    __pysp_serializable__ = True


def _validate_location(value: Any, *, already_centered: bool = False) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim != 1 or raw.size < 2 or not np.all(np.isfinite(raw)):
        raise ValueError("sigma must be a finite one-dimensional vector with at least two entries.")
    dim = len(raw)
    if already_centered:
        # __pysp_setstate__ (SpearmanRankingDistribution, below) restores a `sigma` that was
        # already canonicalized once, at the original object's construction time, before
        # serialization -- shifting a float64 array by its own (near-)mean a second time is not
        # exactly idempotent (the residual left over from the first canonicalization is
        # generically a tiny nonzero value, not exactly 0.0), so redoing it here could shift one
        # or more elements by 1-few ULP relative to the state being restored. That is invisible
        # to log_density/sampling but flips the raw bytes mixle.data.hashing.model_hash
        # fingerprints. Restoring state means reproducing it exactly, not reapplying a
        # transformation that was already applied -- this is for __pysp_setstate__'s exclusive
        # use, never for ordinary construction. The permutahedron check below still runs, since
        # it only reads `sigma` rather than re-deriving it.
        sigma = np.array(raw, dtype=np.float64, copy=True)
    else:
        # A common shift changes every assignment score by the same constant. Canonicalizing the
        # location sum removes that non-identifiability before checking the permutahedron
        # constraints.
        sigma = np.array(raw - raw.mean() + (dim - 1) / 2.0, dtype=np.float64, copy=True)
    ordered = np.sort(sigma)
    tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(sigma))))
    for count in range(1, dim):
        minimum = count * (count - 1) / 2.0
        if float(np.sum(ordered[:count])) < minimum - tolerance:
            raise ValueError("sigma must be a compatible mean rank vector in the permutation permutahedron.")
    sigma.setflags(write=False)
    return sigma


def _rank_vectors(orderings: np.ndarray) -> np.ndarray:
    """Invert each ``x[rank] = item`` row to ``rank[item] = rank``."""
    return np.argsort(orderings, axis=1).astype(np.int64, copy=False)


def _log_weights(sigma: np.ndarray, rho: float) -> np.ndarray:
    ranks = np.arange(len(sigma), dtype=np.float64)
    return -rho * (ranks[:, None] - sigma[None, :]) ** 2


def _log_partition_and_expected_distance(sigma: np.ndarray, rho: float) -> tuple[float, float]:
    """Return exact log partition and expected squared rank distance in O(n 2**n)."""
    costs = (np.arange(len(sigma), dtype=np.float64)[:, None] - sigma[None, :]) ** 2
    log_weight = -rho * costs
    dim = len(sigma)
    states = 1 << dim
    log_partition = np.full(states, -np.inf, dtype=np.float64)
    expected = np.zeros(states, dtype=np.float64)
    log_partition[0] = 0.0
    for mask in range(1, states):
        rank = mask.bit_count() - 1
        combined_log = -np.inf
        combined_expected = 0.0
        for item in range(dim):
            bit = 1 << item
            if not mask & bit:
                continue
            previous = mask ^ bit
            candidate_log = log_partition[previous] + log_weight[rank, item]
            candidate_expected = expected[previous] + costs[rank, item]
            if combined_log == -np.inf:
                combined_log = candidate_log
                combined_expected = candidate_expected
            elif candidate_log > combined_log:
                old_weight = math.exp(combined_log - candidate_log)
                combined_expected = (candidate_expected + old_weight * combined_expected) / (1.0 + old_weight)
                combined_log = candidate_log + math.log1p(old_weight)
            else:
                new_weight = math.exp(candidate_log - combined_log)
                combined_expected = (combined_expected + new_weight * candidate_expected) / (1.0 + new_weight)
                combined_log += math.log1p(new_weight)
        log_partition[mask] = combined_log
        expected[mask] = combined_expected
    return float(log_partition[-1]), float(expected[-1])


def _estimate_rho_from_mean_distance(
    sigma: np.ndarray,
    mean_distance: float,
    max_rho: float,
    tolerance: float = 1.0e-10,
    max_iter: int = 100,
) -> tuple[float, int, bool]:
    """Solve ``E_rho[D] = mean_distance`` by monotone bisection."""
    _, uniform_mean = _log_partition_and_expected_distance(sigma, 0.0)
    if mean_distance >= uniform_mean - tolerance:
        return 0.0, 0, True
    if mean_distance <= tolerance:
        return max_rho, 0, True
    lo = 0.0
    hi = 1.0
    while hi < max_rho:
        _, expected = _log_partition_and_expected_distance(sigma, hi)
        if expected <= mean_distance:
            break
        lo = hi
        hi = min(max_rho, 2.0 * hi)
    _, at_cap = _log_partition_and_expected_distance(sigma, hi)
    if hi == max_rho and at_cap > mean_distance:
        return max_rho, 0, True
    iterations = 0
    for iterations in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        _, expected = _log_partition_and_expected_distance(sigma, mid)
        if expected > mean_distance:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tolerance * max(1.0, hi):
            break
    return 0.5 * (lo + hi), iterations, False


def _spearman_statistics(value: Any, dim: int) -> tuple[float, np.ndarray]:
    if isinstance(value, (str, bytes)):
        raise TypeError("Spearman statistics must be a two-item tuple.")
    try:
        if len(value) != 2:
            raise TypeError("Spearman statistics must be a two-item tuple.")
    except TypeError as exc:
        raise TypeError("Spearman statistics must be a two-item tuple.") from exc
    count = finite_nonnegative(value[0], label="Spearman statistic observation weight")
    rank_sum = np.asarray(value[1], dtype=np.float64)
    if rank_sum.shape != (dim,) or not np.all(np.isfinite(rank_sum)) or np.any(rank_sum < 0.0):
        raise ValueError(f"Spearman rank sum must be a finite nonnegative vector of length {dim}.")
    tolerance = 1.0e-10 * max(1.0, count)
    if np.any(rank_sum > count * (dim - 1) + tolerance):
        raise ValueError("Spearman rank-sum components exceed the support bound.")
    expected_total = count * dim * (dim - 1) / 2.0
    if not math.isclose(float(rank_sum.sum()), expected_total, rel_tol=1.0e-10, abs_tol=tolerance):
        raise ValueError("Spearman rank-sum total is incompatible with the observation weight.")
    ordered = np.sort(rank_sum)
    for prefix in range(1, dim):
        minimum = count * prefix * (prefix - 1) / 2.0
        if float(np.sum(ordered[:prefix])) < minimum - tolerance:
            raise ValueError("Spearman rank sum lies outside the scaled permutation permutahedron.")
    return count, rank_sum.copy()


def _backend_rank_vectors(x: Any, dim: int, engine: Any) -> Any:
    xx = engine.asarray(x)
    items = engine.arange(dim)
    ranks = engine.arange(dim)
    matches = xx[..., :, None] == items
    return engine.sum(matches * ranks[None, :, None], axis=-2)


class SpearmanRankingDistribution(SequenceEncodableProbabilityDistribution):
    """Exact finite Spearman law over item orderings under a dimension budget."""

    @classmethod
    def compute_capabilities(cls):
        """Declare generated NumPy and Torch scoring support."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="spearman_ranking",
            distribution_type=cls,
            parameters=(
                ParameterSpec("sigma", constraint="real_vector"),
                ParameterSpec("rho"),
                ParameterSpec("log_const", constraint="real", differentiable=False),
            ),
            statistics=(StatisticSpec("count"), StatisticSpec("sum", kind="vector_moment")),
            support="permutation",
            differentiable=False,
            legacy_sufficient_statistics=cls.backend_legacy_sufficient_statistics,
        )

    @staticmethod
    def backend_legacy_sufficient_statistics(x: Any, params: dict[str, Any], engine: Any) -> tuple[Any, ...]:
        """Return row-wise count and item-indexed rank-vector statistics."""
        xx = engine.asarray(x)
        ranks = _backend_rank_vectors(xx, int(params["sigma"].shape[-1]), engine)
        one = engine.sum(ranks * 0.0, axis=1) + engine.asarray(1.0)
        return one, ranks

    @staticmethod
    def backend_log_density_from_params(x: Any, sigma: Any, rho: Any, log_const: Any, engine: Any) -> Any:
        """Score item orderings after explicit conversion to rank vectors.

        Callers arrive under two shape conventions. ``backend_seq_log_density`` passes the natural
        ones -- ranks ``(n, d)`` against a single ``sigma`` ``(d,)``. The generated scorer in
        :mod:`mixle.stats.compute.declarations` pre-aligns both sides instead: data picks up a
        component axis (``(n, 1, d)``) and every parameter a leading row axis (``(1, k, d)``), so the
        two already broadcast. Inserting a component axis unconditionally, as this did, double-counts
        it on that path and returns an ``(n, 1, k)`` score where the contract is ``(n, k)``. Only add
        the axis when the ranks really are still two-dimensional.
        """
        ranks = _backend_rank_vectors(x, int(sigma.shape[-1]), engine)
        if len(sigma.shape) > 1 and len(ranks.shape) == 2:
            ranks = ranks[:, None, :]
        diff = ranks - sigma
        return -rho * engine.sum(diff * diff, axis=-1) - log_const

    def __init__(
        self,
        sigma: Sequence[float] | np.ndarray,
        rho: float = 1.0,
        name: str | None = None,
        keys: str | None = None,
        max_dim: int = _DEFAULT_MAX_DIM,
        fit_diagnostics: SpearmanRankingFitDiagnostics | None = None,
        *,
        _sigma_already_centered: bool = False,
    ) -> None:
        self._sigma = _validate_location(sigma, already_centered=_sigma_already_centered)
        self.dim = len(self._sigma)
        self.rho = finite_nonnegative(rho, label="rho")
        self.max_dim = positive_integer(max_dim, label="max_dim", minimum=2)
        if self.dim > self.max_dim:
            raise ValueError(
                f"SpearmanRankingDistribution dim={self.dim} exceeds max_dim={self.max_dim} "
                "(exact subset permanent budget)."
            )
        if self.dim > _HARD_EXACT_DIM:
            raise ValueError(f"exact Spearman permanent evaluation is limited to dimension {_HARD_EXACT_DIM}.")
        self.log_weights = _log_weights(self._sigma, self.rho)
        self.log_weights.setflags(write=False)
        self.log_const, _ = _log_partition_and_expected_distance(self._sigma, self.rho)
        if fit_diagnostics is not None and not isinstance(fit_diagnostics, SpearmanRankingFitDiagnostics):
            raise TypeError("fit_diagnostics must be a SpearmanRankingFitDiagnostics record.")
        self.fit_diagnostics = fit_diagnostics
        self.name = name
        self.keys = keys

    @property
    def sigma(self) -> np.ndarray:
        """Return an ownership-safe copy of the canonical mean rank vector."""
        return self._sigma.copy()

    def __pysp_getstate__(self) -> dict[str, Any]:
        """Return the constructor-owned state used by the safe JSON codec.

        The raw ``__dict__`` stores the location under ``_sigma`` while the constructor takes
        ``sigma``, so the generic constructor-validated decoder encoded this family cleanly and
        then refused every read back (a write-only artifact). ``log_weights``, ``log_const``, and
        ``dim`` are re-derived deterministically in ``__init__`` and must not be serialized;
        ``fit_diagnostics`` is constructor-carried estimation provenance and rides along as is.
        """
        return {
            "sigma": self._sigma,
            "rho": self.rho,
            "name": self.name,
            "keys": self.keys,
            "max_dim": self.max_dim,
            "fit_diagnostics": self.fit_diagnostics,
        }

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild from constructor-owned state, re-deriving the partition tables.

        ``state["sigma"]`` was already canonicalized once, in ``__init__`` at the original
        object's construction time, before ``__pysp_getstate__`` serialized it verbatim -- so
        this must restore that exact array rather than re-canonicalizing it (see
        ``_sigma_already_centered`` on ``__init__`` / ``_validate_location``). A second pass is
        not bit-exact idempotent, and the mismatch is exactly the kind of silent divergence
        :func:`mixle.data.hashing.model_hash` is meant to catch, not produce -- see the identical
        fix on :class:`mixle.stats.rankings.thurstone.ThurstoneDistribution`.
        """
        required = {"sigma", "rho", "name", "keys", "max_dim"}
        missing = required - set(state)
        if missing:
            raise ValueError("SpearmanRankingDistribution state is missing %s" % ", ".join(sorted(missing)))
        self.__init__(
            state["sigma"],
            rho=state["rho"],
            name=state["name"],
            keys=state["keys"],
            max_dim=state["max_dim"],
            fit_diagnostics=state.get("fit_diagnostics"),
            _sigma_already_centered=True,
        )

    def __str__(self) -> str:
        return "SpearmanRankingDistribution(sigma=%s, rho=%s, name=%s, keys=%s, max_dim=%s)" % (
            repr([float(value) for value in self.sigma]),
            repr(self.rho),
            repr(self.name),
            repr(self.keys),
            repr(self.max_dim),
        )

    def density(self, x: Sequence[int] | ItemOrdering) -> float:
        """Return the probability of an item ordering."""
        return float(math.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int] | ItemOrdering) -> float:
        """Return the log-probability of one ``x[rank] = item`` ordering."""
        ordering = permutation(x, self.dim, label="Spearman item ordering")
        ranks = np.argsort(ordering)
        difference = ranks - self._sigma
        return -self.rho * float(np.dot(difference, difference)) - self.log_const

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for item orderings."""
        checked = permutation_batch(x, self.dim, label="Spearman item orderings")
        difference = _rank_vectors(checked) - self._sigma
        return -self.rho * np.sum(difference * difference, axis=1) - self.log_const

    def backend_seq_log_density(self, x: np.ndarray, engine: Any) -> Any:
        """Engine-neutral vectorized scoring for encoded item orderings."""
        return self.backend_log_density_from_params(
            engine.asarray(x),
            engine.asarray(owned_backend_parameter(self._sigma)),
            engine.asarray(self.rho),
            engine.asarray(self.log_const),
            engine,
        )

    @classmethod
    def backend_stacked_params(cls, dists: Sequence[SpearmanRankingDistribution], engine: Any) -> dict[str, Any]:
        """Stack equal-dimensional Spearman parameters."""
        dim = int(dists[0].dim)
        if any(int(dist.dim) != dim for dist in dists):
            raise ValueError("stacked Spearman components require equal dimensions.")
        return {
            "__pysp_component_axis__": {"sigma": 0, "rho": 0, "log_const": 0},
            "sigma": engine.asarray([dist.sigma for dist in dists]),
            "rho": engine.asarray([dist.rho for dist in dists]),
            "log_const": engine.asarray([dist.log_const for dist in dists]),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: np.ndarray, params: dict[str, Any], engine: Any) -> Any:
        """Return an observation-by-component score matrix."""
        ranks = _backend_rank_vectors(x, len(params["sigma"][0]), engine)
        difference = ranks[:, None, :] - params["sigma"][None, :, :]
        return -params["rho"][None, :] * engine.sum(difference * difference, axis=2) - params["log_const"][None, :]

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: np.ndarray, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any]:
        """Return component-stacked item-indexed rank-vector statistics."""
        weights_array = engine.asarray(weights)
        ranks = _backend_rank_vectors(x, len(params["sigma"][0]), engine)
        ranks = engine.asarray(ranks, dtype=getattr(weights_array, "dtype", None))
        return engine.sum(weights_array, axis=0), engine.matmul(weights_array.T, ranks)

    def sampler(self, seed: int | None = None) -> SpearmanRankingSampler:
        """Return an exact conditional-permanent sampler."""
        return SpearmanRankingSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> SpearmanRankingEstimator:
        """Return an estimator preserving the dimension and resource budget."""
        return SpearmanRankingEstimator(
            self.dim,
            rho=None,
            pseudo_count=pseudo_count,
            name=self.name,
            keys=self.keys,
            max_dim=self.max_dim,
        )

    def dist_to_encoder(self) -> SpearmanRankingDataEncoder:
        """Return the item-ordering encoder."""
        return SpearmanRankingDataEncoder(dim=self.dim)

    def enumerator(self) -> SpearmanRankingEnumerator:
        """Return a lazy descending-probability enumerator."""
        return SpearmanRankingEnumerator(self)

    def support_size(self) -> int:
        """Return the number of full item orderings."""
        return math.factorial(self.dim)


class SpearmanRankingEnumerator(DistributionEnumerator):
    """Enumerate item orderings by increasing rank-to-item assignment cost."""

    def __init__(self, dist: SpearmanRankingDistribution) -> None:
        super().__init__(dist)
        self._generator = k_best_assignments(-dist.log_weights)

    def __next__(self) -> tuple[list[int], float]:
        total, rows, items = next(self._generator)
        ordering = [0] * self.dist.dim
        for rank, item in zip(rows, items):
            ordering[int(rank)] = int(item)
        return ordering, float(-total - self.dist.log_const)


class SpearmanRankingSampler(DistributionSampler):
    """Draw exact Spearman item orderings using conditional permanents."""

    def __init__(self, dist: SpearmanRankingDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _sample_one(self) -> list[int]:
        available = list(range(self.dist.dim))
        ordering = [0] * self.dist.dim
        for rank in range(self.dist.dim):
            remaining_ranks = list(range(rank + 1, self.dist.dim))
            log_probabilities = np.empty(len(available), dtype=np.float64)
            for index, item in enumerate(available):
                remaining_items = [candidate for candidate in available if candidate != item]
                minor = self.dist.log_weights[np.ix_(remaining_ranks, remaining_items)]
                log_probabilities[index] = self.dist.log_weights[rank, item] + log_matrix_permanent(minor)
            shift = float(np.max(log_probabilities))
            probabilities = np.exp(log_probabilities - shift)
            probabilities /= probabilities.sum()
            choice = int(self.rng.choice(len(available), p=probabilities))
            ordering[rank] = available.pop(choice)
        return ordering

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[int] | list[list[int]]:
        """Draw one item ordering or ``size`` iid item orderings."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(sample_size(size))]


class SpearmanRankingAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate item-indexed rank-vector sums from item orderings."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.count = 0.0
        self.keys = keys
        self.name = name

    def update(
        self,
        x: Sequence[int] | ItemOrdering,
        weight: float,
        estimate: SpearmanRankingDistribution | None,
    ) -> None:
        """Update from one item ordering."""
        ordering = permutation(x, self.dim, label="Spearman item ordering")
        self.seq_update(ordering[None, :], np.asarray([weight], dtype=np.float64), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize from one item ordering."""
        self.update(x, weight, None)

    def seq_update(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        estimate: SpearmanRankingDistribution | None,
    ) -> None:
        """Update from encoded item orderings."""
        checked = permutation_batch(x, self.dim, label="Spearman item orderings")
        checked_weights = validate_weights(weights, len(checked))
        self.sum += np.dot(_rank_vectors(checked).T, checked_weights)
        self.count += float(np.sum(checked_weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize from encoded item orderings."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray]) -> SpearmanRankingAccumulator:
        """Merge validated rank-vector statistics."""
        count, rank_sum = _spearman_statistics(suff_stat, self.dim)
        self.count += count
        self.sum += rank_sum
        return self

    def value(self) -> tuple[float, np.ndarray]:
        """Return an ownership-safe sufficient-statistic snapshot."""
        return self.count, self.sum.copy()

    def from_value(self, x: tuple[float, np.ndarray]) -> SpearmanRankingAccumulator:
        """Restore validated rank-vector statistics."""
        self.count, self.sum = _spearman_statistics(x, self.dim)
        return self

    def acc_to_encoder(self) -> SpearmanRankingDataEncoder:
        """Return an encoder with the matching item count."""
        return SpearmanRankingDataEncoder(dim=self.dim)


class SpearmanRankingAccumulatorFactory(StatisticAccumulatorFactory):
    """Create Spearman ranking accumulators."""

    def __init__(self, dim: int, name: str | None = None, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.keys = keys
        self.name = name

    def make(self) -> SpearmanRankingAccumulator:
        """Return a fresh accumulator."""
        return SpearmanRankingAccumulator(dim=self.dim, name=self.name, keys=self.keys)


class SpearmanRankingEstimator(ParameterEstimator):
    """Estimate consensus mean ranks and nonnegative Spearman concentration."""

    def __init__(
        self,
        dim: int,
        rho: float | None = None,
        pseudo_count: float | None = None,
        suff_stat: tuple[float, np.ndarray] | None = None,
        name: str | None = None,
        keys: str | None = None,
        max_rho: float = 1.0e6,
        max_dim: int = _DEFAULT_MAX_DIM,
    ) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.max_dim = positive_integer(max_dim, label="max_dim", minimum=2)
        if self.dim > self.max_dim:
            raise ValueError(
                f"SpearmanRankingEstimator dim={self.dim} exceeds max_dim={self.max_dim} "
                "(exact subset permanent budget)."
            )
        if self.dim > _HARD_EXACT_DIM:
            raise ValueError(f"exact Spearman permanent evaluation is limited to dimension {_HARD_EXACT_DIM}.")
        self.rho = None if rho is None else finite_nonnegative(rho, label="rho")
        self.max_rho = finite_positive(max_rho, label="max_rho")
        self.pseudo_count = None if pseudo_count is None else finite_nonnegative(pseudo_count, label="pseudo_count")
        self.suff_stat = None if suff_stat is None else _spearman_statistics(suff_stat, self.dim)
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> SpearmanRankingAccumulatorFactory:
        """Return a factory for item-ordering sufficient statistics."""
        return SpearmanRankingAccumulatorFactory(self.dim, self.name, self.keys)

    def estimate(
        self,
        nobs: float | None,
        suff_stat: tuple[float, np.ndarray],
    ) -> SpearmanRankingDistribution:
        """Estimate a distribution from validated item-indexed rank sums."""
        count, rank_sum = _spearman_statistics(suff_stat, self.dim)
        if nobs is not None:
            checked_nobs = finite_nonnegative(nobs, label="nobs")
            if not math.isclose(checked_nobs, count, rel_tol=1.0e-10, abs_tol=1.0e-10):
                raise ValueError("nobs must equal Spearman statistic observation weight.")
        pseudo_count = 0.0 if self.pseudo_count is None else self.pseudo_count
        if pseudo_count > 0.0:
            if self.suff_stat is None:
                prior_count = 1.0
                prior_sum = np.full(self.dim, (self.dim - 1) / 2.0)
            else:
                prior_count, prior_sum = self.suff_stat
            count += pseudo_count * prior_count
            rank_sum = rank_sum + pseudo_count * prior_sum
        if count <= 0.0:
            sigma = np.arange(self.dim, dtype=np.float64)
            rho = 0.0
            iterations = 0
            boundary = True
        else:
            sigma = np.argsort(np.argsort(rank_sum, kind="stable"), kind="stable").astype(np.float64)
            if self.rho is None:
                rank_norm = float(np.dot(np.arange(self.dim), np.arange(self.dim)))
                total_distance = 2.0 * count * rank_norm - 2.0 * float(np.dot(rank_sum, sigma))
                mean_distance = max(0.0, total_distance / count)
                rho, iterations, boundary = _estimate_rho_from_mean_distance(
                    sigma,
                    mean_distance,
                    self.max_rho,
                )
            else:
                rho = self.rho
                iterations = 0
                boundary = False
        diagnostics = SpearmanRankingFitDiagnostics(
            concentration_algorithm="exact_assignment_moment_bisection" if self.rho is None else "fixed",
            converged=True,
            iterations=iterations,
            boundary_solution=boundary,
            regularized=pseudo_count > 0.0,
            pseudo_count=pseudo_count,
        )
        return SpearmanRankingDistribution(
            sigma,
            rho,
            name=self.name,
            keys=self.keys,
            max_dim=self.max_dim,
            fit_diagnostics=diagnostics,
        )


class SpearmanRankingDataEncoder(DataSequenceEncoder):
    """Validate and encode full ``x[rank] = item`` orderings."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = None if dim is None else positive_integer(dim, label="dim", minimum=2)

    def __str__(self) -> str:
        return "SpearmanRankingDataEncoder(dim=%s)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SpearmanRankingDataEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[Sequence[int] | ItemOrdering]) -> np.ndarray:
        """Return a dense, support-checked item-ordering matrix."""
        raw = np.asarray([list(row) for row in x])
        if self.dim is None:
            if raw.ndim != 2 or raw.shape[0] == 0:
                raise ValueError("SpearmanRankingDataEncoder requires a non-empty ordering batch.")
            return permutation_batch(raw, raw.shape[1], label="Spearman item orderings", allow_empty=False)
        return permutation_batch(raw, self.dim, label="Spearman item orderings", allow_empty=False)

    def row_count(self, x: np.ndarray) -> int:
        """Return the number of encoded item orderings."""
        return len(x)


__all__ = [
    "ItemOrdering",
    "RankVector",
    "SpearmanRankingFitDiagnostics",
    "SpearmanRankingDistribution",
    "SpearmanRankingEnumerator",
    "SpearmanRankingSampler",
    "SpearmanRankingAccumulator",
    "SpearmanRankingAccumulatorFactory",
    "SpearmanRankingEstimator",
    "SpearmanRankingDataEncoder",
]
