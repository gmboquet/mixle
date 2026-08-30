"""Thurstone ranking model as a normalized Gaussian random-utility approximation.

Each item has latent utility ``U_i ~ Normal(mu_i, 1)`` and a ranking is the
descending ordering of those utilities. Exact ranking probabilities are
Gaussian orthant probabilities and have no general closed form.

This implementation exposes an explicitly labelled, deterministic
finite-sample approximation. ``n_mc`` common random-utility draws are converted
to ranking counts and combined with positive symmetric Dirichlet smoothing.
That construction is a proper categorical distribution: probabilities are
exactly normalized, a datum's score cannot depend on its batch neighbors, and
the sampler draws from the same represented law. Every distribution records
the approximation provenance and conservative binomial error scale.

Data type: ``List[int]`` -- a full ordering of ``0..n-1`` with ``x[rank]`` the
item at that rank, best first.
"""

from __future__ import annotations

import math
from collections import Counter
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
    count_matrix_statistics,
    finite_nonnegative,
    finite_positive,
    nonnegative_integer,
    permutation,
    permutation_batch,
    positive_integer,
    sample_size,
)
from mixle.stats.rankings._contracts import weights as validate_weights

_SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class ThurstoneApproximationDiagnostics:
    """Provenance and uncertainty scale for the approximating categorical law."""

    method: str
    draws: int
    seed: int
    smoothing: float
    distinct_rankings: int
    support_size: int
    maximum_binomial_standard_error: float


@dataclass(frozen=True)
class ThurstoneFitDiagnostics:
    """Provenance for an estimated Thurstone utility vector."""

    method: str
    exact_mle: bool
    regularized: bool
    pseudo_count: float

    # Estimation provenance rides on fitted distributions as a constructor argument, so it cannot
    # be re-derived from parameters and must round-trip through the closed JSON registry. Not a
    # distribution or estimator, so it opts in explicitly -- the same mechanism as other
    # serializable value classes discovered by the mixle.stats registry walk. Unannotated on
    # purpose: an annotated name would become a dataclass field.
    __pysp_serializable__ = True


def _checked_seed(value: Any) -> int:
    result = nonnegative_integer(value, label="seed")
    if result > np.iinfo(np.uint32).max:
        raise ValueError("seed must be in [0, 2**32 - 1].")
    return result


def _thurstone_statistics(value: Any, dim: int) -> tuple[float, np.ndarray]:
    """Validate full-ranking pairwise-precedence statistics."""
    count, precede = count_matrix_statistics(
        value,
        dim,
        label="Thurstone statistics",
        entries_per_observation=dim * (dim - 1) / 2.0,
    )
    tolerance = 1.0e-10 * max(1.0, count)
    if not np.allclose(np.diag(precede), 0.0, rtol=0.0, atol=tolerance):
        raise ValueError("Thurstone precedence-count diagonal must be zero.")
    pair_totals = precede + precede.T
    off_diagonal = ~np.eye(dim, dtype=bool)
    if not np.allclose(pair_totals[off_diagonal], count, rtol=1.0e-10, atol=tolerance):
        raise ValueError("each Thurstone item pair must have total precedence weight equal to the observation weight.")
    return count, precede


class ThurstoneDistribution(SequenceEncodableProbabilityDistribution):
    """Normalized finite-sample approximation to a Case V Thurstone law."""

    @classmethod
    def compute_capabilities(cls):
        """Declare the NumPy execution path used to build the approximation."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason=(
                "The labelled common-random-number approximation stores discrete ranking counts "
                "and cannot be represented by generic tensor kernels."
            ),
        )

    def __init__(
        self,
        mu: Sequence[float] | np.ndarray,
        name: str | None = None,
        keys: str | None = None,
        n_mc: int = 4000,
        seed: int = 0,
        smoothing: float = 0.5,
        fit_diagnostics: ThurstoneFitDiagnostics | None = None,
        *,
        _mu_already_centered: bool = False,
    ) -> None:
        raw_mu = np.asarray(mu, dtype=np.float64)
        if raw_mu.ndim != 1 or raw_mu.size < 2 or not np.all(np.isfinite(raw_mu)):
            raise ValueError("mu must be a finite length-K vector with K >= 2.")
        self.dim = int(raw_mu.size)
        if _mu_already_centered:
            # __pysp_setstate__ (below) restores a `mu` that was already centered once, at the
            # original object's construction time, before serialization -- centering a float64
            # array a second time is not exactly idempotent (the residual mean left over from
            # the first centering is generically a tiny nonzero value, e.g. -4.44e-17, not
            # exactly 0.0), so re-centering here would shift one or more elements by 1-few ULP
            # relative to the state being restored. That is invisible to log_density/sample but
            # flips the raw bytes mixle.data.hashing.model_hash fingerprints, so a same-process,
            # zero-tampering deploy()+load() round trip could raise a false-positive integrity
            # warning. Restoring state means reproducing it exactly, not reapplying a
            # transformation that was already applied -- this is for __pysp_setstate__'s
            # exclusive use, never for ordinary construction.
            centered = np.array(raw_mu, dtype=np.float64, copy=True)
        else:
            centered = np.array(raw_mu - raw_mu.mean(), dtype=np.float64, copy=True)
        centered.setflags(write=False)
        self.mu = centered
        self.n_mc = positive_integer(n_mc, label="n_mc")
        self.seed = _checked_seed(seed)
        self.smoothing = finite_positive(smoothing, label="smoothing")
        if fit_diagnostics is not None and not isinstance(fit_diagnostics, ThurstoneFitDiagnostics):
            raise TypeError("fit_diagnostics must be a ThurstoneFitDiagnostics record.")
        self.fit_diagnostics = fit_diagnostics
        self.name = name
        self.keys = keys

        rng = RandomState(self.seed)
        utilities = self.mu + rng.standard_normal((self.n_mc, self.dim))
        draws = np.asarray(np.argsort(-utilities, axis=1), dtype=np.int64)
        draws.setflags(write=False)
        self._approximation_draws = draws
        self._ranking_counts = Counter(map(tuple, draws))
        self._support_size = math.factorial(self.dim)
        log_empirical_mass = math.log(self.n_mc)
        log_smoothing_mass = math.log(self.smoothing) + math.lgamma(self.dim + 1.0)
        self._log_normalizer = float(np.logaddexp(log_empirical_mass, log_smoothing_mass))
        self._empirical_mixture_probability = math.exp(log_empirical_mass - self._log_normalizer)
        self.approximation_diagnostics = ThurstoneApproximationDiagnostics(
            method="common_random_utility_counts_with_symmetric_dirichlet_smoothing",
            draws=self.n_mc,
            seed=self.seed,
            smoothing=self.smoothing,
            distinct_rankings=len(self._ranking_counts),
            support_size=self._support_size,
            maximum_binomial_standard_error=0.5 / math.sqrt(self.n_mc),
        )

    def __pysp_getstate__(self) -> dict[str, Any]:
        """Return the constructor-owned state used by the safe JSON codec.

        The approximation tables (``_approximation_draws``, ``_ranking_counts``, the normalizer
        split) and ``approximation_diagnostics`` are all re-derived deterministically in
        ``__init__`` from ``(mu, n_mc, seed, smoothing)``, so none of them is serialized: the
        diagnostics record is derived provenance rather than a parameter, and shipping it would
        require registering a class whose every field the constructor recomputes anyway.
        ``fit_diagnostics`` is the one record that cannot be re-derived -- it documents how ``mu``
        was estimated -- so it rides along as the constructor argument it is.
        """
        return {
            "mu": self.mu,
            "n_mc": self.n_mc,
            "seed": self.seed,
            "smoothing": self.smoothing,
            "name": self.name,
            "keys": self.keys,
            "fit_diagnostics": self.fit_diagnostics,
        }

    def __pysp_setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild from constructor-owned state, re-deriving the approximation tables.

        ``state["mu"]`` was already centered once, in ``__init__`` at the original object's
        construction time, before ``__pysp_getstate__`` serialized it verbatim -- so this must
        restore that exact array rather than centering it again (see ``_mu_already_centered`` on
        ``__init__``). A second centering pass is not bit-exact idempotent, and the mismatch is
        exactly the kind of silent divergence :func:`mixle.data.hashing.model_hash` is meant to
        catch, not produce.
        """
        required = {"mu", "n_mc", "seed", "smoothing", "name", "keys"}
        missing = required - set(state)
        if missing:
            raise ValueError("ThurstoneDistribution state is missing %s" % ", ".join(sorted(missing)))
        self.__init__(
            state["mu"],
            name=state["name"],
            keys=state["keys"],
            n_mc=state["n_mc"],
            seed=state["seed"],
            smoothing=state["smoothing"],
            fit_diagnostics=state.get("fit_diagnostics"),
            _mu_already_centered=True,
        )

    def __str__(self) -> str:
        return "ThurstoneDistribution(%s, n_mc=%r, seed=%r, smoothing=%r, name=%s, keys=%s)" % (
            repr([float(v) for v in self.mu]),
            self.n_mc,
            self.seed,
            self.smoothing,
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Sequence[int]) -> float:
        """Return the normalized approximating probability of a full ordering."""
        return float(math.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int]) -> float:
        """Return the normalized approximating log-probability of one ordering."""
        checked = permutation(x, self.dim, label="Thurstone ordering")
        count = self._ranking_counts.get(tuple(checked), 0)
        return math.log(count + self.smoothing) - self._log_normalizer

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return batch-invariant log-probabilities for encoded full orderings."""
        checked = permutation_batch(x, self.dim, label="Thurstone orderings")
        return np.fromiter(
            (
                math.log(self._ranking_counts.get(tuple(row), 0) + self.smoothing) - self._log_normalizer
                for row in checked
            ),
            dtype=np.float64,
            count=len(checked),
        )

    def sampler(self, seed: int | None = None) -> ThurstoneSampler:
        """Return a sampler for this exact approximating categorical law."""
        return ThurstoneSampler(self, seed)

    def support_size(self) -> int:
        """Return the number of full rankings in the smoothed support."""
        return self._support_size

    def estimator(self, pseudo_count: float | None = None) -> ThurstoneEstimator:
        """Return a labelled Thurstone-Mosteller pairwise-moment estimator."""
        return ThurstoneEstimator(
            dim=self.dim,
            n_mc=self.n_mc,
            seed=self.seed,
            smoothing=self.smoothing,
            pseudo_count=pseudo_count,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> ThurstoneDataEncoder:
        """Return the dense full-ranking encoder used by vectorized methods."""
        return ThurstoneDataEncoder(dim=self.dim)


class ThurstoneSampler(DistributionSampler):
    """Draw exactly from a distribution's empirical-plus-uniform mixture."""

    def __init__(self, dist: ThurstoneDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def _sample_one(self) -> list[int]:
        if self.rng.random_sample() < self.dist._empirical_mixture_probability:
            index = int(self.rng.randint(self.dist.n_mc))
            return [int(value) for value in self.dist._approximation_draws[index]]
        return [int(value) for value in self.rng.permutation(self.dist.dim)]

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[int] | list[list[int]]:
        """Draw one ordering or ``size`` iid orderings."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(sample_size(size))]


class ThurstoneAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate weighted pairwise-precedence counts from full rankings."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.precede = np.zeros((self.dim, self.dim))
        self.count = 0.0
        self.keys = keys

    def update(self, x: Sequence[int], weight: float, estimate: Any) -> None:
        """Update pairwise-precedence counts from one full ordering."""
        checked = permutation(x, self.dim, label="Thurstone ordering")
        self.seq_update(checked[None, :], np.asarray([weight], dtype=np.float64), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize precedence counts from one ordering."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Update pairwise-precedence counts from encoded orderings."""
        checked = permutation_batch(x, self.dim, label="Thurstone orderings")
        checked_weights = validate_weights(weights, len(checked))
        rank, later_rank = np.triu_indices(self.dim, 1)
        for row, weight in zip(checked, checked_weights):
            np.add.at(self.precede, (row[rank], row[later_rank]), weight)
        self.count += float(np.sum(checked_weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize precedence counts from a batch of encoded orderings."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray]) -> ThurstoneAccumulator:
        """Merge validated count and pairwise-precedence statistics."""
        count, precede = _thurstone_statistics(suff_stat, self.dim)
        self.count += count
        self.precede += precede
        return self

    def value(self) -> tuple[float, np.ndarray]:
        """Return an ownership-safe statistics snapshot."""
        return self.count, self.precede.copy()

    def from_value(self, x: tuple[float, np.ndarray]) -> ThurstoneAccumulator:
        """Restore validated accumulator state."""
        self.count, self.precede = _thurstone_statistics(x, self.dim)
        return self

    def acc_to_encoder(self) -> ThurstoneDataEncoder:
        """Return the ranking encoder compatible with this accumulator."""
        return ThurstoneDataEncoder(dim=self.dim)


class ThurstoneAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for Thurstone pairwise-precedence statistics."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.keys = keys

    def make(self) -> ThurstoneAccumulator:
        """Create an empty Thurstone accumulator."""
        return ThurstoneAccumulator(dim=self.dim, keys=self.keys)


class ThurstoneEstimator(ParameterEstimator):
    """Labelled pairwise-moment estimator for Case V Thurstone utilities."""

    def __init__(
        self,
        dim: int,
        n_mc: int = 4000,
        seed: int = 0,
        smoothing: float = 0.5,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.n_mc = positive_integer(n_mc, label="n_mc")
        self.seed = _checked_seed(seed)
        self.smoothing = finite_positive(smoothing, label="smoothing")
        self.pseudo_count = None if pseudo_count is None else finite_nonnegative(pseudo_count, label="pseudo_count")
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> ThurstoneAccumulatorFactory:
        """Return a factory for Thurstone sufficient-statistic accumulators."""
        return ThurstoneAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray]) -> ThurstoneDistribution:
        """Estimate centered utilities from validated pairwise-precedence statistics."""
        count, precede = _thurstone_statistics(suff_stat, self.dim)
        if nobs is not None:
            checked_nobs = finite_nonnegative(nobs, label="nobs")
            if not math.isclose(checked_nobs, count, rel_tol=1.0e-10, abs_tol=1.0e-10):
                raise ValueError("nobs must equal Thurstone statistic observation weight.")
        pseudo_count = 0.0 if self.pseudo_count is None else self.pseudo_count
        regularized = pseudo_count > 0.0
        if regularized:
            precede = precede + 0.5 * pseudo_count * (1.0 - np.eye(self.dim))
        pair_totals = precede + precede.T
        with np.errstate(invalid="ignore", divide="ignore"):
            pair_probability = np.where(pair_totals > 0.0, precede / pair_totals, 0.5)
        np.fill_diagonal(pair_probability, 0.5)
        pair_probability = np.clip(pair_probability, 1.0e-8, 1.0 - 1.0e-8)

        from scipy.special import ndtri

        pair_difference = _SQRT2 * ndtri(pair_probability)
        mu = pair_difference.mean(axis=1)
        diagnostics = ThurstoneFitDiagnostics(
            method="pairwise_moment_least_squares",
            exact_mle=False,
            regularized=regularized,
            pseudo_count=pseudo_count,
        )
        return ThurstoneDistribution(
            mu - mu.mean(),
            n_mc=self.n_mc,
            seed=self.seed,
            smoothing=self.smoothing,
            fit_diagnostics=diagnostics,
            name=self.name,
            keys=self.keys,
        )


class ThurstoneDataEncoder(DataSequenceEncoder):
    """Encode full item orderings into a dense integer matrix."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = None if dim is None else positive_integer(dim, label="dim", minimum=2)

    def __str__(self) -> str:
        return "ThurstoneDataEncoder(dim=%s)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ThurstoneDataEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        """Validate and encode full orderings as a dense integer matrix."""
        raw = np.asarray([list(row) for row in x])
        if self.dim is None:
            if raw.ndim != 2 or raw.shape[0] == 0:
                raise ValueError("ThurstoneDistribution requires a non-empty sequence of orderings.")
            return permutation_batch(raw, raw.shape[1], label="Thurstone orderings", allow_empty=False)
        return permutation_batch(raw, self.dim, label="Thurstone orderings", allow_empty=False)

    def row_count(self, x: np.ndarray) -> int:
        """Return the number of encoded ranking rows."""
        return len(x)


__all__ = [
    "ThurstoneApproximationDiagnostics",
    "ThurstoneFitDiagnostics",
    "ThurstoneDistribution",
    "ThurstoneSampler",
    "ThurstoneAccumulator",
    "ThurstoneAccumulatorFactory",
    "ThurstoneEstimator",
    "ThurstoneDataEncoder",
]
