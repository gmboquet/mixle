"""Mallows ranking distributions over permutations using Kendall tau distance.

Data type: List[int] (a full ranking/ordering of n items, given as a permutation of 0,...,n-1 where
``x[r]`` is the item placed at rank r, best first).

The Mallows model concentrates probability around a central permutation ``sigma0`` with a dispersion
``theta >= 0``:

    p(sigma) = exp(-theta * d(sigma, sigma0)) / Z(theta),

where ``d`` is the Kendall tau distance (the number of discordant item pairs) and the normalizer has
the closed form Z(theta) = prod_{i=1}^{n-1} (1 - phi^{i+1}) / (1 - phi) with phi = exp(-theta) (and
Z = n! at theta = 0, the uniform distribution). Larger theta concentrates mass on sigma0.

Sampling uses the Repeated Insertion Model: the central items are inserted one at a time, each jumping
back a geometric number of places, which produces an exact Mallows draw in O(n^2). Estimation recovers
the central permutation by Copeland/Borda aggregation of the pairwise-precedence counts (the sufficient
statistic) and fits theta by matching the mean Kendall distance to its closed-form expectation.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations

import numpy as np
from numpy.random import RandomState

from mixle.enumeration.algorithms import BufferedStream, ProductEnumerator
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
    log_truncated_geometric_normalizer,
    permutation,
    permutation_batch,
    positive_integer,
    sample_size,
    truncated_geometric_mean,
)
from mixle.stats.rankings._contracts import (
    weights as validate_weights,
)

_MAX_THETA = 700.0


@dataclass(frozen=True)
class MallowsFitDiagnostics:
    """Center-search and regularization evidence attached to a fitted Mallows law."""

    center_algorithm: str
    center_exact: bool
    centers_evaluated: int
    kendall_objective: float
    regularized: bool
    pseudo_count: float


def _log_normalizer(theta: float, n: int) -> float:
    """Return log Z(theta) for the Kendall Mallows model on n items."""
    if n <= 1:
        return 0.0
    return float(sum(log_truncated_geometric_normalizer(theta, i) for i in range(1, n)))


def _expected_distance(theta: float, n: int) -> float:
    """Return E_theta[d] = sum_{i=1}^{n-1} E[V_i] for the Kendall Mallows model on n items."""
    if n <= 1:
        return 0.0
    return float(sum(truncated_geometric_mean(theta, i) for i in range(1, n)))


def _solve_theta(mean_distance: float, n: int) -> float:
    """Return the theta whose expected Kendall distance matches ``mean_distance`` (bisection)."""
    uniform_mean = n * (n - 1) / 4.0
    if n <= 1 or mean_distance >= uniform_mean:
        return 0.0
    if mean_distance <= 0.0:
        return _MAX_THETA
    lo, hi = 0.0, 1.0
    while _expected_distance(hi, n) > mean_distance and hi < _MAX_THETA:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _expected_distance(mid, n) > mean_distance:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class MallowsDistribution(SequenceEncodableProbabilityDistribution):
    """Mallows distribution over permutations of 0,...,n-1 with central permutation sigma0 and dispersion theta.

    Data type: List[int] (an ordering: a permutation of 0,...,n-1, best-ranked item first).
    """

    @classmethod
    def compute_capabilities(cls):
        """Return compute-backend metadata for the Mallows distribution."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="Kendall tau distance over permutation pairs is numpy-native.",
        )

    def __init__(
        self,
        sigma0: Sequence[int] | np.ndarray,
        theta: float = 1.0,
        name: str | None = None,
        keys: str | None = None,
        fit_diagnostics: MallowsFitDiagnostics | None = None,
    ) -> None:
        """Create a Mallows distribution around a central permutation.

        Args:
            sigma0 (Union[Sequence[int], np.ndarray]): Central permutation (an ordering of 0,...,n-1).
            theta (float): Non-negative dispersion. theta = 0 is uniform; larger theta concentrates on sigma0.
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            sigma0 (np.ndarray): Central permutation.
            theta (float): Dispersion parameter.
            dim (int): Number of items n.
            rank0 (np.ndarray): rank0[item] = position of item in sigma0.
            log_z (float): log normalizer.

        """
        raw_center = np.asarray(sigma0)
        if raw_center.ndim != 1 or len(raw_center) < 2:
            raise ValueError("MallowsDistribution sigma0 must contain at least two items.")
        self.dim = len(raw_center)
        self.sigma0 = permutation(raw_center, self.dim, label="sigma0").copy()
        self.sigma0.setflags(write=False)
        self.theta = finite_nonnegative(theta, label="theta")
        self.rank0 = np.empty(self.dim, dtype=int)
        self.rank0[self.sigma0] = np.arange(self.dim)
        self.rank0.setflags(write=False)
        self.log_z = _log_normalizer(self.theta, self.dim)
        self.name = name
        self.keys = keys
        if fit_diagnostics is not None and not isinstance(fit_diagnostics, MallowsFitDiagnostics):
            raise TypeError("fit_diagnostics must be a MallowsFitDiagnostics record.")
        self.fit_diagnostics = fit_diagnostics

    def __str__(self) -> str:
        """Return a constructor-style representation of the Mallows distribution."""
        return "MallowsDistribution(%s, theta=%s, name=%s, keys=%s)" % (
            repr([int(v) for v in self.sigma0]),
            repr(self.theta),
            repr(self.name),
            repr(self.keys),
        )

    def kendall_distance(self, x: Sequence[int]) -> int:
        """Return the Kendall tau distance between ordering x and the central permutation.

        Raises:
            ValueError: If x is not a permutation of 0,...,n-1 (wrong length, a repeated
                element, a non-integer entry, or an element outside that range). A malformed x
                is not merely an unlikely ordering -- it isn't an ordering at all, so it is
                rejected here rather than silently scored (e.g. a repeated index would otherwise
                still produce a finite, meaningless distance, an out-of-range index can alias a
                valid rank via numpy's negative-index wraparound instead of failing loudly, and
                a fractional entry like 0.5 would otherwise be silently truncated to 0 by the
                int cast before this check ever saw it).

        """
        xa = permutation(x, self.dim, label="Mallows ordering")
        y = self.rank0[xa]
        return int(np.sum(y[:, None] > y[None, :], where=np.triu(np.ones((self.dim, self.dim), dtype=bool), 1)))

    def density(self, x: Sequence[int]) -> float:
        """Return the probability of an ordering x."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int]) -> float:
        """Return the log-probability of an ordering x (a permutation of 0,...,n-1)."""
        return -self.theta * self.kendall_distance(x) - self.log_z

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for an (N, n) array of orderings."""
        checked = permutation_batch(x, self.dim, label="Mallows orderings")
        y = self.rank0[checked]  # (N, n) ranks under sigma0
        mask = np.triu(np.ones((self.dim, self.dim), dtype=bool), 1)
        # discordant pairs per row: r < r' with y[r] > y[r'].
        dist = np.sum((y[:, :, None] > y[:, None, :]) & mask[None, :, :], axis=(1, 2))
        return -self.theta * dist - self.log_z

    def sampler(self, seed: int | None = None) -> "MallowsSampler":
        """Return a sampler for drawing orderings from this distribution."""
        return MallowsSampler(self, seed)

    def enumerator(self) -> "MallowsEnumerator":
        """Return an exact finite enumerator over all orderings in decreasing probability order."""
        return MallowsEnumerator(self)

    def support_size(self) -> int:
        """Return the number of full rankings."""
        return math.factorial(self.dim)

    def estimator(self, pseudo_count: float | None = None) -> "MallowsEstimator":
        """Return an estimator that keeps the item count fixed at this distribution's n."""
        prior = np.zeros((self.dim, self.dim), dtype=np.float64)
        ranks, later = np.triu_indices(self.dim, 1)
        prior[self.sigma0[ranks], self.sigma0[later]] = 1.0
        return MallowsEstimator(
            dim=self.dim,
            pseudo_count=pseudo_count,
            prior_precede=prior if pseudo_count is not None else None,
            prior_center=self.sigma0,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "MallowsDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return MallowsDataEncoder(dim=self.dim)


class MallowsEnumerator(DistributionEnumerator):
    """Enumerate Mallows orderings in descending probability order, lazily.

    Kendall distance is separable in the Lehmer code: an ordering's distance is the sum of digits
    ``L_i in {0,...,n-1-i}`` (inversions contributed at each rank), each weighted ``-theta*L_i``. So the support
    is a product over the digits and ``ProductEnumerator`` streams it in increasing distance (descending
    probability) without materializing the n! permutations; each digit tuple decodes (factorial number system)
    to a permutation of the identity, relabeled through the central permutation sigma0.
    """

    def __init__(self, dist: MallowsDistribution) -> None:
        super().__init__(dist)
        n = dist.dim
        theta = dist.theta
        sigma0 = dist.sigma0

        def combine(digits: tuple[int, ...]) -> list[int]:
            unused = list(range(n))
            perm = [unused.pop(d) for d in digits]  # factorial-number-system decode -> permutation of identity
            return [int(sigma0[v]) for v in perm]  # relabel through the central permutation

        # digit i ranges over 0..n-1-i with log-weight -theta*L (descending weight == ascending L for theta>=0)
        streams = [BufferedStream((d, -theta * d) for d in range(n - i)) for i in range(n)]
        self._prod = ProductEnumerator(streams, combine=combine, offset=-dist.log_z)

    def __next__(self) -> tuple[list[int], float]:
        return self._prod.__next__()  # (permutation, log_density); StopIteration propagates


class MallowsSampler(DistributionSampler):
    """Draw iid Mallows orderings via the Repeated Insertion Model."""

    def __init__(self, dist: MallowsDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _sample_one(self) -> list[int]:
        n = self.dist.dim
        phi = math.exp(-self.dist.theta)
        perm: list[int] = []
        for i in range(n):
            if self.dist.theta <= 0.0:
                j = self.rng.randint(0, i + 1)
            else:
                # V_i in {0..i} with P(k) ∝ phi^k; inverse-CDF sample.
                weights = phi ** np.arange(i + 1)
                cdf = np.cumsum(weights)
                j = int(np.searchsorted(cdf, self.rng.rand() * cdf[-1]))
            perm.insert(i - j, int(self.dist.sigma0[i]))
        return perm

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[int] | list[list[int]]:
        """Draw orderings (permutations of 0,...,n-1); a single list when size is None."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(sample_size(size))]


class MallowsAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted pairwise-precedence matrix for Mallows estimation.

    ``precede[a, b]`` is the weighted number of orderings in which item ``a`` is ranked before item
    ``b``; this is the sufficient statistic for both the central permutation (via Copeland scores) and
    the mean Kendall distance.
    """

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.precede = np.zeros((self.dim, self.dim))
        self.count = 0.0
        self.keys = keys

    def update(self, x: Sequence[int], weight: float, estimate: MallowsDistribution | None) -> None:
        """Accumulate weighted pairwise-precedence counts for one ordering."""
        checked = permutation(x, self.dim, label="Mallows ordering")
        self.seq_update(checked[None, :], np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted ordering."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: MallowsDistribution | None) -> None:
        """Accumulate weighted pairwise-precedence counts for encoded orderings."""
        checked = permutation_batch(x, self.dim, label="Mallows orderings")
        checked_weights = validate_weights(weights, len(checked))
        n = self.dim
        r_idx, rp_idx = np.triu_indices(n, 1)  # all rank pairs r < r'
        for row, weight in zip(checked, checked_weights):
            # the earlier-ranked item precedes the later-ranked item for every pair r < r'.
            np.add.at(self.precede, (row[r_idx], row[rp_idx]), weight)
        self.count += float(np.sum(checked_weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded orderings."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray]) -> "MallowsAccumulator":
        """Merge serialized precedence-count statistics into this accumulator."""
        count, precede = count_matrix_statistics(
            suff_stat,
            self.dim,
            label="Mallows statistics",
            entries_per_observation=self.dim * (self.dim - 1) / 2.0,
        )
        self.count += count
        self.precede += precede
        return self

    def value(self) -> tuple[float, np.ndarray]:
        """Return the accumulated weight and pairwise-precedence matrix."""
        return self.count, self.precede.copy()

    def from_value(self, x: tuple[float, np.ndarray]) -> "MallowsAccumulator":
        """Restore the accumulator from serialized precedence-count statistics."""
        self.count, self.precede = count_matrix_statistics(
            x,
            self.dim,
            label="Mallows statistics",
            entries_per_observation=self.dim * (self.dim - 1) / 2.0,
        )
        return self

    def acc_to_encoder(self) -> "MallowsDataEncoder":
        """Return an encoder compatible with the accumulated ordering dimension."""
        return MallowsDataEncoder(dim=self.dim)


class MallowsAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for MallowsAccumulator."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.keys = keys

    def make(self) -> MallowsAccumulator:
        """Create an empty Mallows accumulator."""
        return MallowsAccumulator(dim=self.dim, keys=self.keys)


class MallowsEstimator(ParameterEstimator):
    """Estimator for the Mallows central permutation (Copeland aggregation) and dispersion theta."""

    def __init__(
        self,
        dim: int,
        theta: float | None = None,
        pseudo_count: float | None = None,
        prior_precede: np.ndarray | None = None,
        prior_center: Sequence[int] | np.ndarray | None = None,
        center_exact_cap: int = 9,
        allow_approximate_center: bool = False,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.theta = None if theta is None else finite_nonnegative(theta, label="theta")
        self.pseudo_count = None if pseudo_count is None else finite_nonnegative(pseudo_count, label="pseudo_count")
        if prior_precede is None:
            prior = 0.5 * (1.0 - np.eye(self.dim))
        else:
            prior = np.asarray(prior_precede, dtype=np.float64)
            if prior.shape != (self.dim, self.dim) or not np.all(np.isfinite(prior)) or np.any(prior < 0.0):
                raise ValueError("prior_precede must be a finite nonnegative dim-by-dim matrix.")
            if not np.allclose(prior + prior.T, 1.0 - np.eye(self.dim)):
                raise ValueError("prior_precede must assign one unit across each unordered pair.")
            prior = prior.copy()
        self.prior_precede = prior
        self.prior_center = (
            np.arange(self.dim, dtype=np.int64)
            if prior_center is None
            else permutation(prior_center, self.dim, label="prior_center").copy()
        )
        self.center_exact_cap = positive_integer(center_exact_cap, label="center_exact_cap", minimum=2)
        if not isinstance(allow_approximate_center, (bool, np.bool_)):
            raise TypeError("allow_approximate_center must be a Boolean.")
        self.allow_approximate_center = bool(allow_approximate_center)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> MallowsAccumulatorFactory:
        """Return a factory for Mallows sufficient-statistic accumulators."""
        return MallowsAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray]) -> MallowsDistribution:
        """Estimate the central permutation and dispersion from precedence counts."""
        if nobs is not None:
            finite_nonnegative(nobs, label="nobs")
        count, precede = count_matrix_statistics(
            suff_stat,
            self.dim,
            label="Mallows statistics",
            entries_per_observation=self.dim * (self.dim - 1) / 2.0,
        )
        if self.pseudo_count is not None and self.pseudo_count > 0.0:
            precede += self.pseudo_count * self.prior_precede
            count += self.pseudo_count
        n = self.dim
        if count <= 0.0:
            return MallowsDistribution(np.arange(n), 0.0, name=self.name, keys=self.keys)

        exact = n <= self.center_exact_cap
        if not exact and not self.allow_approximate_center:
            raise ValueError(
                f"exact Mallows center search is capped at {self.center_exact_cap} items; "
                "set allow_approximate_center=True for a labeled Copeland approximation."
            )

        def objective(center: np.ndarray) -> float:
            total = 0.0
            for earlier in range(n):
                for later in range(earlier + 1, n):
                    total += precede[center[later], center[earlier]]
            return total

        if exact:
            sigma0: np.ndarray | None = None
            best_objective = math.inf
            centers_evaluated = 0
            for candidate_tuple in permutations(range(n)):
                candidate = np.asarray(candidate_tuple, dtype=np.int64)
                candidate_objective = objective(candidate)
                centers_evaluated += 1
                if candidate_objective < best_objective:
                    sigma0, best_objective = candidate, candidate_objective
            if sigma0 is None:
                raise RuntimeError("Mallows exact center search produced no candidates.")
            center_algorithm = "exact_kemeny_enumeration"
        else:
            scores = precede.sum(axis=1) - precede.sum(axis=0)
            sigma0 = np.argsort(-scores, kind="stable")
            best_objective = objective(sigma0)
            centers_evaluated = 1
            center_algorithm = "copeland_approximation"
        rank0 = np.empty(n, dtype=int)
        rank0[sigma0] = np.arange(n)

        # Mean Kendall distance to sigma0: discordant weight summed over sigma0-ordered pairs.
        discordant = 0.0
        for a in range(n):
            for b in range(n):
                if rank0[a] < rank0[b]:
                    discordant += precede[b, a]
        mean_distance = discordant / count

        theta = self.theta if self.theta is not None else _solve_theta(mean_distance, n)
        diagnostics = MallowsFitDiagnostics(
            center_algorithm=center_algorithm,
            center_exact=exact,
            centers_evaluated=centers_evaluated,
            kendall_objective=best_objective,
            regularized=self.pseudo_count is not None and self.pseudo_count > 0.0,
            pseudo_count=0.0 if self.pseudo_count is None else self.pseudo_count,
        )
        return MallowsDistribution(
            sigma0,
            theta,
            name=self.name,
            keys=self.keys,
            fit_diagnostics=diagnostics,
        )


class MallowsDataEncoder(DataSequenceEncoder):
    """Encode a sequence of orderings (permutations of 0,...,n-1) into an (N, n) integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = None if dim is None else positive_integer(dim, label="dim", minimum=2)

    def __str__(self) -> str:
        return "MallowsDataEncoder(dim=%s)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MallowsDataEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        """Validate and encode orderings as a two-dimensional integer array."""
        raw = np.asarray([list(row) for row in x])
        if self.dim is None:
            if raw.ndim != 2 or raw.shape[0] == 0:
                raise ValueError("MallowsDistribution requires a non-empty sequence of orderings.")
            return permutation_batch(raw, raw.shape[1], label="Mallows orderings", allow_empty=False)
        return permutation_batch(raw, self.dim, label="Mallows orderings", allow_empty=False)
