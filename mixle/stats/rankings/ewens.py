"""Ewens distribution over permutations -- the canonical cycle-structure (random-permutation) law.

A permutation is weighted by its number of cycles:

    p(sigma) = theta^{cycles(sigma)} / Z,    Z = theta (theta + 1) ... (theta + n - 1) = Gamma(theta+n)/Gamma(theta).

``theta = 1`` is uniform over the ``n!`` permutations; ``theta -> 0`` concentrates on single ``n``-cycles
(few cycles); large ``theta`` concentrates on the identity (every point a fixed cycle). This is the
permutation form of the Ewens sampling formula (population genetics, random-permutation theory); its
induced *partition* by cycle sizes is what :class:`ChineseRestaurantProcessDistribution` models.

Sufficient statistic: the cycle count, computed from the shared Cayley kernel
(``cycles = n - cayley_distance(sigma, identity)``). Sampling is an exact numba Chinese-restaurant /
Feller construction; ``theta`` is fit by matching the mean cycle count ``E[cycles] = sum_i theta/(theta+i)``.

Data type: ``List[int]`` -- a permutation of ``0..n-1`` read as a function ``i -> sigma[i]`` (cycles of
that map are what the model scores).
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
from mixle.stats.rankings._contracts import (
    bounded_sum_statistics,
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
from mixle.stats.rankings._permutation_kernels import seq_distance_to_center
from mixle.utils.optional_deps import numba

_MAX_THETA = 1e6


@numba.njit("int64[:, :](int64, float64, int64, int64)", cache=True)
def _ewens_sample(n, theta, n_samples, seed):
    """Exact Ewens(theta) permutations via the Chinese-restaurant / Feller insertion construction."""
    np.random.seed(seed)
    out = np.empty((n_samples, n), dtype=np.int64)
    for s in range(n_samples):
        sigma = out[s]
        for i in range(n):
            if np.random.random() < theta / (theta + i):
                sigma[i] = i  # i opens a new cycle (a fixed point for now)
            else:
                j = np.random.randint(0, i)  # splice i into the cycle through a uniform earlier element
                sigma[i] = sigma[j]
                sigma[j] = i
    return out


def _log_normalizer(theta: float, n: int) -> float:
    if n <= 1 or theta <= 0.0:
        return 0.0 if n <= 1 else -math.inf
    return float(math.lgamma(theta + n) - math.lgamma(theta))


def _expected_cycles(theta: float, n: int) -> float:
    return float(sum(theta / (theta + i) for i in range(n)))


def _solve_theta(mean_cycles: float, n: int) -> float:
    """Theta matching a target mean cycle count (E[cycles] is increasing in theta)."""
    if mean_cycles <= 1.0:
        return 1e-6
    if mean_cycles >= n:
        return _MAX_THETA
    lo, hi = 1e-6, 1.0
    while _expected_cycles(hi, n) < mean_cycles and hi < _MAX_THETA:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _expected_cycles(mid, n) < mean_cycles:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _cycle_counts(x: np.ndarray, n: int) -> np.ndarray:
    """Number of cycles of each permutation row: ``n - cayley_distance(sigma, identity)``."""
    return n - seq_distance_to_center(x, np.arange(n, dtype=np.int64), "cayley")


class EwensDistribution(SequenceEncodableProbabilityDistribution):
    """Ewens distribution over permutations of ``0..n-1`` with cycle-weight parameter ``theta > 0``."""

    @classmethod
    def compute_capabilities(cls):
        """Declare the NumPy and numba execution path used by Ewens kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="Cycle counts run through the shared numba Cayley kernel.",
        )

    def __init__(self, dim: int, theta: float = 1.0, name: str | None = None, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.theta = finite_positive(theta, label="theta")
        self.log_theta = math.log(self.theta)
        self.log_z = _log_normalizer(self.theta, self.dim)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "EwensDistribution(dim=%d, theta=%s, name=%s, keys=%s)" % (
            self.dim,
            repr(self.theta),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: Sequence[int]) -> float:
        """Return the probability of one permutation."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int]) -> float:
        """Return the log-probability of one permutation."""
        checked = permutation(x, self.dim, label="Ewens permutation")
        return float(self.seq_log_density(checked[None, :])[0])

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for encoded permutations."""
        checked = permutation_batch(x, self.dim, label="Ewens permutations")
        cycles = _cycle_counts(checked, self.dim)
        return cycles * self.log_theta - self.log_z

    def sampler(self, seed: int | None = None) -> EwensSampler:
        """Return an exact Ewens permutation sampler."""
        return EwensSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> EwensEstimator:
        """Return a cycle-count moment estimator for this dimension."""
        return EwensEstimator(
            dim=self.dim,
            pseudo_count=pseudo_count,
            prior_mean_cycles=_expected_cycles(self.theta, self.dim),
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> EwensDataEncoder:
        """Return the permutation encoder used by vectorized methods."""
        return EwensDataEncoder(dim=self.dim)


class EwensSampler(DistributionSampler):
    """Exact Ewens draws via the numba Chinese-restaurant construction."""

    def __init__(self, dist: EwensDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[int] | list[list[int]]:
        """Draw one permutation or ``size`` iid permutations."""
        checked_size = sample_size(size)
        k = 1 if checked_size is None else checked_size
        seed = int(self.rng.randint(0, 2**31 - 1))
        arr = _ewens_sample(self.dist.dim, self.dist.theta, k, seed)
        draws = [[int(v) for v in row] for row in arr]
        return draws[0] if size is None else draws


class EwensAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the total (weighted) cycle count and observation weight."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.cycle_sum = 0.0
        self.count = 0.0
        self.keys = keys

    def update(self, x: Sequence[int], weight: float, estimate: Any) -> None:
        """Update cycle-count statistics from one weighted permutation."""
        checked = permutation(x, self.dim, label="Ewens permutation")
        self.seq_update(checked[None, :], np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize cycle-count statistics from one weighted permutation."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Any) -> None:
        """Update cycle-count statistics from encoded permutations."""
        checked = permutation_batch(x, self.dim, label="Ewens permutations")
        checked_weights = validate_weights(weights, len(checked))
        cycles = _cycle_counts(checked, self.dim)
        self.cycle_sum += float(np.sum(cycles * checked_weights))
        self.count += float(np.sum(checked_weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize cycle-count statistics from encoded permutations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat) -> EwensAccumulator:
        """Merge cycle-count totals and observation weight from another accumulator."""
        cycle_sum, count = bounded_sum_statistics(
            suff_stat,
            label="Ewens statistics",
            minimum_per_observation=1.0,
            maximum_per_observation=float(self.dim),
        )
        self.cycle_sum += cycle_sum
        self.count += count
        return self

    def value(self):
        """Return total weighted cycle count and total observation weight."""
        return self.cycle_sum, self.count

    def from_value(self, x) -> EwensAccumulator:
        """Restore accumulator state from ``value`` output."""
        self.cycle_sum, self.count = bounded_sum_statistics(
            x,
            label="Ewens statistics",
            minimum_per_observation=1.0,
            maximum_per_observation=float(self.dim),
        )
        return self

    def acc_to_encoder(self) -> EwensDataEncoder:
        """Return the encoder compatible with Ewens cycle-count statistics."""
        return EwensDataEncoder(dim=self.dim)


class EwensAccumulatorFactory(StatisticAccumulatorFactory):
    """Create accumulators for Ewens cycle-count statistics."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.keys = keys

    def make(self) -> EwensAccumulator:
        """Create an empty Ewens accumulator."""
        return EwensAccumulator(dim=self.dim, keys=self.keys)


class EwensEstimator(ParameterEstimator):
    """Fit ``theta`` by matching the mean cycle count ``E[cycles] = sum_i theta/(theta+i)``."""

    def __init__(
        self,
        dim: int,
        theta: float | None = None,
        pseudo_count: float | None = None,
        prior_mean_cycles: float | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dim = positive_integer(dim, label="dim", minimum=2)
        self.theta = None if theta is None else finite_positive(theta, label="theta")
        self.pseudo_count = None if pseudo_count is None else finite_nonnegative(pseudo_count, label="pseudo_count")
        default_prior = _expected_cycles(1.0, self.dim)
        self.prior_mean_cycles = (
            default_prior
            if prior_mean_cycles is None
            else finite_nonnegative(prior_mean_cycles, label="prior_mean_cycles")
        )
        if self.prior_mean_cycles < 1.0 or self.prior_mean_cycles > self.dim:
            raise ValueError("prior_mean_cycles must be in [1, dim].")
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> EwensAccumulatorFactory:
        """Return a factory for Ewens sufficient-statistic accumulators."""
        return EwensAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat) -> EwensDistribution:
        """Estimate ``theta`` by matching the accumulated mean cycle count."""
        if nobs is not None:
            finite_nonnegative(nobs, label="nobs")
        cycle_sum, count = bounded_sum_statistics(
            suff_stat,
            label="Ewens statistics",
            minimum_per_observation=1.0,
            maximum_per_observation=float(self.dim),
        )
        if self.pseudo_count is not None and self.pseudo_count > 0.0:
            cycle_sum += self.pseudo_count * self.prior_mean_cycles
            count += self.pseudo_count
        if count <= 0.0:
            return EwensDistribution(self.dim, 1.0, name=self.name, keys=self.keys)
        theta = self.theta if self.theta is not None else _solve_theta(cycle_sum / count, self.dim)
        return EwensDistribution(self.dim, theta, name=self.name, keys=self.keys)


class EwensDataEncoder(DataSequenceEncoder):
    """Encode a sequence of permutations of 0,...,n-1 into an (N, n) integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = None if dim is None else positive_integer(dim, label="dim", minimum=2)

    def __str__(self) -> str:
        return "EwensDataEncoder(dim=%s)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EwensDataEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        """Validate and encode permutations as a dense integer matrix."""
        raw = np.asarray([list(row) for row in x])
        if self.dim is None:
            if raw.ndim != 2 or raw.shape[0] == 0:
                raise ValueError("EwensDistribution requires a non-empty sequence of permutations.")
            return permutation_batch(raw, raw.shape[1], label="Ewens permutations", allow_empty=False)
        return permutation_batch(raw, self.dim, label="Ewens permutations", allow_empty=False)


__all__ = [
    "EwensDistribution",
    "EwensSampler",
    "EwensAccumulator",
    "EwensAccumulatorFactory",
    "EwensEstimator",
    "EwensDataEncoder",
]
