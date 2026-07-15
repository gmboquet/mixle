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

_MAX_THETA = 700.0


def _log_normalizer(theta: float, n: int) -> float:
    """Return log Z(theta) for the Kendall Mallows model on n items."""
    if n <= 1:
        return 0.0
    if theta <= 0.0:
        return float(math.lgamma(n + 1))  # log n!
    phi = math.exp(-theta)
    if phi <= 0.0:
        return 0.0  # theta -> inf, Z -> 1 (all mass on sigma0)
    log1m_phi = math.log1p(-phi)
    total = 0.0
    for i in range(1, n):
        total += math.log1p(-(phi ** (i + 1))) - log1m_phi
    return total


def _expected_distance(theta: float, n: int) -> float:
    """Return E_theta[d] = sum_{i=1}^{n-1} E[V_i] for the Kendall Mallows model on n items."""
    if n <= 1:
        return 0.0
    if theta <= 0.0:
        return n * (n - 1) / 4.0
    phi = math.exp(-theta)
    total = 0.0
    for i in range(1, n):
        # V_i in {0..i} with P(k) ∝ phi^k: E[V_i] = phi/(1-phi) - (i+1) phi^{i+1} / (1 - phi^{i+1}).
        total += phi / (1.0 - phi) - (i + 1) * phi ** (i + 1) / (1.0 - phi ** (i + 1))
    return total


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
        s0 = np.asarray(sigma0, dtype=int)
        n = len(s0)
        if n < 2 or not np.array_equal(np.sort(s0), np.arange(n)):
            raise ValueError("MallowsDistribution sigma0 must be a permutation of 0,...,n-1 with n >= 2.")
        if theta < 0.0 or not np.isfinite(theta):
            raise ValueError("MallowsDistribution requires theta >= 0.")
        self.sigma0 = s0
        self.theta = float(theta)
        self.dim = n
        self.rank0 = np.empty(n, dtype=int)
        self.rank0[s0] = np.arange(n)
        self.log_z = _log_normalizer(self.theta, n)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-style representation of the Mallows distribution."""
        return "MallowsDistribution(%s, theta=%s, name=%s, keys=%s)" % (
            repr([int(v) for v in self.sigma0]),
            repr(self.theta),
            repr(self.name),
            repr(self.keys),
        )

    def kendall_distance(self, x: Sequence[int]) -> int:
        """Return the Kendall tau distance between ordering x and the central permutation."""
        y = self.rank0[np.asarray(x, dtype=int)]
        return int(np.sum(y[:, None] > y[None, :], where=np.triu(np.ones((self.dim, self.dim), dtype=bool), 1)))

    def density(self, x: Sequence[int]) -> float:
        """Return the probability of an ordering x."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int]) -> float:
        """Return the log-probability of an ordering x (a permutation of 0,...,n-1)."""
        return -self.theta * self.kendall_distance(x) - self.log_z

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return vectorized log-probabilities for an (N, n) array of orderings."""
        y = self.rank0[x]  # (N, n) ranks under sigma0
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

    def estimator(self, pseudo_count: float | None = None) -> "MallowsEstimator":
        """Return an estimator that keeps the item count fixed at this distribution's n."""
        return MallowsEstimator(dim=self.dim, name=self.name, keys=self.keys)

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
        return [self._sample_one() for _ in range(size)]


class MallowsAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted pairwise-precedence matrix for Mallows estimation.

    ``precede[a, b]`` is the weighted number of orderings in which item ``a`` is ranked before item
    ``b``; this is the sufficient statistic for both the central permutation (via Copeland scores) and
    the mean Kendall distance.
    """

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.precede = np.zeros((dim, dim))
        self.count = 0.0
        self.keys = keys

    def update(self, x: Sequence[int], weight: float, estimate: MallowsDistribution | None) -> None:
        """Accumulate weighted pairwise-precedence counts for one ordering."""
        self.seq_update(np.asarray([x], dtype=int), np.asarray([weight], dtype=float), estimate)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted ordering."""
        self.update(x, weight, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: MallowsDistribution | None) -> None:
        """Accumulate weighted pairwise-precedence counts for encoded orderings."""
        n = self.dim
        r_idx, rp_idx = np.triu_indices(n, 1)  # all rank pairs r < r'
        for row, w in zip(x, weights):
            # the earlier-ranked item precedes the later-ranked item for every pair r < r'.
            np.add.at(self.precede, (row[r_idx], row[rp_idx]), w)
        self.count += float(np.sum(weights, dtype=np.float64))

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded orderings."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray]) -> "MallowsAccumulator":
        """Merge serialized precedence-count statistics into this accumulator."""
        self.count += suff_stat[0]
        self.precede += suff_stat[1]
        return self

    def value(self) -> tuple[float, np.ndarray]:
        """Return the accumulated weight and pairwise-precedence matrix."""
        return self.count, self.precede

    def from_value(self, x: tuple[float, np.ndarray]) -> "MallowsAccumulator":
        """Restore the accumulator from serialized precedence-count statistics."""
        self.count, self.precede = x[0], np.asarray(x[1])
        self.dim = self.precede.shape[0]
        return self

    def acc_to_encoder(self) -> "MallowsDataEncoder":
        """Return an encoder compatible with the accumulated ordering dimension."""
        return MallowsDataEncoder(dim=self.dim)


class MallowsAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for MallowsAccumulator."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
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
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if dim is None or dim < 2:
            raise ValueError("MallowsEstimator requires the number of items dim >= 2.")
        self.dim = int(dim)
        self.theta = theta
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> MallowsAccumulatorFactory:
        """Return a factory for Mallows sufficient-statistic accumulators."""
        return MallowsAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray]) -> MallowsDistribution:
        """Estimate the central permutation and dispersion from precedence counts."""
        count, precede = suff_stat
        n = self.dim
        if count <= 0.0:
            return MallowsDistribution(np.arange(n), 0.0, name=self.name, keys=self.keys)

        # Copeland scores: how often each item precedes the others; order descending for sigma0.
        scores = precede.sum(axis=1) - precede.sum(axis=0)
        sigma0 = np.argsort(-scores, kind="stable")
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
        return MallowsDistribution(sigma0, theta, name=self.name, keys=self.keys)


class MallowsDataEncoder(DataSequenceEncoder):
    """Encode a sequence of orderings (permutations of 0,...,n-1) into an (N, n) integer array."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim

    def __str__(self) -> str:
        return "MallowsDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MallowsDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[int]]) -> np.ndarray:
        """Validate and encode orderings as a two-dimensional integer array."""
        rv = np.asarray([list(row) for row in x], dtype=int)
        if rv.ndim != 2 or rv.shape[0] == 0:
            raise ValueError("MallowsDistribution requires a non-empty sequence of orderings.")
        expected = np.arange(rv.shape[1])
        for row in rv:
            if not np.array_equal(np.sort(row), expected):
                raise ValueError("MallowsDistribution orderings must be permutations of 0,...,n-1.")
        return rv
