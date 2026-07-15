"""Chinese Restaurant Process -- a Bayesian-nonparametric distribution over partitions.

The CRP with concentration ``alpha`` is the exchangeable distribution over partitions of ``n`` items
induced by the sequential rule "item ``i`` joins an existing block of size ``m`` with probability
``m / (i - 1 + alpha)`` or starts a new block with probability ``alpha / (i - 1 + alpha)``". A partition
with blocks of sizes ``n_1, ..., n_K`` has the Ewens probability

    P = alpha^K * Gamma(alpha) / Gamma(alpha + n) * prod_k Gamma(n_k),

so larger ``alpha`` favours more, smaller blocks. It is the partition prior underlying Dirichlet-process
mixtures; an observation here is a partition of ``n`` items given as a label vector (the labels are
arbitrary -- the density is relabeling-invariant). ``alpha`` is fit by maximum likelihood, the
monotone solve ``alpha (psi(alpha + n) - psi(alpha)) = mean number of blocks``.


Reference: Pitman, *Combinatorial Stochastic Processes* (Springer, 2006).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import digamma, gammaln

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


def _block_sizes(labels: np.ndarray) -> np.ndarray:
    """Return the block sizes of the partition induced by an integer label vector."""
    return np.unique(np.asarray(labels), return_counts=True)[1]


class ChineseRestaurantProcessDistribution(SequenceEncodableProbabilityDistribution):
    """CRP distribution over partitions of ``n`` items with concentration ``alpha > 0``."""

    def __init__(self, alpha: float, n: int, name: str | None = None, keys: str | None = None) -> None:
        if alpha <= 0.0 or not np.isfinite(alpha) or int(n) <= 0:
            raise ValueError("ChineseRestaurantProcessDistribution requires alpha > 0 and n >= 1.")
        self.alpha = float(alpha)
        self.n = int(n)
        self.name = name
        self.keys = keys
        self._log_alpha = math.log(self.alpha)
        self._log_norm = gammaln(self.alpha) - gammaln(self.alpha + self.n)  # Gamma(alpha)/Gamma(alpha+n)

    def __str__(self) -> str:
        return "ChineseRestaurantProcessDistribution(%s, %s, name=%s, keys=%s)" % (
            repr(self.alpha),
            repr(self.n),
            repr(self.name),
            repr(self.keys),
        )

    def density(self, x: np.ndarray) -> float:
        """Return the probability of the partition encoded by label vector ``x``."""
        return math.exp(self.log_density(x))

    def log_density(self, x: np.ndarray) -> float:
        """Return the Ewens log-probability of the partition that label vector ``x`` induces."""
        sizes = _block_sizes(x)
        if sizes.sum() != self.n:
            return -np.inf
        k = sizes.shape[0]
        return float(k * self._log_alpha + self._log_norm + np.sum(gammaln(sizes.astype(np.float64))))

    def seq_log_density(self, x: list[np.ndarray]) -> np.ndarray:
        """Return the Ewens log-probability for a list of partition label vectors."""
        return np.array([self.log_density(z) for z in x], dtype=np.float64)

    def sampler(self, seed: int | None = None) -> "ChineseRestaurantProcessSampler":
        """Return a sampler that draws partitions by the sequential CRP rule."""
        return ChineseRestaurantProcessSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ChineseRestaurantProcessEstimator":
        """Return a maximum-likelihood estimator for the concentration ``alpha`` at fixed ``n``."""
        return ChineseRestaurantProcessEstimator(self.n, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "ChineseRestaurantProcessDataEncoder":
        """Return the data encoder (passes label vectors through)."""
        return ChineseRestaurantProcessDataEncoder()


class ChineseRestaurantProcessSampler(DistributionSampler):
    """Draw partitions by the sequential CRP seating rule; returns first-appearance label vectors."""

    def __init__(self, dist: ChineseRestaurantProcessDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _one(self) -> np.ndarray:
        d = self.dist
        labels = np.empty(d.n, dtype=np.int64)
        sizes: list[int] = []
        for i in range(d.n):
            weights = np.array(sizes + [d.alpha], dtype=np.float64)
            t = int(self.rng.choice(len(weights), p=weights / weights.sum()))
            if t == len(sizes):
                sizes.append(1)
            else:
                sizes[t] += 1
            labels[i] = t
        return labels

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw one partition or a list of independent partitions."""
        if size is None:
            return self._one()
        return [self._one() for _ in range(int(size))]


class ChineseRestaurantProcessAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the total number of blocks and observation count (the CRP sufficient statistics)."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.sum_k = 0.0  # total number of blocks across observations
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: np.ndarray, weight: float, estimate: Any) -> None:
        """Accumulate the weighted block count for one partition."""
        self.sum_k += weight * _block_sizes(x).shape[0]
        self.count += weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted partition."""
        self.update(x, weight, None)

    def seq_update(self, x: list[np.ndarray], weights: np.ndarray, estimate: Any) -> None:
        """Accumulate weighted block counts for encoded partition label vectors."""
        for z, w in zip(x, np.asarray(weights, dtype=np.float64)):
            self.sum_k += float(w) * _block_sizes(z).shape[0]
            self.count += float(w)

    def seq_initialize(self, x: list[np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded partitions."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float]) -> "ChineseRestaurantProcessAccumulator":
        """Merge serialized block-count statistics into this accumulator."""
        self.sum_k += suff_stat[0]
        self.count += suff_stat[1]
        return self

    def value(self) -> tuple[float, float]:
        """Return the total weighted block count and observation weight."""
        return self.sum_k, self.count

    def from_value(self, x: tuple[float, float]) -> "ChineseRestaurantProcessAccumulator":
        """Restore the accumulator from serialized block-count statistics."""
        self.sum_k, self.count = float(x[0]), float(x[1])
        return self

    def acc_to_encoder(self) -> "ChineseRestaurantProcessDataEncoder":
        """Return an encoder for CRP partition label vectors."""
        return ChineseRestaurantProcessDataEncoder()


class ChineseRestaurantProcessAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for ChineseRestaurantProcessAccumulator."""

    def __init__(self, name: str | None = None, keys: str | None = None) -> None:
        self.name = name
        self.keys = keys

    def make(self) -> ChineseRestaurantProcessAccumulator:
        """Create an empty CRP accumulator."""
        return ChineseRestaurantProcessAccumulator(name=self.name, keys=self.keys)


class ChineseRestaurantProcessEstimator(ParameterEstimator):
    """Maximum-likelihood estimator for the CRP concentration via the monotone expected-blocks equation."""

    def __init__(
        self,
        n: int,
        alpha_min: float = 1.0e-6,
        alpha_max: float = 1.0e6,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.n = int(n)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> ChineseRestaurantProcessAccumulatorFactory:
        """Return a factory for CRP sufficient-statistic accumulators."""
        return ChineseRestaurantProcessAccumulatorFactory(name=self.name, keys=self.keys)

    def _expected_blocks(self, alpha: float) -> float:
        return alpha * float(digamma(alpha + self.n) - digamma(alpha))

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> ChineseRestaurantProcessDistribution:
        """Estimate the CRP concentration from the observed mean block count."""
        sum_k, count = suff_stat
        if count <= 0.0:
            return ChineseRestaurantProcessDistribution(1.0, self.n, name=self.name, keys=self.keys)
        mean_k = sum_k / count
        # solve E[K | alpha] = alpha (psi(alpha+n) - psi(alpha)) = mean_k (monotone increasing in alpha)
        lo, hi = self.alpha_min, self.alpha_max
        if mean_k <= self._expected_blocks(lo):
            alpha = lo
        elif mean_k >= self._expected_blocks(hi):
            alpha = hi
        else:
            for _ in range(200):
                mid = math.sqrt(lo * hi)
                if self._expected_blocks(mid) < mean_k:
                    lo = mid
                else:
                    hi = mid
            alpha = math.sqrt(lo * hi)
        return ChineseRestaurantProcessDistribution(alpha, self.n, name=self.name, keys=self.keys)


class ChineseRestaurantProcessDataEncoder(DataSequenceEncoder):
    """Encode a sequence of partition label vectors (passthrough as integer arrays)."""

    def __str__(self) -> str:
        return "ChineseRestaurantProcessDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ChineseRestaurantProcessDataEncoder)

    def seq_encode(self, x: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Encode partition label vectors as integer arrays without relabeling them."""
        return [np.asarray(z, dtype=np.int64) for z in x]
