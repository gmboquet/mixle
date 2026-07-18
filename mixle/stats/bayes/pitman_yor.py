"""Pitman-Yor process distributions over exchangeable set partitions.

Data type: List[int] (a partition of n elements given as a cluster-label vector; ``x[i]`` is the
cluster id of element i, e.g. ``[0, 0, 1, 0, 2, 1]`` partitions six elements into blocks of sizes
3, 2, 1). Labels are arbitrary -- the distribution is exchangeable and depends only on the block sizes.

The Pitman-Yor process PY(alpha, discount) is the two-parameter generalization of the Dirichlet
process (discount = 0 recovers the DP / Chinese Restaurant Process). Its exchangeable partition
probability function (EPPF) for a partition of n elements into k blocks of sizes n_1, ..., n_k is

    p = [prod_{i=1}^{k-1} (alpha + i*discount)] / [(alpha + 1)_{n-1}] * prod_j (1 - discount)_{n_j - 1},

with (x)_m the rising factorial. In log form (via lgamma) this is computed in ``log_density``. Larger
``alpha`` and ``discount`` favor more blocks; ``discount`` controls the heavy tail of the block-size
distribution (power-law for discount > 0, exponential for discount = 0).

Sampling uses the sequential "Chinese restaurant" construction over ``num_elements`` elements.
Estimation fits ``(alpha, discount)`` by maximizing the aggregated EPPF log-likelihood; the sufficient
statistic is three integer-indexed histograms that capture the (alpha + i)/(alpha + i*discount)/(l -
discount) factors exactly across partitions of arbitrary sizes.
"""

import math
from collections.abc import Sequence

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
from mixle.utils.special import gammaln


def _block_sizes(labels: Sequence[int]) -> np.ndarray:
    """Return the block sizes of a cluster-label vector (descending)."""
    _, counts = np.unique(np.asarray(labels, dtype=int), return_counts=True)
    return np.sort(counts)[::-1]


def _merge_hist(dst: dict[int, float], src: dict[int, float]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + v


class PitmanYorProcessDistribution(SequenceEncodableProbabilityDistribution):
    """Pitman-Yor process over set partitions with concentration alpha and discount in [0, 1).

    Data type: List[int] (a cluster-label vector partitioning n elements). discount = 0 is the
    Dirichlet process / Chinese Restaurant Process.
    """

    @classmethod
    def compute_capabilities(cls):
        """Return compute-backend metadata for Pitman-Yor EPPF evaluation."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="exchangeable partition probability function over block sizes is numpy-native.",
        )

    def __init__(
        self,
        alpha: float = 1.0,
        discount: float = 0.0,
        num_elements: int | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a Pitman-Yor process partition distribution.

        Args:
            alpha (float): Concentration parameter; requires alpha > -discount.
            discount (float): Discount parameter in [0, 1). discount = 0 gives the Dirichlet process.
            num_elements (Optional[int]): Number of elements a draw partitions (sampling only). The
                density is defined for partitions of any size.
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        if not (0.0 <= discount < 1.0) or not np.isfinite(discount):
            raise ValueError("PitmanYorProcessDistribution requires discount in [0, 1).")
        if alpha <= -discount or not np.isfinite(alpha):
            raise ValueError("PitmanYorProcessDistribution requires alpha > -discount.")
        self.alpha = float(alpha)
        self.discount = float(discount)
        self.num_elements = num_elements
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-style representation of the Pitman-Yor process distribution."""
        return "PitmanYorProcessDistribution(alpha=%s, discount=%s, num_elements=%s, name=%s, keys=%s)" % (
            repr(self.alpha),
            repr(self.discount),
            repr(self.num_elements),
            repr(self.name),
            repr(self.keys),
        )

    def _log_eppf(self, sizes: np.ndarray) -> float:
        n = int(sizes.sum())
        k = int(len(sizes))
        a, d = self.alpha, self.discount
        if n == 0:
            return 0.0

        # term1 = sum_{i=1}^{k-1} log(alpha + i*discount)
        if d > 0.0:
            term1 = (k - 1) * math.log(d) + gammaln(a / d + k) - gammaln(a / d + 1.0)
        else:
            term1 = (k - 1) * math.log(a)
        # term2 = -sum_{i=1}^{n-1} log(alpha + i) = lgamma(alpha+1) - lgamma(alpha+n)
        term2 = gammaln(a + 1.0) - gammaln(a + n)
        # term3 = sum_j sum_{l=1}^{n_j-1} log(l - discount) = sum_j [lgamma(n_j - d) - lgamma(1 - d)]
        term3 = float(np.sum(gammaln(sizes - d))) - k * gammaln(1.0 - d)
        return term1 + term2 + term3

    def density(self, x: Sequence[int]) -> float:
        """Return the probability of a partition (cluster-label vector) x."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[int]) -> float:
        """Return the log-probability of a partition (cluster-label vector) x."""
        return self._log_eppf(_block_sizes(x))

    def seq_log_density(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Return vectorized log-probabilities for a sequence of block-size arrays."""
        return np.asarray([self._log_eppf(sizes) for sizes in x], dtype=float)

    def sampler(self, seed: int | None = None) -> "PitmanYorProcessSampler":
        """Return a sampler for drawing partitions from this distribution."""
        return PitmanYorProcessSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "PitmanYorProcessEstimator":
        """Return an estimator that fits alpha (and optionally the discount)."""
        return PitmanYorProcessEstimator(
            discount=self.discount, estimate_discount=False, name=self.name, keys=self.keys
        )

    def dist_to_encoder(self) -> "PitmanYorProcessDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return PitmanYorProcessDataEncoder()


class PitmanYorProcessSampler(DistributionSampler):
    """Draw iid partitions via the sequential Chinese-restaurant construction."""

    def __init__(self, dist: PitmanYorProcessDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def _sample_one(self, n: int) -> list[int]:
        a, d = self.dist.alpha, self.dist.discount
        labels = [0]
        counts = [1.0]
        for i in range(1, n):
            probs = np.empty(len(counts) + 1)
            probs[:-1] = [(c - d) for c in counts]
            probs[-1] = a + len(counts) * d
            probs /= a + i
            choice = int(self.rng.choice(len(probs), p=probs))
            if choice == len(counts):
                counts.append(1.0)
            else:
                counts[choice] += 1.0
            labels.append(choice)
        return labels

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[int] | list[list[int]]:
        """Draw partitions of ``num_elements`` elements; a single label vector when size is None."""
        n = self.dist.num_elements
        if n is None or n < 1:
            raise ValueError("PitmanYorProcessSampler requires the distribution's num_elements to be set (>= 1).")
        if size is None:
            return self._sample_one(n)
        return [self._sample_one(n) for _ in range(size)]


class PitmanYorProcessAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the EPPF histogram sufficient statistics for Pitman-Yor estimation.

    Three integer-indexed weighted histograms make the aggregated log-likelihood exact:
    ``a_hist[i]`` counts partitions with n > i (the ``-log(alpha + i)`` factors), ``b_hist[i]`` counts
    partitions with k > i (the ``log(alpha + i*discount)`` factors), and ``d_hist[l]`` counts blocks with
    size > l (the ``log(l - discount)`` factors).
    """

    def __init__(self, keys: str | None = None) -> None:
        self.a_hist: dict[int, float] = {}
        self.b_hist: dict[int, float] = {}
        self.d_hist: dict[int, float] = {}
        self.count = 0.0
        self.keys = keys

    def _accumulate(self, sizes: np.ndarray, weight: float) -> None:
        n = int(sizes.sum())
        k = int(len(sizes))
        for i in range(1, n):
            self.a_hist[i] = self.a_hist.get(i, 0.0) + weight
        for i in range(1, k):
            self.b_hist[i] = self.b_hist.get(i, 0.0) + weight
        for nj in sizes:
            for l in range(1, int(nj)):
                self.d_hist[l] = self.d_hist.get(l, 0.0) + weight
        self.count += weight

    def update(self, x: Sequence[int], weight: float, estimate: PitmanYorProcessDistribution | None) -> None:
        """Accumulate exact EPPF histograms for one partition."""
        self._accumulate(_block_sizes(x), weight)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted partition."""
        self.update(x, weight, None)

    def seq_update(
        self, x: Sequence[np.ndarray], weights: np.ndarray, estimate: PitmanYorProcessDistribution | None
    ) -> None:
        """Accumulate exact EPPF histograms from encoded block-size arrays."""
        for sizes, w in zip(x, weights):
            self._accumulate(sizes, float(w))

    def seq_initialize(self, x: Sequence[np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded partitions."""
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: tuple[float, dict[int, float], dict[int, float], dict[int, float]]
    ) -> "PitmanYorProcessAccumulator":
        """Merge serialized Pitman-Yor histogram statistics into this accumulator."""
        self.count += suff_stat[0]
        _merge_hist(self.a_hist, suff_stat[1])
        _merge_hist(self.b_hist, suff_stat[2])
        _merge_hist(self.d_hist, suff_stat[3])
        return self

    def value(self) -> tuple[float, dict[int, float], dict[int, float], dict[int, float]]:
        """Return the observation weight and exact EPPF histogram statistics."""
        return self.count, dict(self.a_hist), dict(self.b_hist), dict(self.d_hist)

    def from_value(
        self, x: tuple[float, dict[int, float], dict[int, float], dict[int, float]]
    ) -> "PitmanYorProcessAccumulator":
        """Restore the accumulator from serialized histogram statistics."""
        self.count, self.a_hist, self.b_hist, self.d_hist = x[0], dict(x[1]), dict(x[2]), dict(x[3])
        return self

    def acc_to_encoder(self) -> "PitmanYorProcessDataEncoder":
        """Return an encoder that converts partitions to block-size arrays."""
        return PitmanYorProcessDataEncoder()


class PitmanYorProcessAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for PitmanYorProcessAccumulator."""

    def __init__(self, keys: str | None = None) -> None:
        self.keys = keys

    def make(self) -> PitmanYorProcessAccumulator:
        """Create an empty Pitman-Yor process accumulator."""
        return PitmanYorProcessAccumulator(keys=self.keys)


class PitmanYorProcessEstimator(ParameterEstimator):
    """Maximum-likelihood estimator for the Pitman-Yor concentration alpha and (optionally) discount."""

    def __init__(
        self,
        discount: float = 0.0,
        estimate_discount: bool = False,
        max_alpha: float = 1.0e6,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if not (0.0 <= discount < 1.0):
            raise ValueError("PitmanYorProcessEstimator requires discount in [0, 1).")
        self.discount = float(discount)
        self.estimate_discount = bool(estimate_discount)
        self.max_alpha = float(max_alpha)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> PitmanYorProcessAccumulatorFactory:
        """Return a factory for Pitman-Yor sufficient-statistic accumulators."""
        return PitmanYorProcessAccumulatorFactory(keys=self.keys)

    @staticmethod
    def _grad_alpha(a: float, d: float, a_hist, b_hist, d_hist) -> float:
        g = 0.0
        for i, w in b_hist.items():
            g += w / (a + i * d)
        for i, w in a_hist.items():
            g -= w / (a + i)
        return g

    @staticmethod
    def _grad_discount(a: float, d: float, a_hist, b_hist, d_hist) -> float:
        g = 0.0
        for i, w in b_hist.items():
            g += w * i / (a + i * d)
        for l, w in d_hist.items():
            g -= w / (l - d)
        return g

    def _solve_alpha(self, d: float, a_hist, b_hist, d_hist) -> float:
        lo, hi = 1.0e-9, 1.0
        # grad_alpha is strictly decreasing in alpha; grow hi until the gradient turns non-positive.
        while self._grad_alpha(hi, d, a_hist, b_hist, d_hist) > 0.0 and hi < self.max_alpha:
            hi *= 2.0
        if self._grad_alpha(lo, d, a_hist, b_hist, d_hist) <= 0.0:
            return lo
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if self._grad_alpha(mid, d, a_hist, b_hist, d_hist) > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _solve_discount(self, a: float, a_hist, b_hist, d_hist) -> float:
        lo, hi = 0.0, 1.0 - 1.0e-9
        if self._grad_discount(a, lo, a_hist, b_hist, d_hist) <= 0.0:
            return 0.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if self._grad_discount(a, mid, a_hist, b_hist, d_hist) > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, dict, dict, dict]) -> PitmanYorProcessDistribution:
        """Estimate the concentration and optional discount from EPPF histograms."""
        count, a_hist, b_hist, d_hist = suff_stat
        if count <= 0.0:
            return PitmanYorProcessDistribution(1.0, self.discount, name=self.name, keys=self.keys)

        d = self.discount
        a = self._solve_alpha(d, a_hist, b_hist, d_hist)
        if self.estimate_discount:
            for _ in range(50):
                d_new = self._solve_discount(a, a_hist, b_hist, d_hist)
                a_new = self._solve_alpha(d_new, a_hist, b_hist, d_hist)
                if abs(d_new - d) < 1.0e-9 and abs(a_new - a) < 1.0e-9:
                    a, d = a_new, d_new
                    break
                a, d = a_new, d_new
        return PitmanYorProcessDistribution(a, d, name=self.name, keys=self.keys)


class PitmanYorProcessDataEncoder(DataSequenceEncoder):
    """Encode a sequence of partitions (cluster-label vectors) into per-observation block-size arrays."""

    def __str__(self) -> str:
        return "PitmanYorProcessDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PitmanYorProcessDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """Encode cluster-label vectors as sorted block-size arrays."""
        return [_block_sizes(row) for row in x]
