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
import operator
from collections.abc import Mapping, Sequence
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
from mixle.utils.special import gammaln

_PY_ALPHA_FLOOR = 1.0e-12
_PY_DISCOUNT_CEILING = 1.0 - 1.0e-9
_PY_SOLVER_TOLERANCE = 1.0e-9


class PitmanYorConvergenceError(RuntimeError):
    """Raised when a validated Pitman-Yor fit cannot establish an optimum."""


def _block_sizes(labels: Sequence[Any]) -> np.ndarray:
    """Return descending block sizes without changing cluster-label identity."""
    if isinstance(labels, (str, bytes)):
        raise ValueError("Pitman-Yor partitions must be sequences of hashable labels.")
    try:
        label_values = list(labels)
    except TypeError as exc:
        raise ValueError("Pitman-Yor partitions must be sequences of hashable labels.") from exc
    counts: dict[Any, int] = {}
    for label in label_values:
        try:
            hash(label)
        except TypeError as exc:
            raise ValueError("Pitman-Yor cluster labels must be hashable.") from exc
        try:
            reflexive = label == label
            if isinstance(reflexive, np.ndarray) or not bool(reflexive):
                raise ValueError("Pitman-Yor cluster labels must have reflexive scalar equality.")
        except (TypeError, ValueError) as exc:
            raise ValueError("Pitman-Yor cluster labels must have reflexive scalar equality.") from exc
        counts[label] = counts.get(label, 0) + 1
    return np.asarray(sorted(counts.values(), reverse=True), dtype=np.int64)


def _validated_block_sizes(value: Any) -> np.ndarray:
    """Return a canonical positive-integer block-size vector."""
    if isinstance(value, (str, bytes)):
        raise ValueError("Encoded Pitman-Yor partitions must be one-dimensional block-size sequences.")
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("Encoded Pitman-Yor partitions must be array-like.") from exc
    if raw.ndim != 1:
        raise ValueError("Encoded Pitman-Yor partitions must be one-dimensional.")
    sizes = np.empty(raw.size, dtype=np.int64)
    for index, item in enumerate(raw):
        if isinstance(item, (bool, np.bool_)):
            raise ValueError("Pitman-Yor block sizes must be positive integers.")
        try:
            size = operator.index(item)
        except TypeError as exc:
            raise ValueError("Pitman-Yor block sizes must be positive integers.") from exc
        if size <= 0:
            raise ValueError("Pitman-Yor block sizes must be positive integers.")
        sizes[index] = size
    if sizes.size > 1 and np.any(sizes[:-1] < sizes[1:]):
        raise ValueError("Encoded Pitman-Yor block sizes must be sorted in non-increasing order.")
    return sizes


def _validated_encoded_partitions(value: Any) -> list[np.ndarray]:
    if isinstance(value, (str, bytes)):
        raise ValueError("Encoded Pitman-Yor data must be a sequence of block-size vectors.")
    try:
        partitions = list(value)
    except TypeError as exc:
        raise ValueError(
            "Encoded Pitman-Yor data must be a sequence of block-size vectors."
        ) from exc
    return [_validated_block_sizes(sizes) for sizes in partitions]


def _validated_weight(value: Any, name: str = "weight") -> float:
    if np.ndim(value) != 0:
        raise ValueError("Pitman-Yor %s must be a finite non-negative scalar." % name)
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Pitman-Yor %s must be a finite non-negative scalar." % name) from exc
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("Pitman-Yor %s must be a finite non-negative scalar." % name)
    return weight


def _validated_weights(value: Any, count: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Pitman-Yor weights must be numeric.") from exc
    if weights.shape != (count,):
        raise ValueError("Pitman-Yor weights must have exact shape (%d,)." % count)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Pitman-Yor weights must be finite and non-negative.")
    return weights


def _validated_histogram(value: Any, name: str) -> dict[int, float]:
    if not isinstance(value, Mapping):
        raise ValueError("Pitman-Yor %s histogram must be a mapping." % name)
    result: dict[int, float] = {}
    for raw_index, raw_weight in value.items():
        if isinstance(raw_index, (bool, np.bool_)):
            raise ValueError("Pitman-Yor histogram indices must be positive integers.")
        try:
            index = operator.index(raw_index)
        except TypeError as exc:
            raise ValueError("Pitman-Yor histogram indices must be positive integers.") from exc
        if index <= 0:
            raise ValueError("Pitman-Yor histogram indices must be positive integers.")
        result[int(index)] = _validated_weight(raw_weight, "%s histogram weight" % name)
    positive_support = sorted(index for index, weight in result.items() if weight > 0.0)
    if positive_support and positive_support != list(range(1, positive_support[-1] + 1)):
        raise ValueError("Pitman-Yor %s histogram positive support must be contiguous from one." % name)
    for left, right in zip(positive_support, positive_support[1:]):
        if result[right] > result[left] + _PY_SOLVER_TOLERANCE * max(1.0, result[left]):
            raise ValueError("Pitman-Yor %s histogram weights must be non-increasing." % name)
    return result


def _validated_statistics(
    value: Any,
) -> tuple[float, dict[int, float], dict[int, float], dict[int, float]]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("Pitman-Yor sufficient statistics must be a four-item tuple.")
    count = _validated_weight(value[0], "sufficient-statistic count")
    a_hist = _validated_histogram(value[1], "element")
    b_hist = _validated_histogram(value[2], "block")
    d_hist = _validated_histogram(value[3], "block-size")
    scale = max(
        1.0,
        count,
        sum(a_hist.values()),
        sum(b_hist.values()),
        sum(d_hist.values()),
    )
    tolerance = _PY_SOLVER_TOLERANCE * scale
    if any(weight > count + tolerance for weight in a_hist.values()):
        raise ValueError("Pitman-Yor element-histogram weights cannot exceed total observation weight.")
    if any(weight > count + tolerance for weight in b_hist.values()):
        raise ValueError("Pitman-Yor block-histogram weights cannot exceed total observation weight.")
    for index, weight in b_hist.items():
        if weight > a_hist.get(index, 0.0) + tolerance:
            raise ValueError("Pitman-Yor block histograms are incompatible with element histograms.")
    if abs(sum(a_hist.values()) - sum(b_hist.values()) - sum(d_hist.values())) > tolerance:
        raise ValueError("Pitman-Yor histograms violate the partition-size conservation identity.")
    return count, a_hist, b_hist, d_hist


def _merge_hist(dst: dict[int, float], src: Mapping[int, float]) -> None:
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
        if np.ndim(discount) != 0 or not (0.0 <= discount < 1.0) or not np.isfinite(discount):
            raise ValueError("PitmanYorProcessDistribution requires discount in [0, 1).")
        if np.ndim(alpha) != 0 or alpha <= -discount or not np.isfinite(alpha):
            raise ValueError("PitmanYorProcessDistribution requires alpha > -discount.")
        if num_elements is not None:
            if isinstance(num_elements, (bool, np.bool_)):
                raise TypeError("PitmanYorProcessDistribution num_elements must be a positive integer.")
            try:
                num_elements = operator.index(num_elements)
            except TypeError as exc:
                raise TypeError(
                    "PitmanYorProcessDistribution num_elements must be a positive integer."
                ) from exc
            if num_elements <= 0:
                raise ValueError("PitmanYorProcessDistribution num_elements must be positive.")
        self.alpha = float(alpha)
        self.discount = float(discount)
        self.num_elements = None if num_elements is None else int(num_elements)
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

    def log_density(self, x: Sequence[Any]) -> float:
        """Return the log-probability of a partition (cluster-label vector) x."""
        return self._log_eppf(_block_sizes(x))

    def seq_log_density(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Return vectorized log-probabilities for a sequence of block-size arrays."""
        return np.asarray(
            [self._log_eppf(sizes) for sizes in _validated_encoded_partitions(x)],
            dtype=float,
        )

    def sampler(self, seed: int | None = None) -> "PitmanYorProcessSampler":
        """Return a sampler for drawing partitions from this distribution."""
        return PitmanYorProcessSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "PitmanYorProcessEstimator":
        """Return an estimator that fits alpha (and optionally the discount)."""
        if pseudo_count is not None:
            raise ValueError("Pitman-Yor pseudo-count regularization is not implemented.")
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
        if n is None:
            raise ValueError("PitmanYorProcessSampler requires the distribution's num_elements to be set (>= 1).")
        if size is None:
            return self._sample_one(n)
        if isinstance(size, (bool, np.bool_)):
            raise TypeError("Pitman-Yor sample size must be a non-negative integer.")
        try:
            sample_count = operator.index(size)
        except TypeError as exc:
            raise TypeError("Pitman-Yor sample size must be a non-negative integer.") from exc
        if sample_count < 0:
            raise ValueError("Pitman-Yor sample size must be non-negative.")
        return [self._sample_one(n) for _ in range(sample_count)]


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
        sizes = _validated_block_sizes(sizes)
        weight = _validated_weight(weight)
        if weight == 0.0:
            return
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

    def update(self, x: Sequence[Any], weight: float, estimate: PitmanYorProcessDistribution | None) -> None:
        """Accumulate exact EPPF histograms for one partition."""
        self._accumulate(_block_sizes(x), weight)

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted partition."""
        self.update(x, weight, None)

    def seq_update(
        self, x: Sequence[np.ndarray], weights: np.ndarray, estimate: PitmanYorProcessDistribution | None
    ) -> None:
        """Accumulate exact EPPF histograms from encoded block-size arrays."""
        partitions = _validated_encoded_partitions(x)
        checked_weights = _validated_weights(weights, len(partitions))
        for sizes, weight in zip(partitions, checked_weights):
            self._accumulate(sizes, weight)

    def seq_initialize(self, x: Sequence[np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded partitions."""
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: tuple[float, dict[int, float], dict[int, float], dict[int, float]]
    ) -> "PitmanYorProcessAccumulator":
        """Merge serialized Pitman-Yor histogram statistics into this accumulator."""
        count, a_hist, b_hist, d_hist = _validated_statistics(suff_stat)
        combined = (
            self.count + count,
            {**self.a_hist},
            {**self.b_hist},
            {**self.d_hist},
        )
        _merge_hist(combined[1], a_hist)
        _merge_hist(combined[2], b_hist)
        _merge_hist(combined[3], d_hist)
        checked_count, checked_a, checked_b, checked_d = _validated_statistics(combined)
        self.count = checked_count
        self.a_hist = checked_a
        self.b_hist = checked_b
        self.d_hist = checked_d
        return self

    def value(self) -> tuple[float, dict[int, float], dict[int, float], dict[int, float]]:
        """Return the observation weight and exact EPPF histogram statistics."""
        return self.count, dict(self.a_hist), dict(self.b_hist), dict(self.d_hist)

    def from_value(
        self, x: tuple[float, dict[int, float], dict[int, float], dict[int, float]]
    ) -> "PitmanYorProcessAccumulator":
        """Restore the accumulator from serialized histogram statistics."""
        self.count, self.a_hist, self.b_hist, self.d_hist = _validated_statistics(x)
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
        if np.ndim(discount) != 0 or not np.isfinite(discount) or not (0.0 <= discount < 1.0):
            raise ValueError("PitmanYorProcessEstimator requires discount in [0, 1).")
        if not isinstance(estimate_discount, (bool, np.bool_)):
            raise TypeError("PitmanYorProcessEstimator estimate_discount must be bool.")
        if np.ndim(max_alpha) != 0:
            raise ValueError("PitmanYorProcessEstimator max_alpha must be a finite positive scalar.")
        try:
            checked_max_alpha = float(max_alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "PitmanYorProcessEstimator max_alpha must be a finite positive scalar."
            ) from exc
        if not np.isfinite(checked_max_alpha) or checked_max_alpha <= 0.0:
            raise ValueError("PitmanYorProcessEstimator max_alpha must be a finite positive scalar.")
        self.discount = float(discount)
        self.estimate_discount = bool(estimate_discount)
        self.max_alpha = checked_max_alpha
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

    @staticmethod
    def _objective(a: float, d: float, a_hist, b_hist, d_hist) -> float:
        result = 0.0
        for index, weight in b_hist.items():
            argument = a + index * d
            if argument <= 0.0:
                return -np.inf
            result += weight * math.log(argument)
        for index, weight in a_hist.items():
            argument = a + index
            if argument <= 0.0:
                return -np.inf
            result -= weight * math.log(argument)
        for index, weight in d_hist.items():
            argument = index - d
            if argument <= 0.0:
                return -np.inf
            result += weight * math.log(argument)
        return float(result)

    def _solve_alpha(self, d: float, a_hist, b_hist, d_hist) -> tuple[float, dict[str, Any]]:
        lo = min(_PY_ALPHA_FLOOR, self.max_alpha * 1.0e-6)
        hi = min(1.0, self.max_alpha)
        grad_lo = self._grad_alpha(lo, d, a_hist, b_hist, d_hist)
        grad_hi = self._grad_alpha(hi, d, a_hist, b_hist, d_hist)
        if not np.isfinite(grad_lo) or not np.isfinite(grad_hi):
            raise PitmanYorConvergenceError("Pitman-Yor alpha solver produced a non-finite bracket gradient.")
        while grad_hi > 0.0 and hi < self.max_alpha:
            hi = min(self.max_alpha, hi * 2.0)
            grad_hi = self._grad_alpha(hi, d, a_hist, b_hist, d_hist)
            if not np.isfinite(grad_hi):
                raise PitmanYorConvergenceError(
                    "Pitman-Yor alpha solver produced a non-finite bracket gradient."
                )
        if grad_lo <= 0.0:
            return lo, {
                "converged": True,
                "boundary": "lower",
                "iterations": 0,
                "gradient": float(grad_lo),
            }
        if grad_hi >= 0.0:
            return hi, {
                "converged": True,
                "boundary": "upper",
                "iterations": 0,
                "gradient": float(grad_hi),
            }
        iterations = 0
        for iterations in range(1, 101):
            mid = 0.5 * (lo + hi)
            gradient = self._grad_alpha(mid, d, a_hist, b_hist, d_hist)
            if not np.isfinite(gradient):
                raise PitmanYorConvergenceError("Pitman-Yor alpha solver produced a non-finite gradient.")
            if gradient > 0.0:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) <= _PY_SOLVER_TOLERANCE * max(1.0, mid):
                break
        result = 0.5 * (lo + hi)
        gradient = self._grad_alpha(result, d, a_hist, b_hist, d_hist)
        if not np.isfinite(result) or not np.isfinite(gradient) or not (0.0 < result <= self.max_alpha):
            raise PitmanYorConvergenceError("Pitman-Yor alpha solver returned an invalid optimum.")
        return result, {
            "converged": True,
            "boundary": None,
            "iterations": iterations,
            "gradient": float(gradient),
        }

    def _solve_discount(self, a: float, a_hist, b_hist, d_hist) -> tuple[float, dict[str, Any]]:
        lo, hi = 0.0, _PY_DISCOUNT_CEILING
        grad_lo = self._grad_discount(a, lo, a_hist, b_hist, d_hist)
        grad_hi = self._grad_discount(a, hi, a_hist, b_hist, d_hist)
        if not np.isfinite(grad_lo) or not np.isfinite(grad_hi):
            raise PitmanYorConvergenceError(
                "Pitman-Yor discount solver produced a non-finite bracket gradient."
            )
        if grad_lo <= 0.0:
            return lo, {
                "converged": True,
                "boundary": "lower",
                "iterations": 0,
                "gradient": float(grad_lo),
            }
        if grad_hi >= 0.0:
            return hi, {
                "converged": True,
                "boundary": "upper",
                "iterations": 0,
                "gradient": float(grad_hi),
            }
        iterations = 0
        for iterations in range(1, 101):
            mid = 0.5 * (lo + hi)
            gradient = self._grad_discount(a, mid, a_hist, b_hist, d_hist)
            if not np.isfinite(gradient):
                raise PitmanYorConvergenceError(
                    "Pitman-Yor discount solver produced a non-finite gradient."
                )
            if gradient > 0.0:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) <= _PY_SOLVER_TOLERANCE:
                break
        result = 0.5 * (lo + hi)
        gradient = self._grad_discount(a, result, a_hist, b_hist, d_hist)
        if not np.isfinite(result) or not np.isfinite(gradient) or not (0.0 <= result < 1.0):
            raise PitmanYorConvergenceError("Pitman-Yor discount solver returned an invalid optimum.")
        return result, {
            "converged": True,
            "boundary": None,
            "iterations": iterations,
            "gradient": float(gradient),
        }

    def estimate(self, nobs: float | None, suff_stat: tuple[float, dict, dict, dict]) -> PitmanYorProcessDistribution:
        """Estimate the concentration and optional discount from EPPF histograms."""
        count, a_hist, b_hist, d_hist = _validated_statistics(suff_stat)
        if not a_hist and not b_hist and not d_hist:
            result = PitmanYorProcessDistribution(
                min(1.0, self.max_alpha),
                self.discount,
                name=self.name,
                keys=self.keys,
            )
            result.fit_metadata = {
                "solver": "none",
                "converged": True,
                "iterations": 0,
                "boundary": {},
                "repairs": (),
                "reason": "statistics contain no parameter information",
                "observation_weight": count,
            }
            return result

        d = self.discount
        a, alpha_status = self._solve_alpha(d, a_hist, b_hist, d_hist)
        coordinate_iterations = 0
        discount_status: dict[str, Any] | None = None
        if self.estimate_discount:
            converged = False
            for coordinate_iterations in range(1, 51):
                d_new, discount_status = self._solve_discount(a, a_hist, b_hist, d_hist)
                a_new, alpha_status = self._solve_alpha(d_new, a_hist, b_hist, d_hist)
                if abs(d_new - d) < 1.0e-9 and abs(a_new - a) < 1.0e-9:
                    a, d = a_new, d_new
                    converged = True
                    break
                a, d = a_new, d_new
            if not converged:
                raise PitmanYorConvergenceError(
                    "Pitman-Yor coordinate solver did not converge within 50 iterations."
                )
        objective = self._objective(a, d, a_hist, b_hist, d_hist)
        if not np.isfinite(objective):
            raise PitmanYorConvergenceError("Pitman-Yor estimator returned a non-finite objective.")
        result = PitmanYorProcessDistribution(a, d, name=self.name, keys=self.keys)
        result.fit_metadata = {
            "solver": "coordinate_bisection" if self.estimate_discount else "alpha_bisection",
            "converged": True,
            "iterations": {
                "coordinate": coordinate_iterations,
                "alpha": alpha_status["iterations"],
                "discount": 0 if discount_status is None else discount_status["iterations"],
            },
            "boundary": {
                "alpha": alpha_status["boundary"],
                "discount": None if discount_status is None else discount_status["boundary"],
            },
            "gradient": {
                "alpha": alpha_status["gradient"],
                "discount": None if discount_status is None else discount_status["gradient"],
            },
            "objective": objective,
            "repairs": (),
            "observation_weight": count,
        }
        return result


class PitmanYorProcessDataEncoder(DataSequenceEncoder):
    """Encode a sequence of partitions (cluster-label vectors) into per-observation block-size arrays."""

    def __str__(self) -> str:
        return "PitmanYorProcessDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PitmanYorProcessDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """Encode cluster-label vectors as sorted block-size arrays."""
        if isinstance(x, (str, bytes)):
            raise ValueError("Pitman-Yor encoder input must be a sequence of partitions.")
        try:
            partitions = list(x)
        except TypeError as exc:
            raise ValueError("Pitman-Yor encoder input must be a sequence of partitions.") from exc
        return [_block_sizes(row) for row in partitions]

    def row_count(self, x: Any) -> int:
        """Return the number of encoded partitions after validating their geometry."""
        return len(_validated_encoded_partitions(x))
