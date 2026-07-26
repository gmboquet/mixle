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
import operator
from collections.abc import Sequence
from typing import Any, NamedTuple

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


class ChineseRestaurantProcessStatistics(NamedTuple):
    """Versioned CRP block-count statistics bound to partition size."""

    sum_k: float
    count: float
    n: int

    @property
    def schema_version(self) -> int:
        """Return the serialized-statistic schema version."""
        return 1


def _exact_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an exact integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an exact integer") from exc


def _finite_nonnegative_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError(f"{label} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a real scalar") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _canonical_partition(
    labels: Any,
    *,
    n: int,
    fail_closed: bool,
) -> np.ndarray | None:
    try:
        array = np.asarray(labels, dtype=object)
    except (TypeError, ValueError) as exc:
        if fail_closed:
            raise ValueError("CRP labels must form a sequence") from exc
        return None
    if array.ndim != 1 or array.size != n:
        if fail_closed:
            raise ValueError(
                f"CRP partition must be a one-dimensional length-{n} label vector"
            )
        return None
    representatives: list[Any] = []
    canonical = np.empty(n, dtype=np.int64)
    for index, label in enumerate(array.tolist()):
        try:
            reflexive = label == label
            if not isinstance(reflexive, (bool, np.bool_)) or not reflexive:
                raise ValueError
            match = None
            for block, representative in enumerate(representatives):
                equal = label == representative
                if not isinstance(equal, (bool, np.bool_)):
                    raise ValueError
                if equal:
                    match = block
                    break
        except (TypeError, ValueError):
            if fail_closed:
                raise ValueError(
                    "CRP labels must have scalar, reflexive equality"
                ) from None
            return None
        if match is None:
            match = len(representatives)
            representatives.append(label)
        canonical[index] = match
    canonical.setflags(write=False)
    return canonical


def _block_sizes(labels: np.ndarray) -> np.ndarray:
    """Return block sizes from a canonical non-negative label vector."""
    return np.bincount(labels)


def _validated_weights(value: Any, rows: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("CRP weights must be numeric") from exc
    if weights.shape != (rows,):
        raise ValueError(f"CRP weights must have exact shape ({rows},)")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("CRP weights must be finite and non-negative")
    return weights


def _validated_statistics(
    value: Any,
    *,
    n: int,
) -> ChineseRestaurantProcessStatistics:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("CRP statistics must be (sum_k, count, n)")
    sum_k = _finite_nonnegative_scalar(value[0], label="CRP block total")
    count = _finite_nonnegative_scalar(value[1], label="CRP count")
    statistic_n = _exact_integer(value[2], label="CRP statistic size")
    if statistic_n != n:
        raise ValueError("CRP statistic size does not match estimator")
    if count == 0.0:
        if sum_k != 0.0:
            raise ValueError("CRP zero-count statistics must have zero blocks")
    elif sum_k < count or sum_k > n * count:
        raise ValueError(
            "CRP block total must lie between count and n * count"
        )
    return ChineseRestaurantProcessStatistics(sum_k, count, n)


class ChineseRestaurantProcessDistribution(SequenceEncodableProbabilityDistribution):
    """CRP distribution over partitions of ``n`` items with concentration ``alpha > 0``."""

    def __init__(self, alpha: float, n: int, name: str | None = None, keys: str | None = None) -> None:
        checked_n = _exact_integer(n, label="CRP partition size")
        if alpha <= 0.0 or not np.isfinite(alpha) or checked_n <= 0:
            raise ValueError("ChineseRestaurantProcessDistribution requires alpha > 0 and n >= 1.")
        self.alpha = float(alpha)
        self.n = checked_n
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
        labels = _canonical_partition(
            x,
            n=self.n,
            fail_closed=False,
        )
        if labels is None:
            return -np.inf
        sizes = _block_sizes(labels)
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
        if pseudo_count is not None:
            checked_pseudo = _finite_nonnegative_scalar(
                pseudo_count,
                label="CRP pseudo-count",
            )
            if checked_pseudo != 0.0:
                raise NotImplementedError(
                    "CRP does not define an implicit pseudo-count prior; "
                    "fit an explicit prior over alpha instead"
                )
        return ChineseRestaurantProcessEstimator(self.n, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "ChineseRestaurantProcessDataEncoder":
        """Return the data encoder (passes label vectors through)."""
        return ChineseRestaurantProcessDataEncoder(self.n)


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
        checked_size = _exact_integer(size, label="CRP sample size")
        if checked_size < 0:
            raise ValueError("CRP sample size must be non-negative")
        return [self._one() for _ in range(checked_size)]


class ChineseRestaurantProcessAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the total number of blocks and observation count (the CRP sufficient statistics)."""

    def __init__(self, n: int, name: str | None = None, keys: str | None = None) -> None:
        self.n = _exact_integer(n, label="CRP partition size")
        if self.n <= 0:
            raise ValueError("CRP partition size must be strictly positive")
        self.sum_k = 0.0  # total number of blocks across observations
        self.count = 0.0
        self.name = name
        self.keys = keys

    def update(self, x: np.ndarray, weight: float, estimate: Any) -> None:
        """Accumulate the weighted block count for one partition."""
        labels = _canonical_partition(x, n=self.n, fail_closed=True)
        if labels is None:  # pragma: no cover
            raise AssertionError("unreachable")
        checked_weight = _finite_nonnegative_scalar(
            weight,
            label="CRP weight",
        )
        self.sum_k += checked_weight * _block_sizes(labels).shape[0]
        self.count += checked_weight

    def initialize(self, x: np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted partition."""
        self.update(x, weight, None)

    def seq_update(self, x: list[np.ndarray], weights: np.ndarray, estimate: Any) -> None:
        """Accumulate weighted block counts for encoded partition label vectors."""
        rows = list(x)
        checked_weights = _validated_weights(weights, len(rows))
        for labels, weight in zip(rows, checked_weights):
            self.update(labels, float(weight), estimate)

    def seq_initialize(self, x: list[np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded partitions."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Any) -> "ChineseRestaurantProcessAccumulator":
        """Merge serialized block-count statistics into this accumulator."""
        checked = _validated_statistics(suff_stat, n=self.n)
        self.sum_k += checked.sum_k
        self.count += checked.count
        return self

    def value(self) -> ChineseRestaurantProcessStatistics:
        """Return the total weighted block count and observation weight."""
        return ChineseRestaurantProcessStatistics(
            self.sum_k,
            self.count,
            self.n,
        )

    def from_value(self, x: Any) -> "ChineseRestaurantProcessAccumulator":
        """Restore the accumulator from serialized block-count statistics."""
        checked = _validated_statistics(x, n=self.n)
        self.sum_k, self.count = checked.sum_k, checked.count
        return self

    def scale(self, c: float) -> "ChineseRestaurantProcessAccumulator":
        """Scale all weight-linear statistics."""
        checked = _finite_nonnegative_scalar(c, label="CRP scale")
        self.sum_k *= checked
        self.count *= checked
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge statistics under the configured shared key."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace statistics from the configured shared key."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> "ChineseRestaurantProcessDataEncoder":
        """Return an encoder for CRP partition label vectors."""
        return ChineseRestaurantProcessDataEncoder(self.n)


class ChineseRestaurantProcessAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for ChineseRestaurantProcessAccumulator."""

    def __init__(self, n: int, name: str | None = None, keys: str | None = None) -> None:
        self.n = n
        self.name = name
        self.keys = keys

    def make(self) -> ChineseRestaurantProcessAccumulator:
        """Create an empty CRP accumulator."""
        return ChineseRestaurantProcessAccumulator(
            self.n,
            name=self.name,
            keys=self.keys,
        )


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
        self.n = _exact_integer(n, label="CRP partition size")
        if self.n <= 0:
            raise ValueError("CRP partition size must be strictly positive")
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        if (
            not np.isfinite(self.alpha_min)
            or not np.isfinite(self.alpha_max)
            or self.alpha_min <= 0.0
            or self.alpha_min >= self.alpha_max
        ):
            raise ValueError(
                "CRP alpha bounds must be finite and satisfy 0 < min < max"
            )
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> ChineseRestaurantProcessAccumulatorFactory:
        """Return a factory for CRP sufficient-statistic accumulators."""
        return ChineseRestaurantProcessAccumulatorFactory(
            self.n,
            name=self.name,
            keys=self.keys,
        )

    def _expected_blocks(self, alpha: float) -> float:
        return alpha * float(digamma(alpha + self.n) - digamma(alpha))

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> ChineseRestaurantProcessDistribution:
        """Estimate the CRP concentration from the observed mean block count."""
        checked = _validated_statistics(suff_stat, n=self.n)
        if checked.count <= 0.0:
            raise ValueError("CRP fitting requires positive observation weight")
        mean_k = checked.sum_k / checked.count
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

    def __init__(self, n: int) -> None:
        self.n = _exact_integer(n, label="CRP partition size")
        if self.n <= 0:
            raise ValueError("CRP partition size must be strictly positive")

    def __str__(self) -> str:
        return "ChineseRestaurantProcessDataEncoder(%s)" % repr(self.n)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ChineseRestaurantProcessDataEncoder)
            and self.n == other.n
        )

    def seq_encode(self, x: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Encode partition label vectors as integer arrays without relabeling them."""
        result = []
        for labels in x:
            canonical = _canonical_partition(
                labels,
                n=self.n,
                fail_closed=True,
            )
            if canonical is None:  # pragma: no cover
                raise AssertionError("unreachable")
            result.append(canonical)
        return result

    def row_count(self, x: Any) -> int:
        """Return the number of validated encoded partitions."""
        rows = list(x)
        self.seq_encode(rows)
        return len(rows)
