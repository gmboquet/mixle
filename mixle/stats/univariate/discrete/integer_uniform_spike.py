r"""Integer-valued uniform distributions with extra mass at one spike value.

Observations are integers in ``[min_val, max_val]``. The distribution assigns probability ``p`` to the
spike value ``k`` and spreads the remaining mass uniformly over the other values:

    P(x_i = k) = p,
    P(x_i = x) = (1-p)/(b-a), x in [a,b] \ {k},
    P(x_i = else) = 0.0.

"""

from typing import Any, Optional

import numpy as np
from numpy.random import RandomState

import mixle.utils.vector as vec
from mixle.engines.arithmetic import *
from mixle.enumeration.algorithms import QuantizedCrossIndex, QuantizedEnumerationIndex
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class IntegerUniformSpikeDistribution(SequenceEncodableProbabilityDistribution):
    """Uniform integer distribution with extra probability mass at one value."""

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for integer-uniform-spike generated kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic_table")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the integer uniform spike."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="integer_uniform_spike",
            distribution_type=cls,
            parameters=(
                ParameterSpec("k", constraint="integer", differentiable=False),
                ParameterSpec("num_vals", constraint="positive_integer", differentiable=False),
                ParameterSpec("p", constraint="unit_interval"),
                ParameterSpec("min_val", constraint="integer", differentiable=False),
            ),
            statistics=(
                StatisticSpec("min_val", kind="metadata", additive=False, scales=False),
                StatisticSpec("count_vec"),
            ),
            support="bounded_integer_spike",
            differentiable=False,
        )

    def __init__(self, k: int, num_vals: int, p: float, min_val: int | None = 0, name: str | None = None) -> None:
        """Create a uniform integer distribution with a spike at ``k``.

        Args:
            k (int): Integer value to place spike on. Must be within [min_val,min_val+num_vals)
            num_vals (int): Number of integers in the range.
            p (float): Probability of drawing k. (1-p)/(num_vals-1) to draw any other integer in range.
            min_val (Optional[int]): Defaults to 0. Set bottom of integer range.
            name (Optional[str]): Optional distribution name.

        Attributes:
            p (float): Probability of drawing from k.
            min_val (int): Lower bound for the range.
            max_val (int): Max value for the range.
            k (int): Integer to place the spike on.
            log_p (float): Log of p.
            log_1p (float): Log of 1-p
            num_vals (int): Total number of integers in range.
            name (Optional[str]): Optional distribution name.

        """
        self.p = p
        self.min_val = min_val
        self.max_val = min_val + num_vals - 1

        if not self.min_val <= k <= self.max_val:
            raise ValueError("Spike value k must be between [%s, %s]." % (repr(self.min_val), repr(self.max_val)))
        else:
            self.k = k

        self.log_p = np.log(p)
        self.num_vals = num_vals
        # With a single value there is no non-spike category, so the off-spike log-mass is
        # -inf (the spike carries all probability); avoids log(num_vals - 1) = log(0) = -inf
        # feeding a +inf into log_1p.
        if num_vals == 1:
            self.log_1p = -np.inf
        else:
            self.log_1p = np.log1p(-self.p) - np.log(self.num_vals - 1)
        self.name = name

    def __str__(self) -> str:
        s1 = str(self.min_val)
        s2 = str(self.num_vals)
        s3 = repr(self.p)
        s4 = repr(self.k)
        s5 = repr(self.name)

        return "IntegerUniformSpikeDistribution(p=%s, min_val=%s, num_vals=%s,k=%s, name=%s)" % (s3, s1, s2, s4, s5)

    def density(self, x: int) -> float:
        """Density of the integer uniform spike distribution at observation x.

        See log_density() for details.

        Args:
            x (int): Integer observation.

        Returns:
            Density at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        """Log-density of the integer uniform spike distribution at observation x.

        Returns log(p) if x equals the spike value k, log((1-p)/(num_vals-1)) for any
        other integer in [min_val, max_val], and -inf outside the range.

        Args:
            x (int): Integer observation.

        Returns:
            Log-density at observation x.

        """
        if self.max_val >= x >= self.min_val:
            return self.log_p if x == self.k else self.log_1p
        else:
            return -np.inf

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (np.ndarray): Numpy array of integer observations.

        Returns:
            Numpy array of log-density (float) of len(x).

        """

        rv = np.zeros(len(x), dtype=float)
        rv.fill(-np.inf)

        in_range = np.bitwise_and(x >= self.min_val, x <= self.max_val)
        in_range_k = x[in_range] == self.k

        rv1 = rv[in_range]
        rv1[in_range_k] = self.log_p
        rv1[~in_range_k] = self.log_1p
        rv[in_range] = rv1

        return rv

    def backend_seq_log_density(self, x: np.ndarray, engine: Any) -> Any:
        """Engine-neutral log-density for encoded integer spike observations."""
        xx = engine.asarray(x)
        in_range = (xx >= self.min_val) & (xx <= self.max_val)
        is_spike = xx == self.k
        return engine.where(
            in_range,
            engine.where(is_spike, engine.asarray(self.log_p), engine.asarray(self.log_1p)),
            engine.asarray(-np.inf),
        )

    @classmethod
    def backend_stacked_params(cls, dists: list["IntegerUniformSpikeDistribution"], engine: Any) -> dict[str, Any]:
        """Return stacked integer-uniform-spike parameters for a shared support."""
        min_val = int(dists[0].min_val)
        num_vals = int(dists[0].num_vals)
        if any(int(dist.min_val) != min_val or int(dist.num_vals) != num_vals for dist in dists):
            raise ValueError("Stacked IntegerUniformSpikeDistribution components require shared support.")
        return {
            "__pysp_component_axis__": {"k": 0, "log_p": 0, "log_1p": 0},
            "min_val": min_val,
            "max_val": min_val + num_vals - 1,
            "num_vals": num_vals,
            "k": engine.asarray(np.asarray([dist.k for dist in dists], dtype=np.int64)),
            "log_p": engine.asarray(np.asarray([dist.log_p for dist in dists], dtype=np.float64)),
            "log_1p": engine.asarray(np.asarray([dist.log_1p for dist in dists], dtype=np.float64)),
            "num_components": len(dists),
        }

    @classmethod
    def backend_stacked_log_density(cls, x: np.ndarray, params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of integer-uniform-spike log densities."""
        xx = engine.asarray(x)
        in_range = (xx >= params["min_val"]) & (xx <= params["max_val"])
        is_spike = xx[:, None] == params["k"][None, :]
        rv = engine.where(is_spike, params["log_p"][None, :], params["log_1p"][None, :])
        return engine.where(in_range[:, None], rv, engine.asarray(-np.inf))

    @classmethod
    def backend_stacked_sufficient_statistics(
        cls, x: np.ndarray, weights: Any, params: dict[str, Any], engine: Any
    ) -> tuple[Any, Any]:
        """Return component-stacked legacy ``(min_val, count_vec)`` statistics."""
        xx = engine.asarray(x)
        ww = engine.asarray(weights)
        rel = xx - engine.asarray(params["min_val"])
        rows = []
        zero_rows = ww * engine.asarray(0.0)
        for value_index in range(int(params["num_vals"])):
            mask = rel == engine.asarray(value_index)
            rows.append(engine.sum(engine.where(mask[:, None], ww, zero_rows), axis=0))
        count_mat = engine.stack(rows, axis=1)
        min_vals = engine.asarray(np.full(int(params["num_components"]), int(params["min_val"])))
        return min_vals, count_mat

    def sampler(self, seed: int | None = None) -> "IntegerUniformSpikeSampler":
        """Create an IntegerUniformSpikeSampler from parameters of this distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            IntegerUniformSpikeSampler object.

        """
        return IntegerUniformSpikeSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "IntegerUniformSpikeEstimator":
        """Create an IntegerUniformSpikeEstimator for the current integer range.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.

        Returns:
            IntegerUniformSpikeEstimator object.

        """
        if pseudo_count is None:
            return IntegerUniformSpikeEstimator(min_val=self.min_val, max_val=self.max_val, name=self.name)

        else:
            return IntegerUniformSpikeEstimator(
                min_val=self.min_val, max_val=self.max_val, pseudo_count=pseudo_count, name=self.name
            )

    def dist_to_encoder(self) -> "IntegerUniformSpikeDataEncoder":
        """Returns an IntegerUniformSpikeDataEncoder for encoding sequences of iid integer observations."""
        return IntegerUniformSpikeDataEncoder()

    def enumerator(self) -> "IntegerUniformSpikeEnumerator":
        """Returns an IntegerUniformSpikeEnumerator iterating the support in descending probability order."""
        return IntegerUniformSpikeEnumerator(self)

    def quantized_index(self, max_bits: float, bin_width_bits: float = 1.0) -> QuantizedEnumerationIndex:
        """Build a bounded bit-quantized index directly from the finite integer support."""
        items = []
        if self.p > 0.0:
            items.append((self.k, float(self.log_p)))
        if self.num_vals > 1 and self.log_1p > -np.inf:
            items.extend((v, float(self.log_1p)) for v in range(self.min_val, self.max_val + 1) if v != self.k)
        return QuantizedEnumerationIndex.from_items(items, max_bits=max_bits, bin_width_bits=bin_width_bits)

    def quantized_multi_cross_index(self, others, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view over finite integer spike supports."""
        dists = [self] + list(others)
        if any(not isinstance(dist, IntegerUniformSpikeDistribution) for dist in dists):
            return super().quantized_multi_cross_index(others, max_bits=max_bits, bin_width_bits=bin_width_bits)

        lo = min(dist.min_val for dist in dists)
        hi = max(dist.max_val for dist in dists)
        items = []
        for value in range(lo, hi + 1):
            items.append((value, tuple(float(dist.log_density(value)) for dist in dists)))
        return QuantizedCrossIndex.from_items(items, max_bits=max_bits, bin_width_bits=bin_width_bits)

    def quantized_cross_index(self, other, max_bits, bin_width_bits: float = 1.0) -> QuantizedCrossIndex:
        """Build an exact aligned cross-bin view over two integer spike supports."""
        return self.quantized_multi_cross_index([other], max_bits=max_bits, bin_width_bits=bin_width_bits)


class IntegerUniformSpikeEnumerator(DistributionEnumerator):
    """Enumerates the support [min_val, max_val] in descending probability order.

    The spike value k is yielded first when p >= (1-p)/(num_vals-1), otherwise last; the
    remaining values share the same probability and are yielded in ascending integer
    order. Zero-probability values are skipped.
    """

    def __init__(self, dist: IntegerUniformSpikeDistribution) -> None:
        """Create an enumerator for the finite support.

        Args:
            dist (IntegerUniformSpikeDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        spike = [(dist.k, float(dist.log_p))] if dist.p > 0.0 else []
        rest = []
        if dist.num_vals > 1 and dist.log_1p > -np.inf:
            rest = [(v, float(dist.log_1p)) for v in range(dist.min_val, dist.max_val + 1) if v != dist.k]
        if spike and rest and spike[0][1] < rest[0][1]:
            self._items = rest + spike
        else:
            self._items = spike + rest
        self._pos = 0

    def __next__(self) -> tuple[int, float]:
        if self._pos >= len(self._items):
            raise StopIteration
        item = self._items[self._pos]
        self._pos += 1
        return item


class IntegerUniformSpikeSampler(DistributionSampler):
    """Sampler for an integer-uniform-spike distribution.

    Attributes:
        dist (IntegerUniformSpikeDistribution): Distribution to sample from.
        rng (RandomState): Seeded RandomState for sampling.
        non_k (np.ndarray): Integers of the support excluding the spike value k.

    """

    def __init__(self, dist: "IntegerUniformSpikeDistribution", seed: int | None = None) -> None:
        """Create a sampler for an integer-uniform-spike distribution.

        Args:
            dist (IntegerUniformSpikeDistribution): Distribution to sample from.
            seed (Optional[int]): Seed to set for sampling with RandomState.

        """
        self.rng = RandomState(seed)
        self.dist = dist
        self.non_k = np.delete(np.arange(self.dist.min_val, self.dist.max_val + 1), self.dist.k - self.dist.min_val)

    def sample(self, size: int | None = None, *, batched: bool = True) -> int | np.ndarray:
        """Draw iid samples from the integer uniform spike distribution.

        Args:
            size (Optional[int]): Number of iid samples to draw.

        Returns:
            A single int if size is None, else a numpy array of ints with length size.

        """

        if size is None:
            z = self.rng.binomial(n=1, p=self.dist.p)
            if z == 1:
                return self.dist.k
            else:
                return self.rng.choice(self.non_k)
        else:
            rv = np.zeros(size, dtype=int)
            rv.fill(self.dist.k)
            z = self.rng.binomial(n=1, p=self.dist.p, size=size)
            idx = np.flatnonzero(z == 0)

            if len(idx) > 0:
                rv[idx] = self.rng.choice(self.non_k, replace=True, size=len(idx))

            return rv


class IntegerUniformSpikeAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for weighted integer counts over a growing range.

    Attributes:
        min_val (Optional[int]): Smallest integer observed (or configured) so far.
        max_val (Optional[int]): Largest integer observed (or configured) so far.
        count_vec (Optional[np.ndarray]): Weighted counts for each integer in [min_val, max_val].
        count (float): Total weighted observation count.
        key (Optional[str]): Key for merging sufficient statistics across accumulators.
        name (Optional[str]): Optional accumulator name.

    """

    def __init__(
        self, min_val: int | None, max_val: int | None, keys: str | None = None, name: str | None = None
    ) -> None:
        """Create an accumulator for integer-uniform-spike sufficient statistics.

        Args:
            min_val (Optional[int]): Smallest integer value in the range, if known.
            max_val (Optional[int]): Largest integer value in the range, if known.
            keys (Optional[str]): Set key for merging sufficient statistics.
            name (Optional[str]): Optional accumulator name.

        """
        self.min_val = min_val
        self.max_val = max_val

        if self.min_val is not None and self.max_val is not None:
            self.num_vals = self.max_val - self.min_val + 1
            self.count_vec = np.zeros(self.max_val - self.min_val + 1, dtype=float)
        else:
            self.count_vec = None

        self.count = 0.0
        self.keys = keys
        self.name = name

    def update(self, x: int, weight: float, estimate: Optional["IntegerUniformSpikeDistribution"]) -> None:
        """Add weight to the count for integer x, growing the count vector if x is out of range.

        Args:
            x (int): Integer observation.
            weight (float): Weight on the observation.
            estimate (Optional[IntegerUniformSpikeDistribution]): Unused previous estimate.

        """

        if self.count_vec is None:
            self.min_val = x
            self.max_val = x
            self.count_vec = np.asarray([weight])

        elif self.max_val < x:
            temp_vec = self.count_vec
            self.max_val = x
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[: len(temp_vec)] = temp_vec
            self.count_vec[x - self.min_val] += weight

        elif self.min_val > x:
            temp_vec = self.count_vec
            temp_diff = self.min_val - x
            self.min_val = x
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[temp_diff:] = temp_vec
            self.count_vec[x - self.min_val] += weight

        else:
            self.count_vec[x - self.min_val] += weight

    def initialize(self, x: int, weight: float, rng: RandomState) -> None:
        """Initialize the accumulator with observation x and weight (delegates to update)."""
        return self.update(x, weight, None)

    def seq_initialize(self, x: tuple[int, np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization from encoded observations x (delegates to seq_update)."""
        return self.seq_update(x, weights, None)

    def seq_update(
        self, x: np.ndarray, weights: np.ndarray, estimate: Optional["IntegerUniformSpikeDistribution"]
    ) -> None:
        """Vectorized accumulation of weighted counts from encoded observations x.

        Args:
            x (np.ndarray): Sequence encoded integer observations.
            weights (np.ndarray): Weights on the observations.
            estimate (Optional[IntegerUniformSpikeDistribution]): Unused previous estimate.

        """

        min_x = x.min()
        max_x = x.max()

        loc_cnt = np.bincount(x - min_x, weights=weights)

        if self.count_vec is None:
            self.count_vec = np.zeros(max_x - min_x + 1)
            self.min_val = min_x
            self.max_val = max_x

        if self.min_val > min_x or self.max_val < max_x:
            prev_min = self.min_val
            self.min_val = min(min_x, self.min_val)
            self.max_val = max(max_x, self.max_val)
            temp = self.count_vec
            prev_diff = prev_min - self.min_val
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[prev_diff : (prev_diff + len(temp))] = temp

        min_diff = min_x - self.min_val
        self.count_vec[min_diff : (min_diff + len(loc_cnt))] += loc_cnt

    def seq_update_engine(
        self, x: np.ndarray, weights: Any, estimate: Optional["IntegerUniformSpikeDistribution"], engine: Any
    ) -> None:
        """Engine-resident accumulation: the weighted value histogram is reduced on the active
        engine (numpy or torch); the dynamic support range is host bookkeeping. Matches seq_update.
        """
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        xv = np.asarray(x)
        min_x = int(xv.min())
        max_x = int(xv.max())

        idx = engine.asarray((xv - min_x).astype(np.int64))
        loc_cnt = np.asarray(
            engine.to_numpy(engine.bincount(idx, weights=engine.asarray(weights_np), minlength=max_x - min_x + 1)),
            dtype=np.float64,
        )

        if self.count_vec is None:
            self.count_vec = np.zeros(max_x - min_x + 1)
            self.min_val = min_x
            self.max_val = max_x

        if self.min_val > min_x or self.max_val < max_x:
            prev_min = self.min_val
            self.min_val = min(min_x, self.min_val)
            self.max_val = max(max_x, self.max_val)
            temp = self.count_vec
            prev_diff = prev_min - self.min_val
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[prev_diff : (prev_diff + len(temp))] = temp

        min_diff = min_x - self.min_val
        self.count_vec[min_diff : (min_diff + len(loc_cnt))] += loc_cnt

    def combine(self, suff_stat: tuple[int, np.ndarray]) -> "IntegerUniformSpikeAccumulator":
        """Combine sufficient statistics (min_val, count_vec) with this accumulator, aligning ranges.

        Args:
            suff_stat (Tuple[int, np.ndarray]): Minimum value and count vector of another accumulator.

        Returns:
            This IntegerUniformSpikeAccumulator.

        """
        if self.count_vec is None and suff_stat[1] is not None:
            self.min_val = suff_stat[0]
            self.max_val = suff_stat[0] + len(suff_stat[1]) - 1
            self.count_vec = suff_stat[1]

        elif self.count_vec is not None and suff_stat[1] is not None:
            if self.min_val == suff_stat[0] and len(self.count_vec) == len(suff_stat[1]):
                self.count_vec += suff_stat[1]

            else:
                min_val = min(self.min_val, suff_stat[0])
                max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

                count_vec = vec.zeros(max_val - min_val + 1)

                i0 = self.min_val - min_val
                i1 = self.max_val - min_val + 1
                count_vec[i0:i1] = self.count_vec

                i0 = suff_stat[0] - min_val
                i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
                count_vec[i0:i1] += suff_stat[1]

                self.min_val = min_val
                self.max_val = max_val
                self.count_vec = count_vec

        return self

    def value(self) -> tuple[int, np.ndarray]:
        """Returns sufficient statistics as a tuple (min_val, count_vec)."""
        return self.min_val, self.count_vec

    def from_value(self, x: tuple[int, np.ndarray]) -> "IntegerUniformSpikeAccumulator":
        """Set sufficient statistics from a (min_val, count_vec) tuple.

        Args:
            x (Tuple[int, np.ndarray]): Minimum value and count vector.

        Returns:
            This IntegerUniformSpikeAccumulator.

        """
        self.min_val = x[0]
        self.max_val = x[0] + len(x[1]) - 1
        self.count_vec = x[1]

        return self

    def scale(self, c: float) -> "IntegerUniformSpikeAccumulator":
        """Scale linear counts while preserving the integer support offset."""
        if self.count_vec is not None:
            self.count_vec *= c
        self.count *= c
        return self

    def acc_to_encoder(self) -> "IntegerUniformSpikeDataEncoder":
        """Returns an IntegerUniformSpikeDataEncoder for encoding sequences of iid integer observations."""
        return IntegerUniformSpikeDataEncoder()


class IntegerUniformSpikeAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for integer-uniform-spike accumulators.

    Args:
        min_val (Optional[int]): Smallest integer value in the range, if known.
        max_val (Optional[int]): Largest integer value in the range, if known.
        keys (Optional[str]): Set key for merging sufficient statistics.
        name (Optional[str]): Optional name assigned to created accumulators.

    Attributes:
        min_val (Optional[int]): Smallest integer value in the range, if known.
        max_val (Optional[int]): Largest integer value in the range, if known.
        keys (Optional[str]): Key for merging sufficient statistics.
        name (Optional[str]): Optional name assigned to created accumulators.

    """

    def __init__(
        self,
        min_val: int | None = None,
        max_val: int | None = None,
        keys: str | None = None,
        name: str | None = None,
    ) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.keys = keys
        self.name = name

    def make(self) -> "IntegerUniformSpikeAccumulator":
        """Return a fresh integer-uniform-spike accumulator."""
        return IntegerUniformSpikeAccumulator(
            min_val=self.min_val, max_val=self.max_val, keys=self.keys, name=self.name
        )


class IntegerUniformSpikeEstimator(ParameterEstimator):
    """Estimator for integer-uniform-spike distributions from weighted counts."""

    def __init__(
        self,
        min_val: int | None = None,
        max_val: int | None = None,
        pseudo_count: float | None = None,
        suff_stat: tuple[int, float | None] | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Estimator for an integer-uniform-spike distribution.

        Args:
            min_val (Optional[int]): Smallest integer value in the range.
            pseudo_count (Optional[float]): Regularize value k.
            suff_stat (Optional[Tuple[int, Optional[float]]]): Tuple of k to regularize and optional value of p for k.
            name (Optional[str]): Optional name assigned to the estimated distribution.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            pseudo_count (Optional[float]): Regularize value k.
            min_val (int): Smallest integer value in the range. Defaults to 0.
            max_val (int): Set to the min val plus number of values - 1.
            suff_stat (Optional[Tuple[int, Optional[float]]]): Tuple of k to regularize and optional value of p for k.
            name (Optional[str]): Optional name assigned to the estimated distribution.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        """
        self.pseudo_count = pseudo_count
        self.min_val = min_val
        self.max_val = max_val
        self.suff_stat = suff_stat if suff_stat is not None else (None, None)
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> "IntegerUniformSpikeAccumulatorFactory":
        """Returns an IntegerUniformSpikeAccumulatorFactory consistent with this estimator."""
        return IntegerUniformSpikeAccumulatorFactory(
            min_val=self.min_val, max_val=self.max_val, keys=self.keys, name=self.name
        )

    def estimate(self, nobs: float | None, suff_stat: tuple[int, np.ndarray]) -> "IntegerUniformSpikeDistribution":
        """Estimate an IntegerUniformSpikeDistribution by maximizing the spike location and weight.

        The spike location k is chosen to maximize the likelihood of the accumulated counts
        (with optional pseudo_count regularization from the estimator configuration).

        Args:
            nobs (Optional[float]): Weighted number of observations.
            suff_stat (Tuple[int, np.ndarray]): Minimum value and count vector.

        Returns:
            IntegerUniformSpikeDistribution object.

        """
        min_val, count_vec = suff_stat

        with np.errstate(divide="ignore"):
            if self.pseudo_count is None:
                count = np.sum(count_vec)
                p_vec = count_vec / count
                ll = np.log1p(-p_vec)
                ll -= np.log(len(count_vec) - 1)
                ll *= count - count_vec
                ll += count_vec * np.log(p_vec)
                k = np.argmax(ll)
                p = p_vec[k]

                return IntegerUniformSpikeDistribution(
                    k=k if min_val is None else k + min_val,
                    min_val=min_val,
                    num_vals=len(count_vec),
                    p=p,
                    name=self.name,
                )
            if self.pseudo_count is not None:
                # Copy so the pseudo_count adjustments below do not mutate the caller's array.
                count_vec = np.array(count_vec, dtype=np.float64)
                if self.suff_stat[0] is not None and self.suff_stat[1] is None:
                    k_pseudo = self.suff_stat[0] if min_val is None else self.suff_stat[0] - min_val
                    count_vec[k_pseudo] += self.pseudo_count
                    count = np.sum(count_vec)
                    p_vec = count_vec / count
                    ll = np.log1p(-p_vec)
                    ll -= np.log(len(count_vec) - 1)
                    ll *= count - count_vec
                    ll += count_vec * np.log(p_vec)
                    k = np.argmax(ll)
                    p = p_vec[k]

                    return IntegerUniformSpikeDistribution(
                        k=k if min_val is None else k + min_val,
                        min_val=min_val,
                        num_vals=len(count_vec),
                        p=p,
                        name=self.name,
                    )

                elif self.suff_stat[0] is not None and self.suff_stat[1] is not None:
                    k_pseudo = self.suff_stat[0] if min_val is None else self.suff_stat[0] - min_val
                    count_vec[k_pseudo] += self.pseudo_count * self.suff_stat[1]
                    count = np.sum(count_vec)
                    p_vec = count_vec / count
                    ll = np.log1p(-p_vec)
                    ll -= np.log(len(count_vec) - 1)
                    ll *= count - count_vec
                    ll += count_vec * np.log(p_vec)
                    k = np.argmax(ll)
                    p = p_vec[k]

                    return IntegerUniformSpikeDistribution(
                        k=k if min_val is None else k + min_val,
                        min_val=min_val,
                        num_vals=len(count_vec),
                        p=p,
                        name=self.name,
                    )
                else:
                    count_vec += self.pseudo_count
                    count = np.sum(count_vec)
                    p_vec = count_vec / count
                    ll = np.log1p(-p_vec)
                    ll -= np.log(len(count_vec) - 1)
                    ll *= count - count_vec
                    ll += count_vec * np.log(p_vec)
                    k = np.argmax(ll)
                    p = p_vec[k]

                    return IntegerUniformSpikeDistribution(
                        k=k if min_val is None else k + min_val,
                        min_val=min_val,
                        num_vals=len(count_vec),
                        p=p,
                        name=self.name,
                    )


class IntegerUniformSpikeDataEncoder(DataSequenceEncoder):
    """Data encoder for iid integer-uniform-spike observations."""

    def __str__(self) -> str:
        """Return the integer-uniform-spike encoder's display name."""
        return "IntegerUniformSpikeDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return True if other is an IntegerUniformSpikeDataEncoder, False is else."""
        return True if isinstance(other, IntegerUniformSpikeDataEncoder) else False

    def seq_encode(self, x: list[int] | np.ndarray) -> np.ndarray:
        """Encode a sequence of iid integer observations as a numpy integer array.

        Args:
            x (Union[List[int], np.ndarray]): Sequence of iid integer observations.

        Returns:
            Numpy array of ints.

        """
        return np.asarray(x, dtype=int)
