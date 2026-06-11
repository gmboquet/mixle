r"""Evaluate, estimate, and sample from a uniform distribution over integers in range [min_val, max_val] with a spike
  placed on the integer value k.

Defines the IntegerUniformSpikeDistribution, IntegerUniformSpikeSampler, IntegerUniformSpikeAccumulatorFactory,
IntegerUniformSpikeAccumulator, IntegerUniformSpikeEstimator, and the IntegerUniformSpikeDataEncoder classes for use
with pysparkplug.

Data type: (int): The IntegerUniformSpikeDistribution with a range [min_val, max_val] = [a,b], and spike placed
on integer value k with probability p, is given by

    P(x_i = k) = p,
    P(x_i = x) = (1-p)/(b-a), x in [a,b] \ {k},
    P(x_i = else) = 0.0.

"""

from typing import List, Union, Tuple, Optional, Dict, Any

import numpy as np
from numpy.random import RandomState

import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableStatisticAccumulator, SequenceEncodableProbabilityDistribution, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory, \
    DistributionEnumerator
from pysp.utils.enumeration import QuantizedEnumerationIndex


class IntegerUniformSpikeDistribution(SequenceEncodableProbabilityDistribution):
    """IntegerUniformSpikeDistribution object: uniform over an integer range with a spike of mass p at k."""

    def __init__(self, k: int, num_vals: int,  p: float, min_val: Optional[int] = 0, name: Optional[str] = None) \
            -> None:
        """IntegerUniformSpikeDistribution object for creating a uniform integer distribution with a spike on k.

        Args:
            k (int): Integer value to place spike on. Must be within [min_val,min_val+num_vals)
            num_vals (int): Number of integers in the range.
            p (float): Probability of drawing k. (1-p)/(num_vals-1) to draw any other integer in range.
            min_val (Optional[int]): Defaults to 0. Set bottom of integer range.
            name (Optional[str]): Set name for object.

        Attributes:
            p (float): Probability of drawing from k.
            min_val (int): Lower bound for the range.
            max_val (int): Max value for the range.
            k (int): Integer to place the spike on.
            log_p (float): Log of p.
            log_1p (float): Log of 1-p
            num_vals (int): Total number of integers in range.
            name (Optional[str]): Name for object instance.

        """
        self.p = p
        self.min_val = min_val
        self.max_val = min_val + num_vals - 1

        if not self.min_val <= k <= self.max_val:
            raise Exception('Spike value k must be between [%s, %s].' % (repr(self.min_val), repr(self.max_val)))
        else:
            self.k = k

        self.log_p = np.log(p)
        self.num_vals = num_vals
        self.log_1p = np.log1p(-self.p) - np.log(self.num_vals-1)
        self.name = name

    def __str__(self) -> str:
        s1 = str(self.min_val)
        s2 = str(self.num_vals)
        s3 = repr(self.p)
        s4 = repr(self.k)
        s5 = repr(self.name)

        return 'IntegerUniformSpikeDistribution(p=%s, min_val=%s, num_vals=%s,k=%s, name=%s)' % (s3, s1, s2, s4, s5)

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

    def sampler(self, seed: Optional[int] = None) -> 'IntegerUniformSpikeSampler':
        """Create an IntegerUniformSpikeSampler from parameters of this distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            IntegerUniformSpikeSampler object.

        """
        return IntegerUniformSpikeSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'IntegerUniformSpikeEstimator':
        """Create an IntegerUniformSpikeEstimator for the current integer range.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.

        Returns:
            IntegerUniformSpikeEstimator object.

        """
        if pseudo_count is None:
            return IntegerUniformSpikeEstimator(min_val=self.min_val, max_val=self.max_val, name=self.name)

        else:
            return IntegerUniformSpikeEstimator(min_val=self.min_val, max_val=self.max_val,
                                                pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> 'IntegerUniformSpikeDataEncoder':
        """Returns an IntegerUniformSpikeDataEncoder for encoding sequences of iid integer observations."""
        return IntegerUniformSpikeDataEncoder()

    def enumerator(self) -> 'IntegerUniformSpikeEnumerator':
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


class IntegerUniformSpikeEnumerator(DistributionEnumerator):
    """Enumerates the support [min_val, max_val] in descending probability order.

    The spike value k is yielded first when p >= (1-p)/(num_vals-1), otherwise last; the
    remaining values share the same probability and are yielded in ascending integer
    order. Zero-probability values are skipped.
    """

    def __init__(self, dist: IntegerUniformSpikeDistribution) -> None:
        """IntegerUniformSpikeEnumerator object.

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

    def __next__(self) -> Tuple[int, float]:
        if self._pos >= len(self._items):
            raise StopIteration
        item = self._items[self._pos]
        self._pos += 1
        return item


class IntegerUniformSpikeSampler(DistributionSampler):
    """IntegerUniformSpikeSampler object for sampling from an IntegerUniformSpikeDistribution.

    Attributes:
        dist (IntegerUniformSpikeDistribution): Distribution to sample from.
        rng (RandomState): Seeded RandomState for sampling.
        non_k (np.ndarray): Integers of the support excluding the spike value k.

    """

    def __init__(self, dist: 'IntegerUniformSpikeDistribution', seed: Optional[int] = None) -> None:
        """IntegerUniformSpikeSampler object.

        Args:
            dist (IntegerUniformSpikeDistribution): Distribution to sample from.
            seed (Optional[int]): Seed to set for sampling with RandomState.

        """
        self.rng = RandomState(seed)
        self.dist = dist
        self.non_k = np.delete(np.arange(self.dist.min_val, self.dist.max_val + 1), self.dist.k - self.dist.min_val)

    def sample(self, size: Optional[int] = None) -> Union[int, np.array]:
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
    """IntegerUniformSpikeAccumulator object for accumulating weighted integer counts over a growing range.

    Attributes:
        min_val (Optional[int]): Smallest integer observed (or configured) so far.
        max_val (Optional[int]): Largest integer observed (or configured) so far.
        count_vec (Optional[np.ndarray]): Weighted counts for each integer in [min_val, max_val].
        count (float): Total weighted observation count.
        key (Optional[str]): Key for merging sufficient statistics across accumulators.
        name (Optional[str]): Name for object instance.

    """

    def __init__(self, min_val: Optional[int], max_val: Optional[int], keys: Optional[str] = None,
                 name: Optional[str] = None) -> None:
        """IntegerUniformSpikeAccumulator object.

        Args:
            min_val (Optional[int]): Smallest integer value in the range, if known.
            max_val (Optional[int]): Largest integer value in the range, if known.
            keys (Optional[str]): Set key for merging sufficient statistics.
            name (Optional[str]): Set name for object instance.

        """
        self.min_val = min_val
        self.max_val = max_val

        if self.min_val is not None and self.max_val is not None:
            self.num_vals = self.max_val - self.min_val + 1
            self.count_vec = np.zeros(self.max_val-self.min_val + 1, dtype=float)
        else:
            self.count_vec = None

        self.count = 0.0
        self.key = keys
        self.name = name

    def update(self, x: int, weight: float, estimate: Optional['IntegerUniformSpikeDistribution']) -> None:
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
            self.count_vec[:len(temp_vec)] = temp_vec
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

    def seq_initialize(self, x: Tuple[int, np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization from encoded observations x (delegates to seq_update)."""
        return self.seq_update(x, weights, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray,
                   estimate: Optional['IntegerUniformSpikeDistribution']) -> None:
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
            self.count_vec[prev_diff:(prev_diff + len(temp))] = temp

        min_diff = min_x - self.min_val
        self.count_vec[min_diff:(min_diff + len(loc_cnt))] += loc_cnt

    def combine(self, suff_stat: Tuple[int, np.ndarray]) -> 'IntegerUniformSpikeAccumulator':
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

    def value(self) -> Tuple[int, np.ndarray]:
        """Returns sufficient statistics as a tuple (min_val, count_vec)."""
        return self.min_val, self.count_vec

    def from_value(self, x: Tuple[int, np.ndarray]) -> 'IntegerUniformSpikeAccumulator':
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

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge this accumulator's sufficient statistics into stats_dict under its key."""
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace this accumulator's sufficient statistics from stats_dict under its key."""
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> 'IntegerUniformSpikeDataEncoder':
        """Returns an IntegerUniformSpikeDataEncoder for encoding sequences of iid integer observations."""
        return IntegerUniformSpikeDataEncoder()


class IntegerUniformSpikeAccumulatorFactory(StatisticAccumulatorFactory):
    """IntegerUniformSpikeAccumulatorFactory object for creating IntegerUniformSpikeAccumulator objects.

    Args:
        min_val (Optional[int]): Smallest integer value in the range, if known.
        max_val (Optional[int]): Largest integer value in the range, if known.
        keys (Optional[str]): Set key for merging sufficient statistics.
        name (Optional[str]): Set name for object instance.

    Attributes:
        min_val (Optional[int]): Smallest integer value in the range, if known.
        max_val (Optional[int]): Largest integer value in the range, if known.
        keys (Optional[str]): Key for merging sufficient statistics.
        name (Optional[str]): Name for object instance.

    """

    def __init__(self, min_val: Optional[int] = None, max_val: Optional[int] = None, keys: Optional[str] = None,
                 name: Optional[str] = None) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.keys = keys
        self.name = name

    def make(self) -> 'IntegerUniformSpikeAccumulator':
        """Returns a new IntegerUniformSpikeAccumulator object."""
        return IntegerUniformSpikeAccumulator(min_val=self.min_val, max_val=self.max_val, keys=self.keys,
                                              name=self.name)


class IntegerUniformSpikeEstimator(ParameterEstimator):
    """IntegerUniformSpikeEstimator object for estimating IntegerUniformSpikeDistribution objects from counts."""

    def __init__(self, min_val: Optional[int] = None,
                 max_val: Optional[int] = None,
                 pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Tuple[int, Optional[float]]] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """IntegerUniformSpikeEstimator object instance for estimating IntegerUniformSpikeDistribution objects.

        Args:
            min_val (Optional[int]): Smallest integer value in the range.
            pseudo_count (Optional[float]): Regularize value k.
            suff_stat (Optional[Tuple[int, Optional[float]]]): Tuple of k to regularize and optional value of p for k.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set keys for object instance.

        Attributes:
            pseudo_count (Optional[float]): Regularize value k.
            min_val (int): Smallest integer value in the range. Defaults to 0.
            max_val (int): Set to the min val plus number of values - 1.
            suff_stat (Optional[Tuple[int, Optional[float]]]): Tuple of k to regularize and optional value of p for k.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set keys for object instance.

        """
        self.pseudo_count = pseudo_count
        self.min_val = min_val
        self.max_val = max_val
        self.suff_stat = suff_stat if suff_stat is not None else (None, None)
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'IntegerUniformSpikeAccumulatorFactory':
        """Returns an IntegerUniformSpikeAccumulatorFactory consistent with this estimator."""
        return IntegerUniformSpikeAccumulatorFactory(min_val=self.min_val, max_val=self.max_val,
                                                     keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[int, np.ndarray]) -> 'IntegerUniformSpikeDistribution':
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

        with np.errstate(divide='ignore'):
            if self.pseudo_count is None:
                count = np.sum(count_vec)
                p_vec = count_vec / count
                ll = np.log1p(-p_vec)
                ll -= np.log(len(count_vec)-1)
                ll *= (count-count_vec)
                ll += count_vec*np.log(p_vec)
                k = np.argmax(ll)
                p = p_vec[k]

                return IntegerUniformSpikeDistribution(k=k if min_val is None else k+min_val,
                                                       min_val=min_val, num_vals=len(count_vec),
                                                       p=p, name=self.name)
            if self.pseudo_count is not None:
                if self.suff_stat[0] is not None and self.suff_stat[1] is None:
                    k_pseudo = self.suff_stat[0] if min_val is None else self.suff_stat[0] - min_val
                    count_vec[k_pseudo] += self.pseudo_count
                    count = np.sum(count_vec)
                    p_vec = count_vec / count
                    ll = np.log1p(-p_vec)
                    ll -= np.log(len(count_vec) - 1)
                    ll *= (count - count_vec)
                    ll += count_vec * np.log(p_vec)
                    k = np.argmax(ll)
                    p = p_vec[k]

                    return IntegerUniformSpikeDistribution(k=k if min_val is None else k + min_val,
                                                           min_val=min_val, num_vals=len(count_vec),
                                                           p=p, name=self.name)

                elif self.suff_stat[0] is not None and self.suff_stat[1] is not None:
                    k_pseudo = self.suff_stat[0] if min_val is None else self.suff_stat[0] - min_val
                    count_vec[k_pseudo] += self.pseudo_count*self.suff_stat[1]
                    count = np.sum(count_vec)
                    p_vec = count_vec / count
                    ll = np.log1p(-p_vec)
                    ll -= np.log(len(count_vec) - 1)
                    ll *= (count - count_vec)
                    ll += count_vec * np.log(p_vec)
                    k = np.argmax(ll)
                    p = p_vec[k]

                    return IntegerUniformSpikeDistribution(k=k if min_val is None else k + min_val,
                                                           min_val=min_val, num_vals=len(count_vec),
                                                           p=p, name=self.name)
                else:
                    count_vec += self.pseudo_count
                    count = np.sum(count_vec)
                    p_vec = count_vec / count
                    ll = np.log1p(-p_vec)
                    ll -= np.log(len(count_vec) - 1)
                    ll *= (count - count_vec)
                    ll += count_vec * np.log(p_vec)
                    k = np.argmax(ll)
                    p = p_vec[k]

                    return IntegerUniformSpikeDistribution(k=k if min_val is None else k + min_val,
                                                           min_val=min_val, num_vals=len(count_vec),
                                                           p=p, name=self.name)


class IntegerUniformSpikeDataEncoder(DataSequenceEncoder):
    """IntegerUniformSpikeDataEncoder object for encoding sequences of iid integer observations."""

    def __str__(self) -> str:
        """Returns string representation of IntegerUniformSpikeDataEncoder object."""
        return 'IntegerUniformSpikeDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Return True if other is an IntegerUniformSpikeDataEncoder, False is else."""
        return True if isinstance(other, IntegerUniformSpikeDataEncoder) else False

    def seq_encode(self, x: Union[List[int], np.ndarray]) -> np.ndarray:
        """Encode a sequence of iid integer observations as a numpy integer array.

        Args:
            x (Union[List[int], np.ndarray]): Sequence of iid integer observations.

        Returns:
            Numpy array of ints.

        """
        return np.asarray(x, dtype=int)

