"""Evaluate, estimate, and sample from a uniform distribution over integers in range [min_val, max_val] with a spike
  placed on the integer value k.

Defines the IntegerUniformSpikeDistribution, IntegerUniformSpikeSampler, IntegerUniformSpikeAccumulatorFactory,
IntegerUniformSpikeAccumulator, IntegerUniformSpikeEstimator, and the IntegerUniformSpikeDataEncoder classes for use
with pysparkplug.

Data type: (float): The IntegerUniformSpikeDistribution with a range [min_val, max_val] = [a,b], and spike placed
on integer value k with probability p, if given by

    P(x_i=k) = p,
    P(x_i = x) = (1-p)/(b-a-1), x in [a,b] \ {-k},
    P(x_i = else) = 0.0.

"""

from typing import List, Union, Tuple, Optional, Dict, Any

import numpy as np
from numpy.random import RandomState

import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableStatisticAccumulator, SequenceEncodableProbabilityDistribution, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory


class IntegerUniformSpikeDistribution(SequenceEncodableProbabilityDistribution):

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
        return np.exp(self.log_density(x))

    def log_density(self, x: int) -> float:
        if self.max_val >= x >= self.min_val:
            return self.log_p if x == self.k else self.log_1p
        else:
            return -np.inf

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:

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
        return IntegerUniformSpikeSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'IntegerUniformSpikeEstimator':
        if pseudo_count is None:
            return IntegerUniformSpikeEstimator(min_val=self.min_val, max_val=self.max_val, name=self.name)

        else:
            return IntegerUniformSpikeEstimator(min_val=self.min_val, max_val=self.max_val,
                                                pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> 'IntegerUniformSpikeDataEncoder':
        return IntegerUniformSpikeDataEncoder()


class IntegerUniformSpikeSampler(DistributionSampler):

    def __init__(self, dist: 'IntegerUniformSpikeDistribution', seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.non_k = np.delete(np.arange(self.dist.min_val, self.dist.max_val), self.dist.k)

    def sample(self, size: Optional[int] = None) -> Union[int, np.array]:

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

    def __init__(self, min_val: Optional[int], max_val: Optional[int], keys: Optional[str] = None,
                 name: Optional[str] = None) -> None:
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
        return self.update(x, weight, None)

    def seq_initialize(self, x: Tuple[int, np.ndarray, np.ndarray], weights: np.ndarray, rng: RandomState) -> None:
        return self.seq_update(x, weights, None)

    def seq_update(self, x: np.ndarray, weights: np.ndarray,
                   estimate: Optional['IntegerUniformSpikeDistribution']) -> None:

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
        return self.min_val, self.count_vec

    def from_value(self, x: Tuple[int, np.ndarray]) -> 'IntegerUniformSpikeAccumulator':
        self.min_val = x[0]
        self.max_val = x[0] + len(x[1]) - 1
        self.count_vec = x[1]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> 'IntegerUniformSpikeDataEncoder':
        return IntegerUniformSpikeDataEncoder()


class IntegerUniformSpikeAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, min_val: Optional[int] = None, max_val: Optional[int] = None, keys: Optional[str] = None,
                 name: Optional[str] = None) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.keys = keys
        self.name = name

    def make(self) -> 'IntegerUniformSpikeAccumulator':
        return IntegerUniformSpikeAccumulator(min_val=self.min_val, max_val=self.max_val, keys=self.keys,
                                              name=self.name)


class IntegerUniformSpikeEstimator(ParameterEstimator):

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
        return IntegerUniformSpikeAccumulatorFactory(min_val=self.min_val, max_val=self.max_val,
                                                     keys=self.keys, name=self.name)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[int, np.ndarray]) -> 'IntegerUniformSpikeDistribution':
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
    """IntegerCategoricalDataEncoder object for encoding sequences of iid integer categorical observations."""

    def __str__(self) -> str:
        """Returns IntegerCategoricalDataEncoder object for encoding data sequences."""
        return 'IntegerCategoricalDataEncoder'

    def __eq__(self, other: object) -> bool:
        """Return True if other is an IntegerCategoricalDataEncoder, False is else."""
        return True if isinstance(other, IntegerUniformSpikeDataEncoder) else False

    def seq_encode(self, x: Union[List[int], np.ndarray]) -> np.ndarray:
        return np.asarray(x, dtype=int)


