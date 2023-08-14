"""Create, estimate, and sample from a integer set Bernoulli distribution.

Defines the IntegerBernoulliSetDistribution, IntegerBernoulliSetSampler, IntegerBernoulliSetAccumulatorFactory,
IntegerBernoulliSetAccumulator, IntegerBernoulliSetEstimator, and the IntegerBernoulliSetDataEncoder classes for use
with pysparkplug.


Let S = {0,1,2,3...,N-1} be a set if integers. Let x_mat be a random subset of S. The Bernoulli set distribution models
random subset of S as

    p_k = p_mat(k is in x_mat) , k = 0,2,...,N-1.

The density for an observed subset of S, x=(x_1,x_2,..,x_m), for m < N) is given by
    p_mat(x) = sum_{k=0}^{K-1}( p_k*(k in x) + (1-p_k)*(k not in x)).

"""

import numpy as np
from pysp.arithmetic import *
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DataSequenceEncoder, StatisticAccumulatorFactory, DistributionSampler


from typing import Sequence, Optional, Tuple, Union, List, Any, Dict


class IntegerBernoulliSetDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, log_pvec: Union[Sequence[float], np.ndarray],
                 log_nvec: Optional[Union[Sequence[float], np.ndarray]] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """IntegerBernoulliSetDistribution object defining a Bernoulli set distribution on integers [0,len(pvec)).

        Args:
            log_pvec (Union[Sequence[float], np.ndarray]): Probability of integer k being in set.
            log_nvec (Optional[Union[Sequence[float], np.ndarray]]): Optional normalizing probability for each
                integer probability.
            name (Optional[str]): Set name to object instance.
            keys (Optional[str]): Set keys for object instance.

        Attributes:
            name (Optional[str]): Name for object instance.
            log_pvec (np.ndarray): Probability of integer k being in set.
            log_nvec (Optional[Union[Sequence[float], np.ndarray]]): Optional normalizing probability for each
                integer probability.
            log_dvec (np.ndarray): Normalized probability for each integer value.
            log_nsum (float): Sum of normalized probabilities used for easily adding unobserved (missing) integer
                values in an observation.
            key (Optional[str]): Set keys for object instance.

        """

        num_vals = len(log_pvec)
        self.name = name
        self.num_vals = num_vals
        self.log_pvec = np.asarray(log_pvec, dtype=np.float64).copy()
        self.key = keys

        if log_nvec is None:
            '''
            is_one   = log_pvec == 0
            is_zero  = log_pvec == -np.inf
            is_good  = np.bitwise_and(~is_one, ~is_zero)

            log_nvec = np.zeros(len(log_pvec), dtype=np.float64)
            log_dvec = np.zeros(len(log_pvec), dtype=np.float64)
            log_nvec[is_good] = np.log1p(-np.exp(self.log_pvec[is_good]))
            log_dvec[is_good] = self.log_pvec[is_good] - log_nvec[is_good]
            log_dvec[is_zero] = -np.inf

            self.log_nvec = None
            self.log_dvec = log_dvec
            self.log_nsum = np.sum(log_nvec)
            '''
            log_nvec = np.log1p(-np.exp(self.log_pvec))
            self.log_nvec = None
            self.log_dvec = self.log_pvec - log_nvec
            self.log_nsum = np.sum(log_nvec[np.isfinite(log_nvec)])
        else:
            self.log_nvec = np.asarray(log_nvec, dtype=np.float64)
            self.log_dvec = self.log_pvec - self.log_nvec
            self.log_nsum = np.sum(self.log_nvec[np.isfinite(self.log_nvec)])

    def __str__(self) -> str:
        s1 = repr(list(self.log_pvec))
        s2 = repr(None if self.log_nvec is None else list(self.log_nvec))
        s3 = repr(self.name)
        return 'IntegerBernoulliSetDistribution(%s, log_nvec=%s, name=%s)' % (s1, s2, s3)

    def density(self, x: Union[Sequence[int], np.ndarray]) -> float:
        return exp(self.log_density(x))

    def log_density(self, x: Union[Sequence[int], np.ndarray]) -> float:
        xx = np.asarray(x, dtype=int)
        return np.sum(self.log_dvec[xx]) + self.log_nsum

    def seq_log_density(self, x: Tuple[int, np.ndarray, np.ndarray]) -> np.ndarray:
        sz, idx, xs = x
        rv = np.zeros(sz, dtype=np.float64)
        rv += np.bincount(idx, weights=self.log_dvec[xs], minlength=sz)
        rv += self.log_nsum
        return rv

    def sampler(self, seed: Optional[int] = None) -> 'IntegerBernoulliSetSampler':
        return IntegerBernoulliSetSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'IntegerBernoulliSetEstimator':
        return IntegerBernoulliSetEstimator(self.num_vals, pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> 'IntegerBernoulliSetDataEncoder':
        return IntegerBernoulliSetDataEncoder()

class IntegerBernoulliSetSampler(DistributionSampler):

    def __init__(self, dist: IntegerBernoulliSetDistribution, seed: Optional[int] = None) -> None:
        """IntegerBernoulliSetSampler object for sampling from an IntegerBernoulliSetDistribution instance.

        Args:
            dist (IntegerBernoulliSetDistribution): Object instance to sample from.
            seed (Optional[int]): Seed for random number generator.

        Attributes:
            rng (RandomState): RandomState object with seed set if passed in args.
            dist (IntegerBernoulliSetDistribution): Object instance to sample from.

        """
        self.rng = np.random.RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[List[Sequence[int]], Sequence[int]]:
        if size is None:
            log_u = np.log(self.rng.rand(self.dist.num_vals))
            return list(np.flatnonzero(log_u <= self.dist.log_pvec))
        else:
            rv = []
            for i in range(size):
                log_u = np.log(self.rng.rand(self.dist.num_vals))
                rv.append(list(np.flatnonzero(log_u <= self.dist.log_pvec)))
            return rv


class IntegerBernoulliSetAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, num_vals: int, keys: Optional[str] = None) -> None:
        """IntegerBernoulliSetAccumulator object for accumulating sufficient statistics from observed data.

        Args:
            num_vals (int): Number of values in integer range for the set.
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.

        Attributes:
            pcnt (np.ndarray): Used for aggregating weighted counts of integers.
            key (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            num_vals (int): Number of values in integer range for the set.
            tot_sum (float): Sum of weights for observations.

        """
        self.pcnt = np.zeros(num_vals, dtype=np.float64)
        self.key = keys
        self.num_vals = num_vals
        self.tot_sum = 0.0

    def update(self, x: Union[Sequence[int], np.ndarray], weight: float,
               estimate: Optional[IntegerBernoulliSetDistribution]) -> None:
        xx = np.asarray(x, dtype=int)
        self.pcnt[xx] += weight
        self.tot_sum += weight

    def initialize(self, x: Union[Sequence[int], np.ndarray], weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_update(self, x: Tuple[int, np.ndarray, np.ndarray], weights: np.ndarray,
                   estimate: Optional[IntegerBernoulliSetDistribution]) -> None:
        sz, idx, xs = x
        agg_cnt = np.bincount(xs, weights=weights[idx])
        n = len(agg_cnt)
        self.pcnt[:n] += agg_cnt
        self.tot_sum += weights.sum()

    def seq_initialize(self, x: Tuple[int, np.ndarray, np.ndarray], weights: np.ndarray,
                       rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[np.ndarray, float]) -> 'IntegerBernoulliSetAccumulator':
        self.pcnt += suff_stat[0]
        self.tot_sum += suff_stat[1]
        return self

    def value(self) -> Tuple[np.ndarray, float]:
        return self.pcnt, self.tot_sum

    def from_value(self, x: Tuple[np.ndarray, float]) -> 'IntegerBernoulliSetAccumulator':
        self.pcnt = x[0]
        self.tot_sum = x[1]
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                temp = stats_dict[self.key]
                stats_dict[self.key] = (temp[0] + self.pcnt, temp[1] + self.tot_sum)
            else:
                stats_dict[self.key] = (self.pcnt, self.tot_sum)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                self.pcnt, self.tot_sum = stats_dict[self.key]

    def acc_to_encoder(self) -> 'IntegerBernoulliSetDataEncoder':
        return IntegerBernoulliSetDataEncoder()


class IntegerBernoulliSetAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, num_vals: int, keys: Optional[str] = None) -> None:
        """IntegerBernoulliSetAccumulatorFactory for creating IntegerBernoulliSetAccumulator objects.

        Args:
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            num_vals (int): Number of values in integer range for the set.

        Attributes:
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            num_vals (int): Number of values in integer range for the set.

        """
        self.keys = keys
        self.num_vals = num_vals

    def make(self) -> 'IntegerBernoulliSetAccumulator':
        return IntegerBernoulliSetAccumulator(self.num_vals, keys=self.keys)


class IntegerBernoulliSetEstimator(ParameterEstimator):

    def __init__(self, num_vals: int, min_prob: float = 1.0e-128, pseudo_count: Optional[float] = None,
                 suff_stat: Optional[np.ndarray] = None, name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """IntegerBernoulliSetEstimator object for estimating integer Bernoulli set distributions from aggregated
            sufficient statistics.

        Args:
            num_vals (int): Number of values in integer range for the set.
            min_prob (float): Minimum probability for an integer in range of set dist.
            pseudo_count (Optional[float]): Re-weight suff stats in estimation.
            suff_stat (Optional[np.ndarray]): Probability for integer inclusion.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.

        Attributes:
            num_vals (int): Number of values in integer range for the set.
            keys (Optional[str]): Keys for merging sufficient statistics with matching key'd objects.
            pseudo_count (Optional[float]): Re-weight suff stats in estimation.
            suff_stat (Optional[np.ndarray]): Probability for integer inclusion.
            name (Optional[str]): Set name for object instance.
            min_prob (float): Minimum probability for an integer in range of set dist.

        """
        self.num_vals = num_vals
        self.keys = keys
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.name = name
        self.min_prob = min_prob

    def accumulator_factory(self) -> 'IntegerBernoulliSetAccumulatorFactory':
        return IntegerBernoulliSetAccumulatorFactory(self.num_vals, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Optional[np.ndarray] = None) \
            -> 'IntegerBernoulliSetDistribution':
        if self.pseudo_count is not None and self.suff_stat is not None:
            p0 = np.product(self.suff_stat, self.pseudo_count)
            p1 = np.product(np.subtract(1.0, self.suff_stat), self.pseudo_count)
            pvec = np.log(suff_stat[0] + p0)
            nvec = np.log((suff_stat[1] - suff_stat[0]) + p1)
            tsum = np.log(suff_stat[1] + self.pseudo_count)
            log_pvec = np.log(pvec) - tsum
            log_nvec = np.log(nvec) - tsum

        elif self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count
            log_c = np.log(suff_stat[1] + p)
            log_pvec = np.log(suff_stat[0] + (p / 2.0)) - log_c
            log_nvec = np.log((suff_stat[1] - suff_stat[0]) + (p / 2.0)) - log_c

        else:

            if suff_stat[1] == 0:
                log_pvec = np.zeros(self.num_vals, dtype=np.float64) + 0.5
                log_nvec = np.zeros(self.num_vals, dtype=np.float64) + 0.5

            elif self.min_prob > 0:
                log_pvec = np.log(np.maximum(suff_stat[0] / suff_stat[1], self.min_prob))
                log_nvec = np.log(np.maximum((suff_stat[1] - suff_stat[0]) / suff_stat[1], self.min_prob))

            else:
                pvec = suff_stat[0] / suff_stat[1]
                nvec = (suff_stat[1] - suff_stat[0]) / suff_stat[1]

                is_zero = (pvec == 0)
                is_one = (nvec == 0)

                log_pvec = np.zeros(self.num_vals, dtype=np.float64)
                log_nvec = np.zeros(self.num_vals, dtype=np.float64)

                log_pvec[~is_zero] = np.log(pvec[~is_zero])
                log_pvec[is_zero] = -np.inf
                log_nvec[~is_one] = np.log(nvec[~is_one])
                log_nvec[is_one] = -np.inf

        return IntegerBernoulliSetDistribution(log_pvec, log_nvec, name=self.name)


class IntegerBernoulliSetDataEncoder(DataSequenceEncoder):
    """IntegerBernoulliSetDataEncoder object for encoding sequences of iid integer Bernoulli set observations."""

    def __str__(self) -> str:
        return 'IntegerBernoulliSetDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(object, IntegerBernoulliSetDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[int]]) -> Tuple[int, np.ndarray, np.ndarray]:
        """Encode sequences of iid observations for vectorized calculations.

        Returns 'rv':
            rv[0] (int): Total number of observations.
            rv[1] (np.ndarray): Index for flattened values of observations.
            rv[2] (np.ndarray): Flattened numpy array of integer values.

        Args:
            x (Sequence[Sequence[int]]): Sequence of integer set observations.

        Returns:
            See above for details.

        """
        idx = []
        xs = []
        for i, xx in enumerate(x):
            idx.extend([i] * len(xx))
            xs.extend(xx)

        idx = np.asarray(idx, dtype=np.int32)
        xs = np.asarray(xs, dtype=np.int32)

        return len(x), idx, xs
