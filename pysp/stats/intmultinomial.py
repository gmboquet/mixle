"""Evaluate, estimate, and sample from a integer multinomial distribution on range [min_val, max_val].

Defines the IntegerMultinomialDistribution, IntegerMultinomialSampler, IntegerMultinomialAccumulatorFactory,
IntegerMultinomialAccumulator, IntegerMultinomialEstimator, and the IntegerMultinomialDataEncoder classes for use with
pysparkplug.

Data type: Sequence[Tuple[int, float]]: Consider an observation of a multinomial consisting of integer-category
counts of the form x = (x_0,..,x_K), where K is the number of integers in the range [min_val, max_val]. The
IntegerMultinomialDistribution with support [min_val, max_value], number of trials 'N', and probability of success for
the integer-categories given by p = (p_0, ..., p_k), is given by

    log(P(x,N|p)) = -log(n!) - sum_{k=1}^{K} (x_k * log(p_k) + log(x_k!)) + log(P_len(N))

where P_len(N) is a distribution for the number of trials in the multinomial.

"""
import numpy as np
from numpy.random import RandomState
import pysp.utils.vector as vec
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableStatisticAccumulator, SequenceEncodableProbabilityDistribution, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory
from pysp.arithmetic import maxrandint
from pysp.stats.null_dist import NullDistribution, NullEstimator, NullDataEncoder, NullAccumulator, \
    NullAccumulatorFactory

from typing import Optional, Sequence, Tuple, Any, TypeVar, Union, List, Dict


SS0 = TypeVar('SS0')
D = Sequence[Tuple[int, float]]
E0 = TypeVar('E0')
E = Tuple[int, np.ndarray, np.ndarray, np.ndarray, Optional[E0]]


class IntegerMultinomialDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, min_val: int = 0, p_vec: List[float] = None,
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 name: Optional[str] = None, keys: Optional[str] = None) -> None:
        """Create IntegerMultinomialDistribution object.

        Args:
            min_val (int): Set the minimum value on range of values.
            p_vec (Union[List[float],np.ndarray): Probabilities for values. Length determines number of categories.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional length distributions serving as
                for the number of trials.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set key for distribution.

        Attributes:
            p_vec (ndarray): Probability of each integer category for a trial.
            min_val (int): Smallest integer value for category range. Defaults to 0.
            max_val (int): Largest value of category range. Set by min_val + len(p_vec) - 1.
            log_p_vec (ndarray): Log of p_vec member instance.
            num_vals (int): Total number of integer valued categories.
            len_dist (SequenceEncodableProbabilityDistribution): Distribution for number of trials. Set to
                NullDistribution if None.
            keys (Optional[str]): Keys for distribution passed when ParameterEstimator is created.
            name (Optional[str]): Name for object instance.

        """
        super().__init__()
        p_vec = np.empty(0, dtype=np.float64) if p_vec is None else p_vec

        with np.errstate(divide='ignore'):
            self.p_vec = np.asarray(p_vec, dtype=np.float64)
            self.min_val = min_val
            self.max_val = min_val + self.p_vec.shape[0] - 1
            self.log_p_vec = np.log(self.p_vec)
            self.num_vals = self.p_vec.shape[0]
            self.len_dist = len_dist if len_dist is not None else NullDistribution()
            self.keys = keys
            self.name = name

    def __str__(self) -> str:
        """Return string representation of object instance."""
        s1 = repr(self.min_val)
        s2 = repr(list(self.p_vec))
        s3 = str(self.len_dist)
        s4 = repr(self.name)
        return 'IntegerMultinomialDistribution(%s, %s, len_dist=%s, name=%s)' % (s1, s2, s3, s4)

    def density(self, x: Sequence[Tuple[int, float]]) -> float:
        """Evaluate the density of IntegerMultinomialDistribution at observed value x.

        Args:
            x (Sequence[Tuple[int, float]]): Sequence of Tuple(s) containing the integer category value and number of
                successes.

        Returns:
            Density at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[Tuple[int, float]]) -> float:
        """Evaluate the log-density of IntegerMultinomialDistribution at observed value x.

        Log-density given by

        log(p_mat(x)) = log(n!) - sum_k x_k*log(p_k) - log(x_k!), for x having k integer categories.

        Note: n is the total number of trials in observation x and x has k integer values. p_k denotes the probability
        of success for integer-category x_k.

        Args:
            x (Sequence[Tuple[int, float]]): Sequence of Tuple(s) containing the integer category value and number of
                successes.

        Returns:
            Log-density at x.

        """
        rv = 0.0
        for xx, cnt in x:
            rv += (-inf if (xx < self.min_val or xx > self.max_val) else self.log_p_vec[xx - self.min_val]) * cnt
        return rv

    def seq_log_density(self, x: E) -> np.ndarray:
        """Vectorized evaluation of log-density for an encoded sequence of iid observations from integer multinomial
            distribution.

        Arg 'x' is a Tuple of length 5 containing:
            sz (int): Total number of observed integermultinomial samples.
            idx (ndarray): Numpy index array for each Tuple[value, count] in flattened x.
            cnt (ndarray): Number of successes for each value in flattened x.
            val (ndarray): Integer-category value array in flattened x.
            tcnt (Optional[T1]): Sequence encoded number of trials for each sequence (length sz), with type T if
                length DataSequenceEncoder is not NullDataEncoder and returns type T. Else None.

        Args:
            x (See above for details): Sequence encoding of iid integer multinomial observation.

        Returns:
            Numpy array of log-density evaluated at each observation in encoding.

        """
        sz, idx, cnt, val, tcnt = x

        v  = val - self.min_val
        u  = np.bitwise_and(v >= 0, v < self.num_vals)
        rv = np.zeros(len(v))
        rv.fill(-np.inf)
        rv[u] = self.log_p_vec[v[u]]
        rv[u] *= cnt[u]
        ll = np.bincount(idx, weights=rv, minlength=sz)

        if tcnt is not None:
            ll += self.len_dist.seq_log_density(tcnt)

        return ll

    def sampler(self, seed: Optional[int] = None) -> 'IntegerMultinomialSampler':
        """Create an IntegerMultinomialSampler object for sampling from integer multinomial.

        Args:
            seed (Optional[int]): Set seed on random number generator used in sampling.

        Returns:
            IntegerMultinomialSampler object.

        """
        if isinstance(self.len_dist, NullDistribution):
            raise Exception('IntegerMultinomialDistribution must have len_dist set to distribution with support on '
                            'non-negative integers.')
        return IntegerMultinomialSampler(self, seed)

    def estimator(self, pseudo_count: Optional[int] = None) -> 'IntegerMultinomialEstimator':
        """Create and IntegerMultinomialEstimator object for estimating IntegerMultinomialDistribution from aggregated
            sufficient statistics.

        Args:
            pseudo_count (Optional[float]): Used to re-weight sufficient statistics of object instance when estimated.

        Returns:
            IntegerMultinomialEstimator object.

        """
        len_est = NullEstimator() if self.len_dist is None else self.len_dist.estimator(pseudo_count=pseudo_count)

        if pseudo_count is None:
            return IntegerMultinomialEstimator(len_estimator=len_est, name=self.name)
        else:
            return IntegerMultinomialEstimator(min_val=self.min_val, max_val=self.max_val, len_estimator=len_est,
                                               pseudo_count=pseudo_count, suff_stat=(self.min_val, self.p_vec),
                                               name=self.name)

    def dist_to_encoder(self) -> 'IntegerMultinomialDataEncoder':
        """Returns IntegerMultinomialDataEncoder object with len_encoder created from len_dist."""
        len_encoder = self.len_dist.dist_to_encoder()
        return IntegerMultinomialDataEncoder(len_encoder=len_encoder)


class IntegerMultinomialSampler(DistributionSampler):

    def __init__(self, dist: IntegerMultinomialDistribution, seed: Optional[int] = None) -> None:
        """Create IntegerMultinomialSampler object for sampling from IntegerMultinomialDistribution object instance.

        Args:
            dist (IntegerMultinomialDistribution): IntegerMultinomialDistribution object instance to sample from.
            seed (Optional[int]): Optional seed for random number generator.

        Attributes:
            dist (IntegerMultinomialDistribution): IntegerMultinomialDistribution object instance to sample from.
            rng (RandomState): RandomState set with seed if passed.
            len_sampler (DistributionSampler): DistributionSampler object for number of trials.

        """
        self.dist = dist
        self.rng = np.random.RandomState(seed)
        self.len_sampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None) -> Union[List[Tuple[int, float]], List[List[Tuple[int, float]]]]:
        """Draw independent samples from an integer multinomial distribution.

        Args:
            size (Optional[int]): Number of samples to draw.

        Returns:
            List length size containing List[Tuple[int, float]]. If size is None, returns one sample
                List[Tuple[int, float]].

        """
        if size is None:
            cnt = self.len_sampler.sample()
            entry = self.rng.multinomial(cnt, self.dist.p_vec)
            rrv = []
            for j in np.flatnonzero(entry):
                rrv.append((j + self.dist.min_val, entry[j]))
            return rrv

        else:
            cnt = self.len_sampler.sample(size=size)
            rv = []

            for i in range(size):
                rrv = []
                entry = self.rng.multinomial(cnt[i], self.dist.p_vec)
                for j in np.flatnonzero(entry):
                    rrv.append((j + self.dist.min_val, entry[j]))
                rv.append(rrv)
            return rv


class IntegerMultinomialAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, min_val: Optional[int] = None, max_val: Optional[int] = None, name: Optional[str] = None,
                 keys: Optional[str] = None,
                 len_accumulator: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator()) -> None:
        """Create IntegerMultinomialAccumulator object for accumulating sufficient statistics from observed data.

        Args:
            min_val (Optional[int]): Set minimum value for integer multinomial.
            max_val (Optional[int]): Set maximum value for integer multinomial.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set keys for merging sufficient stats with other objects containing matching keys.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator for number of
                integer multinomial trial counts.

        Attributes:
            min_val (Optional[int]): Minimum value for integer multinomial.
            max_val (Optional[int]): Maximum value for integer multinomial.
            name (Optional[str]): Name for object instance.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator for number of
                integer multinomial trial counts. Set to NullAccumulator() if None.
            count_vec (Optional[ndarray]): Set counter for the number of values in integer multinomial range to zero
                ndarray if min_val and max_val are passed. Else, set to none.
            key (Optional[str]): Keys for merging sufficient stats with other objects containing matching key.

        """
        self.min_val = min_val
        self.max_val = max_val
        self.name = name
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()
        self.count_vec = vec.zeros(max_val - min_val + 1) if min_val is not None and max_val is not None else None
        self.key = keys

    def update(self, x: Sequence[Tuple[int, float]], weight: float,
               estimate: Optional[IntegerMultinomialDistribution]) -> None:
        """Update sufficient statistics from single data observation.

        Args:
            x (Sequence[Tuple[int, float]]): Single observation of integer multinomial distribution.
            weight (float): Weight for observation.
            estimate (Optional[IntegerMultinomialDistribution]): Optional previous estimate of integer multinomial
                distribution.

        Returns:
            None.

        """
        cc = 0
        for xx, cnt in x:
            cc += cnt
            if self.count_vec is None:
                self.min_val = xx
                self.max_val = xx
                self.count_vec = vec.make([weight * cnt])
            elif self.max_val < xx:
                temp_vec = self.count_vec
                self.max_val = xx
                self.count_vec = vec.zeros(self.max_val - self.min_val + 1)
                self.count_vec[:len(temp_vec)] = temp_vec
                self.count_vec[xx - self.min_val] += weight * cnt
            elif self.min_val > xx:
                temp_vec = self.count_vec
                temp_diff = self.min_val - xx
                self.min_val = xx
                self.count_vec = vec.zeros(self.max_val - self.min_val + 1)
                self.count_vec[temp_diff:] = temp_vec
                self.count_vec[xx - self.min_val] += weight * cnt
            else:
                self.count_vec[xx - self.min_val] += weight * cnt

        if estimate is None:
            self.len_accumulator.update(cc, weight, None)
        else:
            self.len_accumulator.update(cc, weight, estimate.len_dist)

    def initialize(self, x: Sequence[Tuple[int, float]], weight: float, rng: Optional[RandomState]) -> None:
        """Initialize IntegerMultinomialAccumulator with single observation x.

        Just calls update() method.

        Args:
            x (Sequence[Tuple[int, float]]): Single observation of integer multinomial distribution.
            weight (float): Weight for observation.
            rng (Optional[RandomState]): Optional RandomState object for consistency with
                SequenceEncodableStatisticAccumulator class.

        Returns:
            None.

        """
        self.update(x, weight, None)

    def seq_update(self, x: E, weights: np.ndarray, estimate: Optional[IntegerMultinomialDistribution]) -> None:
        """Vectorized update of IntegerMultinomialAccumulator sufficient statistics for encoded sequence of
            independent observations x.

        Encoded sequence 'x' is a Tuple of length 5 containing:
            sz (int): Total number of observed integermultinomial samples.
            idx (ndarray): Numpy index array for each Tuple[value, count] in flattened x.
            cnt (ndarray): Number of successes for each value in flattened x.
            val (ndarray): Integer-category value array in flattened x.
            tcnt (Optional[E0]): Sequence encoded number of trials for each sequence (length sz), with type E0 if
                length DataSequenceEncoder is not NullDataEncoder and returns type E0.
        Args:
            x (See above): Encoded sequence of iid observations of integer multinomial distribution.
            weights (ndarray): Weights for observations in encoded sequence.
            estimate (Optional[IntegerMultinomialDistribution]): Optional previous estimate of integer multinomial
                distribution.

        Returns:
            None.

        """
        sz, idx, cnt, val, tenc = x

        min_x = val.min()
        max_x = val.max()

        loc_cnt = np.bincount(val-min_x, weights=cnt*weights[idx])

        if self.count_vec is None:
            self.count_vec = np.zeros(max_x - min_x + 1)
            self.min_val = min_x
            self.max_val = max_x

        if self.min_val > min_x or self.max_val < max_x:
            prev_min    = self.min_val
            self.min_val = min(min_x, self.min_val)
            self.max_val = max(max_x, self.max_val)
            temp        = self.count_vec
            prev_diff   = prev_min - self.min_val
            self.count_vec = np.zeros(self.max_val - self.min_val + 1)
            self.count_vec[prev_diff:(prev_diff + len(temp))] = temp

        min_diff = min_x - self.min_val
        self.count_vec[min_diff:(min_diff + len(loc_cnt))] += loc_cnt

        if self.len_accumulator is not None:
            if estimate is None:
                self.len_accumulator.seq_update(tenc, weights, None)
            else:
                self.len_accumulator.seq_update(tenc, weights, estimate.len_dist)

    def seq_initialize(self, x: E, weights: np.ndarray, rng: Optional[RandomState]) -> None:
        """Vectorized initialization of sufficient statistics from encoded sequence of iid observations 'x'.

        Note: Just calls seq_update() method.

        Encoded sequence 'x' is a Tuple of length 5 containing:
            sz (int): Total number of observed integermultinomial samples.
            idx (ndarray): Numpy index array for each Tuple[value, count] in flattened x.
            cnt (ndarray): Number of successes for each value in flattened x.
            val (ndarray): Integer-category value array in flattened x.
            tcnt (Optional[T1]): Sequence encoded number of trials for each sequence (length sz), with type E0 if
                length DataSequenceEncoder is not NullDataEncoder and returns type E0. Else None.

        Args:
            x (See above): Encoded sequence of iid observations of integer multinomial distribution.
            weights (ndarray): Weights for observations in encoded sequence.
            rng (Optional[RandomState]): Optional RandomState object for consistency with
                SequenceEncodableStatisticAccumulator class.

        Returns:
            None.

        """
        self.seq_update(x, weights, None)


    def combine(self, suff_stat: Tuple[int, np.ndarray, Optional[SS0]]) -> 'IntegerMultinomialAccumulator':
        """Combine sufficient statistics in arg 'suff_stat' with sufficient statistics of obejct instance.

        Arg 'suff_stat' contains:
            suff_stat[0] (int): A minimum value for aggregated counts.
            suff_stat[1] (np.ndarray): Numpy array of aggregated counts.
            suff_stat[2] (Optional[SS0]): Optional sufficient statistics for the length accumulator with type SS0.

        Args:
            suff_stat: See above for details.

        Returns:
            IntegerMultinomialAccumulator object.

        """
        if self.count_vec is None and suff_stat[1] is not None:
            self.min_val   = suff_stat[0]
            self.max_val   = suff_stat[0] + len(suff_stat[1]) - 1
            self.count_vec = suff_stat[1]

        elif self.count_vec is not None and suff_stat[1] is not None:

            if self.min_val == suff_stat[0] and len(self.count_vec) == len(suff_stat[1]):
                self.count_vec += suff_stat[1]

            else:
                min_val = min(self.min_val, suff_stat[0])
                max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

                count_vec = vec.zeros(max_val-min_val+1)

                i0 = self.min_val - min_val
                i1 = self.max_val - min_val + 1
                count_vec[i0:i1] = self.count_vec

                i0 = suff_stat[0] - min_val
                i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
                count_vec[i0:i1] += suff_stat[1]

                self.min_val   = min_val
                self.max_val   = max_val
                self.count_vec = count_vec

        self.len_accumulator.combine(suff_stat[2])

        return self

    def value(self) -> Tuple[int, np.ndarray, Optional[Any]]:
        """Returns sufficient statistics 'suff_stat' of object instance..

        Return value 'suff_stat' contains:
            suff_stat[0] (int): A minimum value for aggregated counts.
            suff_stat[1] (np.ndarray): Numpy array of aggregated counts.
            suff_stat[2] (Optional[SS0]): Optional sufficient statistics for the length accumulator with type SS0.

        Returns:
            Tuple[int, ndarray, Optional[SS0]].

        """
        return self.min_val, self.count_vec, self.len_accumulator.value()

    def from_value(self, x: Tuple[int, np.ndarray, Optional[SS0]]) -> 'IntegerMultinomialAccumulator':
        """Sets the object instance sufficient statistics to value of arg 'x'.

        Arg value 'x' contains:
            x[0] (int): A minimum value for aggregated counts.
            x[1] (np.ndarray): Numpy array of aggregated counts.
            x[2] (Optional[SS0]): Optional sufficient statistics for the length accumulator with type SS0.


        Args:
            x (See above for details).

        Returns:
            IntegerMultinomialAccumulator object.

        """
        self.min_val   = x[0]
        self.max_val   = x[0] + len(x[1]) - 1
        self.count_vec = x[1]

        self.len_accumulator.from_value(x[2])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge sufficient statistics of object instance with objects of matching key contained in stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Dictionary with keys for merging sufficient statistics of matching key
                sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

        if self.len_accumulator is not None:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace sufficient statistics of object instance with sufficient statistics in arg 'stats_dict' of matching
            keys.

        Args:
            stats_dict (Dict[str, Any]): Dictionary with keys for merging sufficient statistics of matching key
                sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

        if self.len_accumulator is not None:
            self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'IntegerMultinomialDataEncoder':
        """Returns IntegerMultinomialDataEncoder object with len_encoder created from len_accumulator."""
        len_encoder = self.len_accumulator.acc_to_encoder()
        return IntegerMultinomialDataEncoder(len_encoder=len_encoder)


class IntegerMultinomialAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, min_val: Optional[int] = None, max_val: Optional[int] = None, name: Optional[str] = None,
                 keys: Optional[str] = None,
                 len_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory()) -> None:
        """Create IntegerMultinomialAccumulatorFactory object for creating IntegerMultinomialAccumulator objects.

        Args:
            min_val (Optional[int]): Optional minimum value for IntegerMultinomialAccumulator.
            max_val (Optional[int]): Optional maximum value for IntegerMultinomialAccumulator.
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional keys for merging sufficient statistics of object instance.
            len_factory (Optional[StatisticAccumulatorFactory]): Optional StatisticAccumulatorFactory object for
                creating StatisticAccumulator object for number of trials.

        Attributes:
            min_val (Optional[int]): Optional minimum value for IntegerMultinomialAccumulator.
            max_val (Optional[int]): Optional maximum value for IntegerMultinomialAccumulator.
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional keys for merging sufficient statistics of object instance.
            len_factory (Optional[StatisticAccumulatorFactory]): Optional StatisticAccumulatorFactory object for
                creating StatisticAccumulator object for number of trials. Default to NullAccumulatorFactory()

        """
        self.min_val = min_val
        self.max_val = max_val
        self.name = name
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys

    def make(self) -> 'IntegerMultinomialAccumulator':
        """Returns IntegerMultinomialAccumulator object for """
        len_acc = self.len_factory.make()
        return IntegerMultinomialAccumulator(min_val=self.min_val, max_val=self.max_val, name=self.name,
                                             keys=self.keys, len_accumulator=len_acc)


class IntegerMultinomialEstimator(ParameterEstimator):

    def __init__(self, min_val: Optional[int] = None, max_val: Optional[int] = None,
                 len_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = None,
                 name: Optional[str] = None, pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Tuple[int, np.ndarray]] = None,
                 keys: Optional[str] = None) -> None:
        """IntegerMultinomialEstimator object for estimating integer multinomial distributions from aggregated data.



        Args:
            min_val (Optional[int]): Set minimum value integer multinomial.
            max_val (Optional[int]): Set maximum value for integer multinomial.
            len_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator for number of trials.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional
                SequenceEncodableProbabilityDistribution for fixing distribution on number of trials.
            name (Optional[str]): Set name for object instance.
            pseudo_count (Optional[float]): Used to re-weight sufficient statistics if suff_stat is passed.
            suff_stat (Optional[Tuple[int, np.ndarray]]): Set minimum value and counts for categories.
            keys (Optional[str]): Set key for merging sufficient statistics of objects with matching keys.

        Attributes:
            min_val (Optional[int]): Set minimum value integer multinomial.
            max_val (Optional[int]): Set maximum value for integer multinomial.
            len_estimator (ParameterEstimator): ParameterEstimator for number of trials, set to NullEstimator() if None
                is passed as arg.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional
                SequenceEncodableProbabilityDistribution for fixing distribution on number of trials.
            name (Optional[str]): Set name for object instance.
            pseudo_count (Optional[float]): Used to re-weight sufficient statistics if suff_stat is passed.
            suff_stat (Optional[Tuple[int, np.ndarray]]): Set minimum value and counts for categories. If 'min_val'
                and 'max_val' are both not None, this is ignored in estimation.
            keys (Optional[str]): Set key for merging sufficient statistics of objects with matching keys.

        """
        self.suff_stat = suff_stat
        self.pseudo_count = pseudo_count
        self.min_val = min_val
        self.max_val = max_val
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.len_dist = len_dist
        self.keys = keys
        self.name = name

    def accumulator_factory(self) -> 'IntegerMultinomialAccumulatorFactory':
        """Create an IntegerMultinomialAccumulatorFactory object from IntegerMultinomialEstimator object instance."""
        min_val = None
        max_val = None

        if self.suff_stat is not None:
            min_val = self.suff_stat[0]
            max_val = min_val + len(self.suff_stat[1]) - 1
        elif self.min_val is not None and self.max_val is not None:
            min_val = self.min_val
            max_val = self.max_val

        len_factory = self.len_estimator.accumulator_factory()
        return IntegerMultinomialAccumulatorFactory(min_val=min_val, max_val=max_val, name=self.name, keys=self.keys,
                                                    len_factory=len_factory)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[int, np.ndarray, Optional[SS0]])\
            -> 'IntegerMultinomialDistribution':
        """Estimate an IntegerMultinomialDistribution from aggregated sufficient statistics in arg 'suff_stat'.

        Note: If pseudo_count is not set, member sufficient statistics are ignored in estimation.

        Arg 'suff_stat' contains:
            suff_stat[0] (int): A minimum value for aggregated counts.
            suff_stat[1] (np.ndarray): Numpy array of aggregated counts.
            suff_stat[2] (Optional[SS0]): Optional sufficient statistics for the length accumulator with type SS0.

        Args:
            nobs (Optional[float]): Number of observations in accumulated data.
            suff_stat: See above for details.

        Returns:
            IntegerMultinomialDistribution object.

        """
        len_dist = self.len_dist if self.len_dist is not None else self.len_estimator.estimate(nobs, suff_stat[2])

        if self.pseudo_count is not None and self.suff_stat is None:
            pseudo_count_per_level = self.pseudo_count / float(len(suff_stat[1]))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            return IntegerMultinomialDistribution(suff_stat[0], (suff_stat[1]+pseudo_count_per_level)/adjusted_nobs,
                                                  len_dist=len_dist, name=self.name, keys=self.keys)

        elif self.pseudo_count is not None and self.min_val is not None and self.max_val is not None:
            min_val = min(self.min_val, suff_stat[0])
            max_val = max(self.max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = vec.zeros(max_val - min_val + 1)

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            pseudo_count_per_level = self.pseudo_count / float(len(count_vec))
            adjusted_nobs = suff_stat[1].sum() + self.pseudo_count

            return IntegerMultinomialDistribution(min_val, (count_vec+pseudo_count_per_level)/adjusted_nobs,
                                                  len_dist=len_dist, name=self.name, keys=self.keys)

        elif self.pseudo_count is not None and self.suff_stat is not None:
            s_max_val = self.suff_stat[0] + len(self.suff_stat[1]) - 1
            s_min_val = self.suff_stat[0]

            min_val = min(s_min_val, suff_stat[0])
            max_val = max(s_max_val, suff_stat[0] + len(suff_stat[1]) - 1)

            count_vec = vec.zeros(max_val - min_val + 1)

            i0 = s_min_val - min_val
            i1 = s_max_val - min_val + 1
            count_vec[i0:i1] = self.suff_stat[1]*self.pseudo_count

            i0 = suff_stat[0] - min_val
            i1 = (suff_stat[0] + len(suff_stat[1]) - 1) - min_val + 1
            count_vec[i0:i1] += suff_stat[1]

            return IntegerMultinomialDistribution(min_val, count_vec/(count_vec.sum()), len_dist=len_dist,
                                                  name=self.name, keys=self.keys)
        else:
            return IntegerMultinomialDistribution(suff_stat[0], suff_stat[1]/(suff_stat[1].sum()), len_dist=len_dist,
                                                  name=self.name, keys=self.keys)


class IntegerMultinomialDataEncoder(DataSequenceEncoder):

    def __init__(self, len_encoder: Optional[DataSequenceEncoder] = NullDataEncoder()) -> None:
        """IntegerMultinomialDataEncoder object for encoding sequence of iid integer multinomial observations.

        Args:
            len_encoder (Optional[DataSequenceEncoder]): Optional DataSequenceEncoder for encoding the number of trials
                in each iid observation of integer multinomial.

        Attributes:
            len_encoder (DataSequenceEncoder): DataSequenceEncoder for encoding number of trials in each iid
                observation of integer multinomial. Defaults to NullDataEncoder() if None is passed.

        """
        self.len_encoder = len_encoder if len_encoder is not None else NullDataEncoder()

    def __str__(self) -> str:
        """Return string representation of object instance."""
        return "IntegerMultinomialDataEncoder(len_encoder=" + str(self.len_encoder) + ")"

    def __eq__(self, other: object) -> bool:
        """Checks if an object is an equivalent instance of IntegerMultinomialDataEncoder.

        Note: Instance len_encoder must match as well.

        Args:
            other (object): Object to compare.

        Returns:
            True if other is matching instance of IntegerMultinomialDataEncoder, else False.

        """
        if isinstance(other, IntegerMultinomialDataEncoder):
            return self.len_encoder == other.len_encoder
        else:
            return False

    def seq_encode(self, x: Sequence[Sequence[Tuple[int, float]]]) -> \
            Tuple[int, np.ndarray, np.ndarray, np.ndarray, Optional[Any]]:
        """Encode a sequence of iid integer multinomial observations.

        Returns a Tuple of length 5 containing:
            sz (int): Total number of observed integermultinomial samples.
            idx (ndarray): Numpy index array for each Tuple[value, count] in flattened x.
            cnt (ndarray): Number of successes for each value in flattened x.
            val (ndarray): Integer-category value array in flattened x.
            tcnt (Optional[E0]): Sequence encoded number of trials for each sequence (length sz), with type E0 if
                length DataSequenceEncoder is not NullDataEncoder and returns type E0. Else None.

        Args:
            x (Sequence[Sequence[Tuple[int, float]]]): A sequence of iid integer multinomial observations in the form
                of Sequence of Tuple(s) containing integer-category and float valued number of successes.

        Returns:
            Tuple[int, ndarray[int], ndarray[float], ndarray[int], Optional[T]. See above for details.

        """
        idx = []
        cnt = []
        val = []
        tcnt = []

        for i, y in enumerate(x):
            cc = 0
            for z in y:
                idx.append(i)
                cnt.append(z[1])
                val.append(z[0])
                cc += z[1]
            tcnt.append(cc)

        sz = len(x)
        idx = np.asarray(idx, dtype=np.int32)
        cnt = np.asarray(cnt, dtype=np.float64)
        val = np.asarray(val, dtype=np.int32)
        tcnt = np.asarray(tcnt, dtype=np.int32)

        tcnt = self.len_encoder.seq_encode(tcnt)

        return sz, idx, cnt, val, tcnt


