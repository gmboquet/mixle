"""Create, estimate, and sample from a Markov chain with support on the integers (chain can include a lag).

Defines the IntegerMarkovChainDistribution, IntegerMarkovChainSampler, IntegerMarkovChainAccumulatorFactory,
IntegerMarkovChainAccumulator, IntegerMarkovChainEstimator, and the IntegerMarkovChainDataEncoder classes for use with
pysparkplug.

The data type: Sequence[int].

Consider a sequence of length n > 0 s.t. x = (x[0],x[1],...,x[n-1]). With lag > 0, we have the integer Markov chain
has a log-density given by:

    log(P(x)) = log(P_init(x[0:lag]) + sum_{j=0}^{n-1} log(p_mat(x[j + lag] | x[j], x[j+1],..,x[j+lag-1])) +
                    log(P_len(n)),

where P_len(n) is the density for the length distribution evaluated for length 'n', and P_init() is the density
for the initial distribution. If the sequence length is less than the lag, i.e. len(x) < lag, then

    log(P(x)) = log(P_len(n)).

Note: P_len() should be compatible with non-negative integers. P_init() must be compatible with sequences of ints.

"""
import numpy as np
from numpy.random import RandomState
from pysp.arithmetic import maxrandint
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DataSequenceEncoder, DistributionSampler, StatisticAccumulatorFactory
from pysp.stats.null_dist import NullDistribution, NullAccumulator, NullEstimator, NullDataEncoder, \
    NullAccumulatorFactory
from typing import Union, List, Sequence, Any, Optional, TypeVar, Tuple, Dict


E1 = TypeVar('E1') ## init encoding
E2 = TypeVar('E2') ## len encoding
SS1 = TypeVar('SS1') ## suff stat of init
SS2 = TypeVar('SS2') ## suff-stat of length


class IntegerMarkovChainDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, num_values: int, cond_dist: Union[List[List[float]], np.ndarray],
                 lag: int = 1, init_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 keys: Optional[str] = None,
                 name: Optional[str] = None) -> None:
        """IntegerMarkovChainDistribution object defining Markov chain with lag.


        Args:
            num_values (int): Total number of values in support.
            cond_dist (Array-like): Should be num_vals ** lag by num_vals with transition probabilities for each
                lagged length tuple (v_0,v_1,..,v_{lag}).
            lag (int): Lag length for conditional density.
            init_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional distribution for initial states
                of Markov chain (with length lag). Should be a distribution compatible with Sequences.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional distribution for the length of
                observations.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set keys for merging sufficient statistics, including the sufficient statistics of
                init_dist and len_dist.

        Attributes:
            num_values (int): Total number of values in support.
            cond_dist (Array-like): Should be num_vals ** lag by num_vals with transition probabilities for each
                lagged length tuple (v_0,v_1,..,v_{lag}).
            lag (int): Lag length for conditional density.
            init_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional distribution for initial states
                of Markov chain (with length lag). Should be a distribution compatible with Sequences.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Optional distribution for the length of
                observations.
            name (Optional[str]): Set name for object instance.
            keys (Optional[str]): Set keys for merging sufficient statistics, including the sufficient statistics of
                init_dist and len_dist.

        """
        self.num_values = num_values
        self.cond_dist = np.asarray(cond_dist)
        self.lag = lag
        self.init_dist = init_dist if init_dist is not None else NullDistribution()
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.name = name
        self.key = keys

    def __str__(self) -> str:
        """Return string representation of object instance."""
        s1 = repr(self.num_values)
        s2 = repr(self.cond_dist.tolist())
        s3 = repr(self.lag)
        s4 = repr(self.init_dist) if self.init_dist is None else str(self.init_dist)
        s5 = repr(self.len_dist) if self.len_dist is None else str(self.len_dist)
        s6 = repr(self.name)
        s7 = repr(self.key)

        return 'IntegerMarkovChainDistribution(%s, %s, lag=%s, init_dist=%s, len_dist=%s, name=%s, keys=%s)' % (
        s1, s2, s3, s4, s5, s6, s7)

    def density(self, x: Sequence[int]) -> float:
        """Density of integer Markov chain evaluated at x.

        See log_density() for details.

        Args:
            x (Sequence[int]): An integer markov chain observation.

        Returns:
            Density evaluated at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[int]) -> float:
        """Log-density of integer Markov chain evaluated at x.

        Consider a sequence of length n > 0 s.t. x = (x[0],x[1],...,x[n-1]). With lag > 0, we have log-density
        given by:

            log(P(x)) = log(P_init(x[0:lag]) + sum_{j=0}^{n-1} log(p_mat(x[j + lag] | x[j], x[j+1],..,x[j+lag-1])) +
                log(P_len(n)),

        where P_len(n) is the density for the length distribution evaluated for length 'n', and P_init() is the density
        for the initial distribution. If the sequence length is less than the lag, i.e. len(x) < lag, then

            log(P(x)) = log(P_len(n)).

        Args:
            x (Sequence[int]): An integer markov chain observation.

        Returns:
            Log-density evaluated at x.

        """
        rv = 0.0
        lag = self.lag

        if len(x) >= lag:

            m_shape = [self.num_values] * lag
            rv += self.init_dist.log_density(x[:lag])

            for i in range(len(x) - lag):
                idx = np.ravel_multi_index(x[i:(i + lag)], m_shape)
                rv += np.log(self.cond_dist[idx, x[i + lag]])

        rv += self.len_dist.log_density(len(x))

        return rv

    def seq_log_density(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                       Optional[E1], Optional[E2]]) -> np.ndarray:
        """Vectorized evaluation of log-density at every observation in encoded sequence.

        See log_density() for details on likelihood evaluation.

        Sequence encoded arg 'x' is a Tuple of length 7 containing:
            seq_len (ndarray[int]): Lengths of chains - lag. If less than lag length is 0.
            init_idx (ndarray[int]): Observed sequence index of chains with lengths >= lag.
            seq_idx (ndarray[int]): Observed sequence index of chains with transitions.
            u_seq_idx (ndarray[object]): Numpy array of tuples containing the unique transitions.
            u_seq_values (ndarray[object]): Numpy array of tuples containing the transitions.
            init_enc (Optional[E]): Sequence encoding of initial values (has type E).
            len_enc (Optional[E2]): Sequence encoding of length values (has type E2).

        Args:
            x: See above for details.

        Returns:
            Log-density evaluated at each observation in encoded sequence.

        """
        seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc = x

        left_idx = [np.ravel_multi_index(u[0], [self.num_values] * self.lag) for u in u_seq_values]
        right_idx = np.asarray([u[1] for u in u_seq_values])
        temp_prob = np.log(self.cond_dist[left_idx, right_idx])
        temp_prob = temp_prob[u_seq_idx]

        rv = np.bincount(seq_idx, weights=temp_prob, minlength=len(seq_len))

        if self.init_dist is not None:
            rv[init_idx] += self.init_dist.seq_log_density(init_enc)

        if self.len_dist is not None and len_enc is not None:
            rv += self.len_dist.seq_log_density(len_enc)

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'IntegerMarkovChainSampler':
        """Returns an IntegerMarkovChainSampler object."""
        return IntegerMarkovChainSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None):
        """Returns an IntegerMarkovChainEstimator object."""
        init_est = self.init_dist.estimator()
        len_est = self.len_dist.estimator()

        return IntegerMarkovChainEstimator(num_values=self.num_values, lag=self.lag, init_estimator=init_est,
                                           len_estimator=len_est, pseudo_count=pseudo_count, name=self.name,
                                           keys=self.key)

    def dist_to_encoder(self) -> 'IntegerMarkovChainDataEncoder':
        """Returns an IntegerMarkovChainDataEncoder object for encoding sequences of iid integer Markov chain
            observations."""
        len_encoder = self.len_dist.dist_to_encoder()
        init_encoder = self.init_dist.dist_to_encoder()
        return IntegerMarkovChainDataEncoder(lag=self.lag, len_encoder=len_encoder, init_encoder=init_encoder)


class IntegerMarkovChainSampler(DistributionSampler):

    def __init__(self, dist: IntegerMarkovChainDistribution, seed: Optional[int]) -> None:
        """IntegerMarkovChainSampler object for sampling from an instance of IntegerMarkovChainDistribution.

        Args:
            dist (IntegerMarkovChainDistribution): Integer Markov chain to sample from.
            seed (Optional[int]): Set the seed for random sampling.

        Attributes:
            dist (IntegerMarkovChainDistribution): Integer Markov chain to sample from.
            rng (RandomState): RandomState object with seed set if passed.
            trans_sampler (RandomState): RandomState object for sampling transitions.

        """
        rng = np.random.RandomState(seed)
        seeds = rng.randint(0, maxrandint, size=3)

        self.dist = dist
        self.rng = rng
        self.trans_sampler = np.random.RandomState(seeds[0])

        if isinstance(self.dist.init_dist, NullDistribution):
            raise Exception('IntegerMarkovChainSampler requires init_dist for IntegerMarkovDistribution.')
        else:
            self.init_sampler = dist.init_dist.sampler(seeds[1])

        if isinstance(dist.len_dist, NullDistribution):
            raise Exception('IntegerMarkovChainSampler requires len_dist for IntegerMarkovDistribution.')
        else:
            self.len_sampler = dist.len_dist.sampler(seeds[2])

    def single_sample(self) -> Sequence[int]:
        """Returns a single sample from the integer Markov chain distribution."""
        cnt = self.len_sampler.sample()
        lag = self.dist.lag
        n_val = self.dist.num_values
        m_shape = [n_val] * lag

        if cnt >= lag:
            rv = self.init_sampler.sample() ## must return a list
            for i in range(lag, cnt):
                idx = np.ravel_multi_index(rv[-lag:], m_shape)
                rv.append(self.trans_sampler.choice(n_val, p=self.dist.cond_dist[idx, :]))
            return rv
        else:
            return []

    def sample(self, size: Optional[int] = None) -> Union[List[Sequence[int]], Sequence[int]]:
        """Draw iid samples from an integer Markov chain distribution.

        Args:
            size (Optional[int]): If None, size is taken to be 0.

        Returns:
            Sequence[int] if size is None, else List[Sequence[int]] with length equal to size.

        """
        if size is not None:
            return [self.single_sample() for i in range(size)]
        else:
            return self.single_sample()

    def sample_given(self, x: Sequence[int]) -> int:
        """Sample from the Markov chain conditioned on a given value 'x'.

        Args:
            x (Sequence[int]): Sample from Markov chain conditioned on observing 'x'.

        Returns:
            Single sample transition from integer Markov chain.

        """
        lag = self.dist.lag
        n_val = self.dist.num_values
        m_shape = [n_val] * lag
        idx = np.ravel_multi_index(x[-lag:], m_shape)

        return self.trans_sampler.choice(n_val, p=self.dist.cond_dist[idx, :])


class IntegerMarkovChainAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, lag: int, init_accumulator: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 len_accumulator: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """IntegerMarkovChainAccumulator object for aggregating sufficient statistics from observed data.

        Args:
            lag (int): The lag for the Markov chain.
            init_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator for the initial
                distribution.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator for the length
                of the observed sequences.
            keys (Optional[str]): Set key for merging sufficient statistics with objects possessing matching key.
            name (Optional[str]): Set name for object.

        Attributes:
            lag (int): The lag for the Markov chain.
            trans_count_map (Dict[Tuple[Sequence[int], int], float]): Dictionary for tracking transition counts.
            init_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the initial distribution. Should
                be a sequence compatible accumulator with support on the integers. Defaults to the NullAccumulator.
            len_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the length of the observed
                sequences. Should be a sequence compatible accumulator with support on the non-negative integers.
                Defaults to the NullAccumulator.
            max_value (int): Largest value encountered when accumulating sufficient statistics.
            keys (Optional[str]): Set key for merging sufficient statistics with objects possessing matching key.
            name (Optional[str]): Set name for object.

            _init_rng (bool): True if RandomState objects for accumulator have been initialized.
            _acc_rng (Optional[RandomState]): RandomState object for initializing the init accumulator.
            _len_rng (Optional[RandomState]): RandomState object for initializing the length accumulator.

        """
        self.lag = lag
        self.trans_count_map = dict()
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()
        self.init_accumulator = init_accumulator if init_accumulator is not None else NullAccumulator()
        self.max_value = -1
        self.key = keys

        self._acc_rng = None
        self._len_rng = None
        self._init_rng = False

    def update(self, x: Sequence[int], weight: float, estimate: Optional[IntegerMarkovChainDistribution]) -> None:
        """Update the sufficient statistics of object instance with a single observation.

        Args:
            x (Sequence[int]): An observation from an integer Markov chain.
            weight (float): Observation weight.
            estimate (Optional[IntegerMarkovChainDistribution]): Optional previous estimate.

        Returns:
            None.

        """
        lag = self.lag
        self.len_accumulator.update(max(len(x) - lag + 1, 0), weight,
                                    estimate.len_dist if estimate is not None else None)

        if len(x) >= lag:
            self.init_accumulator.update(x[:lag], weight, estimate.init_dist if estimate is not None else None)

        for i in range(len(x) - lag):
            entry = (tuple(x[i:(i + lag)]), x[i + lag])
            self.trans_count_map[entry] = self.trans_count_map.get(entry, 0) + weight

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize RandomState objects for accumulators from rng.

        This function exists to ensure consistency between initialize() and seq_initialize() functions.

        Args:
            rng (RandomState): Used to generate seed value for _acc_rng and _len_rng.

        Returns:
            None.

        """
        seeds = rng.randint(maxrandint, size=2)
        self._acc_rng = RandomState(seed=seeds[0])
        self._len_rng = RandomState(seed=seeds[1])
        self._init_rng = True

    def initialize(self, x: Sequence[int], weight: float, rng: RandomState) -> None:
        """Initialize sufficient statistics from a single observation.

        Note: Calls _rng_initialize() to ensure consistency with seq_initialize() function.

        Args:
            x (Sequence[int]): An observation from an integer Markov chain.
            weight (float): Observation weight.
            rng (RandomState): RandomState for initializing sufficient statistics.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        lag = self.lag

        if len(x) >= lag:
            self.len_accumulator.initialize(len(x) - lag, weight, self._len_rng)
            self.init_accumulator.initialize(x[:lag], weight, self._acc_rng)

        for i in range(len(x) - lag):
            entry = (tuple(x[i:(i + lag)]), x[i + lag])
            self.trans_count_map[entry] = self.trans_count_map.get(entry, 0) + weight

    def seq_update(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                  Optional[E1], Optional[E2]],
                   weights: np.ndarray,
                   estimate: Optional[IntegerMarkovChainDistribution]) -> None:
        """Vectorized update of sufficient statistics from an encoded sequence of observations 'x'.

        Sequence encoded arg 'x' is a Tuple of length 7 containing:
            seq_len (ndarray[int]): Lengths of chains - lag. If less than lag length is 0.
            init_idx (ndarray[int]): Observed sequence index of chains with lengths >= lag.
            seq_idx (ndarray[int]): Observed sequence index of chains with transitions.
            u_seq_idx (ndarray[object]): Numpy array of tuples containing the unique transitions.
            u_seq_values (ndarray[object]): Numpy array of tuples containing the transitions.
            init_enc (Optional[E]): Sequence encoding of initial values (has type E).
            len_enc (Optional[E2]): Sequence encoding of length values (has type E2).

        Args:
            x: See above for details.
            weights (np.ndarray): Numpy array of observation weights.
            estimate (Optional[IntegerMarkovChainDistribution]): Optional previous estimate.

        Returns:
            None.

        """
        seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc = x

        seq_cnt = np.bincount(u_seq_idx, weights=weights[seq_idx])

        if len(self.trans_count_map) == 0:
            self.trans_count_map = dict(zip(u_seq_values, seq_cnt))
        else:
            for k, v in zip(u_seq_values, seq_cnt):
                self.trans_count_map[k] = self.trans_count_map.get(k, 0) + v

        self.init_accumulator.seq_update(init_enc, weights[init_idx],
                                         estimate.init_dist if estimate is not None else None)

        self.len_accumulator.seq_update(len_enc, weights, estimate.len_dist if estimate is not None else None)

    def seq_initialize(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                      Optional[E1], Optional[E2]], weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of sufficient statistics from an encoded sequence of observations in 'x'.

        Note: Calls _rng_initialize() to ensure consistency with seq_initialize() function.

        Sequence encoded arg 'x' is a Tuple of length 7 containing:
            seq_len (ndarray[int]): Lengths of chains - lag. If less than lag length is 0.
            init_idx (ndarray[int]): Observed sequence index of chains with lengths >= lag.
            seq_idx (ndarray[int]): Observed sequence index of chains with transitions.
            u_seq_idx (ndarray[object]): Numpy array of tuples containing the unique transitions.
            u_seq_values (ndarray[object]): Numpy array of tuples containing the transitions.
            init_enc (Optional[E]): Sequence encoding of initial values (has type E).
            len_enc (Optional[E2]): Sequence encoding of length values (has type E2).

        Args:
            x: See above for details.
            weights (np.ndarray): Numpy array of observation weights.
            rng (RandomState): RandomState for initializing sufficient statistics.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc = x

        seq_cnt = np.bincount(u_seq_idx, weights=weights[seq_idx])

        if len(self.trans_count_map) == 0:
            self.trans_count_map = dict(zip(u_seq_values, seq_cnt))
        else:
            for k, v in zip(u_seq_values, seq_cnt):
                self.trans_count_map[k] = self.trans_count_map.get(k, 0) + v

        self.init_accumulator.seq_initialize(init_enc, weights[init_idx],self._acc_rng)
        self.len_accumulator.seq_initialize(len_enc, weights, self._len_rng)

    def combine(self, suff_stat: Tuple[Dict[Tuple[Tuple[int, ...], int], float], Optional[SS1], Optional[SS2]]) \
            -> 'IntegerMarkovChainAccumulator':
        """Combine sufficient statistics with object instance.

        Arg suff_stat is a Tuple of length 3 containing:
            suff_stat[0] (Dict[Tuple[Tuple[int, ...], int], float]): Dictionary mapping state transition counts.
            suff_stat[1] (Optional[SS1]): Optional sufficient statistics for init accumulator of type SS1.
            suff_stat[2] (Optional[SS2]): Optional sufficient statistics for length accumulator of type SS2.

        Args:
            suff_stat: See above for details.

        Returns:
            IntegerMarkovAccumulator object.

        """
        for k, v in suff_stat[0].items():
            self.trans_count_map[k] = self.trans_count_map.get(k, 0) + v

        if suff_stat[1] is not None:
            self.init_accumulator = self.init_accumulator.combine(suff_stat[1])

        if suff_stat[2] is not None:
            self.len_accumulator = self.len_accumulator.combine(suff_stat[2])

        return self

    def value(self) -> Tuple[Dict[Tuple[Tuple[int, ...], int], float], Optional[Any], Optional[Any]]:
        """Returns sufficient statistics of integer Markov chain.

        Returned suff_stat is a Tuple of length 3 containing:
            suff_stat[0] (Dict[Tuple[Tuple[int, ...], int], float]): Dictionary mapping state transition counts.
            suff_stat[1] (Optional[SS1]): Optional sufficient statistics for init accumulator of type SS1.
            suff_stat[2] (Optional[SS2]): Optional sufficient statistics for length accumulator of type SS2.

        Returns:
            Tuple[Dict[Tuple[Tuple[int, ...], int], float], Optional[SS1], Optional[SS2]].

        """
        return self.trans_count_map, self.init_accumulator.value(), self.len_accumulator.value()

    def from_value(self, x: Tuple[Dict[Tuple[Tuple[int, ...], int], float], Optional[SS1], Optional[SS2]]) \
            -> 'IntegerMarkovChainAccumulator':
        """Set sufficient statistics of object instance to aggregated sufficient statistics in arg 'x'.

        Arg value 'x' is a Tuple of length 3 containing:
            x[0] (Dict[Tuple[Tuple[int, ...], int], float]): Dictionary mapping state transition counts.
            x[1] (Optional[SS1]): Optional sufficient statistics for init accumulator of type SS1.
            x[2] (Optional[SS2]): Optional sufficient statistics for length accumulator of type SS2.

        Args:
            x: See above for details.

        Returns:
            IntegerMarkovChainAccumulator object.

        """
        self.trans_count_map = x[0]
        if x[1] is not None:
            self.init_accumulator = self.init_accumulator.from_value(x[1])

        if x[2] is not None:
            self.len_accumulator = self.len_accumulator.from_value(x[2])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge sufficient statistics of object instance with matching keys in stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to corresponding sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

        self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace the sufficient statistics of object instance with those containing matching keys in 'stats_dict'.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to corresponding sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

        self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'IntegerMarkovChainDataEncoder':
        """Returns IntegerMarkovChainDataEncoder object."""
        len_encoder = self.len_accumulator.acc_to_encoder()
        init_encoder = self.init_accumulator.acc_to_encoder()
        return IntegerMarkovChainDataEncoder(lag=self.lag, len_encoder=len_encoder, init_encoder=init_encoder)


class IntegerMarkovChainAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, lag: int, init_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
                 len_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
                 keys: Optional[str] = None, name: Optional[str] = None) -> None:
        """IntegerMarkovChainAccumulatorFactory object for creating IntegerMarkovChainAccumulator objects.

        Args:
            lag (int): Length of lag in Markov chain.
            init_factory (Optional[StatisticAccumulatorFactory]): Optional StatisticAccumulatorFactory object for the
                init distribution. Should be compatible with sequences of integers.
            len_factory (Optional[StatisticAccumulatorFactory]): Optional StatisticAccumulatorFactory object for the
                length of Markov chain sequence. Should have support on non-negative integers.
            keys (Optional[str]): Set keys for merging sufficient statistics, including the sufficient statistics of
                init_dist and len_dist.
            name (Optional[str]): Set name for object.

        Attributes:
            lag (int): Length of lag in Markov chain.
            init_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory object for the init distribution.
                Should be compatible with sequences of integers. Defaults to NullAccumulatorFactory if None.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory object for the length of Markov
                chain sequence. Requires support on non-negative integers. Defaults to NullAccumulatorFactory if None.
            key (Optional[str]): Set key for merging sufficient statistics, including the sufficient statistics of
                init_dist and len_dist.
            name (Optional[str]): Set name for object.

        """
        self.lag = lag
        self.init_factory = init_factory if init_factory is not None else NullAccumulatorFactory()
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.key = keys
        self.name = name

    def make(self) -> 'IntegerMarkovChainAccumulator':
        """Returns an IntegerMarkovChainAccumulator object from instance."""
        init_acc = self.init_factory.make()
        len_acc = self.len_factory.make()
        return IntegerMarkovChainAccumulator(self.lag, init_acc, len_acc, keys=self.key, name=self.name)


class IntegerMarkovChainEstimator(ParameterEstimator):

    def __init__(self, num_values: int, lag: int = 1,
                 init_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 len_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 init_dist: Optional[SequenceEncodableProbabilityDistribution] = None,
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = None,
                 pseudo_count: Optional[float] = None,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """IntegerMarkovChainEstimator object for estimating integer Markov distribution from aggregated sufficient
            statistics.

        Args:
            num_values (int): Number of values in Markov chain support.
            lag (int): Length of conditional dependence.
            init_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator object compatible with
                sequences of integers.
            len_estimator (Optional[ParameterEstimator]): Optional ParameterEstimator object compatible with the
                non-negative integers.
            init_dist (Optional[SequenceEncodableProbabilityDistribution]): If passed, init_dist is fixed and not
                estimated. Must be compatible with sequences of integers.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): If passed, len_dist is fixed and not
                estimated. Must be compatible with non-negative integers.
            pseudo_count (Optional[float]): If passed sufficient statistics are re-weighted in estimation step.
            name (Optional[str]): Set name to object instance.
            keys (Optional[str]): Set keys for merging sufficient statistics, including the sufficient statistics of
                init_dist and len_dist.

        Attributes:
            num_values (int): Number of values in Markov chain support.
            lag (int): Length of conditional dependence.
            init_estimator (ParameterEstimator): Optional ParameterEstimator object compatible with
                sequences of integers. Defaults to NullEstimator.
            len_estimator (ParameterEstimator): ParameterEstimator object compatible with the non-negative integers.
                Defaults to the NullEstimator.
            init_dist (Optional[SequenceEncodableProbabilityDistribution]): If passed, init_dist is fixed and not
                estimated. Must be compatible with sequences of integers.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): If passed, len_dist is fixed and not
                estimated. Must be compatible with non-negative integers.
            pseudo_count (Optional[float]): If passed sufficient statistics are re-weighted in estimation step.
            name (Optional[str]): Set name to object instance.
            key (Optional[str]): Set key for merging sufficient statistics, including the sufficient statistics of
                init_dist and len_dist.

        """
        self.num_values = num_values
        self.lag = lag
        self.init_estimator = init_estimator
        self.len_estimator = len_estimator
        self.init_dist = init_dist
        self.len_dist = len_dist
        self.pseudo_count = pseudo_count
        self.name = name
        self.key = keys

    def accumulator_factory(self) -> 'IntegerMarkovChainAccumulatorFactory':
        """Returns an IntegerMarkovChainAccumulatorFactory object from attributes values."""
        len_factory = self.len_estimator.accumulator_factory()
        init_factory = self.init_estimator.accumulator_factory()
        return IntegerMarkovChainAccumulatorFactory(self.lag, init_factory, len_factory, keys=self.key)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[Dict[Tuple[Tuple[int, ...],int], float], Optional[SS1],
                                                               Optional[SS2]]) -> 'IntegerMarkovChainDistribution':
        """Estimate IntegerMarkovChainDistribution object from aggregated sufficient statistics in arg 'suff_stat'.

        Arg 'suff_stat' is a Tuple of length 3 containing:
            suff_stat[0] (Dict[Tuple[Tuple[int, ...], int], float]): Dictionary mapping state transition counts.
            suff_stat[1] (Optional[SS1]): Optional sufficient statistics for init accumulator of type SS1.
            suff_stat[2] (Optional[SS2]): Optional sufficient statistics for length accumulator of type SS2.

        Args:
            nobs (Optional[float]): Number of observations used in aggregation of 'suff_stat'.
            suff_stat: See above for details.

        Returns:
            IntegerMarkovChainDistribution object.

        """
        trans_count_map, init_ss, len_ss = suff_stat
        lag = self.lag

        len_dist = self.len_dist if self.len_dist is not None else self.len_estimator.estimate(None, len_ss)
        init_dist = self.init_dist if self.init_dist is not None else self.init_estimator.estimate(None, init_ss)

        num_values = 1 + max([max(max(u[0]), u[1]) for u in trans_count_map.keys()])

        cond_mat = np.zeros((num_values ** lag, num_values), dtype=np.float32)

        vv = list(trans_count_map.items())
        yidx = np.asarray([np.ravel_multi_index(u[0], [num_values] * lag) for u, _ in vv])
        xidx = np.asarray([u[1] for u, _ in vv])
        zidx = np.asarray([u[1] for u in vv])

        cond_mat[yidx, xidx] = zidx

        if self.pseudo_count is not None:
            cond_mat += self.pseudo_count

        cond_mat/= cond_mat.sum(axis=1, keepdims=True)

        return IntegerMarkovChainDistribution(num_values, cond_mat, init_dist=init_dist, lag=lag, len_dist=len_dist,
                                              name=self.name)


class IntegerMarkovChainDataEncoder(DataSequenceEncoder):

    def __init__(self, lag: int, init_encoder: DataSequenceEncoder = NullDataEncoder(),
                 len_encoder: DataSequenceEncoder = NullDataEncoder()) -> None:
        """IntegerMarkovChainDataEncoder object for encoding sequences of iid integer markov chain observations.

        Args:
            lag (int): Integer valued length of lag.
            init_encoder (DataSequenceEncoder): DataSequenceEncoder object for initial lagged value.
            len_encoder (DataSequenceEncoder): DataSequenceEncoder for the length of observed sequences.

        Attributes:
            lag (int): Integer valued length of lag.
            init_encoder (DataSequenceEncoder): DataSequenceEncoder object for initial lagged value. Should be a
                DataSequenceEncoder for a Sequence of distribution with support on integers.
            len_encoder (DataSequenceEncoder): DataSequenceEncoder for the length of observed sequences. Should be
                a DataSequenceEncoder with support on the integers.

        """
        self.lag = lag
        self.init_encoder = init_encoder
        self.len_encoder = len_encoder

    def __str__(self) -> str:
        """Returns a string representation of object instance."""
        rv = 'IntegerMarkovChainDataEncoder(len_encoder=' + str(self.len_encoder)
        rv += ',init_encoder=' + str(self.init_encoder) + ',lag=' + str(self.lag) + ')'
        return rv

    def __eq__(self, other: object) -> bool:
        """Checks if object is equivalent to object instance.

        Note: Must have equivalent init_encoder and len_encoder member attributes.

        Args:
            other (object): Object to compare.

        Returns:
            True if other is equivalent to IntegerMarkovDataEncoder object instance.

        """
        if isinstance(other, IntegerMarkovChainDataEncoder):
            c0 = other.init_encoder == self.init_encoder
            c1 = other.len_encoder == self.len_encoder
            c2 = self.lag == other.lag
            if c0 and c1 and c2:
                return True
            else:
                return False
        else:
            return False

    def seq_encode(self, x: List[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                          Optional[Any], Optional[Any]]:
        """Encode sequence of iid observations from integer Markov chain.

        Returns a Tuple of length 7 containing:
            seq_len (ndarray[int]): Lengths of chains - lag. If less than lag length is 0.
            init_idx (ndarray[int]): Observed sequence index of chains with lengths >= lag.
            seq_idx (ndarray[int]): Observed sequence index of chains with transitions.
            u_seq_idx (ndarray[object]): Numpy array of tuples containing the unique transitions.
            u_seq_values (ndarray[object]): Numpy array of tuples containing the transitions.
            init_enc (Optional[E]): Sequence encoding of initial values (has type E).
            len_enc (Optional[E2]): Sequence encoding of length values (has type E2).

        Args:
            x (List[Sequence[int]]): Sequence of iid observations from integer markov chain distribution.

        Returns:
            See above for details.


        """
        lag = self.lag

        cnt = len(x)
        lens = np.asarray([len(u) for u in x])
        lag_cnt = (lens >= lag).sum()
        step_cnt = np.maximum(lens - lag, 0).sum()

        init_entries = np.zeros(lag_cnt, dtype=object)
        seq_entries = np.zeros(step_cnt, dtype=object)

        init_idx = []
        seq_idx = []
        seq_len = []

        i0 = 0
        i1 = 0

        for i in range(len(x)):
            xx = x[i]
            seq_len.append(max(len(xx) - lag + 1, 0))

            if len(xx) < lag:
                continue

            init_idx.append(i)
            init_entries[i0] = tuple(xx[:lag])
            i0 += 1

            for j in range(len(xx) - lag):
                seq_idx.append(i)
                seq_entries[i1] = (tuple(xx[j:(j + lag)]), xx[j + lag])
                i1 += 1

        u_seq_values, u_seq_idx = np.unique(seq_entries, return_inverse=True)

        init_idx = np.asarray(init_idx, dtype=np.int32)
        seq_idx = np.asarray(seq_idx, dtype=np.int32)
        seq_len = np.asarray(seq_len, dtype=np.int32)

        len_enc = self.len_encoder.seq_encode(seq_len)
        init_enc = self.init_encoder.seq_encode(init_entries)

        return seq_len, init_idx, seq_idx, u_seq_idx, u_seq_values, init_enc, len_enc

