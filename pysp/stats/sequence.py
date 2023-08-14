"""Create, estimate, and sample from a sequence of iid sequence of base distribution 'dist' with data type T. A
length distribution for the lengths of the iid sequences can be specified as a discrete distribution compatible with
non-negative integer values.

Defines the SequenceDistribution, SequenceSampler, SequenceAccumulatorFactory, SequenceAccumulator,
SequenceEstimator, and the SequenceDataEncoder classes for use with pysparkplug.

Data type (T): Assume the sequence distribution has a base distribution 'dist' compatible with data type T and length
distribution compatible with positive integers len_dist with respective densities P_dist() and P_len(). The density
of the sequence distribution is given by

p_mat(x) = P_dist(x[0])*...*P_dist(x[n-1])*P_len(n),

for an observation x of data type Sequence[T] having length n.

"""
import numpy as np
from numpy.random import RandomState
from pysp.arithmetic import maxrandint
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder
from pysp.stats.null_dist import NullDistribution, NullAccumulator, NullEstimator, NullDataEncoder, \
    NullAccumulatorFactory

from typing import Optional, List, Any, Tuple, Sequence, TypeVar, Dict


T = TypeVar('T') # Data type of Sequence distribution dist.
E1 = TypeVar('E1') # Generic type of distribution encoding.
E2 = TypeVar('E2') # Generic type of length encoding.
SS1 = TypeVar('SS1') # Generic type for sufficient statistic of base dist.
SS2 = TypeVar('SS2') # Generic type for sufficient statistics of length dist.

E = Tuple[np.ndarray, np.ndarray, np.ndarray, E1, Optional[E2]]


class SequenceDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, dist: SequenceEncodableProbabilityDistribution,
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 len_normalized: Optional[bool] = False, name: Optional[str] = None) -> None:
        """SequenceDistribution object for sequence of iid observations from distribution a of data type T.

        Args:
            dist (SequenceEncodableProbabilityDistribution): Set base distribution of sequence (compatible with T).
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Length distribution for modeling lengths
                of sequences of observations (compatible with type int).
            len_normalized (Optional[bool]): If True, take geometric mean density for any density evaluation.
            name (Optional[str]): Set name to instance of SequenceDistribution.

        Attributes:
            dist (SequenceEncodableProbabilityDistribution): Base distribution of sequence (compatible with T).
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Length distribution for modeling lengths
                of sequences of observations (compatible with type int). Set to NullDistribution if None is passed.
            len_normalized (Optional[bool]): If True, take geometric mean density for any density evaluation.
            name (Optional[str]): Name to instance of SequenceDistribution.
            null_len_distribution (bool): True if 'len_dist' is set to instance of NullDistribution.

        """
        self.dist = dist
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.len_normalized = len_normalized
        self.name = name

        self.null_len_dist = isinstance(self.len_dist, NullDistribution)

    def __str__(self) -> str:
        """Return string representation of SequenceDistribution instance."""
        s1 = str(self.dist)
        s2 = str(self.len_dist)
        s3 = repr(self.len_normalized)
        s4 = repr(self.name)

        return 'SequenceDistribution(%s, len_dist=%s, len_normalized=%s, name=%s)' % (s1, s2, s3, s4)

    def density(self, x: Sequence[T]) -> float:
        """Evaluate the density of SequenceDistribution at observed sequence x.

        Assume x is a Sequence of data type T with length n > 0. Assume P_dist() is the density for the base
        distribution with data type T of SequenceDistribution, and P_len() is the length distribution with data type
        int. Then,

        P(x) = P_dist(x[0])*...*P_dist(x[n-1])*P_len(n), if len_normalize is False,

        or,

        P(x) = (P_dist(x[0])*...*P_dist(x[n-1])*P_len(n))^(1/n) if len_normalize is True.



        Args:
            x (Sequence[T]): Sequence of iid observations from base distribution of SequenceDistribution.

        Returns:
            Density evaluated at observation x.


        """
        rv = 1.0

        for i in range(len(x)):
            rv *= self.dist.density(x[i])

        if not self.null_len_dist:
            rv *= self.len_dist.density(len(x))

        if self.len_normalized and len(x) > 0:
            rv = np.power(rv, 1.0 / len(x))

        return rv

    def log_density(self, x: Sequence[T]) -> float:
        """Evaluate the log-density of SequenceDistribution at observed sequence x.

        See density() for details.

        Args:
            x (Sequence[T]): Sequence of iid observations from base distribution of SequenceDistribution.

        Returns:
            Log-density evaluated at observation x.

        """
        rv = 0.0

        for i in range(len(x)):
            rv += self.dist.log_density(x[i])

        if self.len_normalized and len(x) > 0:
            rv /= len(x)

        if not self.null_len_dist:
            rv += self.len_dist.log_density(len(x))

        return rv

    def seq_ld_lambda(self):
        rv = self.dist.seq_ld_lambda()

        if not self.null_len_dist:
            rv.extend(self.len_dist.seq_ld_lambda())

        return rv

    def seq_log_density(self, x: E) -> np.ndarray:
        """Vectorized evaluation of SequenceDistribution.log-density evaluated on sequence encoded x.

        Args:
            x (E): Sequence encoded data observation.

        Returns:
            Numpy array of log-density evaluated at each encoded observation value x.

        """
        idx, icnt, inz, enc_seq, enc_nseq = x

        if np.all(icnt == 0):
            ll_sum = np.zeros(len(icnt), dtype=float)

        else:
            ll = self.dist.seq_log_density(enc_seq)
            ll_sum = np.bincount(idx, weights=ll, minlength=len(icnt))

            if self.len_normalized:
                ll_sum = ll_sum * icnt

        if not self.null_len_dist and enc_nseq is not None:
            nll = self.len_dist.seq_log_density(enc_nseq)
            ll_sum += nll

        return ll_sum

    def sampler(self, seed: Optional[int] = None) -> 'SequenceSampler':
        """Create a SequenceSampler object from instance of SequenceDistribution.

        Note: If member len_dist (SequenceEncodableDistribution) is NullDistribution() and or not compatible with
        data type int, an error is thrown.

        Args:
            seed (Optional[int]): Used to set seed of random number generator used to sample.

        Returns:
            SequenceSampler object.

        """
        if self.null_len_dist:
            raise Exception('Error: len_dist cannot be none for SequenceDistribution.sampler(seed:Optional[int]=None).')
        else:
            return SequenceSampler(self.dist, self.len_dist, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'SequenceEstimator':
        """Create SequenceEstimator from instance of SequenceDistribution with pseudo_count passed if not None."""
        len_est = self.len_dist.estimator(pseudo_count=pseudo_count)

        return SequenceEstimator(self.dist.estimator(pseudo_count=pseudo_count), len_estimator=len_est,
                                 len_normalized=self.len_normalized, name=self.name)

    def dist_to_encoder(self) -> 'SequenceDataEncoder':
        """Create SequenceDataEncoder for encoding sequences of iid observations of SequenceDistribution.

        Base distribution DataSequenceEncoder and length distribution DataSequenceEncoder objects are passed.

        Returns:
            SequenceDataEncoder object.

        """
        dist_encoder = self.dist.dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()
        encoders = (dist_encoder, len_encoder)
        return SequenceDataEncoder(encoders=encoders)


class SequenceSampler(DistributionSampler):

    def __init__(self,
                 dist: SequenceEncodableProbabilityDistribution,
                 len_dist: SequenceEncodableProbabilityDistribution,
                 seed: Optional[int] = None) -> None:
        """SequenceSampler object for sampling from an SequenceDistribution instance.

        Args:
            dist (SequenceEncodableProbabilityDistribution): Set the base distribution for the sequences (data type T).
            len_dist (SequenceEncodableProbabilityDistribution): Set the length distribution for the length of the
                sequences (support on positive integers).
            seed (Optional[int]): Set seed of random number generator for sampling.

        Attributes:
            dist (SequenceEncodableProbabilityDistribution): The Base distribution for the sequences (data type T).
            len_dist (SequenceEncodableProbabilityDistribution): Length distribution for the length of the
                sequences (support on positive integers).
            rng (RandomState): RandomState object for random sampling.
            dist_sampler (DistributionSampler): DistributionSampler instance from base distribution.
            len_sampler (DistributionSampler): DistributionSampler instance from length distribution.

        """
        self.dist = dist
        self.len_dist = len_dist
        self.rng = RandomState(seed)
        self.dist_sampler = self.dist.sampler(seed=self.rng.randint(0, maxrandint))
        self.len_sampler = self.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None) -> List[Any]:
        """Generate iid samples from SequenceSampler object.

        If size is None, the length 'n' of the iid sequence is sampled from len_sampler. Then 'n' iid samples are
        drawn from the base dist sampled 'dist_sampler'.

        If size > 0, above is repeated size times and a List of size List[T] is retured.

        Args:
            size (Optional[int]) Number of sequences to be sampled.

        Returns:
            List[T] or List[List[T]] with length(size).

        """
        if size is None:
            n = self.len_sampler.sample()
            return [self.dist_sampler.sample() for i in range(n)]
        else:
            return [self.sample() for i in range(size)]


class SequenceAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self,
                 accumulator: SequenceEncodableStatisticAccumulator,
                 len_accumulator: SequenceEncodableStatisticAccumulator = NullAccumulator(),
                 len_normalized: Optional[bool] = False,
                 keys: Optional[str] = None) -> None:
        """SequenceAccumulator object for aggregating sufficient statistics of sequence distribution from observed data.

        Args:
            accumulator (SequenceEncodableStatisticAccumulator): Set SequenceEncodableStatisticAccumulator object for
                accumulating sufficient statistics of base distribution compatible with data type T.
            len_accumulator (SequenceEncodableStatisticAccumulator): Set SequenceEncodableStatisticAccumulator object
                for accumulating sufficient statistics of length distribution compatible with non-negative integers.
            len_normalized (Optional[bool]): Geometric mean of density taken if set to True. Else ignored.
            keys (Optional[str]): Set keys for merging sufficient statistics of SequenceAccumulator objects with
                matching keys.

        Attributes:
            accumulator (SequenceEncodableStatisticAccumulator): SequenceEncodableStatisticAccumulator object for
                accumulating sufficient statistics of base distribution compatible with data type T.
            len_accumulator (SequenceEncodableStatisticAccumulator): SequenceEncodableStatisticAccumulator object
                for accumulating sufficient statistics of length distribution compatible with non-negative integers.
            len_normalized (Optional[bool]): Geometric mean of density taken if set to True. Else ignored.
            keys (Optional[str]): Set keys for merging sufficient statistics of SequenceAccumulator objects with
                matching keys.
            null_len_accumulator (bool): True if len_accumulator is an instance of NullAccumulator object.
            _init_rng (bool): True if _len_rng has been initialized.
            _len_rng (Optional[RandomState]): None if not initialized. Set to a RandomState object with call from
                initialize or seq_initialize functions.

        """
        self.accumulator = accumulator
        self.len_accumulator = len_accumulator
        self.key = keys
        self.len_normalized = len_normalized

        self.null_len_accumulator = isinstance(self.len_accumulator, NullAccumulator)

        ### Seeds for initialize/seq_initialize consistency
        self._init_rng = False
        self._len_rng: Optional[RandomState] = None

    def update(self, x: Sequence[T], weight: float, estimate: Optional[SequenceDistribution]) -> None:
        """Update sufficient statistics for SequenceAccumulator object.

        Aggregate sufficient statistics of base accumulator and length accumulator with sufficient statistics of
        'estimate' if passed.

        Note: Calls update() of accumulator and len_accumulator.

        Args:
            x (Sequence[T]): A sequence of iid observations of data type T.
            weight (float): Weight for sequence observation.
            estimate (Optional[SequenceDistribution]): SequenceDistribution instance to aggregate sufficient statistics
                with.

        Returns:
            None.

        """
        if estimate is None:
            w = weight / len(x) if (self.len_normalized and len(x) > 0) else weight

            for i in range(len(x)):
                self.accumulator.update(x[i], w, None)

            if not self.null_len_accumulator:
                self.len_accumulator.update(len(x), weight, None)

        else:
            w = weight / len(x) if (self.len_normalized and len(x) > 0) else weight

            for i in range(len(x)):
                self.accumulator.update(x[i], w, estimate.dist)

            if not self.null_len_accumulator:
                self.len_accumulator.update(len(x), weight, estimate.len_dist)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Set the _len_rng for consistency between initialize and seq_initialize methods.

        Args:
            rng (RandomState): RandomState object for initializing _len_rng.

        Returns:
            None.

        """
        self._len_rng = RandomState(seed=rng.randint(2**31))
        self._init_rng = True

    def initialize(self, x: Sequence[T], weight: float, rng: RandomState) -> None:
        """Initialize SequenceAccumulator object with weighted observation.

        Note: Calls _rng_initialize() method if _len_rng has not been set. This ensures consistency between initialize
        and seq_initialize calls.

        Method invokes calls to accumulator.initialize() and len_accumulator.initialize() if len_accumulator is not
        NullAccumulator.

        Args:
            x (Sequence[T]): Sequence of iid observations from base distribution.
            weight (float): Weight for sequence observation.
            rng (RandomState): RandomState object used to set seed in random initialization.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        if len(x) > 0:
            w = weight / len(x) if self.len_normalized else weight
            for xx in x:
                self.accumulator.initialize(xx, w, rng)

        if not self.null_len_accumulator:
            self.len_accumulator.initialize(len(x), weight, self._len_rng)

    def seq_initialize(self, x: E, weights: np.ndarray, rng: RandomState) -> None:
        """Vectorized initialization of SequenceAccumulator sufficient statistics from sequence encoded x.


        Args:
            x (E): Encoded data sequence.
            weights (np.ndarray[float]): Numpy array of floats for weighting observations.
            rng (RandomState): RandomState object used to set seed in random initialization.

        Returns:
            None.

        """
        idx, icnt, inz, enc_seq, enc_nseq = x

        if not self._init_rng:
            self._rng_initialize(rng)

        w = weights[idx] * icnt[idx] if self.len_normalized else weights[idx]

        self.accumulator.seq_initialize(enc_seq, w, rng)

        if not self.null_len_accumulator:
            self.len_accumulator.seq_initialize(enc_nseq, weights, self._len_rng)

    def seq_update(self, x: E, weights: np.ndarray, estimate: Optional['SequenceDistribution']) -> None:
        """Vectorized update of SequenceAccumulator sufficient statistics from sequence encoded x.

        Args:
            x (E): Encoded data sequence.
            weights (np.ndarray[float]): Numpy array of floats for weighting observations.
            estimate (Optional[SequenceDistribution]): SequenceDistribution instance to aggregate sufficient statistics
                with.

        Returns:
            None.

        """
        idx, icnt, inz, enc_seq, enc_nseq = x

        w = weights[idx] * icnt[idx] if self.len_normalized else weights[idx]

        self.accumulator.seq_update(enc_seq, w, estimate.dist if estimate is not None else None)

        if not self.null_len_accumulator:
            self.len_accumulator.seq_update(enc_nseq, weights, estimate.len_dist if estimate is not None else None)

    def combine(self, suff_stat: Tuple[SS1, Optional[SS2]]) -> 'SequenceAccumulator':
        """Combine the sufficient statistics of SequenceAccumulator instance with suff_stat arg.

        Args:
            suff_stat (Tuple[SS1, Optional[SS2]]): Tuple of sufficient statistics of base distribution and value for
                length distribution.

        Returns:
            SequenceAccumulator object.

        """
        self.accumulator.combine(suff_stat[0])

        if not self.null_len_accumulator:
            self.len_accumulator.combine(suff_stat[1])

        return self

    def value(self) -> Tuple[Any, Optional[Any]]:
        """Return Tuple[SS1, Optional[SS2]], sufficient statistics of base accumulator and length accumulator."""
        return self.accumulator.value(), self.len_accumulator.value()

    def from_value(self, x: Tuple[SS1, Optional[SS2]]) -> 'SequenceAccumulator':
        """Set the SequenceAccumulator base accumulator and length accumulator to values of x.

        Args:
            x (Tuple[SS1, Optional[SS2]]): Tuple of sufficient statistics of base distribution and value for length
                distribution.

        Returns:
            SequenceAccumulator object.

        """
        self.accumulator.from_value(x[0])

        if not self.null_len_accumulator:
            self.len_accumulator.from_value(x[1])

        return self

    def get_seq_lambda(self):
        rv = self.accumulator.get_seq_lambda()

        if self.len_accumulator is not None:
            rv.extend(self.len_accumulator.get_seq_lambda())

        return rv

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merges member sufficient statistics with sufficient statistics that contain matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

        self.accumulator.key_merge(stats_dict)

        if not self.null_len_accumulator:
            self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Set member sufficient statistics to values of objects with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to sufficient statistics.

        Returns:
            None.

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

        self.accumulator.key_replace(stats_dict)

        if not self.null_len_accumulator:
            self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'SequenceDataEncoder':
        """Create SequenceDataEncoder for encoding sequences of iid observations of SequenceDistribution.

        Base distribution DataSequenceEncoder and length distribution DataSequenceEncoder objects are passed.

        Returns:
            SequenceDataEncoder object.

        """
        encoder = self.accumulator.acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()
        encoders = (encoder, len_encoder)
        return SequenceDataEncoder(encoders=encoders)


class SequenceAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self,
                 dist_factory: StatisticAccumulatorFactory,
                 len_factory: StatisticAccumulatorFactory = NullAccumulatorFactory(),
                 len_normalized: Optional[bool] = False,
                 keys: Optional[str] = None) -> None:
        """SequenceAccumulatorFactory object for creating SequenceAccumulator objects.

        Args:
            dist_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for base distribution of sequence
                distribution.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for length distribution of sequence
                distribution.
            len_normalized (Optional[bool]): Standardize by length of sequence distribution.
            keys (Optional[str]): Set key for merging/combining sufficient statistics of SequenceAccumulator.

        Attributes:
            dist_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for base distribution of sequence
                distribution.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for length distribution of sequence
                distribution, set to NullAccumulatorFactory() if corresponding SequenceDistribution has no length
                distribution desired to be estimated.
            len_normalized (Optional[bool]): Standardize by length of sequence distribution.
            keys (Optional[str]): Key for merging/combining sufficient statistics of SequenceAccumulator.

        """
        self.dist_factory = dist_factory
        self.len_factory = len_factory
        self.len_normalized = len_normalized
        self.keys = keys

    def make(self) -> 'SequenceAccumulator':
        """Return SequenceAccumulator with SequenceEncodableStatisticAccumulator objects created from dist_factory and
            len_factory."""
        len_acc = self.len_factory.make()
        return SequenceAccumulator(self.dist_factory.make(), len_acc, self.len_normalized, self.keys)


class SequenceEstimator(ParameterEstimator):

    def __init__(self,
                 estimator: ParameterEstimator,
                 len_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 len_normalized: Optional[bool] = False,
                 name: Optional[str] = None,
                 keys: Optional[str] = None) -> None:
        """SequenceEstimator object for estimating SequenceDistribution from aggregated sufficient statistics.

        Requires arg 'estimator' to be ParameterEstimator of data type T, compatible with the observed entry values
        of SequenceDistribution.

        If arg 'len_estimator' is passed, it must be a ParameterEstimator object compatible with non-negative
        integers.

        If len_estimator is NullEstimator() or None, len_dist is used as length distribution in estimation.

        Args:
            estimator (ParameterEstimator): Set ParameterEstimator for base distribution.
            len_estimator (Optional[ParameterEstimator]): Set ParameterEstimator for length distribution.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Set a fixed length distribution.
            len_normalized (Optional[bool]): Take geometric mean of density if True.
            name (Optional[str]): Set name to SequenceEstimator instance.
            keys (Optional[str]): Set key to SequenceEstimator instance for merging sufficient statistics.

        Attributes:
            estimator (ParameterEstimator): ParameterEstimator for base distribution.
            len_estimator (Optional[ParameterEstimator]): ParameterEstimator for length distribution. If None, set to
                NullEstimator.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Set a fixed length distribution.
            len_normalized (Optional[bool]): Take geometric mean of density if True.
            name (Optional[str]): Name of SequenceEstimator instance.
            keys (Optional[str]): Key for SequenceEstimator instance used in aggregating sufficient statistics.

        """
        self.estimator = estimator
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.keys = keys
        self.len_normalized = len_normalized
        self.name = name

    def accumulator_factory(self) -> 'SequenceAccumulatorFactory':
        """Return SequenceAccumulatorFactory from len_estimator and estimator member variables with keys passed."""
        len_factory = self.len_estimator.accumulator_factory()
        dist_factory = self.estimator.accumulator_factory()

        return SequenceAccumulatorFactory(dist_factory, len_factory, self.len_normalized, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[Any, Optional[Any]]) -> 'SequenceDistribution':
        if isinstance(self.len_estimator, NullEstimator):
            return SequenceDistribution(self.estimator.estimate(nobs, suff_stat[0]), len_dist=self.len_dist,
                                        len_normalized=self.len_normalized, name=self.name)

        else:
            return SequenceDistribution(self.estimator.estimate(nobs, suff_stat[0]),
                                        len_dist=self.len_estimator.estimate(nobs, suff_stat[1]),
                                        len_normalized=self.len_normalized, name=self.name)


class SequenceDataEncoder(DataSequenceEncoder):

    def __init__(self,
                 encoders: Tuple[DataSequenceEncoder, DataSequenceEncoder]) -> None:
        """SequenceDataEncoder object for encoding sequences of iid observations from sequence distributions.

        encoders[0] is a DataSequenceEncoder for data type T, producing encoded sequences of type T1.
        encoders[1] is a DataSequenceEncoder for data type int, production encoded sequences of type T2 or None.

        Args:
            encoders (Tuple[DataSequenceEncoder, DataSequenceEncoder]): Tuple of DataSequenceEncoder objects for
                distribution and length distribution of sequence distribution.

        Attributes:
            encoder (DataSequenceEncoder): DataSequenceEncoder object for the distribution of sequence distribution.
            len_encoder (DataSequenceEncoder): DataSequenceEncoder object for the length distribution of sequence
                distribution. Generally NullDataEncoder() object is no intended length distribution.
            null_len_encoder (bool): True if len_encoder is a NullDataEncoder(), else False.
        """
        self.encoder = encoders[0]
        self.len_encoder = encoders[1]

        self.null_len_enc = isinstance(self.len_encoder, NullDataEncoder)

    def __str__(self) -> str:
        """Returns string representation of SequenceDataEncoder object."""
        s = 'SequenceDataEncoder('
        s += str(self.encoder) + ',len_encoder='
        s += str(self.len_encoder) + ')'

        return s

    def __eq__(self, other: object) -> bool:
        """Checks if other object is an equivalent to SequenceDataEncoder instance.

        Checks if other is a SequenceDataEncoder. If it is, the encoder and len_encoder memeber variables must also
        be equivalent.

        Args:
            other (object): Object to compare to SequenceDataEncoder instance.

        Returns:
            True if other is equivalent to SequenceDataEncoder object instance.

        """
        if not isinstance(other, SequenceDataEncoder):
            return False

        else:
            if not self.encoder == other.encoder:
                return False

            if not self.len_encoder == other.len_encoder:
                return False

            return True

    def seq_encode(self, x: Sequence[Sequence[T]])\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[Any, ...], Optional[Any]]:
        """Encode iid observations of Sequence distribution for use with vectorized "seq_" functions.

        Data 'x' must be a Sequence of Sequences containing data types T consistent with the distribution encoder
        (DataSequenceEncoder) object 'encoder'. That is x: Sequence containing 'N' objects of xx: Sequence[T].

        Consider example data x = [ [0,1,2], [],[3,4]]. Then x: Sequence[Sequence[int]].

        Assume the data type returned by 'encoder.seq_encode()' is T1, and 'len_encoder.seq_encode()' is T2.

        rv1 (ndarray[int]): Index for values of positive length sequence entries. I.e. x produces -> [0,0,0,2,2]
        rv2 (ndarray[float]): Inverse of sequence lengths. I.e. x -> [1/3,1/3,1/3,0,1/2,1/2]
        rv3 (ndarray[bool]): True if length of sequence is not 0. I.e. x -> [True,True, True, False, True,True]
        rv4 (T1): Sequence encoding resulting from encoder.seq_encode() on list of all observed values.
        rv5 (Optional[T2]): Sequence encoding resulting len_encoder.seq_encode() on all sequence length values.

        Args:
            x (Sequence[Sequence[T]]): Sequence of Sequence[T], where T is compatible with base distribution of
                sequence distribution. Sequence of iid sequence observations.

        Returns:

        """
        tx = []
        nx = []
        tidx = []

        for i in range(len(x)):
            nx.append(len(x[i]))

            for j in range(len(x[i])):
                tidx.append(i)
                tx.append(x[i][j])

        rv1 = np.asarray(tidx, dtype=int)
        rv2 = np.asarray(nx, dtype=float)
        rv3 = (rv2 != 0)

        rv2[rv3] = 1.0 / rv2[rv3]

        rv4 = self.encoder.seq_encode(tx)

        ### None if NullDataEncoder() for length
        rv5 = self.len_encoder.seq_encode(nx)

        return rv1, rv2, rv3, rv4, rv5

