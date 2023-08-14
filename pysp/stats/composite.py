"""Create, estimate, and sample from a Composite distribution.

Defines the CompositeDistribution, CompositeSampler, CompositeAccumulatorFactory, CompositeAccumulator,
CompositeEstimator, and the CompositeDataEncoder classes for use with pysparkplug.

Data type: (Tuple[T_0, ... T_{n-1}]): The CompositeDistribution of size 'n' is a joint distribution for
independent observations of 'n'-tupled data. Each component 'k' of the CompositeDistribution has data type T_k that
must be compatible with data type T_k.

"""
import numpy as np
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder
from typing import Optional, List, Union, Any, Tuple, Sequence, TypeVar, Dict
from pysp.arithmetic import maxrandint


T = Tuple[Any, ...]
E = TypeVar('E')
SS = TypeVar('SS')


class CompositeDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self,
                 dists: Sequence[SequenceEncodableProbabilityDistribution]) -> None:
        """CompositeDistribution for modeling independent distributions of from (Dist_0,Dist_1,...,Dist_{n-1}).

        Data type must be (T_0, T_1, ..., T_{n-1}), where data type T_k is consistent with distribution Dist_k. The
        density for a single observation tuple x = (x_0,x_1,...,x_{n-1}) is given by,

        p_mat(x) = p_mat(x_0 | Dist_0)*p_mat(x_1 | Dist_1)*...*p_mat(x_{n-1} | Dist_{n-1}).

        Args:
            dists (Sequence[SequenceEncodableProbabilityDistribution]): Distributions given by Dist_k above.

        Attributes:
            dists: (Sequence[SequenceEncodableProbabilityDistribution]): Distributions given by Dist_k above.
            counts (int): Number of components (i.e. len(dists)).

        """
        self.dists = dists
        self.count = len(dists)

    def __str__(self) -> str:
        """Returns str name of CompositeDistribution with each dist as well."""
        return 'CompositeDistribution((%s))' % (','.join(map(str, self.dists)))

    def density(self, x: Tuple[Any, ...]) -> float:
        """Evaluates density of CompositeDistribution for single observation tuple x.

        p_mat(x) = p_mat(x_0 | dist_0)*p_mat(x_1 | dist_1)*...*p_mat(x_{n-1} | dist_{n-1}),

        where dist_k is the k^{th} element of member variable dists and is consistent with data type type(x[k]).

        Args:
            x (Tuple[Any, ...]): Tuple of length = len(dists), the k^{th} data type must be consistent with dists[k].

        Returns:
            Density as float.

        """
        rv = 0.0

        for i in range(1, self.count):
            rv *= self.dists[i].density(x[i])

        return rv

    def log_density(self, x: Tuple[Any, ...]) -> float:
        """Evaluates log-density of CompositeDistribution for single observation tuple x.

        log(p_mat(x)) = log(p_mat(x_0 | dist_0)) + log(p_mat(x_1 | dist_1)) + ... + log(p_mat(x_{n-1} | dist_{n-1})),

        where dist_k is the k^{th} element of member variable dists and is consistent with data type type(x[k]).

        Args:
            x (Tuple[Any, ...]): Tuple of length = len(dists), the k^{th} data type must be consistent with dists[k].

        Returns:
            Log-density as float.

        """
        rv = self.dists[0].log_density(x[0])

        for i in range(1, self.count):
            rv += self.dists[i].log_density(x[i])

        return rv

    def seq_log_density(self, x: E) -> np.ndarray:
        """Vectorized evaluation of log density for Tuple of dist encoded data.

        Each entry of x is an encoded sequence, encoded by the DataSequenceEncoder of dist[k].dist_to_encoder().

        Note: len(x) == len(dists).
        Args:
            x (E): Tuple of length = len(dists), with k^{th} entry given by encoded sequence of dist[k]'s.

        Returns:
            np.ndarray of log_density evaluated at all encoded data points.

        """
        rv = self.dists[0].seq_log_density(x[0])

        for i in range(1, self.count):
            rv += self.dists[i].seq_log_density(x[i])

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'CompositeSampler':
        """Create CompositeSampler for sampling from CompositeDistribution instance.

        Args:
            seed (Optional[int]): Seed to set for sampling with RandomState.

        Returns:
            CompositeSampler object.

        """
        return CompositeSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'CompositeEstimator':
        """Create CompositeEstimator for estimating CompositeDistribution.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics in estimation.

        Returns:
            CompositeEstimator object.

        """
        return CompositeEstimator([d.estimator(pseudo_count=pseudo_count) for d in self.dists])

    def dist_to_encoder(self) -> 'CompositeDataEncoder':
        """Creates CompositeDataEncoder for encoding sequence of tuple data.

        Passes 'encoders', which is a list of DataSequenceEncoders for each component of the CompositeDistribution.

        Returns:
            CompositeDataEncoder object.

        """
        encoders = tuple([d.dist_to_encoder() for d in self.dists])

        return CompositeDataEncoder(encoders=encoders)


class CompositeSampler(DistributionSampler):

    def __init__(self, dist: 'CompositeDistribution', seed: Optional[int] = None) -> None:
        """CompositeSampler used to generate samples from CompositeDistribution.

        Args:
            dist (CompositeDistribution): CompositeDistribution to draw samples from.
            seed (Optional[int]): Seed to set for sampling with RandomState.

        Attributes:
            dist (CompositeDistribution): CompositeDistribution to draw samples from.
            rng (RandomState): RandomState with seed set if provided.
            dist_samplers (List[DistributionSamplers]): List of DistributionSamplers for each component
                (len=len(dists)).
        """
        self.dist = dist
        self.rng = RandomState(seed)
        self.dist_samplers = [d.sampler(seed=self.rng.randint(maxrandint)) for d in dist.dists]

    def sample(self, size: Optional[int] = None) -> Union[List[Tuple[Any, ...]], Tuple[Any, ...]]:
        """Generate independent samples from a CompositeDistribution.

        If size is None, draw one sample and return as Tuple of length = len(dists). If size > 0,
        draw size samples and return a list of length size containing tuples of len(dists).

        Args:
            size (Optional[int]): If None, draw 1 sample. Else, draw size number of iid samples.

        Returns:
            A tuple of length = len(dists) or a list of length size containing tuples of length = len(dists).

        """
        if size is None:
            return tuple([d.sample(size=size) for d in self.dist_samplers])

        else:
            return list(zip(*[d.sample(size=size) for d in self.dist_samplers]))


class CompositeAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, accumulators: Sequence[SequenceEncodableStatisticAccumulator], keys: Optional[str] = None) -> None:
        """CompositeAccumulator object used for aggregating suffcient statistics of each component of the
            CompositeDistribution.

        Args:
            accumulators (List[SequenceEncodableStatisticAccumulator]):
            keys (Optional[str]): All CompositeAccumulators with same keys will have suff-stats merged.

        Attributes:
            accumulators (List[SequenceEncodableStatisticAccumulator]): List of SequenceEncodableStatisticAccumulator
                objects for accumulating sufficient statsitics for each component of the CompositeDistribution.
            count (int): Length of accumulators.
            keys (Optional[str]): All CompositeAccumulators with same keys will have suff-stats merged.
            _init_rng (bool): Is True if _acc_rng has been set by a single function call to initialize.
            _acc_rng (List[RandomState]): List of RandomState objects generated from seeds set by rng in initialize.

        """
        self.accumulators = accumulators
        self.count = len(accumulators)
        self.key = keys

        ### variables for initialization
        self._init_rng = False
        self._acc_rng: Optional[List[RandomState]] = None

    def update(self, x: T, weight: float, estimate: Optional['CompositeDistribution']) -> None:
        """Calls update on each CompositeAccumulator component[k], passing x[k] and weight along with estimate
            if provided.

        Component-wise update() calls to accumulator for each component of x. The same weight is passed to each update
        call, along with the corresponded component-distribution estimate, if estimate is provided.

        Args:
            x (Any): Category label.
            weight (float): Weight for the observation x.
            estimate (Optional['CategoricalDistribution']): Kept for consistency with update method in
                SequenceEncodableStatisticAccumulator.

        Returns:
            None

        """
        if estimate is not None:
            for i in range(0, self.count):
                self.accumulators[i].update(x[i], weight, estimate.dists[i])

        else:
            for i in range(0, self.count):
                self.accumulators[i].update(x[i], weight, None)

    def _rng_initialize(self, rng: RandomState) -> None:
        seeds = rng.randint(2 ** 31, size=self.count)
        self._acc_rng = [RandomState(seed=seed) for seed in seeds]

    def initialize(self, x: Tuple[Any, ...], weight: float, rng: np.random.RandomState) -> None:
        """Initialize each accumulator of CompositeAccumulator with component x[i] of x and weight.

        Note: rng is used to set List[RandomState]: _acc_rng. This is done to ensure iteration over observations of data,
        produces the same initialization as seq_initialize().

        Args:
            x (Tuple[Any, ...]): Observation Tuple of length count, that is component-wise compatible with
                CompositeAccumulator member variable accumulators.
            weight (float): Weight for the observation x.
            rng (RandomState): Used to set seed of _acc_rng if not set.

        Returns:
            None

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        for i in range(0, self.count):
            self.accumulators[i].initialize(x[i], weight, self._acc_rng[i])

    def seq_initialize(self, x: E, weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Vectorized initialization of each accumulator of CompositeAccumulator with encoded data x.

        Note: rng is used to set List[RandomState]: _acc_rng. This is done to ensure iteration over observations of
        data, produces the same initialization as seq_initialize().

        Args:
            x (E): Tuple of component wise sequence encoding of data.
            weights (np.ndarray): Numpy array weights for the encoded observations.
            rng (RandomState): Used to set seed of _acc_rng if not set.

        Returns:
            None

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        for i in range(0, self.count):
            self.accumulators[i].seq_initialize(x[i], weights, self._acc_rng[i])

    def get_seq_lambda(self) -> List[Any]:
        rv = []
        for i in range(self.count):
            rv.extend(self.accumulators[i].get_seq_lambda())
        return rv

    def seq_update(self, x: Tuple[Any, ...], weights: np.ndarray,
                   estimate: Optional['CompositeDistribution']) -> None:
        """Vectorized aggregation of sufficient statistics for each component of CompositeAccumulator.

        Requires sequence encoded input x, from CompositeDataEncoder.seq_encode(data).

        Args:
            x (Tuple[Any, ...]): Encoded sequence Tuple of length count, that is a component wise sequence encoding of
                data.
            weights (np.ndarray): Numpy array weights for the encoded observations.
            estimate:

        Returns:
            None.

        """
        for i in range(self.count):
            self.accumulators[i].seq_update(x[i], weights, estimate.dists[i] if estimate is not None else None)

    def combine(self, suff_stat: SS) -> 'CompositeAccumulator':
        """Aggregate the sufficient statistics of CompositeAccumulator with input suff_stat.

        Args:
            suff_stat (SS): Tuple of sufficient statistics for each component of the CompositeAccumulator.

        Returns:
            None

        """
        for i in range(0, self.count):
            self.accumulators[i].combine(suff_stat[i])

        return self

    def value(self) -> Tuple[Any, ...]:
        """Returns Tuple of length equal to member variable count, containing sufficient statistics for each
            component."""
        return tuple([x.value() for x in self.accumulators])

    def from_value(self, x: SS) -> 'CompositeAccumulator':
        """Set CompositeAccumulator instance sufficient statistics to x.

        Args:
            x (SS): Tuple of length equal to member variable count, containing sufficient statistics
                for each component.

        Returns:
            CompositeAccumulator

        """
        self.accumulators = [self.accumulators[i].from_value(x[i]) for i in range(len(x))]
        self.count = len(x)

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Combines the sufficient statistics of CompositeAccumulators that have the same key value.

        If key is not in the stats_dict (dictionary), the key and accumulator are added to the dict.

        Args:
            stats_dict (Dict[str, Any]): Dictionary for mapping keys to CompositeAccumulators.

        Returns:
            None

        """
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

        for u in self.accumulators:
            u.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Set CompositeAccumulator sufficient statistic attributes values to suff stats with matching keys.

        Args:
            stats_dict (Dict[str, Any]): Maps member variable key to
                CompositeAccumulator with same key.

        Returns:
            None

        """
        if self.key is not None:
            if self.key in stats_dict:
                self.from_value(stats_dict[self.key].value())

        for u in self.accumulators:
            u.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'CompositeDataEncoder':
        """Creates CompositeDataEncoder for encoding sequence of tuple data.

        encoders is a list of DataSequenceEncoders for each component of the CompositeDistribution.

        Returns:
            CompositeDataEncoder

        """
        encoders = tuple([acc.acc_to_encoder() for acc in self.accumulators])

        return CompositeDataEncoder(encoders=encoders)


class CompositeAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, factories: Sequence[StatisticAccumulatorFactory], keys: Optional[str] = None) -> None:
        """CompositeAccumulatorFactory used for lightweight creation of CompositeAccumulator.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): List of StatisticAccumulatorFactory objects for each
                component.
            keys (Optional[str]): Declare keys for merging sufficient statistics of CompositeAccumulator objects.

        Attributes:
            factories (List[StatisticAccumulatorFactory]): List of StatisticAccumulatorFactory objects for each
                component.
            keys (Optional[str]): Declare keys for merging sufficient statistics of CompositeAccumulator objects.
        """
        self.factories = factories
        self.keys = keys

    def make(self) -> 'CompositeAccumulator':
        """Create a CompositeAccumulator object from list of StatisticAccumulatorFactory objects.

        Returns:
            CompositeAccumulator

        """
        return CompositeAccumulator([u.make() for u in self.factories], self.keys)


class CompositeEstimator(ParameterEstimator):

    def __init__(self, estimators: Sequence[ParameterEstimator], keys: Optional[str] = None) -> None:
        """CompositeEstimator object used to estimate CompositeDistribution from sufficient statistics of each
            component.

        Args:
            estimators (List[ParameterEstimator]): List of ParameterEstimator objects for each component of
                CompositeEstimator.
            keys (Optional[str]): Keys used for merging sufficient statistics of CompositeEstimator objects.

        Attributes:
            estimators (List[ParameterEstimator]): List of ParameterEstimator objects for each component of
                CompositeEstimator.
            keys (Optional[str]): Keys used for merging sufficient statistics of CompositeEstimator objects.
            count (int): Number of components in CompositeEstimator.

        """
        self.estimators = estimators
        self.count = len(estimators)
        self.keys = keys

    def accumulator_factory(self) -> 'CompositeAccumulatorFactory':
        """Creates CompositeAccumulatorFactory from each ParameterEstimator in estimators.

        Returns:
            CompositeAccumulatorFactory.

        """
        return CompositeAccumulatorFactory([u.accumulator_factory() for u in self.estimators], self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: SS) -> 'CompositeDistribution':
        """Estimate a CompositeDistribution from an aggregated sufficient statistics Tuple for a given number of
            observations (nobs).

        Args:
            nobs (Optional[float]): Weighted number of observations used to form suff_stat.
            suff_stat (SS): Tuple of sufficient statistics for each ParameterEstimator of estimators.

        Returns:
            CompositeDistribution estimated from argument aggregated sufficient statistics (suff_stat), from a given
                number of observation (nobs).

        """
        return CompositeDistribution(tuple([est.estimate(nobs, ss) for est, ss in zip(self.estimators, suff_stat)]))


class CompositeDataEncoder(DataSequenceEncoder):

    def __init__(self, encoders: Sequence[DataSequenceEncoder]) -> None:
        """CompositeDataEncoder used for encoding data.

        Data must be of form Sequence[Tuple[Any,...]]. Each encoder component must be compatible with each data
            component of the data.

        Args:
            encoders (Sequence[DataSequenceEncoder]): DataSequenceEncoders for each component of the
                CompositeDistribution.

        Attributes:
            encoders (Sequence[DataSequenceEncoder]): DataSequenceEncoders for each component of the
                CompositeDistribution.

        """
        self.encoders = encoders

    def __eq__(self, other: object) -> bool:
        """Check if an object is an equivalent to instance of CompositeDataEncoder.

        If other is CompositeDataEncoder, it must also have equivalent DataSequenceEncoder object for each
        component of encoder member variable.

        Args:
            other (object): Object to be compared to CompositeDataEncoder.

        Returns:
            True if other can produce and equivalent encoding to instance of CompositeDataEncoder.

        """
        if not isinstance(other, CompositeDataEncoder):
            return False

        else:

            for i, encoder in enumerate(self.encoders):
                if not encoder == other.encoders[i]:
                    return False

        return True

    def __str__(self) -> str:
        """Returns string representation of CompositeDataEncoder and DataSequenceEncoder instance
        for each component.
        """

        s = 'CompositeDataEncoder(['

        for d in self.encoders[:-1]:
            s += str(d) + ','

        s += str(self.encoders[-1]) + '])'

        return s

    def seq_encode(self, x: Sequence[Tuple[Any, ...]]) -> Tuple[Any, ...]:
        """Encode Sequence of tuples of data for use with vectorized "seq_" functions.

        The input x must be a Sequence of Tuples of length equal to the length of encoders. Each component tuple
        observation of x, say x[i], must be component-wise compatible with encoders.

        Args:
            x (Sequence[Tuple[Any, ...]]): Sequence of tuples of length equal to len(encoders).

        Returns:
            Tuple of length equal to len(encoders), with entry i, containing the sequence encoding from encoder[i]
            for all observations of component i from x.

        """
        enc_data = []

        for i, encoder in enumerate(self.encoders):
            enc_data.append(encoder.seq_encode([u[i] for u in x]))

        return tuple(enc_data)

