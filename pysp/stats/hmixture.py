"""Create, estimate, and sample from a hierarchical mixture distribution with K components consisting of
sequence mixture distribution with L topics shared across all K components.

Defines the HierarchicalMixtureDistribution, HierarchicalMixtureSampler, HierarchicalMixtureEstimatorAccumulatorFactory,
HierarchicalMixtureEstimatorAccumulator, HierarchicalMixtureEstimator, and the HierarchicalMixtureDataEncoder classes
for use with pysparkplug.

Data type: Sequence[T], where 'T' is the type of the topic distributions.

Note that this is a mixture with K 'outer-mixture' components consisting of L topic distributions
{f_l(theta_l)}_{l=1}^{L}, with 'inner-mixture' weights {tau_{k,l}}_{l=1}^{L} for each of the K components.

Sampling proceeds as follows. Each sample is a sequence of length 'N' (where can be modeled with a length distribution
P_len()) from an outer-mixture component k with probability w_k. Sampling from mixture component 'k' consists of
sampling from a mixture with topics {f_l(theta_l)}_{l=1}^{L} and 'inner-mixture' weights {tau_{k,l}}_{l=1}^{L}.

Example: Let x = (x_1, x_2, x_3, ...., x_N) be an observation from a hierarchical mixture distribution of length 'N'.
Let Z and U be a random variables s.t. p_mat(Z=k) = w_k and p_mat(U=l | Z = k) = tau_{k,l}. Then

    alpha_i = x_i | Z = k ~ sum_{l=1}^{L} f_l(theta_l)*tau_{k,l}, for i = 1,2,...,N.

Further,

    alpha_i | U=l ~ f_l(theta_l), for i = 1,2,3,...,N.

"""
import numpy as np
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, ParameterEstimator, DistributionSampler, \
    StatisticAccumulatorFactory, SequenceEncodableStatisticAccumulator, DataSequenceEncoder
from pysp.stats.null_dist import NullDistribution, NullAccumulator, NullAccumulatorFactory, NullEstimator
from pysp.arithmetic import maxrandint
import pysp.utils.vector as vec
from numpy.random import RandomState
from pysp.stats.mixture import MixtureDistribution
from pysp.stats.sequence import SequenceDistribution

from typing import Optional, List, Union, Any, Tuple, Sequence, TypeVar, Dict

T = TypeVar('T') ## Data type for topics
E1 = TypeVar('E1') ## Encoded sequence from topic encoder.
E2 = TypeVar('E2') ## Encoded sequence from length encoder.
SS1 = TypeVar('SS1') ### Suff stat type for topics.
SS2 = TypeVar('SS2') ## Suff stat type for length distribution.


class HierarchicalMixtureDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self,
                 topics: Sequence[SequenceEncodableProbabilityDistribution],
                 mixture_weights: Union[List[float], np.ndarray],
                 topic_weights: Union[List[List[float]], np.ndarray],
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None)) -> None:
        """HierarchicalMixtureDistribution object defining a hierarchical mixture distribution.

        Args:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Topic distributions shared in hierarchical
                mixture distribution.
            mixture_weights (Union[List[float], np.ndarray]): One-d array of floats for weights on components
                of mixtures. Should sum to 1.0.
            topic_weights (Union[List[List[float]], np.ndarray]): 2-d array with rows containing weights for each
                component mixture distribution. All rows should sum to 1.0.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for the length on the sequence
                distribution for the component mixtures
            name (Optional[str]): Set name for object instance.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Set keys for the weights and topics.

        Attributes:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Topic distributions shared in hierarchical
                mixture distribution.
            num_topics (int): Number of topic distributions (i.e. sets number of inner-mixture weights).
            num_mixtures (int): Number of weights in outter-mixture (i.e. sets numer of top-layer mixture weights.)
            w (np.ndarray): 1-d numpy array of outer-mixture weights. Should sum to 1.
            log_w (np.ndarray): Numpy array of the log of w above.
            taus (np.ndarray): 2-d array of dimension (num_mixtures by num_topics).
            log_taus (np.ndarray): 2-d array of the log of tau above.
            len_dist (SequenceEncodableProbabilityDistribution): Distribution for the sequence length on topics.
                Defaults to the NullDistribution if None is passed.
            name (Optional[str]): Name for object instance.
            keys (Tuple[Optional[str], Optional[str]]): Keys for the weights and topics.

        """
        with np.errstate(divide='ignore'):
            self.topics = topics
            self.num_topics = len(topics)
            self.num_mixtures = len(mixture_weights)
            self.w = np.asarray(mixture_weights, dtype=np.float64)
            self.log_w = np.log(self.w)
            self.taus = np.asarray(topic_weights, dtype=np.float64)
            self.log_taus = np.log(self.taus)
            self.len_dist = len_dist
            self.name = name
            self.keys = keys if keys is not None else (None, None)

    def __str__(self) -> str:
        """Return a string representation for the object instance."""
        s1 = '[' + ','.join([str(u) for u in self.topics]) + ']'
        s2 = repr(list(self.w))
        s3 = repr(list(map(list, self.taus)))
        s4 = repr(self.len_dist) if self.len_dist is None else str(self.len_dist)
        s5 = repr(self.name)
        s6 = repr(self.keys)
        return 'HierarchicalMixtureDistribution(%s, %s, %s, len_dist=%s, name=%s, keys=%s)' % (s1, s2, s3, s4, s5, s6)

    def density(self, x: Sequence[T]) -> float:
        """Evaluate the density of an observation from hierarchical mixture distribution.

        Args:
            x (Sequence[T]): A sequence of type data type T's.

        Returns:
            Density evaluated at x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[T]) -> float:
        """Evaluate the log density of an observation from hierarchical mixture distribution.

        Note: Observation is a sequence.

        Args:
            x (Sequence[T]): A sequence of type data type T's.

        Returns:
            Log-density evaluated at x.

        """
        enc_x = self.dist_to_encoder().seq_encode([x])
        return self.seq_log_density(enc_x)[0]

    def posterior(self, x: Sequence[T]) -> np.ndarray:
        """Compute the posterior over the mixture components for the outer-mixture at observed value x.

        Args:
            x (Sequence[T]): An observed sequence of data type T.

        Returns:
            Numpy array of length 'num_mixtures'.

        """
        enc_x = self.dist_to_encoder().seq_encode([x])
        return self.seq_posterior(enc_x)[0]


    def component_log_density(self, x: Sequence[T]) -> np.ndarray:
        """Evaluate the component-wise log-density for an observation from a hierarchical mixture model.

        Args:
            x (Sequence[T]): An observation from a hierarchical mixture model.

        Returns:
            Numpy array length of 'num_mixtures'.

        """
        n = len(x)
        ll_topic = np.zeros((n, self.num_topics))

        for i in range(n):
            ll_topic[i, :] = np.array([self.topics[j].log_density(x[i]) for j in range(self.num_topics)])

        ll_topic_max = np.max(ll_topic, axis=0, keepdims=True)
        ll_topic -= ll_topic_max
        np.exp(ll_topic, out=ll_topic)
        ll_topic = np.log(np.sum(ll_topic, axis=0, keepdims=False)) + ll_topic_max.flatten()

        rv = np.zeros(self.num_mixtures)
        for k in range(self.num_mixtures):
            ll_k = ll_topic + self.log_taus[k, :]
            ll_k[self.taus[k, :] == 0.0] = -np.inf

            max_k = np.max(ll_k)

            if max_k == -np.inf:
                rv[k] = -np.inf
            else:
                ll_k -= max_k
                rv[k] = np.log(np.sum(np.exp(ll_k))) + max_k

        return rv

    def to_mixture(self) -> MixtureDistribution:
        """Returns a MixtureDistribution object created from object instance. """
        topics = [SequenceDistribution(MixtureDistribution(self.topics, self.taus[i, :]), len_dist=self.len_dist) for i
                  in range(self.num_mixtures)]
        return MixtureDistribution(topics, self.w)

    def seq_component_log_density(self, x: Tuple[int, np.ndarray, np.ndarray, E1, Optional[E2]]) -> np.ndarray:
        """Vectorized evaluation of the outer-mixture component-wise log-density for an encoded sequence x.

        This returns a numpy array with shape (rv[0], 'num_mixtures').

        Note: This density is a Mixture of Sequence of Mixture, so the data must be bin-counted as last step in code.

        Encoded sequence 'x' is a Tuple of length 5 containing:
            x[0] (int): Number of independent observations.
            x[1] (ndarray[int]): Observation sequence index for each value in flattened x.
            x[2] (ndarray[int]): Length of each observation in x.
            x[3] (E): Encoded sequence of flattened observed values (has type E).
            x[4] (Optional[E2]): Encoded sequence of lengths (has type E2).

        Args:
            x: Encoded sequence of iid hierarchical mixture model observations.

        Returns:
            Numpy array of dimensions 'rv[0]' by 'num_mixtures', containing the log-density for each component of the
                outer mixture.

        """
        sz, idx, cnt, enc_data, enc_len = x
        tsz = len(idx)

        if (sz > 0) and np.all(cnt == 0):
            return np.zeros(sz, dtype=np.float64)
        elif sz == 0:
            return np.zeros(0, dtype=np.float64)

        # Compute p_mat(data|topic) for each topic
        ll_mat = np.zeros((tsz, self.num_topics), dtype=np.float64)
        rv = np.zeros((sz, self.num_mixtures), dtype=np.float64)

        for i in range(self.num_topics):
            ll_mat[:, i] = self.topics[i].seq_log_density(enc_data)

        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())
        ll_mat[bad_rows, :] = 0.0
        ll_max[bad_rows] = 0.0

        ll_mat -= ll_max
        np.exp(ll_mat, out=ll_mat) ### (tsz,1)

        # Compute ln p_mat(data | mixture)
        ll_mat = np.dot(ll_mat, self.taus.T) ### (tsz, num_mixtures)
        np.log(ll_mat, out=ll_mat)
        ll_mat += ll_max

        # Compute ln p_mat(bag of data | mixture)
        for i in range(self.num_mixtures):
            rv[:, i] = np.bincount(idx, weights=ll_mat[:, i], minlength=sz)

        return rv

    def seq_log_density(self, x: Tuple[int, np.ndarray, np.ndarray, E1, Optional[E2]]) -> np.ndarray:
        """Vectorized evaluation of the log-density for an encoded sequence of observations in x.

        Encoded sequence 'x' is a Tuple of length 5 containing:
            x[0] (int): Number of independent observations.
            x[1] (ndarray[int]): Observation sequence index for each value in flattened x.
            x[2] (ndarray[int]): Length of each observation in x.
            x[3] (E): Encoded sequence of flattened observed values (has type E).
            x[4] (Optional[E2]): Encoded sequence of lengths (has type E2).

        Args:
            x: Encoded sequence of observations of hierarchical mixture model.

        Returns:
            Log-density evaluated at each observation in the encoded sequence x.

        """
        sz, idx, cnt, enc_data, enc_len = x
        tsz = len(idx)

        # Compute ln p_mat(bag of data | mixture)
        rv = self.seq_component_log_density(x)

        # Compute ln p_mat(bag of data, mixture)
        rv += self.log_w

        # Compute ln p_mat(bag of data)
        ll_max2 = np.max(rv, axis=1, keepdims=True)
        rv -= ll_max2
        np.exp(rv, out=rv)
        ll_sum = rv.sum(axis=1, keepdims=True)
        np.log(ll_sum, out=ll_sum)
        ll_sum += ll_max2

        rv = ll_sum.flatten()

        if self.len_dist is not None:
            rv += self.len_dist.seq_log_density(enc_len)

        return rv

    def seq_posterior(self, x: Tuple[int, np.ndarray, np.ndarray, E1, Optional[E2]]) -> np.ndarray:
        """Vectorized evaluation of the posterior over each outer-mixture component for an encoded sequence x.

        Encoded sequence 'x' is a Tuple of length 5 containing:
            x[0] (int): Number of independent observations.
            x[1] (ndarray[int]): Observation sequence index for each value in flattened x.
            x[2] (ndarray[int]): Length of each observation in x.
            x[3] (E): Encoded sequence of flattened observed values (has type E).
            x[4] (Optional[E2]): Encoded sequence of lengths (has type E2).

        Args:
            x: See above for details.

        Returns:
            Numpy array of dimension (x[0], 'num_mixtures').

        """
        sz, idx, cnt, enc_data, enc_len = x
        tsz = len(idx)

        # Compute ln p_mat(bag of data | mixture)
        rv = self.seq_component_log_density(x)

        # Compute ln p_mat(bag of data, mixture)
        rv += self.log_w

        # Compute ln p_mat(bag of data)
        ll_max2 = np.max(rv, axis=1, keepdims=True)
        rv -= ll_max2
        np.exp(rv, out=rv)
        rv /= np.sum(rv, axis=1, keepdims=True)

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'HierarchicalMixtureSampler':
        """Return HierarchicalMixtureSampler object created from attribute variables."""
        return HierarchicalMixtureSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'HierarchicalMixtureEstimator':
        """Create an HierarchicalMixtureEstimator object from attributes variables.

        Args:
            pseudo_count (Optional[float]): Re-weight sufficient statistics in estimation step of EM.

        Returns:
            HierarchicalMixtureEstimator object.

        """
        len_est = self.len_dist.estimator(pseudo_count=pseudo_count)
        comp_est = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]

        return HierarchicalMixtureEstimator(comp_est, self.num_mixtures, len_estimator=len_est,
                                            pseudo_count=pseudo_count, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> 'HierarchicalMixtureDataEncoder':
        """Return an HierarchicalMixtureDataEncoder object for encoding sequences of iid observations. """
        topic_encoder = self.topics[0].dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()
        return HierarchicalMixtureDataEncoder(topic_encoder=topic_encoder, len_encoder=len_encoder)


class HierarchicalMixtureSampler(DistributionSampler):

    def __init__(self, dist: HierarchicalMixtureDistribution, seed: Optional[int] = None) -> None:
        """HierarchicalMixtureSampler object for sampling from a hierarchical mixture model.

        Args:
            dist (HierarchicalMixtureDistribution): HierarchicalMixtureDistribution instance to sample from.
            seed (Optional[int]): Set seed for random number generator used in sampling.

        Attributes:
            rng (RandomState): RandomState object with seed set is passed as arg.
            dist (HierarchicalMixtureDistribution): HierarchicalMixtureDistribution instance to sample from.
            sampler (MixtureDistributionSampler): Convert 'dist' to a MixtureDistribution for sampling.

        """
        self.rng = np.random.RandomState(seed)
        self.dist = dist
        self.sampler = dist.to_mixture().sampler(seed)

    def sample(self, size: Optional[int] = None) -> Union[Sequence[Any], Any]:
        """Returns samples from MixtureSampler."""
        return self.sampler.sample(size=size)


class HierarchicalMixtureEstimatorAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, accumulators: Sequence[SequenceEncodableStatisticAccumulator], num_mixtures: int,
                 len_accumulator: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None)) -> None:
        """

        Args:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the topic distributions.
                Each SequenceEncodableStatisticAccumulator should be compatible with data type T.
            num_mixtures (int): Number of outer mixture components.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator object for the
                length of the topic distributions.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys for merging sufficient statistics of
                weights and topics with matching objects containing matching keys.

        Attributes:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the topic distributions.
                Each SequenceEncodableStatisticAccumulator should be compatible with data type T.
            num_topics (int): Number of topic distributions. Length of accumulators above.
            num_mixtures (int): Number of outer mixture components.
            comp_counts (ndarray): Numpy array of shape ('num_mixtures', 'num_topics') for tracking component counts,
                used to estimate the weights.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator object for the
                length of the topic distributions.
            weight_key (Optional[str]): If set, comp_counts are merged with objects containing matching weight_key.
            comp_key (Optional[str]): If set, the components of the outer-mixture are merged with objects containing
                a matching comp_key.
            _init_rng (bool): False if rng for accumulators has not been set.
            _topic_rng (Optional[List[RandomState]]): List of RandomState objects for setting seed on topic accumulator
                initialization.
            _w_rng (Optional[RandomState]): RandomState for initializing draws from components.
            _tau_rng (Optional[RandomState]): RandomState for initializing draws from sequence of mixture component.
            _len_rng (Optional[RandomState]): RandomState for setting seed on length accumulator.

        """
        self.accumulators = accumulators
        self.num_topics = len(accumulators)
        self.num_mixtures = num_mixtures
        self.comp_counts = vec.zeros((self.num_mixtures, self.num_topics))
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()
        keys_temp = keys if keys is not None else (None, None)
        self.weight_key = keys_temp[0]
        self.comp_key = keys_temp[1]

        ### Initializer seeds
        self._init_rng: bool = False
        self._topic_rng: Optional[List[RandomState]] = None
        self._w_rng: Optional[RandomState] = None
        self._tau_rng: Optional[RandomState] = None
        self._len_rng: Optional[RandomState] = None

    def update(self, x, weight, estimate) -> None:
        """Update sufficient statistics with an observation x.

        Args:
            x (Sequence[T]): An observation from hierarchical mixture mode with data type T.
            weight (float): Observation weight.
            estimate (HierarchicalMixtureDistribution): Previous estimate from EM algorithm.

        Returns:
            None.

        """
        pass

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize RandomState objects for accumulators from rng.

        This function exists to ensure consistency between initialize() and seq_initialize() functions.

        Args:
            rng (RandomState): Used to generate seed value for _rng_acc member variable.

        Returns:
            None.

        """
        self._len_rng = RandomState(seed=rng.randint(maxrandint))
        self._w_rng = RandomState(seed=rng.randint(maxrandint))
        self._tau_rng = RandomState(seed=rng.randint(maxrandint))
        self._topic_rng = [RandomState(seed=rng.randint(maxrandint)) for i in range(self.num_topics)]
        self._init_rng = True

    def initialize(self, x: Sequence[T], weight: float, rng: RandomState) -> None:
        """Initialize sufficient statistics with an observation x.

        Args:
            x (Sequence[T]): An observation from hierarchical mixture mode with data type T.
            weight (float): Observation weight.
            rng (RandomState): RandomState object for initializing sufficient statistics.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        idx1 = self._w_rng.choice(self.num_mixtures)

        for j in range(len(x)):
            idx2 = self._tau_rng.choice(self.num_topics)

            for i in range(self.num_topics):
                w = weight if i == idx2 else 0.0
                self.accumulators[i].initialize(x[j], w, self._topic_rng[i])
                self.comp_counts[idx1, i] += w

        self.len_accumulator.initialize(len(x), weight, self._len_rng)

    def seq_initialize(self, x: Tuple[int, np.ndarray, np.ndarray, E1, Optional[E2]], weights: np.ndarray,
                       rng: RandomState) -> None:
        """Vectorized initialization of sufficient statistics from an encoded sequence of observations in x.

        Note: Calls _rng_initialize() to ensure equivalence between seq_initialize() and initialize().

        Encoded sequence 'x' is a Tuple of length 5 containing:
            x[0] (int): Number of independent observations.
            x[1] (ndarray[int]): Observation sequence index for each value in flattened x.
            x[2] (ndarray[int]): Length of each observation in x.
            x[3] (E): Encoded sequence of flattened observed values (has type E).
            x[4] (Optional[E2]): Encoded sequence of lengths (has type E2).

        Args:
            x: Encoded sequence of observations of hierarchical mixture model.
            weights (ndarray): Weights for observations.
            rng (RandomState): RandomState object for initializing sufficient statistics.

        Returns:
            None.

        """
        sz, idx, cnt, enc_data, enc_len = x
        tsz = len(idx)

        if not self._init_rng:
            self._rng_initialize(rng)

        idx1 = self._w_rng.choice(self.num_mixtures, size=sz, replace=True)   # draw component
        idx2 = self._tau_rng.choice(self.num_topics, size=tsz, replace=True)  # draw seqeucne mixture in component
        ww = weights[idx]

        for i in range(self.num_topics):
            w = np.zeros_like(ww)
            w_nz = idx2 == i
            w[w_nz] = ww[w_nz]

            self.accumulators[i].seq_initialize(enc_data, w, self._topic_rng[i])
            self.comp_counts[:, i] = np.bincount(idx1[idx], w)
            #self.comp_counts[idx1, i] += np.sum(w)

        self.len_accumulator.seq_initialize(enc_len, weights, self._len_rng)

    def seq_update(self, x: Tuple[int, np.ndarray, np.ndarray, E1, Optional[E2]], weights: np.ndarray,
                   estimate: HierarchicalMixtureDistribution) -> None:
        """Vectorized update of sufficient statistics from an encoded sequence x.

        Encoded sequence 'x' is a Tuple of length 5 containing:
            x[0] (int): Number of independent observations.
            x[1] (ndarray[int]): Observation sequence index for each value in flattened x.
            x[2] (ndarray[int]): Length of each observation in x.
            x[3] (E): Encoded sequence of flattened observed values (has type E).
            x[4] (Optional[E2]): Encoded sequence of lengths (has type E2).

        Args:
            x: Encoded sequence of observations of hierarchical mixture model.
            weights (ndarray): Weights for observations.
            estimate (HierarchicalMixtureDistribution): Previous estimate from EM algorithm.

        Returns:
            None.

        """
        sz, idx, cnt, enc_data, enc_len = x
        tsz = len(idx)

        ll_mat = np.zeros((tsz, self.num_topics))
        ll_mat.fill(-np.inf)
        rv = np.zeros((sz, self.num_mixtures))
        rv3 = np.zeros((tsz, self.num_topics))

        for i in range(self.num_topics):
            ll_mat[:, i] = estimate.topics[i].seq_log_density(enc_data)

        ll_max = ll_mat.max(axis=1, keepdims=True)

        bad_rows = np.isinf(ll_max.flatten())
        ll_mat[bad_rows, :] = 0.0
        ll_max[bad_rows] = 0.0

        ll_mat -= ll_max

        np.exp(ll_mat, out=ll_mat)

        ll_mat_t = np.dot(ll_mat, estimate.taus.T)
        ll_mat_t2 = np.log(ll_mat_t)

        ll_max = np.bincount(idx, weights=ll_max.flatten(), minlength=sz)
        for i in range(self.num_mixtures):
            rv[:, i] = np.bincount(idx, weights=ll_mat_t2[:, i], minlength=sz)

        rv += estimate.log_w
        rv += ll_max[:, None]
        ll_max2 = np.max(rv, axis=1, keepdims=True)
        rv -= ll_max2

        np.exp(rv, out=rv)
        ll_sum = rv.sum(axis=1, keepdims=True)
        rv /= ll_sum
        rv = rv[idx, :]
        ww = np.reshape(weights[idx], (-1, 1))

        for i in range(self.num_mixtures):
            temp = estimate.taus[i, None, :] * (rv[:, i, None] / ll_mat_t[:, i, None])
            temp *= ll_mat
            temp *= ww
            rv3 += temp
            self.comp_counts[i, :] += temp.sum(axis=0)

        for i in range(self.num_topics):
            self.accumulators[i].seq_update(enc_data, rv3[:, i], estimate.topics[i])

        if self.len_accumulator is not None:
            len_est = None if estimate is None else estimate.len_dist
            self.len_accumulator.seq_update(enc_len, weights, len_est)

    def combine(self, suff_stat: Tuple[np.ndarray, Tuple[SS1, ...], Optional[SS2]]) -> 'HierarchicalMixtureEstimatorAccumulator':
        """Combine the sufficient statistics of 'suff_stat; with attribute variables.

        Arg suff_stat is a Tuple of length 3 containing,
            suff_stat[0] (ndarray[float]): Aggregated component counts with shape (num_mixtures, num_topics).
            suff_stat[1] (Tuple[SS1,...]): Tuple of 'num_topics' sufficient statistics for the topics.
            suff_stat[2] (Optional[SS2]): Optional sufficient statistic for length accumulator.

        Args:
            suff_stat (Tuple[np.ndarray, Tuple[SS1, ...], Optional[SS2]]): See above for details.

        Returns:
            HierarchicalMixtureEstimatorAccumulator object.

        """
        self.comp_counts += suff_stat[0]
        for i in range(self.num_topics):
            self.accumulators[i].combine(suff_stat[1][i])

        self.len_accumulator.combine(suff_stat[2])

        return self

    def value(self) -> Tuple[np.ndarray, Tuple[Any, ...], Optional[Any]]:
        """Returns sufficient statistics of type Tuple[np.ndarray, Tuple[SS1,...], Optional[SS2]]."""
        return self.comp_counts, tuple([u.value() for u in self.accumulators]), self.len_accumulator.value()

    def from_value(self, x: Tuple[np.ndarray, Tuple[SS1, ...], Optional[SS2]]) -> 'HierarchicalMixtureEstimatorAccumulator':
        """Set the attribute variables for sufficient statistics to arg 'x'.

        Arg 'x' is a Tuple of length 3 containing,
            x[0] (ndarray[float]): Aggregated component counts with shape (num_mixtures, num_topics).
            x[1] (Tuple[SS1,...]): Tuple of 'num_topics' sufficient statistics for the topics.
            x[2] (Optional[SS2]): Optional sufficient statistic for length accumulator.

        Args:
            x (Tuple[np.ndarray, Tuple[SS1, ...], Optional[SS2]]): See above for details.

        Returns:
            HierarchicalMixtureEstimatorAccumulator object.

        """
        self.comp_counts = x[0]
        for i in range(self.num_topics):
            self.accumulators[i].from_value(x[1][i])

        self.len_accumulator.from_value(x[2])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge sufficient statistics of object instance with matching keys in stats_dict.

        Merges comp_counts if weight_key is set and found in stats_dict.
        Merges topic accumulators if 'comp_key' is set and found in stats_dict.

        Calls key_merge() of the length accumulator.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to corresponding sufficient statistics.

        Returns:
            None.

        """
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.comp_counts
            else:
                stats_dict[self.weight_key] = self.comp_counts

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                for i in range(len(acc)):
                    acc[i] = acc[i].combine(self.accumulators[i].value())
            else:
                stats_dict[self.comp_key] = self.accumulators

        for u in self.accumulators:
            u.key_merge(stats_dict)

        self.len_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace the sufficient statistics of object instance with those containing matching keys in 'stats_dict'.

        Replaces comp_counts if weight_key is set and found in stats_dict.
        Replaces topic accumulators if 'comp_key' is set and found in stats_dict.

        Calls key_replace() of the length accumulator.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to corresponding sufficient statistics.

        Returns:
            None.

        """
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.comp_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

        self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'HierarchicalMixtureDataEncoder':
        """Return an HierarchicalMixtureDataEncoder object for encoding sequences of iid observations. """
        topic_encoder = self.accumulators[0].acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()
        return HierarchicalMixtureDataEncoder(topic_encoder=topic_encoder, len_encoder=len_encoder)


class HierarchicalMixtureEstimatorAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, factories: Sequence[StatisticAccumulatorFactory], num_mixtures: int,
                 len_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None)):
        """HierarchicalMixtureEstimatorAccumulatorFactory object for creating HierarchicalMixtureEstimatorAccumulator
            objects.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): StatisticAccumulatorFactory objects for the topics.
            num_mixtures (int): Number of outer mixture components.
            len_factory (Optional[StatisticAccumulatorFactory]): Optional StatisticAccumulatorFactory for the length
                distribution.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys for merging sufficient statistics of weights and
                topics with matching objects containing matching keys.

        Attributes:
            factories (Sequence[StatisticAccumulatorFactory]): StatisticAccumulatorFactory objects for the topics.
            num_mixtures (int): Number of outer mixture components.
            dim (int): Number of topics.
            len_factory (StatisticAccumulatorFactory): StatisticAccumulatorFactory for the length distribution.
                Defaults to the NullAccumulatorFactory.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys for merging sufficient statistics of weights and
                topics with matching objects containing matching keys.

        """
        self.factories = factories
        self.num_mixtures = num_mixtures
        self.dim = len(factories)
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys if keys is not None else (None, None)

    def make(self) -> 'HierarchicalMixtureEstimatorAccumulator':
        """Returns an HierarchicalMixtureEstimatorAccumulator object from attributes variables."""
        return HierarchicalMixtureEstimatorAccumulator([self.factories[i].make() for i in range(self.dim)],
                                                       self.num_mixtures, self.len_factory.make(), self.keys)

class HierarchicalMixtureEstimator(ParameterEstimator):

    def __init__(self, estimators: Sequence[ParameterEstimator], num_mixtures: int,
                 len_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = None,
                 suff_stat: Optional[np.ndarray] = None,
                 pseudo_count: Optional[float] = None,
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None)) -> None:
        """HierarchicalMixtureEstimator object for estimating hierarchical mixture distribution for aggregated
            sufficient statistics.

        Note: If pseudo_count is passed, the mixture weights are re-weighted in estimation. If attribute suff_stat
        is set, a suff_stat is re-weighted and combined with new sufficient statistics in estimation.

        Args:
            estimators (Sequence[ParameterEstimator]): ParameterEstimator objects for the topics.
            num_mixtures (int): Number of outer-mixture components.
            len_estimator (Optional[ParameterEstimator]): Estimator for the length of inner mixture sequences.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Fix the length on inner-mixture sequence
                distribution.
            suff_stat (np.ndarray): 2-d numpy array of dimension (num_components, num_mixtures). Represents the
                inner-mixture weights.
            pseudo_count (Optional[float]): Re-weight 'suff_stat' above in estimation.
            name (Optional[str]): Set a name to object instnace.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Set keys for weights and topics.

        Attributes:
            num_components (int): Number of topic distributions (inner-mixture).
            num_mixtures (int): Number of outer-mixture components.
            estimators (Sequence[ParameterEstimator]): ParameterEstimator objects for the topics.
            pseudo_count (Optional[float]): Re-weight 'suff_stat' above in estimation.
            suff_stat (np.ndarray): 2-d numpy array of dimension (num_components, num_mixtures). Represents the
                inner-mixture weights.
            len_estimator (Optional[ParameterEstimator]): Estimator for the length of inner mixture sequences.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys for weights and topics, passed to accumulator
                factory with call to 'accumulator_factory()'.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Fix the length on inner-mixture sequence
                distribution.
            name (Optional[str]): Name for object instance.

        """
        self.num_components = len(estimators)
        self.num_mixtures = num_mixtures
        self.estimators = estimators
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.keys = keys if keys is not None else (None, None)
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.name = name

    def accumulator_factory(self) -> 'HierarchicalMixtureEstimatorAccumulatorFactory':
        """Create an HierarchicalMixtureEstimatorAccumulator from object instance."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        len_factory = self.len_estimator.accumulator_factory()
        return HierarchicalMixtureEstimatorAccumulatorFactory(est_factories, self.num_mixtures, len_factory, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, SS1, Optional[SS2]]) \
            -> 'HierarchicalMixtureDistribution':
        """Estimate HierarchicalMixtureDistribution from aggregated sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations used in accumulation of 'suff_stat'.
            suff_stat:

        Returns:
            HierarchicalMixtureDistribution object.

        """
        num_components = self.num_components
        num_mixtures = self.num_mixtures
        counts, comp_suff_stats, len_suff_stats = suff_stat
        len_dist = self.len_estimator.estimate(None, len_suff_stats) if len_suff_stats is not None else self.len_dist

        components = [self.estimators[i].estimate(None, comp_suff_stats[i]) for i in range(num_components)]

        if self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count / (num_components * num_mixtures)
            taus = counts + p
            w = taus.sum(axis=1, keepdims=True)
            taus /= w
            w /= w.sum()
            w = w.flatten()

        elif self.pseudo_count is not None and self.suff_stat is not None:
            taus = (counts + self.suff_stat*self.pseudo_count) / (counts.sum() + self.pseudo_count)
            w = taus.sum(axis=1, keepdims=True)
            taus /= w
            w /= w.sum()
            w = w.flatten()

        else:
            taus = counts
            w = taus.sum(axis=1, keepdims=True)
            taus /= w
            w /= w.sum()
            w = w.flatten()

        return HierarchicalMixtureDistribution(components, w, taus, len_dist=len_dist, name=self.name, keys=self.keys)

class HierarchicalMixtureDataEncoder(DataSequenceEncoder):

    def __init__(self, topic_encoder: DataSequenceEncoder, len_encoder: DataSequenceEncoder) -> None:
        """HierarchicalMixtureDataEncoder object for encoding sequences of iid hierarchical mixture observations.

        Args:
            topic_encoder (DataSequenceEncoder): DataSequenceEncoder for topic distributions. Must be compatible with
                data type T.
            len_encoder (DataSequenceEncoder): DataSequenceEncoder for length of sequences.

        Attributes:
            topic_encoder (DataSequenceEncoder): DataSequenceEncoder for topic distributions. Must be compatible with
                data type T.
            len_encoder (DataSequenceEncoder): DataSequenceEncoder for length of sequences.

        """
        self.topic_encoder = topic_encoder
        self.len_encoder = len_encoder

    def __str__(self) -> str:
        """Return string representation of object instance."""
        rv = 'HierarchicalMixtureDataEncoder(topic_encoder=' + str(self.topic_encoder) + ','
        rv += 'len_encoder=' + str(self.len_encoder) + ')'
        return rv

    def __eq__(self, other: object) -> bool:
        """Check if object is equivalent to instance of HierarchicalMixtureDataEncoder.

        Note: topic and length encoder objects must be equivalent.

        Args:
            other (object): Object to compare to object instance.

        Returns:
            True if other is equivalent.

        """
        if isinstance(other, HierarchicalMixtureDataEncoder):
            return other.topic_encoder == self.topic_encoder and other.len_encoder == self.len_encoder
        else:
            return False

    def seq_encode(self, x: Sequence[Sequence[T]]) -> Tuple[int, np.ndarray, np.ndarray, Any, Optional[Any]]:
        """Encode a sequence of iid observations from a hierarchical mixture model.

        Returns 'rv' as a Tuple of length 5 containing:
            rv[0] (int): Number of independent observations.
            rv[1] (ndarray[int]): Observation sequence index for each value in flattened x.
            rv[2] (ndarray[int]): Length of each observation in x.
            rv[3] (E): Encoded sequence of flattened observed values (has type E).
            rv[4] (Optional[E2]): Encoded sequence of lengths (has type E2).

        Args:
            x (Sequence[Sequence[T]]): Sequence of hierarchical mixture model observations.

        Returns:
            See above.

        """
        sx = []
        idx = []
        cnt = []

        for i in range(len(x)):
            idx.extend([i] * len(x[i]))
            sx.extend(x[i])
            cnt.append(len(x[i]))

        enc_len = self.len_encoder.seq_encode(cnt)
        idx = np.asarray(idx, dtype=np.int32)
        cnt = np.asarray(cnt, dtype=np.int32)

        enc_data = self.topic_encoder.seq_encode(sx)

        return len(x), idx, cnt, enc_data, enc_len


