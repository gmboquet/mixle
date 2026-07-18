"""Hierarchical mixtures over shared topic distributions.

Data type: ``Sequence[T]``, where ``T`` is the type of the topic distributions.

This model has K outer-mixture components and L shared topic distributions
{f_l(theta_l)}_{l=1}^{L}. Each outer component k has its own inner-mixture weights
{tau_{k,l}}_{l=1}^{L}.

Sampling proceeds by drawing an outer component k with probability w_k, drawing a sequence length N from ``P_len`` when a
length distribution is supplied, and then sampling each item from the topic mixture for component k.

Example: Let x = (x_1, x_2, x_3, ...., x_N) be an observation from a hierarchical mixture distribution of length 'N'.
Let Z and U be a random variables s.t. p_mat(Z=k) = w_k and p_mat(U=l | Z = k) = tau_{k,l}. Then

    alpha_i = x_i | Z = k ~ sum_{l=1}^{L} f_l(theta_l)*tau_{k,l}, for i = 1,2,...,N.

Further,

    alpha_i | U=l ~ f_l(theta_l), for i = 1,2,3,...,N.

"""

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

import mixle.utils.vector as vec
from mixle.capability import Neutral, supports
from mixle.engines.arithmetic import maxrandint
from mixle.enumeration.algorithms import BufferedStream, best_first_union
from mixle.stats.combinator.null_dist import NullAccumulator, NullAccumulatorFactory, NullDistribution, NullEstimator
from mixle.stats.combinator.sequence import SequenceDistribution
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)
from mixle.stats.latent.mixture import MixtureDistribution

T = TypeVar("T")  ## Data type for topics
E1 = TypeVar("E1")  ## Encoded sequence from topic encoder.
E2 = TypeVar("E2")  ## Encoded sequence from length encoder.
SS1 = TypeVar("SS1")  ### Suff stat type for topics.
SS2 = TypeVar("SS2")  ## Suff stat type for length distribution.


class HierarchicalMixtureDistribution(SequenceEncodableProbabilityDistribution):
    """Outer mixture over sequence mixtures with shared topic distributions.

    Data type: ``Sequence[T]``, where ``T`` is the data type of the topic distributions.

    """

    def compute_capabilities(self):
        """Declare generated-compute support inherited from topics and length model."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        children = tuple(self.topics) + ((self.len_dist,) if self.len_dist is not None else ())
        return DistributionCapabilities(engine_ready=intersect_engine_ready(children), kernel_status="generic_latent")

    def compute_declaration(self):
        """Return the generated-compute declaration for the hierarchical mixture."""
        from mixle.stats.compute.declarations import (
            DistributionDeclaration,
            ParameterSpec,
            StatisticSpec,
            declaration_for,
        )

        topic_children = tuple(declaration_for(topic) for topic in self.topics)
        length = None if supports(self.len_dist, Neutral) else declaration_for(self.len_dist)
        children = tuple(
            child for child in topic_children + ((length,) if length is not None else ()) if child is not None
        )
        roles = tuple("topic_%d" % i for i, child in enumerate(topic_children) if child is not None)
        if length is not None:
            roles += ("length",)
        return DistributionDeclaration(
            name="hierarchical_mixture",
            distribution_type=type(self),
            parameters=(
                ParameterSpec("w", constraint="simplex_vector"),
                ParameterSpec("taus", constraint="row_simplex_matrix"),
            ),
            statistics=(
                StatisticSpec("component_counts"),
                StatisticSpec("outer_weight_counts"),  # document-level outer posteriors (the w M-step)
                StatisticSpec("topics", kind="tuple"),
                StatisticSpec("length", kind="child_stat"),
            ),
            support="sequence_mixture",
            children=children,
            child_roles=roles,
            differentiable=False,
        )

    def __init__(
        self,
        topics: Sequence[SequenceEncodableProbabilityDistribution],
        mixture_weights: list[float] | np.ndarray,
        topic_weights: list[list[float]] | np.ndarray,
        len_dist: SequenceEncodableProbabilityDistribution | None = NullDistribution(),
        name: str | None = None,
        keys: tuple[str | None, str | None] | None = (None, None),
    ) -> None:
        """Create a hierarchical mixture distribution.

        Args:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Topic distributions shared in hierarchical
                mixture distribution.
            mixture_weights (Union[List[float], np.ndarray]): Outer-mixture weights. Values should sum to one.
            topic_weights (Union[List[List[float]], np.ndarray]): Matrix whose rows contain topic weights for each outer
                component. Each row should sum to one.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for sequence lengths.
            name (Optional[str]): Optional distribution name.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Optional keys for outer weights and topic statistics.

        Attributes:
            topics (Sequence[SequenceEncodableProbabilityDistribution]): Topic distributions shared in hierarchical
                mixture distribution.
            num_topics (int): Number of shared topic distributions.
            num_mixtures (int): Number of outer-mixture components.
            w (np.ndarray): Outer-mixture weights.
            log_w (np.ndarray): Log outer-mixture weights.
            taus (np.ndarray): Topic-weight matrix with shape ``(num_mixtures, num_topics)``.
            log_taus (np.ndarray): Log topic-weight matrix.
            len_dist (SequenceEncodableProbabilityDistribution): Distribution for the sequence length on topics.
                Defaults to the NullDistribution if None is passed.
            name (Optional[str]): Optional distribution name.
            keys (Tuple[Optional[str], Optional[str]]): Optional keys for outer weights and topic statistics.

        """
        with np.errstate(divide="ignore"):
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
        """Return a constructor-style representation of the distribution."""
        s1 = "[" + ",".join([str(u) for u in self.topics]) + "]"
        s2 = repr(list(self.w))
        s3 = repr(list(map(list, self.taus)))
        s4 = repr(self.len_dist) if self.len_dist is None else str(self.len_dist)
        s5 = repr(self.name)
        s6 = repr(self.keys)
        return "HierarchicalMixtureDistribution(%s, %s, %s, len_dist=%s, name=%s, keys=%s)" % (s1, s2, s3, s4, s5, s6)

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
        if n == 0:
            return np.zeros(self.num_mixtures, dtype=np.float64)

        ll_topic = np.zeros((n, self.num_topics), dtype=np.float64)
        for i in range(n):
            ll_topic[i, :] = np.asarray([self.topics[j].log_density(x[i]) for j in range(self.num_topics)])

        rv = np.zeros(self.num_mixtures, dtype=np.float64)
        for k in range(self.num_mixtures):
            ll_k = ll_topic + self.log_taus[k, :][None, :]
            row_max = np.max(ll_k, axis=1)
            good_rows = np.isfinite(row_max)

            if not np.all(good_rows):
                rv[k] = -np.inf
            else:
                ll_k = ll_k - row_max[:, None]
                rv[k] = np.sum(np.log(np.sum(np.exp(ll_k), axis=1)) + row_max)

        return rv

    def to_mixture(self) -> MixtureDistribution:
        """Return the equivalent flat mixture of sequence distributions."""
        topics = [
            SequenceDistribution(MixtureDistribution(self.topics, self.taus[i, :]), len_dist=self.len_dist)
            for i in range(self.num_mixtures)
        ]
        return MixtureDistribution(topics, self.w)

    def seq_component_log_density(self, x: tuple[int, np.ndarray, np.ndarray, E1, E2 | None]) -> np.ndarray:
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
            return np.zeros((sz, self.num_mixtures), dtype=np.float64)
        elif sz == 0:
            return np.zeros((0, self.num_mixtures), dtype=np.float64)

        # Compute p_mat(data|topic) for each topic
        ll_mat = np.zeros((tsz, self.num_topics), dtype=np.float64)
        rv = np.zeros((sz, self.num_mixtures), dtype=np.float64)

        for i in range(self.num_topics):
            ll_mat[:, i] = self.topics[i].seq_log_density(enc_data)

        ll_max = ll_mat.max(axis=1)
        good_rows = np.isfinite(ll_max)
        ll_exp = np.zeros_like(ll_mat)
        if np.any(good_rows):
            ll_exp[good_rows, :] = np.exp(ll_mat[good_rows, :] - ll_max[good_rows, None])

        # Compute ln p_mat(data | mixture)
        ll_mix = np.dot(ll_exp, self.taus.T)  ### (tsz, num_mixtures)
        ll_mat = np.full_like(ll_mix, -np.inf)
        pos = ll_mix > 0.0
        ll_mat[pos] = np.log(ll_mix[pos])
        ll_mat[good_rows, :] += ll_max[good_rows, None]

        # Compute ln p_mat(bag of data | mixture)
        for i in range(self.num_mixtures):
            rv[:, i] = np.bincount(idx, weights=ll_mat[:, i], minlength=sz)

        return rv

    def seq_log_density(self, x: tuple[int, np.ndarray, np.ndarray, E1, E2 | None]) -> np.ndarray:
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
        good_rows = np.isfinite(ll_max2.flatten())
        out = np.full(sz, -np.inf, dtype=np.float64)
        if np.any(good_rows):
            rv_good = rv[good_rows, :] - ll_max2[good_rows, :]
            np.exp(rv_good, out=rv_good)
            ll_sum = np.sum(rv_good, axis=1)
            out[good_rows] = np.log(ll_sum) + ll_max2[good_rows, 0]
        rv = out

        if self.len_dist is not None:
            rv += self.len_dist.seq_log_density(enc_len)

        return rv

    def backend_seq_component_log_density(
        self, x: tuple[int, np.ndarray, np.ndarray, E1, E2 | None], engine: Any
    ) -> Any:
        """Engine-neutral outer-component log densities for hierarchical-mixture encoded sequences."""
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, idx, cnt, enc_data, enc_len = x
        if sz == 0:
            return engine.zeros((0, self.num_mixtures))
        if np.all(cnt == 0):
            return engine.zeros((sz, self.num_mixtures))

        topic_scores = [backend_seq_log_density(topic, enc_data, engine) for topic in self.topics]
        ll_topics = engine.stack(topic_scores, axis=1)
        log_taus = engine.asarray(self.log_taus)
        item_mix_scores = engine.logsumexp(ll_topics[:, None, :] + log_taus[None, :, :], axis=2)

        rv = engine.zeros((sz, self.num_mixtures))
        return engine.index_add(rv, engine.asarray(idx), item_mix_scores)

    def backend_seq_log_density(self, x: tuple[int, np.ndarray, np.ndarray, E1, E2 | None], engine: Any) -> Any:
        """Engine-neutral hierarchical-mixture log-density for encoded sequences."""
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, idx, cnt, enc_data, enc_len = x
        rv = engine.logsumexp(self.backend_seq_component_log_density(x, engine) + engine.asarray(self.log_w), axis=1)
        if self.len_dist is not None:
            rv = rv + backend_seq_log_density(self.len_dist, enc_len, engine)
        return rv

    def seq_posterior(self, x: tuple[int, np.ndarray, np.ndarray, E1, E2 | None]) -> np.ndarray:
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
        bad_rows = ~np.isfinite(ll_max2.flatten())
        if np.any(bad_rows):
            rv[bad_rows, :] = self.log_w
            ll_max2[bad_rows, :] = np.max(self.log_w)
        rv -= ll_max2
        np.exp(rv, out=rv)
        rv /= np.sum(rv, axis=1, keepdims=True)

        return rv

    def to_fisher(self, **kwargs):
        """Reuse the equivalent flat mixture's Fisher view."""
        if hasattr(self, "to_mixture"):
            return self.to_mixture().to_fisher(**kwargs)
        return super().to_fisher(**kwargs)

    def density_semantics(self):
        """Return exact-or-approximate density semantics joined from child models."""
        from mixle.stats.compute.pdist import DensitySemantics, join_density_semantics

        children = list(self.topics)
        sems = [c.density_semantics() for c in children if hasattr(c, "density_semantics")]
        return join_density_semantics(sems) if sems else DensitySemantics.EXACT

    def sampler(self, seed: int | None = None) -> "HierarchicalMixtureSampler":
        """Return a sampler for this hierarchical mixture."""
        return HierarchicalMixtureSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "HierarchicalMixtureEstimator":
        """Create an estimator initialized from this hierarchical mixture.

        Args:
            pseudo_count (Optional[float]): Prior mass used to smooth mixture and topic weights during estimation.

        Returns:
            HierarchicalMixtureEstimator: Estimator configured with matching topic and length estimators.

        """
        len_est = self.len_dist.estimator(pseudo_count=pseudo_count)
        comp_est = [u.estimator(pseudo_count=pseudo_count) for u in self.topics]

        return HierarchicalMixtureEstimator(
            comp_est,
            self.num_mixtures,
            len_estimator=len_est,
            pseudo_count=pseudo_count,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> "HierarchicalMixtureDataEncoder":
        """Return a data encoder for iid hierarchical-mixture observations."""
        topic_encoder = self.topics[0].dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()
        return HierarchicalMixtureDataEncoder(topic_encoder=topic_encoder, len_encoder=len_encoder)

    def enumerator(self) -> "HierarchicalMixtureEnumerator":
        """Returns a HierarchicalMixtureEnumerator iterating sequences in descending probability order."""
        return HierarchicalMixtureEnumerator(self)


class HierarchicalMixtureEnumerator(DistributionEnumerator):
    """Enumerates the support of a HierarchicalMixtureDistribution in descending probability order."""

    def __init__(self, dist: HierarchicalMixtureDistribution) -> None:
        """Enumerates the union of the outer-component sequence supports in descending probability order.

        Each outer component k is the sequence distribution over the shared topic mixture with
        inner weights taus[k, :] (see to_mixture()). Component supports overlap, so candidates
        pulled from the component enumerations are re-scored exactly with the hierarchical mixture
        log-density and emitted only once their score beats the upper bound on any not-yet-seen
        value (the mixture best-first-union algorithm). Zero-weight outer components are never
        asked to enumerate. Raises EnumerationError when no length distribution is modeled, since
        the sequence support is then ill-defined.

        Args:
            dist (HierarchicalMixtureDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        streams = []
        log_offsets = []

        for k in range(dist.num_mixtures):
            if dist.w[k] <= 0.0:
                continue
            comp = SequenceDistribution(MixtureDistribution(dist.topics, dist.taus[k, :]), len_dist=dist.len_dist)
            streams.append(BufferedStream(child_enumerator(comp, "HierarchicalMixtureDistribution.component[%d]" % k)))
            log_offsets.append(dist.log_w[k])

        def exact_log_density(x):
            with np.errstate(divide="ignore"):
                return float(dist.log_density(x))

        self._union = best_first_union(streams, log_offsets, exact_log_density)

    def __next__(self) -> tuple[Any, float]:
        return next(self._union)


class HierarchicalMixtureSampler(DistributionSampler):
    """Sampler for sequences from a hierarchical mixture distribution."""

    def __init__(self, dist: HierarchicalMixtureDistribution, seed: int | None = None) -> None:
        """Create a sampler for a hierarchical mixture model.

        Args:
            dist (HierarchicalMixtureDistribution): HierarchicalMixtureDistribution instance to sample from.
            seed (Optional[int]): Set seed for random number generator used in sampling.

        Attributes:
            rng (RandomState): Random state initialized from ``seed`` when supplied.
            dist (HierarchicalMixtureDistribution): HierarchicalMixtureDistribution instance to sample from.
            sampler (MixtureDistributionSampler): Convert 'dist' to a MixtureDistribution for sampling.

        """
        self.rng = np.random.RandomState(seed)
        self.dist = dist
        self.sampler = dist.to_mixture().sampler(seed)

    def sample(self, size: int | None = None, *, batched: bool = True) -> Sequence[Any] | Any:
        """Return samples from the underlying mixture sampler."""
        return self.sampler.sample(size=size)


class HierarchicalMixtureEstimatorAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for hierarchical-mixture sufficient statistics."""

    def __init__(
        self,
        accumulators: Sequence[SequenceEncodableStatisticAccumulator],
        num_mixtures: int,
        len_accumulator: SequenceEncodableStatisticAccumulator | None = NullAccumulator(),
        keys: tuple[str | None, str | None] | None = (None, None),
    ) -> None:
        """Create an accumulator for hierarchical-mixture estimation.

        Args:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the topic distributions.
                Each SequenceEncodableStatisticAccumulator should be compatible with data type T.
            num_mixtures (int): Number of outer mixture components.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator for the
                length of the topic distributions.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys for merging sufficient statistics of
                weights and topics with matching objects containing matching keys.

        Attributes:
            accumulators (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the topic distributions.
                Each SequenceEncodableStatisticAccumulator should be compatible with data type T.
            num_topics (int): Number of topic distributions. Length of accumulators above.
            num_mixtures (int): Number of outer mixture components.
            comp_counts (ndarray): Numpy array of shape ('num_mixtures', 'num_topics') for tracking token-level
                component counts, used to estimate the topic weights 'taus'.
            w_counts (ndarray): Numpy array of length 'num_mixtures' with document-level outer-posterior sums,
                used to estimate the outer weights 'w'.
            len_accumulator (Optional[SequenceEncodableStatisticAccumulator]): Optional accumulator for the
                length of the topic distributions.
            weight_key (Optional[str]): If set, comp_counts/w_counts are merged with objects containing matching
                weight_key.
            comp_key (Optional[str]): If set, the components of the outer-mixture are merged with objects containing
                a matching comp_key.
            _init_rng (bool): False if rng for accumulators has not been set.
            _topic_rng (Optional[List[RandomState]]): Random states for topic accumulator
                initialization.
            _w_rng (Optional[RandomState]): RandomState for initializing draws from components.
            _tau_rng (Optional[RandomState]): RandomState for initializing draws from sequence of mixture component.
            _len_rng (Optional[RandomState]): RandomState for setting seed on length accumulator.

        """
        self.accumulators = accumulators
        self.num_topics = len(accumulators)
        self.num_mixtures = num_mixtures
        self.comp_counts = vec.zeros((self.num_mixtures, self.num_topics))
        # Document-level outer-posterior sums, one entry per outer mixture component. The EM
        # maximizer for the outer weights `w` is the sum of DOCUMENT posteriors; deriving `w` from
        # the token-level `comp_counts` row sums weights each document by its length, which is not
        # the EM update and breaks monotonicity on variable-length corpora. `comp_counts` stays
        # token-level (its row-normalization is the correct `taus` update).
        self.w_counts = vec.zeros(self.num_mixtures)
        self.len_accumulator = len_accumulator if len_accumulator is not None else NullAccumulator()
        keys_temp = keys if keys is not None else (None, None)
        self.weight_key = keys_temp[0]
        self.comp_key = keys_temp[1]
        # Data log-likelihood accumulated as a byproduct of the E-step (the posterior normalizer),
        # only when _track_ll is enabled. Used by the fused-EM fast path in
        # optimize(reuse_estep_ll=True); not part of value(). Off by default so the standard path
        # pays nothing.
        self._track_ll = False
        self._seq_ll = 0.0

        ### Initializer seeds
        self._init_rng: bool = False
        self._topic_rng: list[RandomState] | None = None
        self._w_rng: RandomState | None = None
        self._tau_rng: RandomState | None = None
        self._len_rng: RandomState | None = None

    def update(self, x, weight, estimate) -> None:
        """Update sufficient statistics with an observation x.

        Encodes the single observation and delegates to seq_update() so that the scalar and
        vectorized estimation paths agree.

        Args:
            x (Sequence[T]): An observation from hierarchical mixture mode with data type T.
            weight (float): Observation weight.
            estimate (HierarchicalMixtureDistribution): Previous estimate from EM algorithm.

        Returns:
            None.

        """
        enc_x = estimate.dist_to_encoder().seq_encode([x])
        self.seq_update(enc_x, np.asarray([weight]), estimate)

    def _rng_initialize(self, rng: RandomState) -> None:
        """Initialize accumulator random states from ``rng``.

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
            rng (RandomState): Random state for initializing sufficient statistics.

        Returns:
            None.

        """
        if not self._init_rng:
            self._rng_initialize(rng)

        idx1 = self._w_rng.choice(self.num_mixtures)
        self.w_counts[idx1] += weight

        for j in range(len(x)):
            idx2 = self._tau_rng.choice(self.num_topics)

            for i in range(self.num_topics):
                w = weight if i == idx2 else 0.0
                self.accumulators[i].initialize(x[j], w, self._topic_rng[i])
                self.comp_counts[idx1, i] += w

        self.len_accumulator.initialize(len(x), weight, self._len_rng)

    def seq_initialize(
        self, x: tuple[int, np.ndarray, np.ndarray, E1, E2 | None], weights: np.ndarray, rng: RandomState
    ) -> None:
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
            rng (RandomState): Random state for initializing sufficient statistics.

        Returns:
            None.

        """
        sz, idx, cnt, enc_data, enc_len = x
        tsz = len(idx)

        if not self._init_rng:
            self._rng_initialize(rng)

        idx1 = self._w_rng.choice(self.num_mixtures, size=sz, replace=True)  # draw component
        idx2 = self._tau_rng.choice(self.num_topics, size=tsz, replace=True)  # draw seqeucne mixture in component
        ww = weights[idx]

        self.w_counts += np.bincount(idx1, weights=weights, minlength=self.num_mixtures)

        for i in range(self.num_topics):
            w = np.zeros_like(ww)
            w_nz = idx2 == i
            w[w_nz] = ww[w_nz]

            self.accumulators[i].seq_initialize(enc_data, w, self._topic_rng[i])
            self.comp_counts[:, i] += np.bincount(idx1[idx], w, minlength=self.comp_counts.shape[0])

        self.len_accumulator.seq_initialize(enc_len, weights, self._len_rng)

    def seq_update(
        self,
        x: tuple[int, np.ndarray, np.ndarray, E1, E2 | None],
        weights: np.ndarray,
        estimate: HierarchicalMixtureDistribution,
    ) -> None:
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

        ll_max = ll_mat.max(axis=1)
        good_rows = np.isfinite(ll_max)
        ll_exp = np.zeros_like(ll_mat)
        if np.any(good_rows):
            ll_exp[good_rows, :] = np.exp(ll_mat[good_rows, :] - ll_max[good_rows, None])

        ll_mat_t = np.dot(ll_exp, estimate.taus.T)
        ll_mat_t2 = np.full_like(ll_mat_t, -np.inf)
        pos = ll_mat_t > 0.0
        ll_mat_t2[pos] = np.log(ll_mat_t[pos])

        ll_max_sum = np.full(sz, 0.0, dtype=np.float64)
        if tsz > 0:
            ll_max_for_sum = np.where(good_rows, ll_max, -np.inf)
            ll_max_sum = np.bincount(idx, weights=ll_max_for_sum, minlength=sz)
        for i in range(self.num_mixtures):
            rv[:, i] = np.bincount(idx, weights=ll_mat_t2[:, i], minlength=sz)

        rv += estimate.log_w
        rv += ll_max_sum[:, None]
        ll_max2 = np.max(rv, axis=1, keepdims=True)
        bad_seq = ~np.isfinite(ll_max2.flatten())
        if np.any(bad_seq):
            rv[bad_seq, :] = estimate.log_w
            ll_max2[bad_seq, :] = np.max(estimate.log_w)
        rv -= ll_max2

        np.exp(rv, out=rv)
        ll_sum = rv.sum(axis=1, keepdims=True)

        # Capture per-row data log-likelihood (== seq_log_density) by reusing the posterior
        # normalizer already computed here: row_ll = rowmax + log(rowsum), with -inf for invalid
        # sequences seq_log_density would also return -inf for, plus the length-distribution term.
        # Only when the fused-EM fast path requests it (_track_ll); standard path is unaffected.
        if self._track_ll:
            with np.errstate(divide="ignore"):
                row_ll = ll_max2[:, 0] + np.log(ll_sum[:, 0])
            if np.any(bad_seq):
                row_ll[bad_seq] = -np.inf
            if estimate is not None and estimate.len_dist is not None:
                row_ll = row_ll + estimate.len_dist.seq_log_density(enc_len)
            self._seq_ll += float(np.dot(weights, row_ll))

        rv /= ll_sum
        # Outer weights: accumulate the DOCUMENT-level outer posteriors (one row per document,
        # weighted by the document weight) -- the EM maximizer for `w`. The token-level expansion
        # below (rv[idx, :]) feeds only the taus/topic statistics.
        self.w_counts += np.dot(weights, rv)
        rv = rv[idx, :]
        ww = np.reshape(weights[idx], (-1, 1))

        for i in range(self.num_mixtures):
            temp = np.zeros((tsz, self.num_topics), dtype=np.float64)
            valid = ll_mat_t[:, i] > 0.0
            if np.any(valid):
                temp[valid, :] = estimate.taus[i, None, :] * (rv[valid, i, None] / ll_mat_t[valid, i, None])
                temp[valid, :] *= ll_exp[valid, :]
            temp *= ww
            rv3 += temp
            self.comp_counts[i, :] += temp.sum(axis=0)

        for i in range(self.num_topics):
            self.accumulators[i].seq_update(enc_data, rv3[:, i], estimate.topics[i])

        if self.len_accumulator is not None:
            len_est = None if estimate is None else estimate.len_dist
            self.len_accumulator.seq_update(enc_len, weights, len_est)

    def seq_update_engine(self, x, weights, estimate, engine):
        """Engine-resident E-step: topic scoring and the outer/topic posterior arithmetic run on the
        active engine (numpy or torch); component counts and per-item topic responsibilities are
        produced on the engine and fed to the child accumulators. Matches host seq_update.
        """
        from mixle.stats.compute.backend import backend_seq_log_density

        sz, idx, cnt, enc_data, enc_len = x
        tsz = len(idx)
        weights_np = np.asarray(engine.to_numpy(weights) if hasattr(engine, "to_numpy") else weights, dtype=np.float64)
        if tsz == 0:
            if sz and estimate is not None:
                # every document is empty: its outer posterior is the prior (matches host seq_update)
                self.w_counts += float(np.sum(weights_np)) * np.asarray(estimate.w, dtype=np.float64)
            if self.len_accumulator is not None:
                self.len_accumulator.seq_update(enc_len, weights_np, None if estimate is None else estimate.len_dist)
            return

        idx_e = engine.asarray(np.asarray(idx, dtype=np.int64))
        neg = engine.asarray(-np.inf)
        ll_mat = engine.stack(
            [backend_seq_log_density(estimate.topics[i], enc_data, engine) for i in range(self.num_topics)], axis=1
        )  # (tsz, T)
        ll_max = engine.max(ll_mat, axis=1)  # (tsz,)
        finite = ll_max > engine.asarray(-1.0e308)
        ll_exp = engine.where(finite[:, None], engine.exp(ll_mat - ll_max[:, None]), engine.asarray(0.0))  # (tsz, T)

        taus_t = engine.asarray(np.asarray(estimate.taus, dtype=np.float64).T)  # (T, M)
        ll_mat_t = engine.matmul(ll_exp, taus_t)  # (tsz, M)
        pos = ll_mat_t > engine.asarray(0.0)
        ll_mat_t2 = engine.where(pos, engine.log(engine.where(pos, ll_mat_t, engine.asarray(1.0))), neg)

        ll_max_for_sum = engine.where(finite, ll_max, neg)
        ll_max_sum = engine.index_add(engine.zeros(sz), idx_e, ll_max_for_sum)  # (sz,)
        cols = [engine.index_add(engine.zeros(sz), idx_e, ll_mat_t2[:, i]) for i in range(self.num_mixtures)]
        rv = engine.stack(cols, axis=1)
        rv = rv + engine.asarray(estimate.log_w) + ll_max_sum[:, None]
        rv = rv - engine.logsumexp(rv, axis=1, keepdims=True)
        rv = engine.exp(rv)  # outer posteriors (sz, M)

        # Outer weights: document-level outer-posterior sums (matches the host seq_update).
        self.w_counts += np.dot(weights_np, np.asarray(engine.to_numpy(rv)))

        rv_items = rv[idx_e, :]  # (tsz, M)
        ww = engine.asarray(weights_np)[idx_e][:, None]
        taus = engine.asarray(np.asarray(estimate.taus, dtype=np.float64))  # (M, T)
        rv3 = engine.zeros((tsz, self.num_topics))
        comp_counts = engine.zeros((self.num_mixtures, self.num_topics))
        comp_rows = []
        for i in range(self.num_mixtures):
            valid = ll_mat_t[:, i] > 0.0
            ratio = engine.where(
                valid, rv_items[:, i] / engine.where(valid, ll_mat_t[:, i], engine.asarray(1.0)), engine.asarray(0.0)
            )
            temp = taus[i][None, :] * ratio[:, None] * ll_exp * ww  # (tsz, T)
            rv3 = rv3 + temp
            comp_rows.append(engine.sum(temp, axis=0))
        comp_counts = engine.stack(comp_rows, axis=0)

        self.comp_counts += np.asarray(engine.to_numpy(comp_counts))
        rv3_np = np.asarray(engine.to_numpy(rv3))
        for i in range(self.num_topics):
            self.accumulators[i].seq_update(enc_data, rv3_np[:, i], estimate.topics[i])
        if self.len_accumulator is not None:
            self.len_accumulator.seq_update(enc_len, weights_np, None if estimate is None else estimate.len_dist)

    def combine(
        self, suff_stat: tuple[np.ndarray, np.ndarray, tuple[SS1, ...], SS2 | None]
    ) -> "HierarchicalMixtureEstimatorAccumulator":
        """Combine the sufficient statistics of 'suff_stat; with attribute variables.

        Arg suff_stat is a Tuple of length 4 containing,
            suff_stat[0] (ndarray[float]): Aggregated token-level component counts with shape
                (num_mixtures, num_topics).
            suff_stat[1] (ndarray[float]): Aggregated document-level outer-posterior sums with length num_mixtures.
            suff_stat[2] (Tuple[SS1,...]): Tuple of 'num_topics' sufficient statistics for the topics.
            suff_stat[3] (Optional[SS2]): Optional sufficient statistic for length accumulator.

        Args:
            suff_stat (Tuple[np.ndarray, np.ndarray, Tuple[SS1, ...], Optional[SS2]]): See above for details.

        Returns:
            HierarchicalMixtureEstimatorAccumulator object.

        """
        self.comp_counts += suff_stat[0]
        self.w_counts += suff_stat[1]
        for i in range(self.num_topics):
            self.accumulators[i].combine(suff_stat[2][i])

        self.len_accumulator.combine(suff_stat[3])

        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...], Any | None]:
        """Returns sufficient statistics of type Tuple[np.ndarray, np.ndarray, Tuple[SS1,...], Optional[SS2]]."""
        return (
            self.comp_counts,
            self.w_counts,
            tuple([u.value() for u in self.accumulators]),
            self.len_accumulator.value(),
        )

    def from_value(
        self, x: tuple[np.ndarray, np.ndarray, tuple[SS1, ...], SS2 | None]
    ) -> "HierarchicalMixtureEstimatorAccumulator":
        """Set the attribute variables for sufficient statistics to arg 'x'.

        Arg 'x' is a Tuple of length 4 containing,
            x[0] (ndarray[float]): Aggregated token-level component counts with shape (num_mixtures, num_topics).
            x[1] (ndarray[float]): Aggregated document-level outer-posterior sums with length num_mixtures.
            x[2] (Tuple[SS1,...]): Tuple of 'num_topics' sufficient statistics for the topics.
            x[3] (Optional[SS2]): Optional sufficient statistic for length accumulator.

        Args:
            x (Tuple[np.ndarray, np.ndarray, Tuple[SS1, ...], Optional[SS2]]): See above for details.

        Returns:
            HierarchicalMixtureEstimatorAccumulator object.

        """
        self.comp_counts = x[0]
        self.w_counts = x[1]
        for i in range(self.num_topics):
            self.accumulators[i].from_value(x[2][i])

        self.len_accumulator.from_value(x[3])

        return self

    def scale(self, c: float) -> "HierarchicalMixtureEstimatorAccumulator":
        """Scale linear counts and delegate child/length sufficient statistics."""
        self.comp_counts *= c
        self.w_counts *= c
        for acc in self.accumulators:
            acc.scale(c)
        self.len_accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator into keyed sufficient statistics.

        Merges ``comp_counts``/``w_counts`` when ``weight_key`` is set, and merges topic accumulators when
        ``comp_key`` is set.

        Also delegates keyed merging to the topic and length accumulators.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to corresponding sufficient statistics.

        Returns:
            None.

        """
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                keyed_comp_counts, keyed_w_counts = stats_dict[self.weight_key]
                keyed_comp_counts += self.comp_counts
                keyed_w_counts += self.w_counts
            else:
                stats_dict[self.weight_key] = (self.comp_counts, self.w_counts)

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

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator's statistics from matching keyed values.

        Replaces ``comp_counts``/``w_counts`` when ``weight_key`` is set, and replaces topic accumulators when
        ``comp_key`` is set.

        Also delegates keyed replacement to the topic and length accumulators.

        Args:
            stats_dict (Dict[str, Any]): Dictionary mapping keys to corresponding sufficient statistics.

        Returns:
            None.

        """
        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.comp_counts, self.w_counts = stats_dict[self.weight_key]

        if self.comp_key is not None:
            if self.comp_key in stats_dict:
                acc = stats_dict[self.comp_key]
                self.accumulators = acc

        for u in self.accumulators:
            u.key_replace(stats_dict)

        self.len_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> "HierarchicalMixtureDataEncoder":
        """Return a data encoder assembled from topic and length accumulators."""
        topic_encoder = self.accumulators[0].acc_to_encoder()
        len_encoder = self.len_accumulator.acc_to_encoder()
        return HierarchicalMixtureDataEncoder(topic_encoder=topic_encoder, len_encoder=len_encoder)


class HierarchicalMixtureEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for hierarchical-mixture estimator accumulators."""

    def __init__(
        self,
        factories: Sequence[StatisticAccumulatorFactory],
        num_mixtures: int,
        len_factory: StatisticAccumulatorFactory | None = NullAccumulatorFactory(),
        keys: tuple[str | None, str | None] | None = (None, None),
    ):
        """Create a factory for hierarchical-mixture estimator accumulators.

        Args:
            factories (Sequence[StatisticAccumulatorFactory]): Accumulator factories for the topic distributions.
            num_mixtures (int): Number of outer mixture components.
            len_factory (Optional[StatisticAccumulatorFactory]): Accumulator factory for the length distribution.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Optional keys for outer weights and topic statistics.

        Attributes:
            factories (Sequence[StatisticAccumulatorFactory]): Accumulator factories for the topic distributions.
            num_mixtures (int): Number of outer mixture components.
            dim (int): Number of topics.
            len_factory (StatisticAccumulatorFactory): Accumulator factory for the length distribution.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Optional keys for outer weights and topic statistics.

        """
        self.factories = factories
        self.num_mixtures = num_mixtures
        self.dim = len(factories)
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys if keys is not None else (None, None)

    def make(self) -> "HierarchicalMixtureEstimatorAccumulator":
        """Return a new hierarchical-mixture estimator accumulator."""
        return HierarchicalMixtureEstimatorAccumulator(
            [self.factories[i].make() for i in range(self.dim)], self.num_mixtures, self.len_factory.make(), self.keys
        )


class HierarchicalMixtureEstimator(ParameterEstimator):
    """Estimate hierarchical-mixture distributions from sufficient statistics."""

    def __init__(
        self,
        estimators: Sequence[ParameterEstimator],
        num_mixtures: int,
        len_estimator: ParameterEstimator | None = NullEstimator(),
        len_dist: SequenceEncodableProbabilityDistribution | None = None,
        suff_stat: np.ndarray | None = None,
        pseudo_count: float | None = None,
        name: str | None = None,
        keys: tuple[str | None, str | None] | None = (None, None),
    ) -> None:
        """Create an estimator for hierarchical-mixture distributions.

        When ``pseudo_count`` is set, observed mixture statistics are smoothed. When ``suff_stat`` is also supplied,
        those prior statistics are blended with new sufficient statistics during estimation.

        Args:
            estimators (Sequence[ParameterEstimator]): Estimators for the topic distributions.
            num_mixtures (int): Number of outer-mixture components.
            len_estimator (Optional[ParameterEstimator]): Estimator for the length of inner mixture sequences.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Fixed length distribution for sequences.
            suff_stat (np.ndarray): Prior inner-mixture statistics with shape ``(num_mixtures, num_components)``.
            pseudo_count (Optional[float]): Prior mass used to smooth mixture weights.
            name (Optional[str]): Optional name assigned to estimated distributions.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Optional keys for weights and topics.

        Attributes:
            num_components (int): Number of topic distributions (inner-mixture).
            num_mixtures (int): Number of outer-mixture components.
            estimators (Sequence[ParameterEstimator]): Estimators for the topic distributions.
            pseudo_count (Optional[float]): Prior mass used to smooth mixture weights.
            suff_stat (np.ndarray): Prior inner-mixture statistics with shape ``(num_mixtures, num_components)``.
            len_estimator (Optional[ParameterEstimator]): Estimator for the length of inner mixture sequences.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys passed to accumulator factories.
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Fixed length distribution for sequences.
            name (Optional[str]): Optional name assigned to estimated distributions.

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

    def accumulator_factory(self) -> "HierarchicalMixtureEstimatorAccumulatorFactory":
        """Return an accumulator factory configured from this estimator."""
        est_factories = [u.accumulator_factory() for u in self.estimators]
        len_factory = self.len_estimator.accumulator_factory()
        return HierarchicalMixtureEstimatorAccumulatorFactory(est_factories, self.num_mixtures, len_factory, self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray, SS1, SS2 | None]
    ) -> "HierarchicalMixtureDistribution":
        """Estimate a hierarchical mixture from aggregated sufficient statistics.

        ``suff_stat`` is a four-item tuple containing:
            suff_stat[0] (ndarray[float]): Aggregated token-level component counts with shape
                (num_mixtures, num_topics). Row-normalized into the topic weights ``taus``.
            suff_stat[1] (ndarray[float]): Aggregated document-level outer-posterior sums with length
                num_mixtures. Normalized into the outer weights ``w`` -- the EM maximizer for ``w``
                is the document-level posterior sum, NOT the (length-weighted) token-level row sums.
            suff_stat[2] (Tuple[SS1,...]): One sufficient-statistic value per topic.
            suff_stat[3] (Optional[SS2]): Sufficient statistics for the length estimator.

        Args:
            nobs (Optional[float]): Number of observations used in accumulation of 'suff_stat'.
            suff_stat: See above for details.

        Returns:
            HierarchicalMixtureDistribution: Estimated distribution.

        """
        num_components = self.num_components
        num_mixtures = self.num_mixtures
        counts, w_counts, comp_suff_stats, len_suff_stats = suff_stat
        len_dist = self.len_estimator.estimate(None, len_suff_stats) if len_suff_stats is not None else self.len_dist

        components = [self.estimators[i].estimate(None, comp_suff_stats[i]) for i in range(num_components)]

        if self.pseudo_count is not None and self.suff_stat is None:
            p = self.pseudo_count / (num_components * num_mixtures)
            taus = counts + p
            taus /= taus.sum(axis=1, keepdims=True)
            w = w_counts + self.pseudo_count / num_mixtures
            w = w / w.sum()

        elif self.pseudo_count is not None and self.suff_stat is not None:
            taus = (counts + self.suff_stat * self.pseudo_count) / (counts.sum() + self.pseudo_count)
            taus /= taus.sum(axis=1, keepdims=True)
            w = w_counts + self.pseudo_count * np.asarray(self.suff_stat).sum(axis=1)
            w = w / w.sum()

        else:
            taus = counts
            row_sums = taus.sum(axis=1, keepdims=True)
            rows_pos = row_sums[:, 0] > 0
            taus[rows_pos, :] /= row_sums[rows_pos, :]
            taus[~rows_pos, :] = 1.0 / float(num_components)
            w_sum = w_counts.sum()

            if w_sum == 0:
                w = np.ones(num_mixtures) / float(num_mixtures)
            else:
                w = w_counts / w_sum

        return HierarchicalMixtureDistribution(components, w, taus, len_dist=len_dist, name=self.name, keys=self.keys)


class HierarchicalMixtureDataEncoder(DataSequenceEncoder):
    """Encode iid hierarchical-mixture observations for vectorized scoring."""

    def __init__(self, topic_encoder: DataSequenceEncoder, len_encoder: DataSequenceEncoder) -> None:
        """Create an encoder for hierarchical-mixture observations.

        Args:
            topic_encoder (DataSequenceEncoder): Encoder for topic-distribution observations.
            len_encoder (DataSequenceEncoder): Encoder for sequence lengths.

        Attributes:
            topic_encoder (DataSequenceEncoder): Encoder for topic-distribution observations.
            len_encoder (DataSequenceEncoder): Encoder for sequence lengths.

        """
        self.topic_encoder = topic_encoder
        self.len_encoder = len_encoder

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        rv = "HierarchicalMixtureDataEncoder(topic_encoder=" + str(self.topic_encoder) + ","
        rv += "len_encoder=" + str(self.len_encoder) + ")"
        return rv

    def __eq__(self, other: object) -> bool:
        """Return whether another encoder is equivalent to this encoder.

        Topic and length encoders must both be equivalent.

        Args:
            other (object): Object to compare.

        Returns:
            True if other is equivalent.

        """
        if isinstance(other, HierarchicalMixtureDataEncoder):
            return other.topic_encoder == self.topic_encoder and other.len_encoder == self.len_encoder
        else:
            return False

    def seq_encode(self, x: Sequence[Sequence[T]]) -> tuple[int, np.ndarray, np.ndarray, Any, Any | None]:
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


# --- Backward-compatible API naming aliases ---
HierarchicalMixtureAccumulator = HierarchicalMixtureEstimatorAccumulator
HierarchicalMixtureAccumulatorFactory = HierarchicalMixtureEstimatorAccumulatorFactory


def _register_hierarchical_mixture_engine_kernel():
    """Register the engine-resident hierarchical-mixture kernel (idempotent; called at import)."""
    from mixle.stats.compute.kernel import GenericKernel, GenericKernelFactory, KernelFactory, register_kernel_factory

    class HierarchicalMixtureKernel(GenericKernel):
        def accumulate(self, enc, weights):
            if self.estimator is None:
                raise ValueError("HierarchicalMixtureKernel.accumulate requires an estimator.")
            if not getattr(self.engine, "resident_estep", True):
                return super().accumulate(enc, weights)
            host_enc = getattr(enc, "host_payload", enc)
            accumulator = self.estimator.accumulator_factory().make()
            accumulator.seq_update_engine(host_enc, weights, self.dist, self.engine)
            return accumulator.value()

    class HierarchicalMixtureKernelFactory(KernelFactory):
        def build(self, dist, engine, estimator=None):
            if not dist.supports_engine(engine):
                return GenericKernelFactory().build(dist, engine, estimator=estimator)
            return HierarchicalMixtureKernel(dist, engine=engine, estimator=estimator)

    register_kernel_factory(HierarchicalMixtureDistribution, HierarchicalMixtureKernelFactory())


_register_hierarchical_mixture_engine_kernel()
