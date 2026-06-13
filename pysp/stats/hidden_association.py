"""Create, estimate, and sample from a hidden association model.

Defines the HiddenAssociationDistribution, HiddenAssociationSampler, HiddenAssociationAccumulatorFactory,
HiddenAssociationAccumulator, HiddenAssociationEstimator, and the HiddenAssociationDataEncoder classes for use with
pysparkplug.

Consider a set of value V = {v_1,v_2,...,v_K} with data type T. Let the given density be discrete probability density
over the values in V,

        P_g(X_i = v_k) = p_g(k), for k = 1,2,....,K

where sum_k p_g(k) = 1.0. Consider M samples from P_g() denoted x = (x_1,x_2,...,x_M). We then introduce the latent
variable U, where

    p_k(x) = p_mat(U = v_k | x) = (# of x_1,...,x_M that are = to v_k) / M, for k = 1,2,...,K.

We then draw N a positive integer N from distribution P_len(), then draw N samples from the density above to get
z = (z_1, z_2, ...., z_N). Last we sample from the conditional distribution defined for P_c(Y = v_k | z_i) to obtain
y = (y_1,...,y_N).

The log_density is given by,

    log(p_mat(x,y)) = sum_{i=1}^{N} log(sum_{k=1}^{K} p_k(x)*P_c(y_i|v_k)) + log(P_g(x)) + log(P_len(N)).

Note: That in this model we consider grouped-counts. So the given data type is

    x: Tuple[List[Tuple[T, float]], List[Tuple[T, float]]] = [x[0], x[1]],

where x[0] = [(value, count)] for the unique values of x_mat = (X_1,X_2,...,X_M) in V, and x[1] = [(value, count)] for
the unique values of Y = (Y_1,...,Y_N) in V as well.

"""
import numpy as np
import math
from pysp.arithmetic import *
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, StatisticAccumulatorFactory, DistributionSampler, DataSequenceEncoder
from pysp.utils.optsutil import count_by_value
from pysp.arithmetic import maxrandint
from pysp.stats.null_dist import NullDistribution, NullAccumulator, NullEstimator, NullDataEncoder, \
    NullAccumulatorFactory
from pysp.stats.conditional import ConditionalDistribution, ConditionalDistributionAccumulator, \
    ConditionalDistributionEstimator, ConditionalDistributionAccumulatorFactory
from typing import TypeVar, Dict, List, Sequence, Any, Optional, Tuple, Union


T = TypeVar('T') ### value data type
SS1 = TypeVar('SS1') ### Data type for suff stats of conditional
SS2 = TypeVar('SS2') ### Data type for suff stats of given
SS3 = TypeVar('SS3') ### Data type for suff stats of length

class HiddenAssociationDistribution(SequenceEncodableProbabilityDistribution):
    """Hidden association model: values of a second set are emitted conditionally on values drawn from a first set."""

    def __init__(self, cond_dist: ConditionalDistribution,
                 given_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None)) -> None:
        """HiddenAssociationDistribution object for specifying hidden association models.

        Args:
            cond_dist (ConditionalDistribution): ConditionalDistribution defining distributions conditioned on the
                number of states.
            given_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for the previous set. Must
                be compatible with Tuple[T, float].
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for the length of the observed
                emission. (Second set output).
            name (Optional[str]): Name for object instance.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys for weights and transitions.

        Attributes:
            cond_dist (ConditionalDistribution): ConditionalDistribution defining distributions conditioned on the
                number of states.
            given_dist (SequenceEncodableProbabilityDistribution): Distribution for the previous set. Defaults to
                NullDistribution.
            len_dist (SequenceEncodableProbabilityDistribution): Distribution for the length of the observed emission.
            name (Optional[str]): Name for object instance.
            keys (Tuple[Optional[str], Optional[str]]): Keys for weights and transitions.

        """
        self.cond_dist = cond_dist
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.given_dist = given_dist if given_dist is not None else NullDistribution()
        self.name = name
        self.key = keys if keys is not None else (None, None)

    def compute_capabilities(self):
        """Return backend capability metadata for this concrete hidden association model."""
        from pysp.stats.capabilities import DistributionCapabilities, intersect_engine_ready

        return DistributionCapabilities(engine_ready=intersect_engine_ready((
            self.cond_dist, self.given_dist, self.len_dist)),
            kernel_status='generic_latent')

    def compute_declaration(self):
        from pysp.stats.declarations import DistributionDeclaration, StatisticSpec, declaration_for
        conditional = declaration_for(self.cond_dist)
        given = None if isinstance(self.given_dist, NullDistribution) else declaration_for(self.given_dist)
        length = None if isinstance(self.len_dist, NullDistribution) else declaration_for(self.len_dist)
        children = tuple(child for child in (conditional, given, length) if child is not None)
        roles = ()
        if conditional is not None:
            roles += ('conditional',)
        if given is not None:
            roles += ('given',)
        if length is not None:
            roles += ('length',)
        return DistributionDeclaration(
            name='hidden_association',
            distribution_type=type(self),
            parameters=(),
            statistics=(
                StatisticSpec('conditional', kind='child_stat'),
                StatisticSpec('given', kind='child_stat'),
                StatisticSpec('length', kind='child_stat'),
            ),
            support='hidden_association_grouped_counts',
            children=children,
            child_roles=roles,
            differentiable=False,
        )

    def __str__(self) -> str:
        """Returns string representation of HiddenAssociationDistribution object."""
        s1 = repr(self.cond_dist)
        s2 = repr(self.given_dist)
        s3 = repr(self.len_dist)
        s4 = repr(self.name)
        s5 = repr(self.key)
        return 'HiddenAssociationDistribution(%s, given_dist=%s, len_dist=%s, name=%s, keys=%s)' % (s1, s2, s3, s4, s5)

    def density(self, x: Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]) -> float:
        """Density of the hidden association model at observation x.

        See log_density() for details.

        Args:
            x (Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]): Grouped-count observation
                ([(given value, count)], [(emitted value, count)]).

        Returns:
            Density at observation x.

        """
        return exp(self.log_density(x))

    def log_density(self, x: Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]) -> float:
        """Log-density of the hidden association model at observation x.

        For each emitted value in x[1], marginalizes the conditional emission density over the given
        values in x[0] weighted by their counts, then adds the log-density of the given set under
        given_dist and of the total emission count under len_dist.

        Args:
            x (Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]): Grouped-count observation
                ([(given value, count)], [(emitted value, count)]).

        Returns:
            Log-density at observation x.

        """
        rv = 0
        nn = 0
        for x1, c1 in x[1]:
            cc = 0  ## count for counts in given
            nn += c1
            ll = -np.inf
            for x0, c0 in x[0]:
                tt = self.cond_dist.log_density((x0, x1)) + math.log(c0)
                cc += c0

                if tt == -np.inf:
                    continue

                if ll > tt:
                    ll = math.log1p(math.exp(tt - ll)) + ll
                else:
                    ll = math.log1p(math.exp(ll - tt)) + tt

            ll -= math.log(cc)
            rv += ll * c1

        rv += self.given_dist.log_density(x[0])
        rv += self.len_dist.log_density(nn)

        return rv

    def seq_log_density(self, x: List[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]) -> np.ndarray:
        """Evaluation of log-density at sequence encoded input x (loops over log_density).

        Args:
            x (List[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]): Sequence encoded observations from
                HiddenAssociationDataEncoder.seq_encode() (the observations themselves).

        Returns:
            Numpy array of log-density values, one per observation.

        """
        return np.asarray([self.log_density(xx) for xx in x])

    def backend_seq_log_density(self, x: Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]],
                                engine: Any) -> Any:
        """Evaluate encoded log-densities through distribution-owned backend composition."""
        from pysp.stats.backend import backend_seq_log_density

        assoc_scores = []
        emit_lengths = []

        for given_obs, emitted_obs in x:
            emit_counts = [c1 for _, c1 in emitted_obs]
            emit_lengths.append(sum(emit_counts))

            if not emitted_obs:
                assoc_scores.append(engine.asarray(0.0))
                continue
            if not given_obs:
                assoc_scores.append(engine.asarray(float('-inf')))
                continue

            pairs = []
            given_counts = []
            for x0, c0 in given_obs:
                given_counts.append(c0)
            for x1, _ in emitted_obs:
                for x0, _ in given_obs:
                    pairs.append((x0, x1))

            cond_enc = self.cond_dist.dist_to_encoder().seq_encode(pairs)
            pair_scores = backend_seq_log_density(self.cond_dist, cond_enc, engine)
            pair_scores = pair_scores.reshape((len(emitted_obs), len(given_obs)))
            given_count_array = engine.asarray(np.asarray(given_counts, dtype=np.float64))
            emit_count_array = engine.asarray(np.asarray(emit_counts, dtype=np.float64))

            weighted_scores = pair_scores + engine.log(given_count_array).reshape((1, -1))
            per_emitted = engine.logsumexp(weighted_scores, axis=1) - engine.log(engine.sum(given_count_array))
            assoc_scores.append(engine.sum(per_emitted * emit_count_array))

        rv = engine.stack(assoc_scores) if assoc_scores else engine.zeros(0)

        given_enc = self.given_dist.dist_to_encoder().seq_encode([xx[0] for xx in x])
        rv = rv + backend_seq_log_density(self.given_dist, given_enc, engine)

        len_enc = self.len_dist.dist_to_encoder().seq_encode(emit_lengths)
        rv = rv + backend_seq_log_density(self.len_dist, len_enc, engine)

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'HiddenAssociationSampler':
        """Create a HiddenAssociationSampler object from this distribution.

        Requires non-null given_dist and len_dist.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            HiddenAssociationSampler object.

        """
        return HiddenAssociationSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'HiddenAssociationEstimator':
        """Create a HiddenAssociationEstimator from the component distributions' estimators.

        Args:
            pseudo_count (Optional[float]): Unused (kept for protocol consistency).

        Returns:
            HiddenAssociationEstimator object.

        """
        return HiddenAssociationEstimator(cond_estimator=self.cond_dist.estimator(),
                                          given_estimator=self.given_dist.estimator(),
                                          len_estimator=self.len_dist.estimator(),
                                          name=self.name)

    def dist_to_encoder(self) -> 'HiddenAssociationDataEncoder':
        """Returns a HiddenAssociationDataEncoder object for encoding sequences of data."""
        return HiddenAssociationDataEncoder()

class HiddenAssociationSampler(DistributionSampler):
    """HiddenAssociationSampler object for drawing grouped-count set pairs from a HiddenAssociationDistribution."""

    def __init__(self, dist: HiddenAssociationDistribution, seed: Optional[int] = None) -> None:
        """HiddenAssociationSampler object for sampling from a HiddenAssociationDistribution instance.

        Args:
            dist (HiddenAssociationDistribution): Object instance to sample from. Must have non-null
                given_dist and len_dist.
            seed (Optional[int]): Seed for random number generator.

        Attributes:
            rng (RandomState): RandomState object with seed set if passed in args.
            dist (HiddenAssociationDistribution): Object instance to sample from.
            cond_sampler (ConditionalSampler): Sampler for the conditional emission distribution.
            idx_sampler (RandomState): RandomState for drawing latent given-value indices.
            len_sampler (DistributionSampler): Sampler for the number of emitted values.
            given_sampler (DistributionSampler): Sampler for the given set.

        """
        if isinstance(dist.given_dist, NullDistribution):
            raise Exception('HiddenAssociationSampler requires attribute dist.given_dist.')
        if isinstance(dist.len_dist, NullDistribution):
            raise Exception('HiddenAssociationSampler requires attribute dist.len_dist.')

        self.rng = np.random.RandomState(seed)
        self.dist = dist

        self.cond_sampler = dist.cond_dist.sampler(seed=self.rng.randint(0, maxrandint))
        self.idx_sampler = np.random.RandomState(seed=self.rng.randint(0, maxrandint))
        self.len_sampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))
        self.given_sampler = self.dist.given_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample(self, size: Optional[int] = None)\
            -> Union[Sequence[Tuple[List[Tuple[Any, float]], List[Tuple[Any, float]]]],
                     Tuple[List[Tuple[Any, float]], List[Tuple[Any, float]]]]:
        """Draw iid grouped-count observations from the hidden association model.

        Args:
            size (Optional[int]): Number of observations to draw. If None, a single observation is returned.

        Returns:
            A ([(given value, count)], [(emitted value, count)]) tuple if size is None, else a list of
            such tuples of length size.

        """
        if size is None:
            prev_obs = self.given_sampler.sample()
            cnt = self.len_sampler.sample()
            rng = np.random.RandomState(self.idx_sampler.randint(0, maxrandint))
            rv = []
            pp = np.asarray([u[1] for u in prev_obs], dtype=float)
            pp /= pp.sum()

            for i in rng.choice(len(prev_obs), p=pp, size=cnt):
                rv.append(self.cond_sampler.sample_given(prev_obs[i][0]))

            rv = list(count_by_value(rv).items())

            return prev_obs, rv

        else:
            return [self.sample() for i in range(size)]

    def sample_given(self, x: List[Tuple[T, float]]):
        """Draw an emitted grouped-count set conditioned on the given set x.

        Args:
            x (List[Tuple[T, float]]): Given set as (value, count) pairs.

        Returns:
            List of (emitted value, count) pairs.

        """
        cnt = self.len_sampler.sample()
        rng = np.random.RandomState(self.idx_sampler.randint(0, maxrandint))
        rv = []
        pp = np.asarray([u[1] for u in x], dtype=float)
        pp /= pp.sum()

        for i in rng.choice(len(x), p=pp, size=cnt):
            rv.append(self.cond_sampler.sample_given(x[i][0]))

        rv = list(count_by_value(rv).items())

        return rv


class HiddenAssociationAccumulator(SequenceEncodableStatisticAccumulator):
    """HiddenAssociationAccumulator object for accumulating sufficient statistics from observed set pairs."""

    def __init__(self, cond_acc: ConditionalDistributionAccumulator,
                 given_acc: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 size_acc: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str],Optional[str]]] = (None, None)) -> None:
        """HiddenAssociationAccumulator object for accumulating sufficient statistics from observed data.

        Args:
            cond_acc (ConditionalDistributionAccumulator): Accumulator for the conditional emission distribution.
            given_acc (Optional[SequenceEncodableStatisticAccumulator]): Accumulator for the given set.
            size_acc (Optional[SequenceEncodableStatisticAccumulator]): Accumulator for the emission count.
            name (Optional[str]): Name for object instance.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys for weights and transitions.

        Attributes:
            cond_accumulator (ConditionalDistributionAccumulator): Accumulator for the conditional emission
                distribution.
            given_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the given set.
            size_accumulator (SequenceEncodableStatisticAccumulator): Accumulator for the emission count.
            init_key (Optional[str]): Key for the initial-state statistics.
            trans_key (Optional[str]): Key for the transition statistics.
            name (Optional[str]): Name for object instance.

        """
        self.cond_accumulator = cond_acc
        self.given_accumulator = given_acc if given_acc is not None else NullAccumulator()
        self.size_accumulator = size_acc if size_acc is not None else NullAccumulator()
        self.init_key, self.trans_key = keys if keys is not None else (None, None)
        self.name = name

    def update(self, x: Tuple[List[Tuple[T, float]], List[Tuple[T, float]]], weight: float,
               estimate: HiddenAssociationDistribution) -> None:
        """Update sufficient statistics with the posterior assignment of emitted values to given values.

        Args:
            x (Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]): Grouped-count observation
                ([(given value, count)], [(emitted value, count)]).
            weight (float): Weight for the observation.
            estimate (HiddenAssociationDistribution): Previous estimate used to compute posteriors.

        """
        nn = 0
        pv = np.zeros(len(x[0]))

        for x1, c1 in x[1]:
            cc = 0
            nn += c1
            ll = -np.inf

            for i, (x0, c0) in enumerate(x[0]):
                tt = estimate.cond_dist.log_density((x0, x1)) + math.log(c0)
                cc += c0
                pv[i] = tt

                if tt == -np.inf:
                    continue

                if ll > tt:
                    ll = math.log1p(math.exp(tt - ll)) + ll
                else:
                    ll = math.log1p(math.exp(ll - tt)) + tt

            pv -= ll
            np.exp(pv, out=pv)

            for i, (x0, c0) in enumerate(x[0]):
                self.cond_accumulator.update((x0, x1), pv[i] * c1 * weight, estimate.cond_dist)

        if self.given_accumulator is not None:
            given_dist = None if estimate is None else estimate.given_dist
            self.given_accumulator.update(x[0], weight, given_dist)

        if self.size_accumulator is not None:
            len_dist = None if estimate is None else estimate.len_dist
            self.size_accumulator.update(nn, weight, len_dist)

    def initialize(self, x: Tuple[List[Tuple[T, float]], List[Tuple[T, float]]], weight: float,
                   rng: np.random.RandomState) -> None:
        """Initialize sufficient statistics with random (Dirichlet) assignments of emitted to given values.

        Args:
            x (Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]): Grouped-count observation
                ([(given value, count)], [(emitted value, count)]).
            weight (float): Weight for the observation.
            rng (np.random.RandomState): Random number generator for the random assignments.

        """
        w = rng.dirichlet(np.ones(len(x[0])), size=len(x[1]))
        nn = 0
        for j, (x1, c1) in enumerate(x[1]):
            nn += c1
            for i, (x0, c0) in enumerate(x[0]):
                self.cond_accumulator.initialize((x0, x1), w[j, i] * c1 * weight, rng)

        if self.given_accumulator is not None:
            self.given_accumulator.initialize(x[0], weight, rng)

        if self.size_accumulator is not None:
            self.size_accumulator.initialize(nn, weight, rng)

    def seq_initialize(self, x:  Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]],
                       weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Initialize sufficient statistics from sequence encoded observations (loops over initialize()).

        Args:
            x (Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]): Sequence encoded observations.
            weights (np.ndarray): Weights, one per observation.
            rng (np.random.RandomState): Random number generator for the random assignments.

        """
        for i, xx in enumerate(x):
            self.initialize(xx, weights[i], rng)

    def seq_update(self, x: Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]], weights: np.ndarray,
                   estimate: HiddenAssociationDistribution) -> None:
        """Update sufficient statistics from sequence encoded observations (loops over update()).

        Args:
            x (Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]): Sequence encoded observations.
            weights (np.ndarray): Weights, one per observation.
            estimate (HiddenAssociationDistribution): Previous estimate used to compute posteriors.

        """
        for xx, ww in zip(x, weights):
            self.update(xx, ww, estimate)

    def combine(self, suff_stat: Tuple[SS1, Optional[SS2], Optional[SS3]]) -> 'HiddenAssociationAccumulator':
        """Merge sufficient statistics of suff_stat into this accumulator.

        Args:
            suff_stat (Tuple[SS1, Optional[SS2], Optional[SS3]]): Conditional, given, and size suff stats.

        Returns:
            This HiddenAssociationAccumulator.

        """
        cond_acc, given_acc, size_acc = suff_stat

        self.cond_accumulator.combine(cond_acc)
        self.given_accumulator.combine(given_acc)
        self.size_accumulator.combine(size_acc)

        return self

    def value(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Returns the sufficient statistics: (conditional, given, size) accumulator values."""
        return self.cond_accumulator.value(), self.given_accumulator.value(), self.size_accumulator.value()

    def from_value(self, x: Tuple[SS1, Optional[SS2], Optional[SS3]]) -> 'HiddenAssociationAccumulator':
        """Set the sufficient statistics of this accumulator from x.

        Args:
            x (Tuple[SS1, Optional[SS2], Optional[SS3]]): Conditional, given, and size suff stats.

        Returns:
            This HiddenAssociationAccumulator.

        """
        cond_acc, given_acc, size_acc = x

        self.cond_accumulator.from_value(cond_acc)
        self.given_accumulator.from_value(given_acc)
        self.size_accumulator.from_value(size_acc)

        return self

    def scale(self, c: float) -> 'HiddenAssociationAccumulator':
        """Scale sufficient statistics by delegating to child accumulators."""
        self.cond_accumulator.scale(c)
        self.given_accumulator.scale(c)
        self.size_accumulator.scale(c)
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        """Merge keyed statistics of the conditional, given, and size accumulators into stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to merged sufficient statistics.

        """
        self.cond_accumulator.key_merge(stats_dict)
        self.given_accumulator.key_merge(stats_dict)
        self.size_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        """Replace keyed statistics of the conditional, given, and size accumulators with those in stats_dict.

        Args:
            stats_dict (Dict[str, Any]): Maps keys to merged sufficient statistics.

        """
        self.cond_accumulator.key_replace(stats_dict)
        self.given_accumulator.key_replace(stats_dict)
        self.size_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'HiddenAssociationDataEncoder':
        """Returns a HiddenAssociationDataEncoder object for encoding sequences of data."""
        return HiddenAssociationDataEncoder()


class HiddenAssociationAccumulatorFactory(StatisticAccumulatorFactory):
    """HiddenAssociationAccumulatorFactory object for creating HiddenAssociationAccumulator objects."""

    def __init__(self, cond_factory: ConditionalDistributionAccumulatorFactory,
                 given_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
                 len_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None)) -> None:
        """HiddenAssociationAccumulatorFactory for creating HiddenAssociationAccumulator objects.

        Args:
            cond_factory (ConditionalDistributionAccumulatorFactory): Factory for the conditional emission
                accumulator.
            given_factory (Optional[StatisticAccumulatorFactory]): Factory for the given-set accumulator.
            len_factory (Optional[StatisticAccumulatorFactory]): Factory for the emission-count accumulator.
            name (Optional[str]): Name for object instance.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Keys for weights and transitions.

        Attributes:
            cond_factory (ConditionalDistributionAccumulatorFactory): Factory for the conditional emission
                accumulator.
            given_factory (StatisticAccumulatorFactory): Factory for the given-set accumulator.
            len_factory (StatisticAccumulatorFactory): Factory for the emission-count accumulator.
            keys (Tuple[Optional[str], Optional[str]]): Keys for weights and transitions.
            name (Optional[str]): Name for object instance.

        """
        self.cond_factory = cond_factory
        self.given_factory = given_factory if given_factory is not None else NullAccumulatorFactory()
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys if keys is not None else (None, None)
        self.name = name

    def make(self) -> 'HiddenAssociationAccumulator':
        """Returns a new HiddenAssociationAccumulator object."""
        return HiddenAssociationAccumulator(self.cond_factory.make(), self.given_factory.make(),
                                            self.len_factory.make(), self.name, self.keys)


class HiddenAssociationEstimator(ParameterEstimator):
    """HiddenAssociationEstimator object for estimating a HiddenAssociationDistribution from aggregated
    sufficient statistics."""

    def __init__(self, cond_estimator: ConditionalDistributionEstimator,
                 given_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 len_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 pseudo_count: Optional[float] = None,
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None)) -> None:
        """HiddenAssociationEstimator for estimating HiddenAssociationDistribution from sufficient statistics.

        Args:
            cond_estimator (ConditionalDistributionEstimator): Estimator for the conditional emission of values in
                set 2 given states.
            given_estimator (Optional[ParameterEstimator]): Estimator for the given values. Should be compatible with
                Tuple[T, float] where T is the type for the values.
            len_estimator (Optional[ParameterEstimator]): Estimator for the length of the observed set 2 values.
            pseudo_count (Optional[float]): Kept for consistency.
            name (Optional[str]): Set name for object instance.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Set keys for weights and transitions.

        Attributes:
            cond_estimator (ConditionalDistributionEstimator): Estimator for the conditional emission of values in
                set 2 given states.
            given_estimator (ParameterEstimator): Estimator for the given values. Should be compatible with
                Tuple[T, float] where T is the type for the values.
            len_estimator (ParameterEstimator): Estimator for the length of the observed set 2 values.
            pseudo_count (Optional[float]): Kept for consistency.
            name (Optional[str]): Set name for object instance.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Set keys for weights and transitions.

        """
        self.keys = keys if keys is not None else (None, None)
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.pseudo_count = pseudo_count
        self.cond_estimator = cond_estimator
        self.given_estimator = given_estimator if given_estimator is not None else NullEstimator()
        self.name = name

    def accumulator_factory(self) -> 'HiddenAssociationAccumulatorFactory':
        """Returns a HiddenAssociationAccumulatorFactory for creating HiddenAssociationAccumulator objects."""
        len_factory = self.len_estimator.accumulator_factory()
        given_factory = self.given_estimator.accumulator_factory()
        cond_factory = self.cond_estimator.accumulator_factory()
        return HiddenAssociationAccumulatorFactory(cond_factory=cond_factory, given_factory=given_factory,
                                                   len_factory=len_factory, name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[SS1, Optional[SS2], Optional[SS3]]) \
            -> 'HiddenAssociationDistribution':
        """Estimate a HiddenAssociationDistribution from aggregated sufficient statistics.

        Args:
            nobs (Optional[float]): Number of observations, passed to the given and length estimators.
            suff_stat (Tuple[SS1, Optional[SS2], Optional[SS3]]): Conditional, given, and size suff stats.

        Returns:
            HiddenAssociationDistribution object.

        """
        cond_stats, given_stats, size_stats = suff_stat

        cond_dist = self.cond_estimator.estimate(None, cond_stats)
        given_dist = self.given_estimator.estimate(nobs, given_stats)
        len_dist = self.len_estimator.estimate(nobs, size_stats)

        return HiddenAssociationDistribution(cond_dist=cond_dist, given_dist=given_dist, len_dist=len_dist,
                                             name=self.name)

class HiddenAssociationDataEncoder(DataSequenceEncoder):
    """HiddenAssociationDataEncoder object for encoding sequences of iid grouped-count set pair observations."""

    def __str__(self) -> str:
        """Returns string representation of HiddenAssociationDataEncoder object."""
        return 'HiddenAssociationDataEncoder'

    def __eq__(self, other) -> bool:
        """Checks if other object is an equivalent HiddenAssociationDataEncoder."""
        return isinstance(other, HiddenAssociationDataEncoder)

    def seq_encode(self, x: Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]])\
            -> Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]:
        """Encode a sequence of iid grouped-count observations (identity encoding).

        Args:
            x (Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]): Sequence of iid
                ([(given value, count)], [(emitted value, count)]) observations.

        Returns:
            The observations unchanged (seq_log_density and seq_update loop over them).

        """
        return x








