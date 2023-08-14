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

    def __str__(self) -> str:
        s1 = repr(self.cond_dist)
        s2 = repr(self.given_dist)
        s3 = repr(self.len_dist)
        s4 = repr(self.name)
        s5 = repr(self.key)
        return 'HiddenAssociationDistribution(%s, given_dist=%s, len_dist=%s, name=%s, keys=%s)' % (s1, s2, s3, s4, s5)

    def density(self, x: Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]) -> float:
        return exp(self.log_density(x))

    def log_density(self, x: Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]) -> float:
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
        return np.asarray([self.log_density(xx) for xx in x])

    def sampler(self, seed: Optional[int] = None) -> 'HiddenAssociationSampler':
        return HiddenAssociationSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'HiddenAssociationEstimator':
        return HiddenAssociationEstimator(cond_estimator=self.cond_dist.estimator(),
                                          given_estimator=self.given_dist.estimator(),
                                          len_estimator=self.len_dist.estimator(),
                                          name=self.name)

    def dist_to_encoder(self) -> 'HiddenAssociationDataEncoder':
        return HiddenAssociationDataEncoder()

class HiddenAssociationSampler(DistributionSampler):

    def __init__(self, dist: HiddenAssociationDistribution, seed: Optional[int] = None) -> None:
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

    def __init__(self, cond_acc: ConditionalDistributionAccumulator,
                 given_acc: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 size_acc: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str],Optional[str]]] = (None, None)) -> None:
        self.cond_accumulator = cond_acc
        self.given_accumulator = given_acc if given_acc is not None else NullAccumulator()
        self.size_accumulator = size_acc if size_acc is not None else NullAccumulator()
        self.init_key, self.trans_key = keys[0], keys[1] if keys is not None else (None, None)
        self.name = name

    def update(self, x: Tuple[List[Tuple[T, float]], List[Tuple[T, float]]], weight: float,
               estimate: HiddenAssociationDistribution) -> None:
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
        w = rng.dirichlet(np.ones(len(x[0])), size=len(x[1]))
        nn = 0
        for j, (x1, c1) in enumerate(x[1]):
            nn += c1
            for i, (x0, c0) in enumerate(x[0]):
                self.cond_accumulator.initialize((x0, x1), w[j, i] * weight, rng)

        if self.given_accumulator is not None:
            self.given_accumulator.initialize(x[0], weight, rng)

        if self.size_accumulator is not None:
            self.size_accumulator.initialize(nn, weight, rng)

    def seq_initialize(self, x:  Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]],
                       weights: np.ndarray, rng: np.random.RandomState) -> None:
        for i, xx in enumerate(x):
            self.initialize(xx, weights[i], rng)

    def seq_update(self, x: Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]], weights: np.ndarray,
                   estimate: HiddenAssociationDistribution) -> None:
        for xx, ww in zip(x, weights):
            self.update(xx, ww, estimate)

    def combine(self, suff_stat: Tuple[SS1, Optional[SS2], Optional[SS3]]) -> 'HiddenAssociationAccumulator':
        cond_acc, given_acc, size_acc = suff_stat

        self.cond_accumulator.combine(cond_acc)
        self.given_accumulator.combine(given_acc)
        self.size_accumulator.combine(size_acc)

        return self

    def value(self) -> Tuple[Any, Optional[Any], Optional[Any]]:
        return self.cond_accumulator.value(), self.given_accumulator.value(), self.size_accumulator.value()

    def from_value(self, x: Tuple[SS1, Optional[SS2], Optional[SS3]]) -> 'HiddenAssociationAccumulator':
        cond_acc, given_acc, size_acc = x

        self.cond_accumulator.from_value(cond_acc)
        self.given_accumulator.from_value(given_acc)
        self.size_accumulator.from_value(size_acc)

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        self.size_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        self.size_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'HiddenAssociationDataEncoder':
        return HiddenAssociationDataEncoder()


class HiddenAssociationAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, cond_factory: ConditionalDistributionAccumulatorFactory,
                 given_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
                 len_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None)) -> None:
        self.cond_factory = cond_factory
        self.given_factory = given_factory if given_factory is not None else NullAccumulatorFactory()
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.keys = keys if keys is not None else (None, None)
        self.name = name

    def make(self) -> 'HiddenAssociationAccumulator':
        return HiddenAssociationAccumulator(self.cond_factory.make(), self.given_factory.make(),
                                            self.len_factory.make(), self.name, self.keys)


class HiddenAssociationEstimator(ParameterEstimator):

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
        len_factory = self.len_estimator.accumulator_factory()
        given_factory = self.given_estimator.accumulator_factory()
        cond_factory = self.cond_estimator.accumulator_factory()
        return HiddenAssociationAccumulatorFactory(cond_factory=cond_factory, given_factory=given_factory,
                                                   len_factory=len_factory, name=self.name, keys=self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[SS1, Optional[SS2], Optional[SS3]]) \
            -> 'HiddenAssociationDistribution':

        cond_stats, given_stats, size_stats = suff_stat

        cond_dist = self.cond_estimator.estimate(None, cond_stats)
        given_dist = self.given_estimator.estimate(nobs, given_stats)
        len_dist = self.len_estimator.estimate(nobs, size_stats)

        return HiddenAssociationDistribution(cond_dist=cond_dist, given_dist=given_dist, len_dist=len_dist,
                                             name=self.name)

class HiddenAssociationDataEncoder(DataSequenceEncoder):

    def __str__(self) -> str:
        return 'HiddenAssociationDataEncoder'

    def __eq__(self, other) -> bool:
        return isinstance(other, HiddenAssociationDataEncoder)

    def seq_encode(self, x: Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]])\
            -> Sequence[Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]]:
        return x











