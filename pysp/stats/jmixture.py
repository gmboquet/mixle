"""Create, estimate, and sample from a Joint mixture distribution.

Defines the JointMixtureDistribution, JointMixtureSampler, JointMixtureAccumulatorFactory, JointMixtureAccumulator,
JointMixtureEstimator, and the JointMixtureDataEncoder classes for use with pysparkplug.

Data type: Tuple[T0, T1].

Consider a random variable X = (X_1, X_2). A joint mixture with N components for X_1, and M components for X_2 is
given by

    P(X) = sum_{i=1}^{N} w_i * f_i(X_1) * sum_{j=1}^{M} tau_{ij}*g_j(X_2),

where w_i is the probability of sampling X_1 from distribution f_i() (data type T0), tau_{ij} is the probability of
sampling X_2 from g_j() (data type T1) given X_1 was sampled from f_i().


"""
from pysp.arithmetic import *
from numpy.random import RandomState
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, StatisticAccumulatorFactory, \
    SequenceEncodableStatisticAccumulator, DataSequenceEncoder, DistributionSampler, ParameterEstimator
import numpy as np
import pysp.utils.vector as vec
from pysp.arithmetic import maxrandint

from typing import Tuple, Union, Any, Optional, TypeVar, Sequence, List, Dict

T0 = TypeVar('T0')
T1 = TypeVar('T1')
E0 = TypeVar('E0')
E1 = TypeVar('E1')
SS0 = TypeVar('SS0')
SS1 = TypeVar('SS1')


class JointMixtureDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, components1: Sequence[SequenceEncodableProbabilityDistribution],
                 components2: Sequence[SequenceEncodableProbabilityDistribution],
                 w1: Union[Sequence[float], np.ndarray],
                 w2: Union[Sequence[float], np.ndarray],
                 taus12: Union[List[List[float]], np.ndarray],
                 taus21: Union[List[List[float]], np.ndarray],
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None),
                 name: Optional[str] = None) -> None:
        """JointMixtureDistribution object for defining a joint mixture distribution.

        Note: Data type is Tuple[T0, T1] where all components1 entries and component2 entries are compatible with
        T0 and T1 respectively.

        Args:
            components1(Sequence[SequenceEncodableProbabilityDistribution]): Mixture components for mixture of X1.
            components2 (Sequence[SequenceEncodableProbabilityDistribution]): Mixture components for mixture X2.
            w1 (np.ndarray): Probability of drawing X1 from component i.
            w2 (np.ndarray): Probability of drawing X2 from component j.
            taus12 (np.ndarray): 2-d Numpy array with probabilities of drawing X2 from comp j given X1 was drawn from
                comp i. Rows are component X1 state.
            taus21 (np.ndarray): 2-d Numpy array with probabilities of drawing X1 from comp i given X2 was drawn from
                comp j. Rows are component X1 state.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        Attributes:
            components1(Sequence[SequenceEncodableProbabilityDistribution]): Mixture components for mixture of X1.
            components2 (Sequence[SequenceEncodableProbabilityDistribution]): Mixture components for mixture X2.
            w1 (np.ndarray): Probability of drawing X1 from component i.
            w2 (np.ndarray): Probability of drawing X2 from component j.
            num_components1 (int): Number of mixture components for X1.
            num_components2 (int): Number of mixture components for X2.
            taus12 (np.ndarray): 2-d Numpy array with probabilities of drawing X2 from comp j given X1 was drawn from
                comp i. Rows are component X1 state.
            taus21 (np.ndarray): 2-d Numpy array with probabilities of drawing X1 from comp i given X2 was drawn from
                comp j. Rows are component X1 state.
            log_w1 (np.ndarray): Log-probability of drawing X1 from component i.
            log_w2 (np.ndarray): Log-probability of drawing X2 from component j.
            log_taus12 (np.ndarray): 2-d Numpy array with log-probabilities of drawing X2 from comp j given X1 was
                drawn from comp i. Rows are component X1 state.
            log_taus21 (np.ndarray): 2-d Numpy array with log-probabilities of drawing X1 from comp i given X2 was
                drawn from comp j. Rows are component X1 state.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        """
        with np.errstate(divide='ignore'):
            self.components1 = components1
            self.components2 = components2
            self.w1 = vec.make(w1)
            self.w2 = vec.make(w2)
            self.num_components1 = len(components1)
            self.num_components2 = len(components2)
            self.taus12 = np.reshape(taus12, (self.num_components1, self.num_components2))
            self.taus21 = np.reshape(taus21, (self.num_components1, self.num_components2))
            self.log_w1 = np.log(self.w1)
            self.log_w2 = np.log(self.w2)
            self.log_taus12 = np.log(self.taus12)
            self.log_taus21 = np.log(self.taus21)
            self.keys = keys if keys is not None else (None, None, None)
            self.name = name

    def __str__(self) -> str:
        s1 = ','.join([str(u) for u in self.components1])
        s2 = ','.join([str(u) for u in self.components2])
        s3 = ','.join(map(str, self.w1))
        s4 = ','.join(map(str, self.w2))
        s5 = ','.join(map(str, self.taus12.flatten()))
        s6 = ','.join(map(str, self.taus21.flatten()))
        s7 = repr(self.name)

        return 'JointMixtureDistribution([%s], [%s], [%s], [%s], [%s], [%s], name=%s)' % (s1, s2, s3, s4, s5, s6, s7)

    def density(self, x: Tuple[T0, T1]) -> float:
        return exp(self.log_density(x))

    def log_density(self, x: Tuple[T0, T1]) -> float:
        ll1 = np.zeros((1, self.num_components1))
        ll2 = np.zeros((1, self.num_components2))

        for i in range(self.num_components1):
            ll1[0, i] = self.components1[i].log_density(x[0]) + self.log_w1[i]
        for i in range(self.num_components2):
            ll2[0, i] += self.components2[i].log_density(x[1])

        max1 = ll1.max()
        ll1 -= max1
        np.exp(ll1, out=ll1)

        max2 = np.max(ll2)
        ll2 -= max2
        np.exp(ll2, out=ll2)

        ll12 = np.dot(ll1, self.taus12)
        ll2 *= ll12

        rv = np.log(ll2.sum()) + max1 + max2

        return rv

    def seq_log_density(self, x: Tuple[int, E0, E1]) -> np.ndarray:
        sz, enc_data1, enc_data2 = x
        ll_mat1 = np.zeros((sz, self.num_components1))
        ll_mat2 = np.zeros((sz, self.num_components2))

        for i in range(self.num_components1):
            ll_mat1[:, i] = self.components1[i].seq_log_density(enc_data1)
            ll_mat1[:, i] += self.log_w1[i]

        for i in range(self.num_components2):
            ll_mat2[:, i] = self.components2[i].seq_log_density(enc_data2)

        ll_max1 = ll_mat1.max(axis=1, keepdims=True)
        ll_mat1 -= ll_max1
        np.exp(ll_mat1, out=ll_mat1)

        ll_max2 = ll_mat2.max(axis=1, keepdims=True)
        ll_mat2 -= ll_max2
        np.exp(ll_mat2, out=ll_mat2)

        ll_mat12 = np.dot(ll_mat1, self.taus12)
        ll_mat2 *= ll_mat12

        rv = np.log(ll_mat2.sum(axis=1)) + ll_max1[:, 0] + ll_max2[:, 0]

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'JointMixtureSampler':
        return JointMixtureSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'JointMixtureEstimator':
        estimators1 = [comp1.estimator() for comp1 in self.components1]
        estimators2 = [comp2.estimator() for comp2 in self.components2]

        return JointMixtureEstimator(estimators1=estimators1, estimators2=estimators2, pseudo_count=pseudo_count,
                                     keys=self.keys, name=self.name)

    def dist_to_encoder(self) -> 'DataSequenceEncoder':
        encoder1 = self.components1[0].dist_to_encoder()
        encoder2 = self.components2[0].dist_to_encoder()
        return JointMixtureDataEncoder(encoder1=encoder1, encoder2=encoder2)


class JointMixtureSampler(DistributionSampler):
    def __init__(self, dist: JointMixtureDistribution, seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        self.comp_sampler1 = [d.sampler(seed=self.rng.randint(0, maxrandint)) for d in self.dist.components1]
        self.comp_sampler2 = [d.sampler(seed=self.rng.randint(0, maxrandint)) for d in self.dist.components2]

    def sample(self, size: Optional[int] = None) -> Union[Tuple[Any, Any], Sequence[Tuple[Any, Any]]]:

        if size is None:
            comp_state1 = self.rng.choice(range(0, self.dist.num_components1), replace=True, p=self.dist.w1)
            f1 = self.comp_sampler1[comp_state1].sample()
            comp_state2 = self.rng.choice(range(0, self.dist.num_components2), replace=True,
                                         p=self.dist.taus12[comp_state1, :])
            f2 = self.comp_sampler2[comp_state2].sample()

            return f1, f2
        else:
            return [self.sample() for i in range(size)]


class JointMixtureEstimatorAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, accumulators1: Sequence[SequenceEncodableStatisticAccumulator],
                 accumulators2: Sequence[SequenceEncodableStatisticAccumulator],
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None),
                 name: Optional[str] = None) -> None:
        """

        Args:
            accumulators1 (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the mixture components
                of X1.
            accumulators2 (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the mixture components
                of X2.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        Attributes:
            accumulators1 (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the mixture components
                of X1.
            accumulators2 (Sequence[SequenceEncodableStatisticAccumulator]): Accumulators for the mixture components
                of X2.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            num_components1 (int): Number of X1 mixture components.
            num_components2 (int): Number of X2 mixture components.
            comp_counts1 (np.ndarray): Weighted observation counts for states of mixture on X1.
            comp_counts2 (np.ndarray): Weighted observation counts for states of mixture on X2.
            joint_counts (np.ndarray): 2-d Numpy array for counts of state-given-state weights. Row indexed by states
                of X1, cols indexed by states of X2.
            name (Optional[str]): Set name to object.

            _rng_init (bool): Set to True once _rng_ members have been set.
            _idx1_rng (Optional[RandomState]): RandomState for generating states for X1 in initializer.
            _idx2_rng (Optional[RandomState]): RandomState for generating states for X2 in initializer.
            _acc1_rng (Optional[List[RandomState]]): List of RandomStates for initializing each accumulator for
                mixture components of X1.
            _acc2_rng (Optional[List[RandomState]]): List of RandomStates for initializing each accumulator for
                mixture components of X2.


        """
        self.accumulators1 = accumulators1
        self.accumulators2 = accumulators2
        self.keys = keys if keys is not None else (None, None, None)
        self.num_components1 = len(accumulators1)
        self.num_components2 = len(accumulators2)
        self.comp_counts1 = vec.zeros(self.num_components1)
        self.comp_counts2 = vec.zeros(self.num_components2)
        self.joint_counts = vec.zeros((self.num_components1, self.num_components2))
        self.name = name

        self._rng_init = False
        self._idx1_rng: Optional[RandomState] = None
        self._idx2_rng: Optional[RandomState] = None
        self._acc1_rng: Optional[List[RandomState]] = None
        self._acc2_rng: Optional[List[RandomState]] = None

    def update(self, x: Tuple[T0, T1], weight: float, estimate: JointMixtureDistribution) -> None:
        pass

    def _rng_initialize(self, rng: RandomState) -> None:
        self._idx1_rng = RandomState(seed=rng.randint(0, maxrandint))
        self._idx2_rng = RandomState(seed=rng.randint(0, maxrandint))
        self._acc1_rng = [RandomState(seed=rng.randint(0, maxrandint)) for i in range(self.num_components1)]
        self._acc2_rng = [RandomState(seed=rng.randint(0, maxrandint)) for i in range(self.num_components2)]
        self._rng_init = True

    def initialize(self, x: Tuple[T0, T1], weight: float, rng: RandomState) -> None:

        if not self._rng_init:
            self._rng_initialize(rng)

        idx1 = self._idx1_rng.choice(self.num_components1)
        idx2 = self._idx2_rng.choice(self.num_components2)

        self.joint_counts[idx1, idx2] += 1.0

        for i in range(self.num_components1):
            w = 1.0 if i == idx1 else 0.0
            self.accumulators1[i].initialize(x[0], w, self._acc1_rng[i])
            self.comp_counts1[i] += w
        for i in range(self.num_components2):
            w = 1.0 if i == idx2 else 0.0
            self.accumulators2[i].initialize(x[1], w, self._acc2_rng[i])
            self.comp_counts2[i] += w

    def seq_initialize(self, x: Tuple[int, E0, E1], weights, rng) -> None:
        sz, enc1, enc2 = x

        if not self._rng_init:
            self._rng_initialize(rng)

        idx1 = self._idx1_rng.choice(self.num_components1, size=sz)
        idx2 = self._idx2_rng.choice(self.num_components2, size=sz)

        temp = np.bincount(idx1*self.num_components1 + idx2, minlength=self.num_components1*self.num_components2)
        self.joint_counts += np.reshape(temp, (self.num_components1, self.num_components2))

        for i in range(self.num_components1):
            w = np.zeros(sz)
            w[idx1 == i] = 1.0
            self.accumulators1[i].seq_initialize(enc1, w, self._acc1_rng[i])
            self.comp_counts1[i] += np.sum(w)

        for i in range(self.num_components2):
            w = np.zeros(sz)
            w[idx2 == i] = 1.0
            self.accumulators2[i].seq_initialize(enc2, w, self._acc2_rng[i])
            self.comp_counts2[i] += np.sum(w)

    def seq_update(self, x: Tuple[int, E0, E1], weights: np.ndarray, estimate: JointMixtureDistribution) -> None:
        sz, enc_data1, enc_data2 = x
        ll_mat1 = np.zeros((sz, self.num_components1, 1))
        ll_mat2 = np.zeros((sz, 1, self.num_components2))
        log_w = estimate.log_w1

        for i in range(estimate.num_components1):
            ll_mat1[:, i, 0] = estimate.components1[i].seq_log_density(enc_data1)
            ll_mat1[:, i, 0] += log_w[i]

        ll_max1 = ll_mat1.max(axis=1, keepdims=True)
        ll_mat1 -= ll_max1
        np.exp(ll_mat1, out=ll_mat1)

        for i in range(estimate.num_components2):
            ll_mat2[:, 0, i] = estimate.components2[i].seq_log_density(enc_data2)

        ll_max2 = ll_mat2.max(axis=2, keepdims=True)
        ll_mat2 -= ll_max2
        np.exp(ll_mat2, out=ll_mat2)

        ll_joint = ll_mat1 * ll_mat2
        ll_joint *= estimate.taus12

        gamma_2 = np.sum(ll_joint, axis=1, keepdims=True)
        sf = np.sum(gamma_2, axis=2, keepdims=True)
        ww = np.reshape(weights, [-1, 1, 1])

        gamma_1 = np.sum(ll_joint, axis=2, keepdims=True)
        gamma_1 *= ww / sf
        gamma_2 *= ww / sf

        ll_joint *= ww / sf

        self.comp_counts1 += np.sum(gamma_1, axis=0).flatten()
        self.comp_counts2 += np.sum(gamma_2, axis=0).flatten()
        self.joint_counts += ll_joint.sum(axis=0)

        for i in range(self.num_components1):
            self.accumulators1[i].seq_update(enc_data1, gamma_1[:, i, 0], estimate.components1[i])

        for i in range(self.num_components2):
            self.accumulators2[i].seq_update(enc_data2, gamma_2[:, 0, i], estimate.components2[i])

    def combine(self, suff_stat: Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[E0, ...], Tuple[E1, ...]]) \
            -> 'JointMixtureEstimatorAccumulator':

        cc1, cc2, jc, s1, s2 = suff_stat

        self.joint_counts += jc
        self.comp_counts1 += cc1
        for i in range(self.num_components1):
            self.accumulators1[i].combine(s1[i])
        self.comp_counts2 += cc2
        for i in range(self.num_components2):
            self.accumulators2[i].combine(s2[i])

        return self

    def value(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[Any,...], Tuple[Any,...]]:
        return self.comp_counts1, self.comp_counts2, self.joint_counts, tuple(
            [u.value() for u in self.accumulators1]), tuple([u.value() for u in self.accumulators2])

    def from_value(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[E0, ...], Tuple[E1, ...]]) \
            -> 'JointMixtureEstimatorAccumulator':

        cc1, cc2, jc, s1, s2 = x

        self.comp_counts1 = cc1
        self.comp_counts2 = cc2
        self.joint_counts = jc

        for i in range(self.num_components1):
            self.accumulators1[i].from_value(s1[i])
        for i in range(self.num_components2):
            self.accumulators2[i].from_value(s2[i])

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        weight_key, acc1_key, acc2_key = self.keys

        if weight_key is not None:
            if weight_key in stats_dict[weight_key]:
                x1, x2, x3 = stats_dict[weight_key]
                self.comp_counts1 += x1
                self.comp_counts2 += x2
                self.joint_counts += x3

        if acc1_key is not None:
            if acc1_key in stats_dict[acc1_key]:
                for i, u in enumerate(stats_dict[acc1_key]):
                    self.accumulators1[i].combine(u)
            else:
                stats_dict[acc1_key] = tuple([acc.value() for acc in self.accumulators1])

        if acc2_key is not None:
            if acc2_key in stats_dict[acc2_key]:
                for i, u in enumerate(stats_dict[acc2_key]):
                    self.accumulators2[i].combine(u)
            else:
                stats_dict[acc2_key] = tuple([acc.value() for acc in self.accumulators2])

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        weight_key, acc1_key, acc2_key = self.keys

        if weight_key is not None:
            if weight_key in stats_dict[weight_key]:
                x1, x2, x3 = stats_dict[weight_key]
                self.comp_counts1 = x1
                self.comp_counts2 = x2
                self.joint_counts = x3

        if acc1_key is not None:
            if acc1_key in stats_dict[acc1_key]:
                for i, u in enumerate(stats_dict[acc1_key]):
                    self.accumulators1[i].from_value(u)

        if acc2_key is not None:
            if acc2_key in stats_dict[acc2_key]:
                for i, u in enumerate(stats_dict[acc2_key]):
                    self.accumulators2[i].from_value(u)

    def acc_to_encoder(self) -> 'DataSequenceEncoder':
        encoder1 = self.accumulators1[0].acc_to_encoder()
        encoder2 = self.accumulators2[0].acc_to_encoder()
        return JointMixtureDataEncoder(encoder1=encoder1, encoder2=encoder2)


class JointMixtureEstimatorAccumulatorFactory(StatisticAccumulatorFactory):
    def __init__(self, factories1: Sequence[StatisticAccumulatorFactory],
                 factories2: Sequence[StatisticAccumulatorFactory],
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None),
                 name: Optional[str] = None) -> None:
        """JointMixtureEstimatorAccumulatorFactory object for creating JointMixtureEstimatorAccumulator objects.

        Args:
            factories1 (Sequence[StatisticAccumulatorFactory]): List of mixture component factories for X1.
            factories2 (Sequence[StatisticAccumulatorFactory]): List of mixture component factories for X2.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        Attributes:
            factories1 (Sequence[StatisticAccumulatorFactory]): List of mixture component factories for X1.
            factories2 (Sequence[StatisticAccumulatorFactory]): List of mixture component factories for X2.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        """
        self.factories1 = factories1
        self.factories2 = factories2
        self.keys = keys if keys is not None else (None, None, None)
        self.name = name

    def make(self) -> 'JointMixtureEstimatorAccumulator':
        f1 = [self.factories1[i].make() for i in range(len(self.factories1))]
        f2 = [self.factories2[i].make() for i in range(len(self.factories2))]
        return JointMixtureEstimatorAccumulator(f1, f2, keys=self.keys)


class JointMixtureEstimator(ParameterEstimator):

    def __init__(self, estimators1: Sequence[ParameterEstimator], estimators2: Sequence[ParameterEstimator],
                 suff_stat: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[E0, ...], Tuple[E1, ...]]] = None,
                 pseudo_count: Optional[Tuple[float, float, float]] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str], Optional[str]]] = (None, None, None),
                 name: Optional[str] = None)  -> None:
        """JointMixtureEstimator object for estimating joint mixture distribution from aggregated sufficient stats.

        Args:
            estimators1 (Sequence[ParameterEstimator]): Estimators for mixture component of X1.
            estimators2 (Sequence[ParameterEstimator]): Estimators for mixture component of X2.
            suff_stat:
            pseudo_count (Optional[Tuple[float, float, float]]): Used to re-weight the state counts in estimation.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        Attributes:
            estimators1 (Sequence[ParameterEstimator]): Estimators for mixture component of X1.
            estimators2 (Sequence[ParameterEstimator]): Estimators for mixture component of X2.
            suff_stat:
            pseudo_count (Optional[Tuple[float, float, float]]): Used to re-weight the state counts in estimation.
            keys (Optional[Tuple[Optional[str], Optional[str], Optional[str]]]): Set keys for weights, mixture
                components of X1, mixture components of X2.
            name (Optional[str]): Set name to object.

        """
        self.num_components1 = len(estimators1)
        self.num_components2 = len(estimators2)
        self.estimators1 = estimators1
        self.estimators2 = estimators2
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys if keys is not None else (None, None, None)
        self.name = name

    def accumulator_factory(self) -> 'JointMixtureEstimatorAccumulatorFactory':
        est_factories1 = [u.accumulator_factory() for u in self.estimators1]
        est_factories2 = [u.accumulator_factory() for u in self.estimators2]
        return JointMixtureEstimatorAccumulatorFactory(est_factories1, est_factories2, self.keys)

    def estimate(self, nobs, suff_stat: Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[E0, ...], Tuple[E1, ...]]) \
            -> 'JointMixtureDistribution':
        """Estimate a Joint mixture distribution from aggregated sufficient statistics.

        suff_stat is a Tuple containing:
            suff_stat[0] (np.ndarray): Component counts for outer mixture.
            suff_stat[1] (np.ndarray): Component counts for the inner mixture.
            suff_stat[2] (np.ndarray): Component counts for the comps of inner mix given an outer mix component.
            suff_stat[3] (Tuple[E0,...]): Suff-stats for outer comps
            suff_stat[4] (Tuple[E1,...]): Suff-stats for the inner comps.

        Args:
            nobs (Optional[float]): Weighted number of observations used in aggregation of suff_stats.
            suff_stat: See above for details.

        Returns:

        """
        num_components1 = self.num_components1
        num_components2 = self.num_components2
        counts1, counts2, joint_counts, comp_suff_stats1, comp_suff_stats2 = suff_stat

        components1 = [self.estimators1[i].estimate(counts1[i], comp_suff_stats1[i]) for i in range(num_components1)]
        components2 = [self.estimators2[i].estimate(counts2[i], comp_suff_stats2[i]) for i in range(num_components2)]

        if self.pseudo_count is not None and self.suff_stat is None:
            p1 = self.pseudo_count[0] / float(self.num_components1)
            p2 = self.pseudo_count[1] / float(self.num_components2)
            p3 = self.pseudo_count[2] / float(self.num_components2 * self.num_components1)

            w1 = (counts1 + p1) / (counts1.sum() + p1)
            w2 = (counts2 + p2) / (counts2.sum() + p2)
            taus = joint_counts + p3

            taus12_sum = np.sum(taus, axis=1, keepdims=True)
            taus12_sum[taus12_sum == 0] = 1.0
            taus12 = taus / taus12_sum

            taus21_sum = np.sum(taus, axis=0, keepdims=True)
            taus21_sum[taus21_sum == 0] = 1.0
            taus21 = taus / taus21_sum

        else:
            w1 = counts1 / counts1.sum()
            w2 = counts2 / counts2.sum()
            taus = joint_counts

            taus12_sum = np.sum(taus, axis=1, keepdims=True)
            taus12_sum[taus12_sum == 0] = 1.0
            taus12 = taus / taus12_sum

            taus21_sum = np.sum(taus, axis=0, keepdims=True)
            taus21_sum[taus21_sum == 0] = 1.0
            taus21 = taus / taus21_sum

        return JointMixtureDistribution(components1, components2, w1, w2, taus12, taus21, name=self.name)


class JointMixtureDataEncoder(DataSequenceEncoder):

    def __init__(self, encoder1: DataSequenceEncoder, encoder2: DataSequenceEncoder) -> None:
        """JointMixtureDataEncoder object for encoding sequences of iid joint mixture observations.

        Args:
            encoder1 (DataSequenceEncoder): DataSequenceEncoder for the components of X1.
            encoder2 (DataSequenceEncoder): DataSequenceEncoder for the components of X2.

        Attributes:
            encoder1 (DataSequenceEncoder): DataSequenceEncoder for the components of X1.
            encoder2 (DataSequenceEncoder): DataSequenceEncoder for the components of X2.

        """
        self.encoder1 = encoder1
        self.encoder2 = encoder2

    def __str__(self) -> str:
        return 'JointMixtureDataEncoder(encoder0=' + str(self.encoder1) + ',encoder1=' + str(self.encoder2) + ')'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, JointMixtureDataEncoder):
            return self.encoder2 == other.encoder2 and self.encoder1 == other.encoder1
        else:
            return False

    def seq_encode(self, x: Sequence[Tuple[T0, T1]]) -> Tuple[int, Any, Any]:
        rv0 = len(x)
        rv1 = self.encoder1.seq_encode([u[0] for u in x])
        rv2 = self.encoder2.seq_encode([u[1] for u in x])

        return rv0, rv1, rv2
