"""Create, estimate, and sample from an integer hidden association model.

Defines the IntegerHiddenAssociationDistribution, IntegerHiddenAssociationSampler,
IntegerHiddenAssociationAccumulatorFactory, IntegerHiddenAssociationAccumulator, IntegerHiddenAssociationEstimator, and
the IntegerHiddenAssociationDataEncoder classes for use with pysparkplug.

The k-rank variant of SparseMarkovAssociation.

Data type:  Tuple[List[Tuple[int, float]], List[Tuple[int, float]]].

The SparseMarkovAssociation model is a generative model for two sets of words S_1 ={w_{1,1},...,w_{1,n}} and
S_2 ={w_{2,1},...,w_{2,m}} over W possible words. The model assumes a hidden set of states
H_2 = {h_{2,1},...,h_{2,m}} where h_{2,j} takes on values in {1,2,...,k} and a hidden set of assignments
A_2 = {a_{2,1},...,a_{2,m}} where a_{2,j} takes on values in {1,2,...,m}. The observed likelihood function is
computed from P(S_1, S_2) = P(S_2 | S_1) P(S_1), where

(1) log(P(S_2|S_1)) = sum_{i=1}^{m} log(P(w_{2,i}|w_{1,1},...,w_{1,n})
    = sum_{i=1}^{m} log( (1/m)*sum_{j=1}^{n} (1-alpha)*sum_{k=1}^{K}P(w_{2,i} | h_{2,k})*P(h_{2,k}|w_{1,j}) + alpha/W).
(2) log(P(S_1)) = sum_{j=1}^{n} log((1-alpha)*P(w_{1,j}) + alpha/W ).

This model is great when the conditional probability matrix is both large and dense. It can also be nested inside other
graphical models like a mixture model.

Note: This is the k-rank equivalent of SparseMarkovAssociationModel.

"""
import math
import numba
import numpy as np

from pysp.arithmetic import *
from pysp.arithmetic import maxrandint
from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DistributionSampler, DataSequenceEncoder, StatisticAccumulatorFactory
from pysp.stats.null_dist import NullDistribution, NullAccumulator, NullEstimator, NullDataEncoder, \
    NullAccumulatorFactory
from pysp.utils.optsutil import count_by_value

from typing import Optional, List, Tuple, Optional, Any, Dict, Union, Sequence, TypeVar

E0 = Tuple[Tuple[List[Tuple[np.ndarray, ...]], Optional[Any], Optional[Any]], None]

E1 = TypeVar('E1')  # Encoded prev
E2 = TypeVar('E2')  # Encoded lengths
E3 = Tuple[Tuple[List[Tuple[np.ndarray, ...]], Optional[E2], Optional[E1]], None]
E4 = Tuple[None, Tuple[Tuple[np.ndarray, ...], Optional[E1], Optional[E2]]]
E = Union[E3, E4]

SS1 = TypeVar('SS1')  # suff stat prev
SS2 = TypeVar('SS2')  # suff stat len


class IntegerHiddenAssociationDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, state_prob_mat: Union[List[List[float]], np.ndarray],
                 cond_weights: Union[List[List[float]], np.ndarray],
                 alpha: float = 0.0,
                 prev_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 len_dist: Optional[SequenceEncodableProbabilityDistribution] = NullDistribution(),
                 name: Optional[str] = None,
                 keys: Tuple[Optional[str], Optional[str]] = (None, None),
                 use_numba: bool = False) -> None:
        """IntegerHiddenAssociationDistribution object for specifying integer Hidden association distribution.

        Args:
            state_prob_mat (Union[List[List[float]], np.ndarray]): States given previous words.
            cond_weights (Union[List[List[float]], np.ndarray]): Words in S2 given states.
            alpha (float): Probability of drawing from uniform vs transition density.
            prev_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for given P(S1).
                Should be compatible with Tuple[int, float].
            len_dist (Optional[SequenceEncodableProbabilityDistribution]): Distribution for length of observations.
                Should be compatible with type Tuple[int, int].
            name (Optional[str]): Set name for object.
            keys (Tuple[Optional[str], Optional[str]): Keys for the weights and states.
            use_numba (bool): If True, numba is used for encoding and estimation.

        Attributes:
            cond_weights (np.ndarray): States given words in S1.
            state_prob_mat (np.ndarray): Words in S2 given States.
            len_dist (SequenceEncodableProbabilityDistribution): Distribution for length of observations.
                Should be compatible with type Tuple[int, int].
            prev_dist (SequenceEncodableProbabilityDistribution): Distribution for given P(S1).
                Should be compatible with Tuple[int, float].
            has_prev_dist (bool): True is there is a non-null prev_dist specified.
            num_vals2 (int): Number of values in S2.
            num_vals1 (int): Number of values in S1.
            num_states (int): Number of hidden states.
            alpha (float): Probability of drawing from uniform vs transition density.
            name (Optional[str]): Set name for object.
            keys (Tuple[Optional[str], Optional[str]): Keys for the weights and states.
            init_prob_vec (np.ndarray): inital prob vector.
            use_numba (bool): If True, numba is used for encoding and estimation.

        """
        self.cond_weights = np.asarray(cond_weights, dtype=np.float64)
        self.state_prob_mat = np.asarray(state_prob_mat, dtype=np.float64)
        self.len_dist = len_dist if len_dist is not None else NullDistribution()
        self.prev_dist = prev_dist if prev_dist is not None else NullDistribution()
        self.has_prev_dist = not isinstance(self.prev_dist, NullDistribution)
        self.num_vals2 = self.state_prob_mat.shape[1]
        self.num_vals1 = self.cond_weights.shape[0]
        self.num_states = self.state_prob_mat.shape[0]
        self.alpha = alpha
        self.name = name
        self.keys = keys
        self.init_prob_vec = np.empty(0, dtype=np.float64)
        self.use_numba = use_numba

    def __str__(self) -> str:
        s1 = ','.join(
            ['[' + ','.join(map(str, self.state_prob_mat[i, :])) + ']' for i in range(len(self.state_prob_mat))])
        s2 = ','.join(['[' + ','.join(map(str, self.cond_weights[i, :])) + ']' for i in range(len(self.cond_weights))])
        s3 = str(self.alpha)
        s4 = repr(self.prev_dist) if self.prev_dist is None else str(self.prev_dist)
        s5 = str(self.len_dist)
        s6 = repr(self.name)
        s7 = repr(self.keys)

        return 'IntegerHiddenAssociationDistribution([%s], [%s], alpha=%s, prev_dist=%s, len_dist=%s, name=%s, ' \
               'keys=%s)' % (s1, s2, s3, s4, s5, s6, s7)

    def density(self, x: Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]) -> float:
        return exp(self.log_density(x))

    def log_density(self, x: Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]) -> float:

        nw = self.num_vals2
        a = self.alpha / nw
        b = 1 - self.alpha

        cx = np.asarray([u[1] for u in x[0]])
        vx = np.asarray([u[0] for u in x[0]])
        cy = np.asarray([u[1] for u in x[1]])
        vy = np.asarray([u[0] for u in x[1]])

        n1 = np.sum(cx)
        n2 = np.sum(cy)

        ll = self.cond_weights[vx, :].T * (cx / np.sum(cx))
        ll = np.dot(ll.T, self.state_prob_mat[:, vy]) * b + a
        log_sum_x = np.log(np.sum(ll, axis=0))
        rv = float(np.dot(log_sum_x, cy))
        # rv += np.dot(np.log(self.init_prob_vec[vx]), cx)

        rv += self.prev_dist.log_density(x[0])
        rv += self.len_dist.log_density(n2)

        return rv

    def seq_log_density(self, x: E) -> np.ndarray:

        nw = self.num_vals2
        a = self.alpha / nw
        b = 1 - self.alpha

        if x[1] is None:

            xx = x[0]
            rv = np.zeros(len(xx[0]), dtype=np.float64)

            for i, entry in enumerate(xx[0]):
                vx, cx, vy, cy = entry

                x_mat = self.cond_weights[vx, :].T * (cx / np.sum(cx))
                x_mat = np.dot(x_mat.T, self.state_prob_mat[:, vy]) * b + a
                rv[i] = np.dot(np.log(np.sum(x_mat, axis=0)), cy)
                # rv[i] += np.dot(np.log(self.init_prob_vec[vx]), cx)

            rv += self.prev_dist.seq_log_density(xx[1])
            rv += self.len_dist.seq_log_density(xx[2])

        else:
            (s0, s1, x0, x1, c0, c1, w0), xv, nn = x[1]

            rv = np.zeros(len(s0), dtype=np.float64)
            t0 = np.concatenate([[0], s0]).cumsum().astype(np.int32)
            t1 = np.concatenate([[0], s1]).cumsum().astype(np.int32)
            max_len = s0.max()
            numba_seq_log_density(self.num_states, max_len, t0, t1, x0, x1, c0, c1, w0, self.cond_weights,
                                  self.state_prob_mat, self.init_prob_vec, a, b, rv)

            rv += self.prev_dist.seq_log_density(xv)
            rv += self.len_dist.seq_log_density(nn)

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'IntegerHiddenAssociationSampler':
        if isinstance(self.prev_dist, NullDistribution):
            raise Exception('HiddenAssociationSampler requires attribute dist.prev_dist.')
        if isinstance(self.len_dist, NullDistribution):
            raise Exception('HiddenAssociationSampler requires attribute dist.size_dist.')
        return IntegerHiddenAssociationSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'IntegerHiddenAssociationEstimator':
        n_vals = (self.num_vals1, self.num_vals2)
        prev_est = self.prev_dist.estimator()
        len_est = self.len_dist.estimator()

        return IntegerHiddenAssociationEstimator(num_vals=n_vals, num_states=self.num_states, alpha=self.alpha,
                                                 prev_estimator=prev_est, len_estimator=len_est, name=self.name,
                                                 keys=self.keys, use_numba=self.use_numba)

    def dist_to_encoder(self) -> 'IntegerHiddenAssociationDataEncoder':
        prev_encoder = self.prev_dist.dist_to_encoder()
        len_encoder = self.len_dist.dist_to_encoder()
        return IntegerHiddenAssociationDataEncoder(prev_encoder, len_encoder, self.use_numba)

class IntegerHiddenAssociationSampler(DistributionSampler):

    def __init__(self, dist: IntegerHiddenAssociationDistribution, seed: Optional[int] = None) -> None:
        self.rng = np.random.RandomState(seed)
        self.dist = dist

        if isinstance(self.dist.prev_dist, NullDistribution):
            raise Exception('HiddenAssociationSampler requires attribute dist.prev_dist.')
        else:
            self.prev_sampler = self.dist.prev_dist.sampler(seed=self.rng.randint(0, maxrandint))

        if isinstance(self.dist.len_dist, NullDistribution):
            raise Exception('HiddenAssociationSampler requires attribute dist.size_dist.')
        else:
            self.size_sampler = self.dist.len_dist.sampler(seed=self.rng.randint(0, maxrandint))

    def sample_given(self, x: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        slen = self.size_sampler.sample()
        rng = np.random.RandomState(self.rng.randint(0, maxrandint))

        x0 = np.asarray([xx[0] for xx in x])
        x1 = np.asarray([xx[1] for xx in x], dtype=float)
        s1 = np.sum(x1)

        if s1 > 0:
            x1 /= s1
        else:
            return []

        v2 = []
        z1 = rng.choice(len(x0), p=x1, replace=True, size=slen)
        ns = self.dist.num_states
        nw = self.dist.num_vals2

        for zz1 in z1:

            if rng.rand() >= self.dist.alpha:
                u = rng.choice(ns, p=self.dist.cond_weights[x0[zz1], :])
                v2.append(rng.choice(nw, p=self.dist.state_prob_mat[u, :]))
            else:
                v2.append(rng.choice(nw))

        return list(count_by_value(v2).items())

    def sample(self, size: Optional[int] = None) -> Union[Sequence[List[Tuple[int, float]]], List[Tuple[int, float]]]:

        if size is None:
            x = self.prev_sampler.sample()
            return x, self.sample_given(x)
        else:
            return [self.sample() for i in range(size)]


class IntegerHiddenAssociationAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, num_vals1: int, num_vals2: int, num_states: int,
                 prev_acc: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 size_acc: Optional[SequenceEncodableStatisticAccumulator] = NullAccumulator(),
                 use_numba: bool = False,
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None)) -> None:
        self.init_count = np.zeros(num_vals1, dtype=np.float64)
        self.weight_count = np.zeros((num_vals1, num_states), dtype=np.float64)
        self.state_count = np.zeros((num_states, num_vals2), dtype=np.float64)
        self.size_accumulator = size_acc if size_acc is not None else NullAccumulator()
        self.prev_accumulator = prev_acc if prev_acc is not None else NullAccumulator()
        self.num_vals1 = num_vals1
        self.num_vals2 = num_vals2
        self.num_states = num_states
        self.use_numba = use_numba
        self.weight_key, self.state_key = keys if keys is not None else (None, None)

        self._init_rng = False
        self._rng_prev = None
        self._rng_size = None
        self._rng_weight = None
        self._rng_state = None

    def update(self, x: Tuple[List[Tuple[int, float]], List[Tuple[int, float]]], weight: float,
               estimate: IntegerHiddenAssociationDistribution) -> None:
        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)
        nx = np.sum(cx)

        x_mat = (estimate.cond_weights[vx, :].T * (cx / nx)).T
        y_mat = estimate.state_prob_mat[:, vy]
        z_mat = x_mat[:, :, None] * y_mat[None, :, :]

        # [old word] x [state] x [new word]

        ss = np.sum(np.sum(z_mat, axis=0, keepdims=True), axis=1, keepdims=True)
        z_mat /= ss

        self.weight_count[vx, :] += np.dot(z_mat, cy) * weight
        self.state_count[:, vy] += np.sum(z_mat, axis=0) * cy * weight
        self.init_count[vx] += cx

        self.prev_accumulator.update(x[0], weight, None if estimate is None else estimate.prev_dist)
        self.size_accumulator.update(cy.sum(), weight, None if estimate is None else estimate.len_dist)

    def _rng_initialize(self, rng: np.random.RandomState) -> None:
        if not self._init_rng:
            seeds = rng.randint(low=0, high=maxrandint, size=4)
            self._rng_state = np.random.RandomState(seed=seeds[0])
            self._rng_weight = np.random.RandomState(seed=seeds[1])
            self._rng_prev = np.random.RandomState(seed=seeds[2])
            self._rng_size = np.random.RandomState(seed=seeds[3])
            self._init_rng = True

    def initialize(self, x: Tuple[List[Tuple[int, float]], List[Tuple[int, float]]], weight: float,
                   rng: np.random.RandomState) -> None:

        if not self._init_rng:
            self._rng_initialize(rng)

        vx = np.asarray([u[0] for u in x[0]], dtype=int)
        cx = np.asarray([u[1] for u in x[0]], dtype=float)
        vy = np.asarray([u[0] for u in x[1]], dtype=int)
        cy = np.asarray([u[1] for u in x[1]], dtype=float)

        self.weight_count[vx, :] += self._rng_weight.dirichlet(np.ones(self.num_states), size=len(vx)) * weight
        self.state_count[:, vy] += self._rng_state.dirichlet(np.ones(self.num_states), size=len(vy)).T * cy * weight
        self.init_count[vx] += cx

        self.prev_accumulator.initialize(x[0], weight, self._rng_prev)
        self.size_accumulator.initialize(cy.sum(), weight, self._rng_size)

    def seq_initialize(self, x: E, weights: np.ndarray, rng: np.random.RandomState) -> None:
        if not self._init_rng:
            self._rng_initialize(rng)

        if x[1] is None:
            xx = x[0]

            for i, (entry, weight) in enumerate(zip(xx[0], weights)):
                vx, cx, vy, cy = entry

                self.weight_count[vx, :] += self._rng_weight.dirichlet(np.ones(self.num_states), size=len(vx)) * weight
                self.state_count[:, vy] += self._rng_state.dirichlet(np.ones(self.num_states), size=len(vy)).T * cy * \
                                           weight
                self.init_count[vx] += cx

            self.prev_accumulator.seq_initialize(xx[1], weights, self._rng_prev)
            self.size_accumulator.seq_initialize(xx[2], weights, self._rng_size)

        else:
            (s0, s1, x0, x1, c0, c1, w0), xv, nn = x[1]
            weights_0 = []
            weights_1 = []

            for i in range(len(s0)):
                weights_0.extend([weights[i]]*s0[i]*self.num_states)
                weights_1.extend([weights[i]]*s1[i])

            weights_0 = np.asarray(weights_0)
            weights_1 = np.asarray(weights_1)
            ww0 = self._rng_weight.dirichlet(np.ones(self.num_states), size=len(x0)).flatten() * weights_0
            ww0 = np.reshape(ww0, (len(x0), self.num_states))

            self.weight_count += vec_bincount1(x=x0, w=ww0, out=np.zeros_like(self.weight_count, dtype=np.float64))

            ww1 = self._rng_state.dirichlet(np.ones(self.num_states), size=len(x1)).T
            ww1 *= np.reshape(c1*weights_1, (-1, len(x1)))

            self.state_count += vec_bincount2(x=x1, w=ww1, out=np.zeros_like(self.state_count, dtype=np.float64))

            self.init_count += np.bincount(x0, weights=c0*weights_0[np.arange(0, len(weights_0), self.num_states)],
                                           minlength=len(self.init_count))

            self.prev_accumulator.seq_initialize(xv, weights, self._rng_prev)
            self.size_accumulator.seq_initialize(nn, weights, self._rng_size)

    def seq_update(self, x: E, weights: np.ndarray, estimate: IntegerHiddenAssociationDistribution) -> None:

        if x[1] is None:
            xx = x[0]

            for i, (entry, weight) in enumerate(zip(xx[0], weights)):
                vx, cx, vy, cy = entry
                nx = np.sum(cx)
                x_mat = (estimate.cond_weights[vx, :].T * (cx / nx)).T
                y_mat = estimate.state_prob_mat[:, vy]
                z_mat = x_mat[:, :, None] * y_mat[None, :, :]

                # [old word] x [state] x [new word]

                ss = np.sum(np.sum(z_mat, axis=0, keepdims=True), axis=1, keepdims=True)
                z_mat /= ss

                self.weight_count[vx, :] += np.dot(z_mat, cy) * weight
                self.state_count[:, vy] += np.sum(z_mat, axis=0) * cy * weight
                self.init_count[vx] += cx

            self.prev_accumulator.seq_update(xx[1], weights, None if estimate is None else estimate.prev_dist)
            self.size_accumulator.seq_update(xx[2], weights, None if estimate is None else estimate.len_dist)
        else:

            (s0, s1, x0, x1, c0, c1, w0), xv, nn = x[1]

            t0 = np.concatenate([[0], s0]).cumsum().astype(np.int32)
            t1 = np.concatenate([[0], s1]).cumsum().astype(np.int32)
            max_len = s0.max()

            numba_seq_update(self.num_states, max_len, t0, t1, x0, x1, c0, c1, w0, estimate.cond_weights,
                             estimate.state_prob_mat, self.weight_count, self.state_count, self.init_count, weights)

            self.prev_accumulator.seq_update(xv, weights, None if estimate is None else estimate.prev_dist)
            self.size_accumulator.seq_update(nn, weights, None if estimate is None else estimate.len_dist)

    def combine(self, suff_stat: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1], Optional[SS2]]) \
            -> 'IntegerHiddenAssociationAccumulator':

        init_count, weight_count, state_count, prev_acc, size_acc = suff_stat

        self.prev_accumulator.combine(prev_acc)
        self.size_accumulator.combine(size_acc)

        self.init_count += init_count
        self.weight_count += weight_count
        self.state_count += state_count

        return self

    def value(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Any], Optional[Any]]:
        pval = self.prev_accumulator.value()
        sval = self.size_accumulator.value()

        return self.init_count, self.weight_count, self.state_count, pval, sval

    def from_value(self, x: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[SS1], Optional[SS2]]) \
            -> 'IntegerHiddenAssociationAccumulator':

        init_count, weight_count, state_count, prev_acc, size_acc = x

        self.init_count = init_count
        self.weight_count = weight_count
        self.state_count = state_count

        self.prev_accumulator.from_value(prev_acc)
        self.size_accumulator.from_value(size_acc)

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:

        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                stats_dict[self.weight_key] += self.weight_count
            else:
                stats_dict[self.weight_key] = self.weight_count.copy()

        if self.state_key is not None:
            if self.state_key in stats_dict:
                stats_dict[self.state_key] += self.state_count
            else:
                stats_dict[self.state_key] = self.state_count.copy()

        self.prev_accumulator.key_merge(stats_dict)
        self.size_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:

        if self.weight_key is not None:
            if self.weight_key in stats_dict:
                self.weight_count = stats_dict[self.weight_key].copy()

        if self.state_key is not None:
            if self.state_key in stats_dict:
                self.state_count = stats_dict[self.state_key].copy()

        self.prev_accumulator.key_replace(stats_dict)
        self.size_accumulator.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'DataSequenceEncoder':
        prev_encoder = self.prev_accumulator.acc_to_encoder()
        len_encoder = self.size_accumulator.acc_to_encoder()
        return IntegerHiddenAssociationDataEncoder(prev_encoder, len_encoder, self.use_numba)


class IntegerHiddenAssociationAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, num_vals1: int, num_vals2: int, num_states: int,
                 prev_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
                 len_factory: Optional[StatisticAccumulatorFactory] = NullAccumulatorFactory(),
                 use_numba: bool = False,
                 keys: Tuple[Optional[str], Optional[str]] = (None, None)) -> None:
        self.len_factory = len_factory if len_factory is not None else NullAccumulatorFactory()
        self.prev_factory = prev_factory if prev_factory is not None else NullAccumulatorFactory()
        self.keys = keys
        self.use_numba = use_numba
        self.num_vals1 = num_vals1
        self.num_vals2 = num_vals2
        self.num_states = num_states

    def make(self) -> 'IntegerHiddenAssociationAccumulator':
        len_acc = self.len_factory.make()
        prev_acc = self.prev_factory.make()
        return IntegerHiddenAssociationAccumulator(num_vals1=self.num_vals1, num_vals2=self.num_vals2,
                                                   num_states=self.num_states, prev_acc=prev_acc, size_acc=len_acc,
                                                   use_numba=self.use_numba, keys=self.keys)


class IntegerHiddenAssociationEstimator(ParameterEstimator):

    def __init__(self, num_vals: Union[List[int], Tuple[int, int], int], num_states: int, alpha: float = 0.0,
                 prev_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 len_estimator: Optional[ParameterEstimator] = NullEstimator(),
                 suff_stat: Optional[Any] = None,
                 pseudo_count: Optional[float] = None,
                 use_numba: bool = False,
                 name: Optional[str] = None,
                 keys: Optional[Tuple[Optional[str], Optional[str]]] = (None, None)) -> None:
        """IntegerHiddenAssociationEstimator object for estimating IntegerHiddenAssociationDistribution from aggregated
            sufficient statistics.

        Args:
            num_vals (Union[List[int], Tuple[int, int], int]): Number of values in S1 and S2. Either length 2, if int
                value is set to num_vals1 and num_vals2.
            num_states (int): Number of hidden states.
            alpha (float): Prob of drawing from uniform, (1-alpha) draw from transition density.
            prev_estimator (Optional[ParameterEstimator]): Estimator for the previous word set. Must be compatible with
                Tuple[int, float].
            len_estimator (Optional[ParameterEstimator]): Estimator for the length of observations. Must be compatible
                with Tuple[int, int].
            suff_stat (Optional[Any]): Kept for consistency.
            pseudo_count (Optional[float]): Kept for consistency.
            use_numba (bool): If true Numba is used for encoding and vectorized function calls.
            name (Optional[str]): Set a name to the object instance.
            keys (Optional[Tuple[Optional[str], Optional[str]]]): Set the keys for weights and transitions.

        Attributes:
            num_vals (Union[List[int], Tuple[int, int], int]): Number of values in S1 and S2. Either length 2, if int
                value is set to num_vals1 and num_vals2.
            num_states (int): Number of hidden states.
            alpha (float): Prob of drawing from uniform, (1-alpha) draw from transition density.
            prev_estimator (ParameterEstimator): Estimator for the previous word set. Must be compatible with
                Tuple[int, float]. Defaults to NullEstimator().
            len_estimator (ParameterEstimator): Estimator for the length of observations. Must be compatible
                with Tuple[int, int]. Defaults to NullEstimator().
            suff_stat (Optional[Any]): Kept for consistency.
            pseudo_count (Optional[float]): Kept for consistency.
            use_numba (bool): If true Numba is used for encoding and vectorized function calls.
            name (Optional[str]): Set a name to the object instance.
            keys (Tuple[Optional[str], Optional[str]]): Set the keys for weights and transitions.
            num_vals1 (int): Number of values in set 1.
            num_vals2 (int): Number of values in set 2.

        """
        self.prev_estimator = prev_estimator if prev_estimator is not None else NullEstimator()
        self.len_estimator = len_estimator if len_estimator is not None else NullEstimator()
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.num_vals = num_vals
        self.num_states = num_states
        self.alpha = alpha
        self.use_numba = use_numba
        self.name = name
        self.keys = keys if keys is not None else (None, None)

        if isinstance(num_vals, (tuple, list)):
            if len(num_vals) >= 2:
                self.num_vals1 = num_vals[0]
                self.num_vals2 = num_vals[1]
            elif len(num_vals) == 1:
                self.num_vals1 = num_vals[0]
                self.num_vals2 = num_vals[0]
        else:
            self.num_vals1 = num_vals
            self.num_vals2 = num_vals

    def accumulator_factory(self) -> 'IntegerHiddenAssociationAccumulatorFactory':
        len_factory = self.len_estimator.accumulator_factory()
        prev_factory = self.prev_estimator.accumulator_factory()

        return IntegerHiddenAssociationAccumulatorFactory(self.num_vals1, self.num_vals2, self.num_states, prev_factory,
                                                          len_factory, self.use_numba, self.keys)

    def estimate(self, nobs: Optional[float], suff_stat: Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                               Optional[SS1], Optional[SS2]]) \
            -> 'IntegerHiddenAssociationDistribution':

        init_count, weight_count, state_count, prev_stats, size_stats = suff_stat

        len_dist = self.len_estimator.estimate(nobs, size_stats)
        prev_dist = self.prev_estimator.estimate(nobs, prev_stats)

        if self.pseudo_count is not None:
            init_count += self.pseudo_count / len(init_count)
            state_count += self.pseudo_count / (self.num_states * self.num_vals2)
            weight_count += self.pseudo_count / (self.num_states * self.num_vals1)

        # init_prob = init_count / np.sum(init_count)

        wsum = np.sum(weight_count, axis=1, keepdims=True)
        ssum = np.sum(state_count, axis=1, keepdims=True)
        ssum[ssum == 0] = 1.0
        wsum[wsum == 0] = 1.0

        weight_prob = weight_count / wsum
        state_prob = state_count / ssum

        # return IntegerHiddenAssociationDistribution(init_prob, state_prob, weight_prob, self.alpha, len_dist)
        return IntegerHiddenAssociationDistribution(state_prob_mat=state_prob, cond_weights=weight_prob,
                                                    alpha=self.alpha, prev_dist=prev_dist, use_numba=self.use_numba,
                                                    len_dist=len_dist, name=self.name, keys=self.keys)


class IntegerHiddenAssociationDataEncoder(DataSequenceEncoder):

    def __init__(self, prev_encoder: DataSequenceEncoder, len_encoder: DataSequenceEncoder, use_numba: bool) -> None:
        self.prev_encoder = prev_encoder
        self.len_encoder = len_encoder
        self.use_numba = use_numba

    def __str__(self) -> str:
        s = 'IntegerHiddenAssociationDataEncoder(prev_encoder=' + str(self.prev_encoder) + ',len_encoder='
        s += str(self.len_encoder) + ',use_numba=' + str(self.use_numba) + ')'
        return s

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IntegerHiddenAssociationDataEncoder):
            cond0 = self.prev_encoder == other.prev_encoder
            cond1 = self.len_encoder == other.len_encoder
            cond2 = self.use_numba == other.use_numba
            return cond0 and cond1 and cond2
        else:
            return False

    def _seq_encode(self, x: Sequence[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]]) \
            -> Tuple[Tuple[List[Tuple[np.ndarray, ...]], Optional[Any], Optional[Any]], None]:
        """Sequence encoding for use with without numba.
        
        Returns 'rv' Tuple of 
            rv[0] (List[Tuple[ndarray[int], ndarray[float], ndarray[int], ndarray[float]]]): List of Tuples containing 
                Flattened numpy arrays of x0 values, x0 counts, x1 values, x1 counts. 
            rv[1] (E1): Sequence encoded output from list of Tuples containing sum of counts for 
                x0 and x1.
            rv[2] (E2): Sequence encoding of x0 from prev_encoder. 
        
        Args:
            x: Sequence of iid integer hidden association observations. 

        Returns:
            See rv above. 

        """
        rv = []
        nn = []

        for xx in x:
            rv0 = []
            for c_vec in xx:
                rv0.append(np.asarray([v for v, c in c_vec], dtype=int))
                rv0.append(np.asarray([c for v, c in c_vec], dtype=float))
            nn0 = np.sum(rv0[-1])

            rv.append(tuple(rv0))
            nn.append(nn0)

        nn = self.len_encoder.seq_encode(nn)
        xv = self.prev_encoder.seq_encode([x[0] for x in x])

        return (rv, xv, nn), None

    def seq_encode(self, x: Sequence[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]]) \
            -> Union[Tuple[Tuple[List[Tuple[np.ndarray, ...]], Optional[Any], Optional[Any]], None],
                     Tuple[None, Tuple[np.ndarray, ...], Optional[Any], Optional[Any]]]:
        """Sequence encoding for integer hidden association observations.

        If numba is not used see _seq_encode(). Else the following is returned a Tuple of the following form is returned
        None, ((s0, s1, x0, x1, c0, c1, w0), xv, nn) with,

            s0 (np.ndarray): Numpy array of lengths for length of x[i][0]
            s1 (np.ndarray): Numpy array of lengths for length of x[i][1].
            x0 (np.ndarray): Flattened numpy array of values from x[i][0].
            x1 (np.ndarray): Flattened numpy array of values from x[i][1].
            c0 (np.ndarray): Flattened numpy array of counts from x[i][0].
            c1 (np.ndarray): Flattened numpy array of counts from x[i][1].
            w0 (np.ndarray): Numpy array of sum of counts for each x[i][0].
            xv (E1): Sequence encoded flattened values of x[i][0].
            nn (E2): Sequence encoded values of lengths (counts).

        Args:
            x: Sequence of iid integer hidden association observations.

        Returns:
            See above.

        """

        if not self.use_numba:
            enc_rv = self._seq_encode(x)
        else:
            x1 = []
            x0 = []
            s1 = []
            s0 = []
            c0 = []
            c1 = []
            w0 = []
            nn = []

            for i, xx in enumerate(x):
                xx0 = [v for v, c in xx[0]]
                cc0 = [c for v, c in xx[0]]
                xx1 = [v for v, c in xx[1]]
                cc1 = [c for v, c in xx[1]]

                x0.extend(xx0)
                x1.extend(xx1)
                c0.extend(cc0)
                c1.extend(cc1)
                w0.append(sum(cc0))
                s1.append(len(xx1))
                s0.append(len(xx0))
                nn.append(sum(cc1))

            nn = self.len_encoder.seq_encode(nn)
            xv = self.prev_encoder.seq_encode([x[0] for x in x])

            x0 = np.asarray(x0, dtype=np.int32)
            x1 = np.asarray(x1, dtype=np.int32)
            c0 = np.asarray(c0, dtype=np.float64)
            c1 = np.asarray(c1, dtype=np.float64)
            s0 = np.asarray(s0, dtype=np.int32)
            s1 = np.asarray(s1, dtype=np.int32)
            w0 = np.asarray(w0, dtype=np.float64)

            enc_rv = tuple([None, ((s0, s1, x0, x1, c0, c1, w0), xv, nn)])

        return enc_rv

@numba.njit(
    'void(int64, int64, int32[:], int32[:], int32[:], int32[:], float64[:], float64[:], float64[:], float64[:,:], '
    'float64[:,:], float64[:], float64, float64, float64[:])')
def numba_seq_log_density(num_states, max_len1, t0, t1, x0, x1, c0, c1, w0, cond_weights, state_prob_mat, init_prob_vec,
                          a, b, out):
    x_mat = np.zeros((max_len1, num_states), dtype=np.float64)

    for i in range(len(t0) - 1):

        vx = x0[t0[i]:t0[i + 1]]
        cx = c0[t0[i]:t0[i + 1]]
        vy = x1[t1[i]:t1[i + 1]]
        cy = c1[t1[i]:t1[i + 1]]
        sx = w0[i]

        l1 = t0[i + 1] - t0[i]
        l2 = t1[i + 1] - t1[i]

        for j in range(l1):
            temp = cx[j] / sx
            # out[i] += math.log(init_prob_vec[vx[j]])*cx[j]
            for k in range(num_states):
                x_mat[j, k] = cond_weights[vx[j], k] * temp

        for w in range(l2):
            wid = vy[w]
            temp_sum = 0
            for j in range(l1):
                for k in range(num_states):
                    temp_sum += x_mat[j, k] * state_prob_mat[k, wid]
            out[i] += math.log(temp_sum) * cy[w]


@numba.njit(
    'void(int64, int64, int32[:], int32[:], int32[:], int32[:], float64[:], float64[:], float64[:], float64[:,:], '
    'float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:])')
def numba_seq_update(num_states, max_len1, t0, t1, x0, x1, c0, c1, w0, cond_weights, state_prob_mat, weight_count,
                     state_count, init_count, weights):
    x_mat = np.zeros((max_len1, num_states), dtype=np.float64)
    z_mat = np.zeros((max_len1, num_states), dtype=np.float64)

    for i in range(len(t0) - 1):

        weight = weights[i]
        vx = x0[t0[i]:t0[i + 1]]
        cx = c0[t0[i]:t0[i + 1]]
        vy = x1[t1[i]:t1[i + 1]]
        cy = c1[t1[i]:t1[i + 1]]

        l1 = t0[i + 1] - t0[i]
        l2 = t1[i + 1] - t1[i]

        nx = w0[i]

        for j in range(l1):
            temp = cx[j] / nx
            # init_count[vx[j]] += cx[j] * weight
            for k in range(num_states):
                x_mat[j, k] = cond_weights[vx[j], k] * temp

        for w in range(l2):
            wid = vy[w]
            temp_sum = 0
            for j in range(l1):
                for k in range(num_states):
                    temp = x_mat[j, k] * state_prob_mat[k, wid]
                    z_mat[j, k] = temp
                    temp_sum += temp

            temp_weight = cy[w] * weight / temp_sum
            for j in range(l1):
                for k in range(num_states):
                    temp = temp_weight * z_mat[j, k]
                    weight_count[vx[j], k] += temp
                    state_count[k, wid] += temp


@numba.njit('float64[:,:](int32[:], float64[:,:], float64[:,:])')
def vec_bincount1(x, w, out):
    """Numba bincount on the rows of matrix w for groups x.

    Args:
        x (np.ndarray[np.float64]): Group ids of rows
        w (np.ndarray[np.float64]): N by S numpy array with rows corresponding to x
        out (np.ndarray[np.float64]): Unique values in support of x by S.

    Returns:
        Numpy 2-d array.

    """
    for i in range(len(x)):
        out[x[i], :] += w[i, :]
    return out


@numba.njit('float64[:,:](int32[:], float64[:,:], float64[:,:])')
def vec_bincount2(x, w, out):
    """Numba bincount on the rows of matrix w for groups x.

    N = len(x)
    S = number of states.
    U = unique values in x can take on.

    Args:
        x (np.ndarray[np.float64]): Group ids of columns of w.
        w (np.ndarray[np.float64]): S by N numpy array with cols corresponding to x
        out (np.ndarray[np.float64]): S by U matrix.

    Returns:
        Numpy 2-d array.

    """
    for j in range(len(x)):
        out[:, x[j]] += w[:, j]
    return out

