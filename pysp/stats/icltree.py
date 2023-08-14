"""Create, estimate, and sample from an integer Chow Liu Tree distribution.

Defines the ICLTreeDistribution, ICLTreeSampler, ICLTreeAccumulatorFactory, ICLTreeAccumulator, ICLTreeEstimator, and
the ICLTreeDataEncoder classes for use with pysparkplug.

PySparkPlug supports Chow & Liu trees [1] through the ICLTree (Integer Chow Liu Tree) class of objects. ICLTrees model
non-Markov conditional dependence for fixed-length sequences of integers with the likelihood functions of the form

    P(x_1, x_2,..,x_n) = P(x_i1) P(x_{i_2}|x_{j_2})*...*P(x_{i_n}|x_{j_n}),

where j_k < i_k for all k = 1,2,3,..N.

Data type: Union[Sequence[int], np.ndarray] .

"""
import numpy as np
from numpy.random import RandomState
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order

from pysp.stats.pdist import SequenceEncodableProbabilityDistribution, SequenceEncodableStatisticAccumulator, \
    ParameterEstimator, DataSequenceEncoder, DistributionSampler, StatisticAccumulatorFactory

from typing import Sequence, Tuple, Any, List, Dict, Union, Optional


class ICLTreeDistribution(SequenceEncodableProbabilityDistribution):

    def __init__(self, dependency_list: List[Tuple[int, Optional[int]]],
                 conditional_log_densities: Union[Sequence[float], np.ndarray],
                 feature_order: Optional[Sequence[int]] = None, name: Optional[str] = None) -> None:
        """ICLTreeDistribution object for integer Chow Liu tree distribution.

        Args:
            dependency_list (List[Tuple[int, Optional[int]]]): List of Tuples containing node id and parent dependence
                if any dependence is present.
            conditional_log_densities (Union[Sequence[float], np.ndarray]): Conditional log densities for each features
                dependency split.
            feature_order (Optional[Sequence[int]]): Ordering of features. If None, ordering is assumed as entered.
            name (Optional[str]): Set name to object.

        Attributes:
            feature_order (Sequence[int]): Ordering of features. If None, ordering is assumed as entered.
            dependency_list (List[ Tuple[int, Tuple[int, Optional[int]]]]): List of Tuples containing features
                order id and Tuple of feature and feature dep.
            conditional_log_densities (Union[Sequence[float], np.ndarray]): Conditional log densities for each features
                dependency split.
            conditional_densities (np.ndarray): Conditional densities as numpy array.
            num_features (int): Total number of features.
            name (Optional[str]): Name for object isntance.

        """
        self.feature_order = range(len(dependency_list)) if feature_order is None else feature_order
        self.dependency_list = list(zip(self.feature_order, dependency_list))
        self.conditional_log_densities = conditional_log_densities
        self.conditional_densities = [np.exp(u) for u in conditional_log_densities]
        self.num_features = len(dependency_list)
        self.name = name

    def __str__(self) -> str:
        f1 = ','.join([str(u[1]) for u in self.dependency_list])
        f3 = ','.join([str(u[0]) for u in self.dependency_list])
        f2 = ['[' + ','.join(map(str, u.flatten())) + ']' for u in self.conditional_log_densities]
        f4 = ','.join(repr(self.name))
        return 'ICLTreeDistribution([%s], [%s], feature_order=[%s], name=%s)' % (f1, f2, f3, f4)

    def density(self, x: Union[Sequence[int], np.ndarray]) -> float:
        return np.exp(self.log_density(x))

    def log_density(self, x: Union[Sequence[int], np.ndarray]) -> float:
        rv = 0
        for i, (j, k) in enumerate(self.dependency_list):
            if k is None:
                rv += self.conditional_log_densities[i][x[j]]
            else:
                rv += self.conditional_log_densities[i][x[k], x[j]]

        return rv

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        rv = np.zeros(x.shape[0])
        for i, (j, k) in enumerate(self.dependency_list):
            if k is None:
                rv += self.conditional_log_densities[i][x[:, j]]
            else:
                rv += self.conditional_log_densities[i][x[:, k], x[:, j]]

        return rv

    def sampler(self, seed: Optional[int] = None) -> 'ICLTreeSampler':
        return ICLTreeSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'ICLTreeEstimator':
        return ICLTreeEstimator(name=self.name)

    def dist_to_encoder(self) -> 'ICLTreeDataEncoder':
        return ICLTreeDataEncoder()


class ICLTreeSampler(DistributionSampler):

    def __init__(self, dist: ICLTreeDistribution, seed: Optional[int] = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: Optional[int] = None) -> Union[List[Optional[int]], Sequence[List[Optional[int]]]]:

        if size is None:
            rv = [None] * self.dist.num_features

            for i, (j, k) in enumerate(self.dist.dependency_list):

                if k is None:
                    pmat = self.dist.conditional_densities[i]
                else:
                    pmat = self.dist.conditional_densities[i][rv[k], :]

                rv[j] = self.rng.choice(len(pmat), p=pmat)

            return rv
        else:
            return [self.sample() for i in range(size)]


class ICLTreeAccumulator(SequenceEncodableStatisticAccumulator):

    def __init__(self, num_features: int, num_states: int, keys: Optional[str] = None, name: Optional[str] = None):
        self.num_states = num_states
        self.num_features = num_features

        if num_states is not None and num_features is not None:
            self.counts = np.zeros((num_features, num_features, num_states, num_states))
            self.marginal_counts = np.zeros((num_features, num_states))
        else:
            self.counts = None
            self.marginal_counts = None

        self.key = keys
        self.name = name

    def _expand_states(self, num_states: int, num_features: int):

        if (self.counts is None) and (num_states is not None) and (num_features is not None):
            self.num_features = num_features
            self.num_states = num_states
            self.counts = np.zeros((num_features, num_features, num_states, num_states))
            self.marginal_counts = np.zeros((num_features, num_states))

        elif (self.counts is not None) and (num_states is not None) and (num_features is not None):
            old_num_states = self.num_states
            new_counts = np.zeros((num_features, num_features, num_states, num_states))
            new_marginal = np.zeros((num_features, num_states))
            new_counts[:, :, :old_num_states, :old_num_states] = self.counts
            new_marginal[:, :old_num_states] = self.marginal_counts
            self.num_features = num_features
            self.num_states = num_states
            self.counts = new_counts
            self.marginal_counts = new_marginal

    def update(self, x: Union[Sequence[int], np.ndarray], weight: float,
               estimate: Optional[ICLTreeDistribution]) -> None:

        if (self.counts is None) or (self.num_states <= np.max(x)):
            self._expand_states(max(x) + 1, len(x))

        xx = np.asarray(x)
        ff = np.arange(self.num_features)

        self.marginal_counts[ff, xx] += weight
        for i in range(self.num_features):
            self.counts[i, ff, xx[i], xx] += weight

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: Optional[ICLTreeDistribution]) -> None:

        max_x = np.max(x)

        if (self.counts is None) or (self.num_states <= max_x):
            self._expand_states(max_x + 1, x.shape[1])

        num_states = self.num_states

        for i in range(self.num_features):
            self.marginal_counts[i, :] += np.bincount(x[:, i], weights=weights, minlength=num_states)

            for j in range(i + 1, self.num_features):
                joint_idx = x[:, i] * num_states + x[:, j]
                joint_cnt = np.bincount(joint_idx, weights=weights, minlength=(num_states * num_states))
                joint_cnt = np.reshape(joint_cnt, (num_states, num_states))

                self.counts[i, j, :, :] += joint_cnt

    def initialize(self, x: Union[Sequence[int], np.ndarray], weight: float, rng: Optional[RandomState]) -> None:
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: Optional[RandomState]) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: Tuple[int, int, np.ndarray, np.ndarray]) -> 'ICLTreeAccumulator':

        num_features, num_states, counts, marginal_counts = suff_stat

        if self.counts is None and counts is None:
            return self

        elif (self.counts is None) and (counts is not None):
            self.counts = counts
            self.marginal_counts = marginal_counts
            self.num_states = suff_stat.shape[-1]
            self.num_features = suff_stat.shape[0]

        elif self.counts is not None and counts is None:
            pass

        else:
            if self.num_states < num_states:
                self._expand_states(num_states, num_features)
                self.counts += counts
                self.marginal_counts += marginal_counts

            elif self.num_states > num_states:
                self.counts[:, :, :num_states, :num_states] += counts
                self.marginal_counts[:, :num_states] += marginal_counts

            else:
                self.counts += counts
                self.marginal_counts += marginal_counts

        return self

    def value(self) -> Tuple[int, int, np.ndarray, np.ndarray]:
        return self.num_features, self.num_states, self.counts, self.marginal_counts

    def from_value(self, x: Tuple[int, int, np.ndarray, np.ndarray]) -> 'ICLTreeAccumulator':
        self.num_features = x[0]
        self.num_states = x[1]
        self.counts = x[2]
        self.marginal_counts = x[3]

        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        pass

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        pass

    def acc_to_encoder(self) -> 'ICLTreeDataEncoder':
        return ICLTreeDataEncoder()

class ICLTreeAccumulatorFactory(StatisticAccumulatorFactory):

    def __init__(self, num_features: Optional[int] = None, num_states: Optional[int] = None,
                 keys: Optional[str] = None,
                 name: Optional[str] = None) -> None:
        self.num_features = num_features
        self.num_states = num_states
        self.keys = keys
        self.name = name

    def make(self) -> 'ICLTreeAccumulator':
        return ICLTreeAccumulator(self.num_features, self.num_states, self.keys)

class ICLTreeEstimator(ParameterEstimator):

    def __init__(self, num_features: Optional[int] = None, num_states: Optional[int] = None,
                 pseudo_count: Optional[float] = None,
                 suff_stat: Optional[Any] = None,
                 keys: Optional[str] = None, name: Optional[str] = None):
        self.num_features = num_features
        self.num_states = num_states
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self):
        return ICLTreeAccumulatorFactory(self.num_features, self.num_states, self.keys)

    def estimate(self, nobs, suff_stat):

        num_features, num_states, counts, marginal_counts = suff_stat

        mi_mat = np.zeros((num_features, num_features))

        pseudo_count = self.pseudo_count if self.pseudo_count is not None else 0.0
        pseudo_count_adj0 = pseudo_count / num_states
        pseudo_count_adj1 = pseudo_count / (num_states * num_states)

        for i in range(num_features - 1):
            for j in range(i + 1, num_features):
                joint_ij = counts[i, j, :, :] + pseudo_count_adj1
                indep_ij = np.outer(marginal_counts[i, :], marginal_counts[j, :]) + pseudo_count_adj1

                joint_ij_sum = joint_ij.sum()
                indep_ij_sum = indep_ij.sum()

                if joint_ij_sum > 0:
                    joint_ij /= joint_ij_sum
                if indep_ij_sum > 0:
                    indep_ij /= indep_ij_sum

                good = np.bitwise_and(joint_ij > 0, indep_ij > 0)

                if good.sum() > 0:
                    mi_val = (joint_ij[good] * (np.log(joint_ij[good]) - np.log(indep_ij[good]))).sum()
                    mi_mat[i, j] = 1.0 + mi_val

                else:
                    mi_mat[i, j] = 1.0

        cost_mat = np.abs((mi_mat.max() - mi_mat))
        cost_mat[mi_mat > 0] += 1.0
        cost_mat[mi_mat == 0] = 0

        span_tree = minimum_spanning_tree(cost_mat)

        root_node = 0
        feature_order, deps = breadth_first_order(span_tree, root_node, directed=False, return_predecessors=True)

        deps = [deps[i] for i in feature_order]
        tmats = [None] * num_features

        with np.errstate(divide='ignore'):

            root_marginal = marginal_counts[root_node, :] + pseudo_count_adj0
            tmats[0] = np.log(root_marginal / (root_marginal.sum()))
            deps[0] = None

            for i in range(1, num_features):
                n = feature_order[i]
                p = deps[i]

                if p < n:
                    tmat = counts[p, n, :, :]
                else:
                    tmat = counts[n, p, :, :].T

                tmat = tmat + pseudo_count_adj1
                tmat_sum = np.sum(tmat, axis=1, keepdims=True)
                tmat_sum[tmat_sum == 0] = 1.0
                tmat /= tmat_sum

                tmats[i] = np.log(tmat)

        return ICLTreeDistribution(deps, tmats, feature_order=feature_order)


class ICLTreeDataEncoder(DataSequenceEncoder):

    def __str__(self) -> str:
        return 'ICLTreeDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ICLTreeDataEncoder)

    def seq_encode(self, x: Union[List[int], np.ndarray]) -> np.ndarray:
        return np.asarray(x, dtype=int)




