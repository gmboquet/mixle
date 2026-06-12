"""Generic Chow-Liu tree distribution for fixed-length tuple observations.

The integer-only :mod:`pysp.stats.icltree` implementation uses dense integer
count tables.  This module keeps the Chow-Liu structure learning step generic:
parent variables are represented by their observed/enumerable values, while
each child conditional distribution is estimated with the user-supplied
estimator for that coordinate.

Data type: ``Sequence[Any]`` with fixed length.
"""
import itertools
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.random import RandomState
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

from pysp.arithmetic import maxrandint
from pysp.stats.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)
from pysp.utils.enumeration import freeze


SS = Tuple[
    float,
    int,
    List[Dict[Hashable, float]],
    List[Dict[Hashable, Any]],
    Dict[Tuple[int, int], Dict[Tuple[Hashable, Hashable], float]],
    Tuple[Any, ...],
    Dict[Tuple[int, int], Dict[Hashable, Any]],
]


def _as_estimator(obj: Any, pseudo_count: Optional[float]) -> ParameterEstimator:
    if isinstance(obj, ParameterEstimator) or (
            hasattr(obj, 'accumulator_factory') and hasattr(obj, 'estimate')):
        return obj
    if hasattr(obj, 'estimator'):
        return obj.estimator(pseudo_count=pseudo_count)
    raise TypeError('Expected a ParameterEstimator or distribution with estimator().')


def _pseudo_for_index(pseudo_count: Any, idx: int) -> Optional[float]:
    if isinstance(pseudo_count, (list, tuple)):
        return pseudo_count[idx]
    return pseudo_count


class ChowLiuTreeDistribution(SequenceEncodableProbabilityDistribution):
    """Chow-Liu tree over fixed-position fields with generic conditional models.

    ``parents[i]`` gives the parent feature of feature ``i``; exactly one entry
    must be ``None`` and is treated as the root.  The root uses its marginal
    distribution.  Non-root features use ``conditional_dists[i][freeze(x[parent])]``
    when present, otherwise ``default_dists[i]`` if one was supplied.
    """

    def __init__(self,
                 parents: Sequence[Optional[int]],
                 marginal_dists: Sequence[SequenceEncodableProbabilityDistribution],
                 conditional_dists: Sequence[Optional[Dict[Hashable, SequenceEncodableProbabilityDistribution]]],
                 default_dists: Optional[Sequence[Optional[SequenceEncodableProbabilityDistribution]]] = None,
                 feature_order: Optional[Sequence[int]] = None,
                 parent_values: Optional[Sequence[Dict[Hashable, Any]]] = None,
                 name: Optional[str] = None) -> None:
        self.parents = [None if u is None else int(u) for u in parents]
        self.marginal_dists = list(marginal_dists)
        self.conditional_dists = [
            {} if d is None else {freeze(k): v for k, v in d.items()}
            for d in conditional_dists
        ]
        if default_dists is None:
            self.default_dists = [None] * len(self.parents)
        else:
            self.default_dists = list(default_dists)
        self.feature_order = list(range(len(self.parents))) if feature_order is None else [int(u) for u in feature_order]
        self.parent_values = [{} for _ in self.parents] if parent_values is None else list(parent_values)
        self.num_features = len(self.parents)
        self.name = name

        if len(self.marginal_dists) != self.num_features:
            raise ValueError('marginal_dists length must match parents length.')
        if len(self.conditional_dists) != self.num_features:
            raise ValueError('conditional_dists length must match parents length.')
        if len(self.default_dists) != self.num_features:
            raise ValueError('default_dists length must match parents length.')
        if sorted(self.feature_order) != list(range(self.num_features)):
            raise ValueError('feature_order must be a permutation of feature indices.')
        if sum(parent is None for parent in self.parents) != 1:
            raise ValueError('parents must contain exactly one root entry set to None.')

    def __str__(self) -> str:
        return (
            'ChowLiuTreeDistribution(parents=%s, marginal_dists=%s, conditional_dists=%s, '
            'default_dists=%s, feature_order=%s, name=%s)'
        ) % (
            repr(self.parents),
            repr(self.marginal_dists),
            repr(self.conditional_dists),
            repr(self.default_dists),
            repr(self.feature_order),
            repr(self.name),
        )

    def density(self, x: Sequence[Any]) -> float:
        """Return the probability density or mass at a single observation."""
        return float(np.exp(self.log_density(x)))

    def conditional_dist(self, child: int, parent_value: Any) -> Optional[SequenceEncodableProbabilityDistribution]:
        """Return the conditional distribution associated with a child and parent assignment."""
        key = freeze(parent_value)
        return self.conditional_dists[child].get(key, self.default_dists[child])

    def log_density(self, x: Sequence[Any]) -> float:
        """Return the log-density or log-mass at a single observation."""
        if len(x) != self.num_features:
            raise ValueError('Observation length does not match ChowLiuTreeDistribution.')

        root = self.feature_order[0]
        rv = self.marginal_dists[root].log_density(x[root])
        if rv == -np.inf:
            return -np.inf

        for child in self.feature_order[1:]:
            parent = self.parents[child]
            if parent is None:
                raise ValueError('feature_order contains a second root.')
            dist = self.conditional_dist(child, x[parent])
            if dist is None:
                return -np.inf
            rv += dist.log_density(x[child])
            if rv == -np.inf:
                return -np.inf
        return rv

    def seq_log_density(self, x: Sequence[Sequence[Any]]) -> np.ndarray:
        """Return vectorized log-density values for sequence-encoded observations."""
        return np.asarray([self.log_density(u) for u in x], dtype=float)

    def sampler(self, seed: Optional[int] = None) -> 'ChowLiuTreeSampler':
        """Return a sampler for drawing observations from this distribution."""
        return ChowLiuTreeSampler(self, seed)

    def estimator(self, pseudo_count: Optional[float] = None) -> 'ChowLiuTreeEstimator':
        """Return an estimator for fitting this distribution from data."""
        estimators = [
            dist.estimator(pseudo_count=_pseudo_for_index(pseudo_count, i))
            for i, dist in enumerate(self.marginal_dists)
        ]
        root = self.feature_order[0]
        return ChowLiuTreeEstimator(estimators, root=root, pseudo_count=pseudo_count, name=self.name)

    def dist_to_encoder(self) -> 'ChowLiuTreeDataEncoder':
        """Return the data encoder used by this distribution for vectorized methods."""
        return ChowLiuTreeDataEncoder()

    def enumerator(self) -> 'ChowLiuTreeEnumerator':
        """Return an enumerator over the distribution support when available."""
        return ChowLiuTreeEnumerator(self)


class ChowLiuTreeEnumerator(DistributionEnumerator):
    """Finite-support enumerator for a ChowLiuTreeDistribution."""

    def __init__(self, dist: ChowLiuTreeDistribution) -> None:
        super().__init__(dist)
        supports = []
        for i, child in enumerate(dist.marginal_dists):
            enum = child_enumerator(child, 'ChowLiuTreeDistribution.marginal_dists[%d]' % i)
            supports.append([value for value, _ in enum])
        entries = []
        for value in itertools.product(*supports):
            lp = float(dist.log_density(value))
            if lp > -np.inf:
                entries.append((tuple(value), lp))
        entries.sort(key=lambda u: -u[1])
        self._entries = entries
        self._pos = 0

    def __next__(self) -> Tuple[Tuple[Any, ...], float]:
        if self._pos >= len(self._entries):
            raise StopIteration
        rv = self._entries[self._pos]
        self._pos += 1
        return rv


class ChowLiuTreeSampler(DistributionSampler):
    """Sampler for a generic Chow-Liu tree."""

    def __init__(self, dist: ChowLiuTreeDistribution, seed: Optional[int] = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)
        self.root_samplers = [d.sampler(seed=self.rng.randint(maxrandint)) for d in dist.marginal_dists]
        self.conditional_samplers = [
            {k: v.sampler(seed=self.rng.randint(maxrandint)) for k, v in dmap.items()}
            for dmap in dist.conditional_dists
        ]
        self.default_samplers = [
            None if d is None else d.sampler(seed=self.rng.randint(maxrandint))
            for d in dist.default_dists
        ]

    def sample(self, size: Optional[int] = None) -> Union[Tuple[Any, ...], List[Tuple[Any, ...]]]:
        if size is not None:
            return [self.sample() for _ in range(int(size))]

        rv: List[Any] = [None] * self.dist.num_features
        root = self.dist.feature_order[0]
        rv[root] = self.root_samplers[root].sample()

        for child in self.dist.feature_order[1:]:
            parent = self.dist.parents[child]
            key = freeze(rv[parent])
            sampler = self.conditional_samplers[child].get(key, self.default_samplers[child])
            if sampler is None:
                raise RuntimeError('No conditional sampler for feature %d and parent value %r.' % (child, rv[parent]))
            rv[child] = sampler.sample()

        return tuple(rv)


class ChowLiuTreeAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for generic Chow-Liu tree sufficient statistics."""

    def __init__(self, estimators: Sequence[ParameterEstimator], keys: Optional[str] = None,
                 name: Optional[str] = None) -> None:
        self.estimators = list(estimators)
        self.num_features = len(self.estimators)
        self.key = keys
        self.name = name
        self.total_weight = 0.0

        self.marginal_counts: List[Dict[Hashable, float]] = [dict() for _ in range(self.num_features)]
        self.marginal_values: List[Dict[Hashable, Any]] = [dict() for _ in range(self.num_features)]
        self.joint_counts: Dict[Tuple[int, int], Dict[Tuple[Hashable, Hashable], float]] = dict()

        self.marginal_accumulators = [
            est.accumulator_factory().make() for est in self.estimators
        ]
        self.conditional_accumulators: Dict[Tuple[int, int], Dict[Hashable, SequenceEncodableStatisticAccumulator]] = {
            (p, c): {} for p in range(self.num_features) for c in range(self.num_features) if p != c
        }

    def _conditional_accumulator(self, parent: int, child: int, parent_key: Hashable) \
            -> SequenceEncodableStatisticAccumulator:
        accs = self.conditional_accumulators.setdefault((parent, child), {})
        if parent_key not in accs:
            accs[parent_key] = self.estimators[child].accumulator_factory().make()
        return accs[parent_key]

    @staticmethod
    def _joint_key(i: int, j: int) -> Tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def _previous_child_estimate(self, estimate: Optional[ChowLiuTreeDistribution],
                                 parent: int, child: int, parent_value: Any):
        if estimate is None:
            return None
        if estimate.parents[child] == parent:
            dist = estimate.conditional_dist(child, parent_value)
            if dist is not None:
                return dist
        return estimate.marginal_dists[child]

    def update(self, x: Sequence[Any], weight: float, estimate: Optional[ChowLiuTreeDistribution]) -> None:
        if len(x) != self.num_features:
            raise ValueError('Observation length does not match ChowLiuTreeEstimator.')

        xx = tuple(x)
        keys = [freeze(u) for u in xx]
        self.total_weight += weight

        for i, value in enumerate(xx):
            key = keys[i]
            self.marginal_counts[i][key] = self.marginal_counts[i].get(key, 0.0) + weight
            self.marginal_values[i].setdefault(key, value)
            prev = None if estimate is None else estimate.marginal_dists[i]
            self.marginal_accumulators[i].update(value, weight, prev)

        for i in range(self.num_features - 1):
            for j in range(i + 1, self.num_features):
                pair = (keys[i], keys[j])
                counts = self.joint_counts.setdefault((i, j), {})
                counts[pair] = counts.get(pair, 0.0) + weight

        for parent in range(self.num_features):
            parent_value = xx[parent]
            parent_key = keys[parent]
            for child in range(self.num_features):
                if child == parent:
                    continue
                acc = self._conditional_accumulator(parent, child, parent_key)
                prev = self._previous_child_estimate(estimate, parent, child, parent_value)
                acc.update(xx[child], weight, prev)

    def initialize(self, x: Sequence[Any], weight: float, rng: RandomState) -> None:
        if len(x) != self.num_features:
            raise ValueError('Observation length does not match ChowLiuTreeEstimator.')

        xx = tuple(x)
        keys = [freeze(u) for u in xx]
        self.total_weight += weight

        for i, value in enumerate(xx):
            key = keys[i]
            self.marginal_counts[i][key] = self.marginal_counts[i].get(key, 0.0) + weight
            self.marginal_values[i].setdefault(key, value)
            self.marginal_accumulators[i].initialize(value, weight, rng)

        for i in range(self.num_features - 1):
            for j in range(i + 1, self.num_features):
                pair = (keys[i], keys[j])
                counts = self.joint_counts.setdefault((i, j), {})
                counts[pair] = counts.get(pair, 0.0) + weight

        for parent in range(self.num_features):
            parent_key = keys[parent]
            for child in range(self.num_features):
                if child == parent:
                    continue
                self._conditional_accumulator(parent, child, parent_key).initialize(xx[child], weight, rng)

    def seq_update(self, x: Sequence[Sequence[Any]], weights: np.ndarray,
                   estimate: Optional[ChowLiuTreeDistribution]) -> None:
        for value, weight in zip(x, weights):
            self.update(value, float(weight), estimate)

    def seq_initialize(self, x: Sequence[Sequence[Any]], weights: np.ndarray, rng: RandomState) -> None:
        for value, weight in zip(x, weights):
            self.initialize(value, float(weight), rng)

    def combine(self, suff_stat: SS) -> 'ChowLiuTreeAccumulator':
        total_weight, num_features, marginal_counts, marginal_values, joint_counts, marginal_stats, cond_stats = suff_stat
        if num_features != self.num_features:
            raise ValueError('Cannot combine Chow-Liu statistics with different feature counts.')

        self.total_weight += total_weight
        for i in range(self.num_features):
            for key, count in marginal_counts[i].items():
                self.marginal_counts[i][key] = self.marginal_counts[i].get(key, 0.0) + count
            for key, value in marginal_values[i].items():
                self.marginal_values[i].setdefault(key, value)
            self.marginal_accumulators[i].combine(marginal_stats[i])

        for pair_key, counts in joint_counts.items():
            dst = self.joint_counts.setdefault(pair_key, {})
            for value_key, count in counts.items():
                dst[value_key] = dst.get(value_key, 0.0) + count

        for pair_key, by_parent in cond_stats.items():
            for parent_key, child_stat in by_parent.items():
                self._conditional_accumulator(pair_key[0], pair_key[1], parent_key).combine(child_stat)

        return self

    def value(self) -> SS:
        return (
            self.total_weight,
            self.num_features,
            [d.copy() for d in self.marginal_counts],
            [d.copy() for d in self.marginal_values],
            {k: v.copy() for k, v in self.joint_counts.items()},
            tuple(acc.value() for acc in self.marginal_accumulators),
            {
                pair_key: {parent_key: acc.value() for parent_key, acc in by_parent.items()}
                for pair_key, by_parent in self.conditional_accumulators.items()
                if by_parent
            },
        )

    def from_value(self, x: SS) -> 'ChowLiuTreeAccumulator':
        total_weight, num_features, marginal_counts, marginal_values, joint_counts, marginal_stats, cond_stats = x
        if num_features != self.num_features:
            raise ValueError('Cannot load Chow-Liu statistics with different feature counts.')
        self.total_weight = total_weight
        self.marginal_counts = [d.copy() for d in marginal_counts]
        self.marginal_values = [d.copy() for d in marginal_values]
        self.joint_counts = {k: v.copy() for k, v in joint_counts.items()}
        self.marginal_accumulators = [
            self.estimators[i].accumulator_factory().make().from_value(marginal_stats[i])
            for i in range(self.num_features)
        ]
        self.conditional_accumulators = {
            (p, c): {} for p in range(self.num_features) for c in range(self.num_features) if p != c
        }
        for pair_key, by_parent in cond_stats.items():
            for parent_key, child_stat in by_parent.items():
                acc = self.estimators[pair_key[1]].accumulator_factory().make()
                acc.from_value(child_stat)
                self.conditional_accumulators.setdefault(pair_key, {})[parent_key] = acc
        return self

    def key_merge(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self
        for acc in self.marginal_accumulators:
            acc.key_merge(stats_dict)
        for by_parent in self.conditional_accumulators.values():
            for acc in by_parent.values():
                acc.key_merge(stats_dict)

    def key_replace(self, stats_dict: Dict[str, Any]) -> None:
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())
        for acc in self.marginal_accumulators:
            acc.key_replace(stats_dict)
        for by_parent in self.conditional_accumulators.values():
            for acc in by_parent.values():
                acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> 'ChowLiuTreeDataEncoder':
        return ChowLiuTreeDataEncoder()


class ChowLiuTreeAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for ChowLiuTreeAccumulator."""

    def __init__(self, estimators: Sequence[ParameterEstimator], keys: Optional[str] = None,
                 name: Optional[str] = None) -> None:
        self.estimators = list(estimators)
        self.keys = keys
        self.name = name

    def make(self) -> ChowLiuTreeAccumulator:
        return ChowLiuTreeAccumulator(self.estimators, keys=self.keys, name=self.name)


class ChowLiuTreeEstimator(ParameterEstimator):
    """Estimate a generic Chow-Liu tree from fixed-length tuple observations.

    Structure learning uses empirical mutual information over observed field
    values, so it is best suited to finite/enumerable or intentionally
    discretized coordinates.  Once an edge is chosen, the child conditional
    distribution for each parent value is estimated with that child's supplied
    estimator.
    """

    def __init__(self,
                 estimators: Sequence[Union[ParameterEstimator, SequenceEncodableProbabilityDistribution]],
                 root: int = 0,
                 pseudo_count: Optional[float] = None,
                 mi_pseudo_count: Optional[float] = None,
                 default_policy: str = 'marginal',
                 keys: Optional[str] = None,
                 name: Optional[str] = None) -> None:
        self.pseudo_count = pseudo_count
        self.mi_pseudo_count = pseudo_count if mi_pseudo_count is None else mi_pseudo_count
        self.estimators = [
            _as_estimator(est, _pseudo_for_index(pseudo_count, i))
            for i, est in enumerate(estimators)
        ]
        self.num_features = len(self.estimators)
        self.root = int(root)
        self.default_policy = default_policy
        self.keys = keys
        self.name = name

        if self.num_features == 0:
            raise ValueError('ChowLiuTreeEstimator requires at least one feature estimator.')
        if self.root < 0 or self.root >= self.num_features:
            raise ValueError('root must be a valid feature index.')
        if default_policy not in ('marginal', 'none'):
            raise ValueError("default_policy must be 'marginal' or 'none'.")

    def accumulator_factory(self) -> ChowLiuTreeAccumulatorFactory:
        return ChowLiuTreeAccumulatorFactory(self.estimators, keys=self.keys, name=self.name)

    @staticmethod
    def _mutual_information(i: int, j: int,
                            marginal_counts: List[Dict[Hashable, float]],
                            joint_counts: Dict[Tuple[int, int], Dict[Tuple[Hashable, Hashable], float]],
                            pseudo_count: float) -> float:
        pair_counts = joint_counts.get((i, j), {})
        total = sum(pair_counts.values())
        if total <= 0.0:
            return 0.0

        keys_i = list(marginal_counts[i].keys())
        keys_j = list(marginal_counts[j].keys())
        if len(keys_i) == 0 or len(keys_j) == 0:
            return 0.0

        if pseudo_count > 0.0:
            denom = total + pseudo_count
            joint_alpha = pseudo_count / float(len(keys_i) * len(keys_j))
            marg_i_alpha = pseudo_count / float(len(keys_i))
            marg_j_alpha = pseudo_count / float(len(keys_j))
            mi = 0.0
            for key_i in keys_i:
                p_i = (marginal_counts[i].get(key_i, 0.0) + marg_i_alpha) / denom
                for key_j in keys_j:
                    p_j = (marginal_counts[j].get(key_j, 0.0) + marg_j_alpha) / denom
                    p_ij = (pair_counts.get((key_i, key_j), 0.0) + joint_alpha) / denom
                    if p_ij > 0.0 and p_i > 0.0 and p_j > 0.0:
                        mi += p_ij * (np.log(p_ij) - np.log(p_i) - np.log(p_j))
            return float(mi)

        mi = 0.0
        for (key_i, key_j), count in pair_counts.items():
            if count <= 0.0:
                continue
            p_ij = count / total
            p_i = marginal_counts[i].get(key_i, 0.0) / total
            p_j = marginal_counts[j].get(key_j, 0.0) / total
            if p_i > 0.0 and p_j > 0.0:
                mi += p_ij * (np.log(p_ij) - np.log(p_i) - np.log(p_j))
        return float(mi)

    def _tree_from_mi(self, marginal_counts, joint_counts) -> Tuple[List[Optional[int]], List[int]]:
        n = self.num_features
        if n == 1:
            return [None], [0]

        mi_mat = np.zeros((n, n), dtype=float)
        pseudo_count = 0.0 if self.mi_pseudo_count is None else float(self.mi_pseudo_count)
        for i in range(n - 1):
            for j in range(i + 1, n):
                mi = self._mutual_information(i, j, marginal_counts, joint_counts, pseudo_count)
                mi_mat[i, j] = mi
                mi_mat[j, i] = mi

        max_mi = float(np.max(mi_mat))
        cost_mat = np.zeros((n, n), dtype=float)
        for i in range(n - 1):
            for j in range(i + 1, n):
                cost = max_mi - mi_mat[i, j] + 1.0
                cost_mat[i, j] = cost
                cost_mat[j, i] = cost

        span_tree = minimum_spanning_tree(cost_mat)
        feature_order, predecessors = breadth_first_order(span_tree, self.root, directed=False,
                                                          return_predecessors=True)
        feature_order = [int(u) for u in feature_order]
        parents: List[Optional[int]] = [None] * n
        for feature in range(n):
            if feature == self.root:
                parents[feature] = None
            else:
                parents[feature] = int(predecessors[feature])
        return parents, feature_order

    def estimate(self, nobs: Optional[float], suff_stat: SS) -> ChowLiuTreeDistribution:
        total_weight, num_features, marginal_counts, marginal_values, joint_counts, marginal_stats, cond_stats = suff_stat
        if num_features != self.num_features:
            raise ValueError('Sufficient statistics feature count does not match estimator.')
        if total_weight <= 0.0:
            raise ValueError('Cannot estimate a Chow-Liu tree with no weighted observations.')

        marginal_dists = [
            self.estimators[i].estimate(None, marginal_stats[i])
            for i in range(self.num_features)
        ]

        parents, feature_order = self._tree_from_mi(marginal_counts, joint_counts)
        conditional_dists: List[Dict[Hashable, SequenceEncodableProbabilityDistribution]] = [
            {} for _ in range(self.num_features)
        ]
        default_dists: List[Optional[SequenceEncodableProbabilityDistribution]] = [None] * self.num_features

        for child in range(self.num_features):
            parent = parents[child]
            if parent is None:
                continue
            by_parent = cond_stats.get((parent, child), {})
            conditional_dists[child] = {
                parent_key: self.estimators[child].estimate(None, child_stat)
                for parent_key, child_stat in by_parent.items()
            }
            if self.default_policy == 'marginal':
                default_dists[child] = marginal_dists[child]

        return ChowLiuTreeDistribution(
            parents=parents,
            marginal_dists=marginal_dists,
            conditional_dists=conditional_dists,
            default_dists=default_dists,
            feature_order=feature_order,
            parent_values=marginal_values,
            name=self.name)


class ChowLiuTreeDataEncoder(DataSequenceEncoder):
    """Raw tuple encoder for generic Chow-Liu tree observations."""

    def __str__(self) -> str:
        return 'ChowLiuTreeDataEncoder'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ChowLiuTreeDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[Any]]) -> Tuple[Tuple[Any, ...], ...]:
        return tuple(tuple(u) for u in x)
