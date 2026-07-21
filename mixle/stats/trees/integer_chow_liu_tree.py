"""Integer Chow-Liu tree distributions for fixed-length integer vectors.

Mixle supports Chow & Liu trees [1] through the IntegerChowLiuTree (Integer Chow Liu Tree) class of objects. IntegerChowLiuTrees model
non-Markov conditional dependence for fixed-length sequences of integers with the likelihood functions of the form

    P(x_1, x_2,..,x_n) = P(x_i1) P(x_{i_2}|x_{j_2})*...*P(x_{i_n}|x_{j_n}),

where j_k < i_k for all k = 1,2,3,..N.

Data type: Union[Sequence[int], np.ndarray] .

"""

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class IntegerChowLiuTreeDistribution(SequenceEncodableProbabilityDistribution):
    """Integer Chow-Liu tree distribution factorizing a joint over fixed-length integer vectors along a tree.

    Data type: Union[Sequence[int], np.ndarray] (fixed-length vector of non-negative integers).
    """

    @classmethod
    def compute_capabilities(cls):
        """Declare backend support for integer Chow-Liu generated kernels."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic_table")

    @classmethod
    def compute_declaration(cls):
        """Return the generated-compute declaration for the integer Chow-Liu tree."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="integer_chow_liu_tree",
            distribution_type=cls,
            parameters=(
                ParameterSpec("conditional_log_densities", constraint="log_probability_tables", differentiable=False),
            ),
            statistics=(
                StatisticSpec("num_features", kind="metadata", additive=False, scales=False),
                StatisticSpec("num_states", kind="metadata", additive=False, scales=False),
                StatisticSpec("counts", kind="pairwise_count_tensor"),
                StatisticSpec("marginal_counts", kind="count_tensor"),
            ),
            support="fixed_integer_tuple_tree",
            differentiable=False,
        )

    def __init__(
        self,
        dependency_list: list[int | None],
        conditional_log_densities: Sequence[float] | np.ndarray,
        feature_order: Sequence[int] | None = None,
        name: str | None = None,
    ) -> None:
        """Create an integer Chow-Liu tree distribution.

        Args:
            dependency_list (List[Optional[int]]): Parent feature id for each feature in feature_order, or None
                for the (exactly one) root feature with no parent.
            conditional_log_densities (Union[Sequence[float], np.ndarray]): Conditional log densities for each features
                dependency split.
            feature_order (Optional[Sequence[int]]): Ordering of features. If None, ordering is assumed as entered.
            name (Optional[str]): Optional distribution name.

        Attributes:
            feature_order (Sequence[int]): Ordering of features. If None, ordering is assumed as entered.
            dependency_list (List[ Tuple[int, Optional[int]]]): List of Tuples containing each feature's
                order id and its parent id (or None for the root).
            conditional_log_densities (Union[Sequence[float], np.ndarray]): Conditional log densities for each features
                dependency split.
            conditional_densities (np.ndarray): Conditional densities as numpy array.
            num_features (int): Total number of features.
            name (Optional[str]): Optional distribution name.

        """
        self.feature_order = range(len(dependency_list)) if feature_order is None else feature_order
        self.dependency_list = list(zip(self.feature_order, dependency_list))
        # Normalized to ndarrays (mirroring MultivariateGaussianDistribution's np.asarray convention) so a
        # feature table is always 2-d-indexable regardless of whether the caller passed lists or arrays --
        # log_density/seq_log_density index these with `table[parent_val, child_val]`, which raises "list
        # indices must be integers or slices, not tuple" on a plain (unconverted) nested list.
        self.conditional_log_densities = [np.asarray(u) for u in conditional_log_densities]
        self.conditional_densities = [np.exp(u) for u in conditional_log_densities]
        self.num_features = len(dependency_list)
        self.name = name

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""

        def _fmt(u: np.ndarray) -> str:
            # Recursively render as a bracketed literal that preserves shape (1-d root tables vs.
            # 2-d parent/child tables), so eval(str(dist)) reconstructs the same table shapes.
            if u.ndim <= 1:
                return "[" + ",".join(map(str, u)) + "]"
            return "[" + ",".join(_fmt(row) for row in u) + "]"

        f1 = ",".join([str(u[1]) for u in self.dependency_list])
        f3 = ",".join([str(u[0]) for u in self.dependency_list])
        f2 = ",".join(_fmt(u) for u in self.conditional_log_densities)
        f4 = repr(self.name)
        return "IntegerChowLiuTreeDistribution([%s], [%s], feature_order=[%s], name=%s)" % (f1, f2, f3, f4)

    def density(self, x: Sequence[int] | np.ndarray) -> float:
        """Density of integer Chow-Liu tree distribution at observation x.

        See log_density() for details.

        Args:
            x (Union[Sequence[int], np.ndarray]): Fixed-length vector of non-negative integers.

        Returns:
            Density at observation x.

        """
        return np.exp(self.log_density(x))

    def log_density(self, x: Sequence[int] | np.ndarray) -> float:
        """Log-density of integer Chow-Liu tree distribution at observation x.

        Sums the conditional log-densities of each feature given its parent in the dependency tree
        (the root feature contributes its marginal log-density).

        Args:
            x (Union[Sequence[int], np.ndarray]): Fixed-length vector of non-negative integers.

        Returns:
            Log-density at observation x.

        """
        rv = 0
        for i, (j, k) in enumerate(self.dependency_list):
            if k is None:
                rv += self.conditional_log_densities[i][x[j]]
            else:
                rv += self.conditional_log_densities[i][x[k], x[j]]

        return rv

    def seq_log_density(self, x: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of log-density at sequence encoded input x.

        Args:
            x (np.ndarray): 2-d numpy array of N integer vectors with num_features columns.

        Returns:
            Numpy array of log-density (float) of length N.

        """
        rv = np.zeros(x.shape[0])
        for i, (j, k) in enumerate(self.dependency_list):
            if k is None:
                rv += self.conditional_log_densities[i][x[:, j]]
            else:
                rv += self.conditional_log_densities[i][x[:, k], x[:, j]]

        return rv

    def backend_seq_log_density(self, x: np.ndarray, engine: Any) -> Any:
        """Engine-neutral vectorized table lookup for fixed integer tree factors."""
        xx = engine.asarray(x)
        rv = engine.zeros(xx.shape[0])
        for i, (j, k) in enumerate(self.dependency_list):
            table = engine.asarray(self.conditional_log_densities[i])
            if k is None:
                rv = rv + table[xx[:, j]]
            else:
                rv = rv + table[xx[:, k], xx[:, j]]
        return rv

    def sampler(self, seed: int | None = None) -> "IntegerChowLiuTreeSampler":
        """Create a sampler for this integer Chow-Liu tree distribution.

        Args:
            seed (Optional[int]): Used to set seed in random sampler.

        Returns:
            IntegerChowLiuTreeSampler: Sampler bound to this distribution.

        """
        return IntegerChowLiuTreeSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "IntegerChowLiuTreeEstimator":
        """Create an estimator initialized from this integer Chow-Liu tree distribution.

        Args:
            pseudo_count (Optional[float]): Used to inflate sufficient statistics.

        Returns:
            IntegerChowLiuTreeEstimator: Estimator configured with the same feature count and state count.

        """
        num_states = len(self.conditional_densities[0])
        return IntegerChowLiuTreeEstimator(
            num_features=self.num_features, num_states=num_states, pseudo_count=pseudo_count, name=self.name
        )

    def dist_to_encoder(self) -> "IntegerChowLiuTreeDataEncoder":
        """Return a data encoder for integer Chow-Liu tree observations."""
        return IntegerChowLiuTreeDataEncoder()

    def enumerator(self) -> "IntegerChowLiuTreeEnumerator":
        """Returns IntegerChowLiuTreeEnumerator iterating fixed-length integer vectors in descending probability order."""
        return IntegerChowLiuTreeEnumerator(self)


class IntegerChowLiuTreeEnumerator(DistributionEnumerator):
    """Enumerates the finite support of an integer Chow-Liu tree."""

    def __init__(self, dist: IntegerChowLiuTreeDistribution) -> None:
        """Create an enumerator for integer Chow-Liu tree observations.

        The support is the Cartesian product of each feature's finite state range, inferred
        from the root marginal and conditional probability tables.

        Args:
            dist (IntegerChowLiuTreeDistribution): Distribution whose support is enumerated.

        """
        super().__init__(dist)
        domain_sizes: list[int | None] = [None] * dist.num_features

        for i, (feature, parent) in enumerate(dist.dependency_list):
            table = np.asarray(dist.conditional_log_densities[i])
            if parent is None:
                if table.ndim != 1:
                    raise EnumerationError(dist, reason="root conditional table must be one-dimensional")
                child_size = table.shape[0]
                parent_size = None
            else:
                if table.ndim != 2:
                    raise EnumerationError(dist, reason="conditional tables must be two-dimensional")
                parent_size, child_size = table.shape
                self._set_domain_size(domain_sizes, parent, parent_size, dist)
            self._set_domain_size(domain_sizes, feature, child_size, dist)

        if any(sz is None for sz in domain_sizes):
            raise EnumerationError(dist, reason="could not infer every feature domain size")

        with np.errstate(divide="ignore"):
            entries = []
            ranges = [range(int(sz)) for sz in domain_sizes]
            for value in itertools.product(*ranges):
                lp = float(dist.log_density(value))
                if lp > -np.inf:
                    entries.append((list(value), lp))
        entries.sort(key=lambda u: -u[1])
        self._entries = entries
        self._pos = 0

    @staticmethod
    def _set_domain_size(
        domain_sizes: list[int | None], idx: int, size: int, dist: IntegerChowLiuTreeDistribution
    ) -> None:
        if idx < 0 or idx >= len(domain_sizes):
            raise EnumerationError(dist, reason="feature index out of range")
        if domain_sizes[idx] is not None and domain_sizes[idx] != size:
            raise EnumerationError(dist, reason="inconsistent feature domain sizes")
        domain_sizes[idx] = int(size)

    def __next__(self) -> tuple[list[int], float]:
        if self._pos >= len(self._entries):
            raise StopIteration
        item = self._entries[self._pos]
        self._pos += 1
        return item


class IntegerChowLiuTreeSampler(DistributionSampler):
    """Sampler for the IntegerChowLiuTreeDistribution. Samples each feature given its sampled parent value."""

    def __init__(self, dist: IntegerChowLiuTreeDistribution, seed: int | None = None) -> None:
        """Create a sampler for integer Chow-Liu tree observations.

        Args:
            dist (IntegerChowLiuTreeDistribution): Distribution to sample from.
            seed (Optional[int]): Seed for random number generator.

        """
        self.rng = RandomState(seed)
        self.dist = dist

    def sample(self, size: int | None = None, *, batched: bool = True) -> list[int | None] | Sequence[list[int | None]]:
        """Draw iid integer vectors from the integer Chow-Liu tree distribution.

        Features are drawn in dependency order: the root from its marginal, each remaining
        feature from its conditional given the sampled parent value.

        Args:
            size (Optional[int]): Number of samples to draw. If None, a single vector is returned.

        Returns:
            A single integer vector (List[int]) if size is None, else a list of size vectors.

        """

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


class IntegerChowLiuTreeAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulator for the IntegerChowLiuTreeDistribution. Tracks pairwise joint and marginal feature-state counts."""

    def __init__(self, num_features: int, num_states: int, keys: str | None = None, name: str | None = None):
        """Create an accumulator for integer Chow-Liu tree sufficient statistics.

        Args:
            num_features (int): Number of features (length of observed integer vectors).
            num_states (int): Number of states (distinct integer values) per feature.
            keys (Optional[str]): Optional key for merging sufficient statistics.
            name (Optional[str]): Optional accumulator name.

        Attributes:
            num_states (int): Number of states per feature.
            num_features (int): Number of features.
            counts (Optional[np.ndarray]): Pairwise joint counts with shape
                (num_features, num_features, num_states, num_states). None until dimensions are known.
            marginal_counts (Optional[np.ndarray]): Marginal counts with shape (num_features, num_states).
            key (Optional[str]): Optional key for merging sufficient statistics.
            name (Optional[str]): Optional accumulator name.

        """
        self.num_states = num_states
        self.num_features = num_features

        if num_states is not None and num_features is not None:
            self.counts = np.zeros((num_features, num_features, num_states, num_states))
            self.marginal_counts = np.zeros((num_features, num_states))
        else:
            self.counts = None
            self.marginal_counts = None

        self.keys = keys
        self.name = name

    def _expand_states(self, num_states: int, num_features: int):
        """Allocate or grow the count arrays to hold num_states states for num_features features.

        Args:
            num_states (int): New number of states per feature.
            num_features (int): Number of features.

        """
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

    def update(
        self, x: Sequence[int] | np.ndarray, weight: float, estimate: IntegerChowLiuTreeDistribution | None
    ) -> None:
        """Update pairwise joint and marginal counts with a weighted observation.

        Args:
            x (Union[Sequence[int], np.ndarray]): Fixed-length vector of non-negative integers.
            weight (float): Weight for observation.
            estimate (Optional[IntegerChowLiuTreeDistribution]): Previous estimate (unused).

        """
        if (self.counts is None) or (self.num_states <= np.max(x)):
            self._expand_states(max(x) + 1, len(x))

        xx = np.asarray(x)
        ff = np.arange(self.num_features)

        self.marginal_counts[ff, xx] += weight
        for i in range(self.num_features):
            self.counts[i, ff, xx[i], xx] += weight

    def seq_update(self, x: np.ndarray, weights: np.ndarray, estimate: IntegerChowLiuTreeDistribution | None) -> None:
        """Vectorized update of pairwise joint and marginal counts from sequence encoded data.

        Args:
            x (np.ndarray): 2-d numpy array of N integer vectors with num_features columns.
            weights (np.ndarray): Weights for each of the N observations.
            estimate (Optional[IntegerChowLiuTreeDistribution]): Previous estimate (unused).

        """
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

    def initialize(self, x: Sequence[int] | np.ndarray, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics with a weighted observation.

        Args:
            x (Union[Sequence[int], np.ndarray]): Fixed-length vector of non-negative integers.
            weight (float): Weight for observation.
            rng (Optional[RandomState]): Random number generator (unused).

        """
        self.update(x, weight, None)

    def seq_initialize(self, x: np.ndarray, weights: np.ndarray, rng: RandomState | None) -> None:
        """Vectorized initialization of sufficient statistics from sequence encoded data.

        Args:
            x (np.ndarray): 2-d numpy array of N integer vectors with num_features columns.
            weights (np.ndarray): Weights for each of the N observations.
            rng (Optional[RandomState]): Random number generator (unused).

        """
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[int, int, np.ndarray, np.ndarray]) -> "IntegerChowLiuTreeAccumulator":
        """Combine sufficient statistics from another accumulator into this one.

        Count arrays are expanded if the incoming statistics track more states.

        Args:
            suff_stat (Tuple[int, int, np.ndarray, np.ndarray]): Tuple of number of features, number of
                states, pairwise joint counts, and marginal counts.

        Returns:
            Self, with aggregated sufficient statistics.

        """
        num_features, num_states, counts, marginal_counts = suff_stat

        if self.counts is None and counts is None:
            return self

        elif (self.counts is None) and (counts is not None):
            self.counts = counts
            self.marginal_counts = marginal_counts
            self.num_states = num_states
            self.num_features = num_features

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

    def value(self) -> tuple[int, int, np.ndarray, np.ndarray]:
        """Returns sufficient statistics as a Tuple of number of features, number of states, pairwise
        joint counts, and marginal counts."""
        return self.num_features, self.num_states, self.counts, self.marginal_counts

    def from_value(self, x: tuple[int, int, np.ndarray, np.ndarray]) -> "IntegerChowLiuTreeAccumulator":
        """Set sufficient statistics of accumulator from value x.

        Args:
            x (Tuple[int, int, np.ndarray, np.ndarray]): Tuple of number of features, number of states,
                pairwise joint counts, and marginal counts.

        Returns:
            Self, with sufficient statistics set to x.

        """
        self.num_features = x[0]
        self.num_states = x[1]
        self.counts = x[2]
        self.marginal_counts = x[3]

        return self

    def scale(self, c: float) -> "IntegerChowLiuTreeAccumulator":
        """Scale all accumulated Chow-Liu sufficient statistics in place."""
        if self.counts is not None:
            self.counts *= c
        if self.marginal_counts is not None:
            self.marginal_counts *= c
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """No-op kept for interface consistency (keyed merging is not supported for IntegerChowLiuTreeAccumulator).

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to shared sufficient statistics (ignored).

        Returns:
            None.

        """
        pass

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """No-op kept for interface consistency (keyed merging is not supported for IntegerChowLiuTreeAccumulator).

        Args:
            stats_dict (Dict[str, Any]): Dict mapping keys to shared sufficient statistics (ignored).

        Returns:
            None.

        """
        pass

    def acc_to_encoder(self) -> "IntegerChowLiuTreeDataEncoder":
        """Return a data encoder for accumulated integer Chow-Liu tree observations."""
        return IntegerChowLiuTreeDataEncoder()


class IntegerChowLiuTreeAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for integer Chow-Liu tree accumulators."""

    def __init__(
        self,
        num_features: int | None = None,
        num_states: int | None = None,
        keys: str | None = None,
        name: str | None = None,
    ) -> None:
        """Create a factory for integer Chow-Liu tree accumulators.

        Args:
            num_features (Optional[int]): Number of features. If None, set from data on first update.
            num_states (Optional[int]): Number of states per feature. If None, set from data.
            keys (Optional[str]): Optional key for merging sufficient statistics.
            name (Optional[str]): Optional accumulator name.

        """
        self.num_features = num_features
        self.num_states = num_states
        self.keys = keys
        self.name = name

    def make(self) -> "IntegerChowLiuTreeAccumulator":
        """Return a new integer Chow-Liu tree accumulator."""
        return IntegerChowLiuTreeAccumulator(self.num_features, self.num_states, self.keys)


class IntegerChowLiuTreeEstimator(ParameterEstimator):
    """Estimator for the IntegerChowLiuTreeDistribution. Learns the dependency tree with the Chow-Liu algorithm."""

    def __init__(
        self,
        num_features: int | None = None,
        num_states: int | None = None,
        pseudo_count: float | None = None,
        suff_stat: Any | None = None,
        keys: str | None = None,
        name: str | None = None,
    ):
        """Create an estimator for integer Chow-Liu tree distributions.

        Args:
            num_features (Optional[int]): Number of features. If None, set from data.
            num_states (Optional[int]): Number of states per feature. If None, set from data.
            pseudo_count (Optional[float]): Smoothing count spread over the marginal and joint counts.
            suff_stat (Optional[Any]): Kept for interface consistency (unused).
            keys (Optional[str]): Optional key for merging sufficient statistics.
            name (Optional[str]): Optional name assigned to estimated distributions.

        """
        self.num_features = num_features
        self.num_states = num_states
        self.pseudo_count = pseudo_count
        self.suff_stat = suff_stat
        self.keys = keys
        self.name = name

    def accumulator_factory(self):
        """Return an accumulator factory configured from this estimator."""
        return IntegerChowLiuTreeAccumulatorFactory(self.num_features, self.num_states, self.keys)

    def estimate(self, nobs, suff_stat):
        """Estimate an IntegerChowLiuTreeDistribution from sufficient statistics via the Chow-Liu algorithm.

        Pairwise mutual information is computed from the (optionally smoothed) joint and marginal
        counts, a maximum mutual information spanning tree is extracted, and conditional densities
        are computed along the tree rooted at feature 0.

        Args:
            nobs (Optional[float]): Number of observations (unused).
            suff_stat (Tuple[int, int, np.ndarray, np.ndarray]): Tuple of number of features, number of
                states, pairwise joint counts, and marginal counts.

        Returns:
            IntegerChowLiuTreeDistribution object.

        """
        num_features, num_states, counts, marginal_counts = suff_stat

        mi_mat = np.zeros((num_features, num_features))

        pseudo_count = self.pseudo_count if self.pseudo_count is not None else 0.0
        pseudo_count_adj0 = pseudo_count / num_states
        pseudo_count_adj1 = pseudo_count / (num_states * num_states)

        for i in range(num_features - 1):
            for j in range(i + 1, num_features):
                if pseudo_count > 0:
                    n_ij = counts[i, j, :, :].sum()
                    joint_ij = (counts[i, j, :, :] + pseudo_count_adj1) / (n_ij + pseudo_count)
                    marg_i = (marginal_counts[i, :] + pseudo_count_adj0) / (n_ij + pseudo_count)
                    marg_j = (marginal_counts[j, :] + pseudo_count_adj0) / (n_ij + pseudo_count)
                    indep_ij = np.outer(marg_i, marg_j)
                else:
                    joint_ij = counts[i, j, :, :].copy()
                    indep_ij = np.outer(marginal_counts[i, :], marginal_counts[j, :])

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

        cost_mat = np.abs(mi_mat.max() - mi_mat)
        cost_mat[mi_mat > 0] += 1.0
        cost_mat[mi_mat == 0] = 0

        span_tree = minimum_spanning_tree(cost_mat)

        root_node = 0
        feature_order, deps = breadth_first_order(span_tree, root_node, directed=False, return_predecessors=True)

        deps = [deps[i] for i in feature_order]
        tmats = [None] * num_features

        with np.errstate(divide="ignore"):
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

        return IntegerChowLiuTreeDistribution(deps, tmats, feature_order=feature_order)


class IntegerChowLiuTreeDataEncoder(DataSequenceEncoder):
    """Data encoder for sequences of fixed-length integer vector observations."""

    def __str__(self) -> str:
        """Return a constructor-style representation of the encoder."""
        return "IntegerChowLiuTreeDataEncoder"

    def __eq__(self, other: object) -> bool:
        """Return true when ``other`` is an integer Chow-Liu tree data encoder.

        Args:
            other (object): Object to compare against.

        Returns:
            True if other is an IntegerChowLiuTreeDataEncoder instance, else False.

        """
        return isinstance(other, IntegerChowLiuTreeDataEncoder)

    def seq_encode(self, x: list[int] | np.ndarray) -> np.ndarray:
        """Encode a sequence of N integer vectors for vectorized functions.

        Args:
            x (Union[List[int], np.ndarray]): Sequence of N fixed-length integer vectors.

        Returns:
            2-d numpy array of ints with N rows and num_features columns.

        """
        return np.asarray(x, dtype=int)
