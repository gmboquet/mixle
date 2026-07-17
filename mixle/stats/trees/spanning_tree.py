"""Weighted spanning-tree distributions over labeled graphs.

Data type: a spanning tree of n labeled nodes given as a sequence of n-1 undirected edges, each an
``(i, j)`` pair (e.g. ``[(0, 1), (1, 2), (1, 3)]``). Unlike ChowLiuTree (a tree-structured distribution
over vectors), this is a distribution over the tree STRUCTURES themselves.

Each undirected edge has a positive weight ``w[i, j]``. A spanning tree T has probability

    p(T) = prod_{(i,j) in T} w[i, j] / Z,    Z = sum over all spanning trees of prod w[e],

and by the Matrix-Tree theorem Z equals any first cofactor of the weighted graph Laplacian
L = diag(W 1) - W, i.e. ``det(L[1:, 1:])``. Sampling uses Wilson's loop-erased-random-walk algorithm,
which draws exactly from this weighted uniform-spanning-tree law. Estimation matches empirical or
smoothed edge frequencies to the model edge marginals (an exponential family over trees, fit by
projected gradient ascent on the log-weights); the per-edge marginal ``w[i,j] * R_eff(i,j)`` is read
from the Laplacian pseudoinverse. Exact finite enumeration scans all positive-edge subsets of size
n-1, keeps the spanning trees, and sorts them by fitted probability.
"""

from collections.abc import Sequence

import numpy as np
from numpy.random import RandomState

from mixle.enumeration.spanning import k_best_spanning_trees
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_MIN_LOG_WEIGHT = -30.0
_MAX_LOG_WEIGHT = 30.0
_DEFAULT_MAX_ENUMERATION_SUBSETS = 200_000


def _weighted_laplacian(weights: np.ndarray) -> np.ndarray:
    return np.diag(weights.sum(axis=1)) - weights


def _log_partition(weights: np.ndarray) -> float:
    """Return log Z via the Matrix-Tree theorem (log-det of a Laplacian cofactor)."""
    lap = _weighted_laplacian(weights)
    sign, logabsdet = np.linalg.slogdet(lap[1:, 1:])
    if sign <= 0.0:
        raise ValueError("SpanningTreeDistribution: weighted Laplacian cofactor is not positive (check weights).")
    return float(logabsdet)


def _edge_marginals(weights: np.ndarray) -> np.ndarray:
    """Return the model edge-inclusion probabilities P((i,j) in T) = w[i,j] * R_eff(i,j)."""
    lap = _weighted_laplacian(weights)
    lap_pinv = np.linalg.pinv(lap)
    diag = np.diag(lap_pinv)
    r_eff = diag[:, None] + diag[None, :] - 2.0 * lap_pinv
    return weights * r_eff


def _smoothed_edge_target(
    edge_counts: np.ndarray,
    count: float,
    candidate: np.ndarray,
    pseudo_count: float | None,
) -> np.ndarray:
    """Return empirical edge marginals, optionally smoothed toward the uniform tree law."""
    target = edge_counts / count
    if pseudo_count:
        prior_marginals = _edge_marginals(np.where(candidate, 1.0, 0.0))
        target = (count * target + pseudo_count * prior_marginals) / (count + pseudo_count)
    return target * candidate


class SpanningTreeDistribution(SequenceEncodableProbabilityDistribution):
    """Weighted spanning-tree distribution over n labeled nodes with symmetric positive edge weights.

    Data type: a sequence of n-1 undirected edges (i, j) forming a spanning tree of 0,...,n-1.
    """

    @classmethod
    def compute_capabilities(cls):
        """Describe backend support for weighted spanning-tree operations."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(
            engine_ready=("numpy",),
            kernel_status="numpy_only",
            numpy_only_reason="Matrix-Tree normalizer and Wilson sampling are numpy-native.",
        )

    def __init__(
        self,
        weights: Sequence[Sequence[float]] | np.ndarray,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a distribution over spanning trees.

        Args:
            weights (Union[Sequence[Sequence[float]], np.ndarray]): Symmetric n-by-n matrix of
                non-negative edge weights (zero diagonal). Positive entries are the candidate edges.
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            weights (np.ndarray): Symmetrized edge-weight matrix with zero diagonal.
            dim (int): Number of nodes n.
            log_weights (np.ndarray): Elementwise log of the (positive) weights (-inf off-support).
            log_z (float): log normalizer from the Matrix-Tree theorem.

        """
        w = np.asarray(weights, dtype=float).copy()
        n = w.shape[0]
        if w.ndim != 2 or w.shape != (n, n) or n < 2:
            raise ValueError("SpanningTreeDistribution requires a square n-by-n weight matrix with n >= 2.")
        w = 0.5 * (w + w.T)
        np.fill_diagonal(w, 0.0)
        if np.any(w < 0.0) or not np.all(np.isfinite(w)):
            raise ValueError("SpanningTreeDistribution requires finite non-negative edge weights.")
        self.weights = w
        self.dim = n
        with np.errstate(divide="ignore"):
            self.log_weights = np.log(w)
        self.log_z = _log_partition(w)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-style representation of the distribution."""
        return "SpanningTreeDistribution(%s, name=%s, keys=%s)" % (
            repr([[float(v) for v in row] for row in self.weights]),
            repr(self.name),
            repr(self.keys),
        )

    def _edge_log_weight_sum(self, edges: np.ndarray) -> float:
        return float(np.sum(self.log_weights[edges[:, 0], edges[:, 1]]))

    def density(self, x: Sequence[Sequence[int]]) -> float:
        """Return the probability of a spanning tree x (a sequence of edges)."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Sequence[Sequence[int]]) -> float:
        """Return the log-probability of a spanning tree x (a sequence of n-1 edges)."""
        edges = _canonical_edges(x, self.dim)
        return self._edge_log_weight_sum(edges) - self.log_z

    def seq_log_density(self, x: Sequence[np.ndarray]) -> np.ndarray:
        """Return vectorized log-probabilities for a sequence of canonical edge arrays."""
        return np.asarray([self._edge_log_weight_sum(edges) - self.log_z for edges in x], dtype=float)

    def sampler(self, seed: int | None = None) -> "SpanningTreeSampler":
        """Return a sampler for drawing spanning trees from this distribution."""
        return SpanningTreeSampler(self, seed)

    def enumerator(
        self,
        max_edge_subsets: int | None = _DEFAULT_MAX_ENUMERATION_SUBSETS,
    ) -> "SpanningTreeEnumerator":
        """Return an exact finite enumerator over all supported spanning trees in probability order."""
        return SpanningTreeEnumerator(self, max_edge_subsets=max_edge_subsets)

    def estimator(self, pseudo_count: float | None = None) -> "SpanningTreeEstimator":
        """Return an estimator that keeps the node count fixed at this distribution's n."""
        return SpanningTreeEstimator(dim=self.dim, pseudo_count=pseudo_count, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> "SpanningTreeDataEncoder":
        """Return the data encoder used by this distribution for vectorized methods."""
        return SpanningTreeDataEncoder(dim=self.dim)


class SpanningTreeEnumerator(DistributionEnumerator):
    """Enumerate supported spanning trees in descending probability order, lazily.

    A tree's probability is the product of its edge weights, so descending probability is increasing total edge
    cost under ``cost = -log(weights)`` (zero-weight edges become +inf, i.e. absent). Gabow's k-best spanning-tree
    algorithm streams the trees in that order from one constrained-MST oracle per node, without scanning the
    exponential set of edge subsets.
    """

    def __init__(
        self,
        dist: SpanningTreeDistribution,
        max_edge_subsets: int | None = _DEFAULT_MAX_ENUMERATION_SUBSETS,
    ) -> None:
        # max_edge_subsets is accepted for backward compatibility but no longer constrains the lazy enumeration.
        super().__init__(dist)
        with np.errstate(divide="ignore"):
            cost = -dist.log_weights  # +inf where the edge weight is 0 (absent edge)
        self._gen = k_best_spanning_trees(cost)
        self._log_z = dist.log_z

    def __next__(self) -> tuple[list[tuple[int, int]], float]:
        total, tree = next(self._gen)  # StopIteration propagates at the end of the support
        canon = _canonical_edges(tree, self.dist.dim)  # same canonical edge representation as log_density
        value = [(int(a), int(b)) for a, b in canon]
        return value, float(-total - self._log_z)


class SpanningTreeSampler(DistributionSampler):
    """Draw iid spanning trees via Wilson's loop-erased-random-walk algorithm."""

    def __init__(self, dist: SpanningTreeDistribution, seed: int | None = None) -> None:
        self.rng = RandomState(seed)
        self.dist = dist
        w = dist.weights
        row = w.sum(axis=1)
        # Random-walk transition probabilities P[u, v] ∝ w[u, v]; isolated rows stay put.
        self.trans = np.divide(w, row[:, None], out=np.zeros_like(w), where=row[:, None] > 0.0)

    def _sample_one(self) -> list[tuple[int, int]]:
        n = self.dist.dim
        in_tree = np.zeros(n, dtype=bool)
        next_node = -np.ones(n, dtype=int)
        in_tree[0] = True
        for i in range(1, n):
            u = i
            while not in_tree[u]:
                v = int(self.rng.choice(n, p=self.trans[u]))
                next_node[u] = v
                u = v
            u = i
            while not in_tree[u]:
                in_tree[u] = True
                u = next_node[u]
        edges = [(min(v, int(next_node[v])), max(v, int(next_node[v]))) for v in range(n) if v != 0]
        return sorted(edges)

    def sample(
        self, size: int | None = None, *, batched: bool = True
    ) -> list[tuple[int, int]] | list[list[tuple[int, int]]]:
        """Draw spanning trees (each a sorted edge list); a single tree when size is None."""
        if size is None:
            return self._sample_one()
        return [self._sample_one() for _ in range(size)]


class SpanningTreeAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted edge-appearance counts (the sufficient statistic for the tree weights)."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.edge_counts = np.zeros((dim, dim))
        self.count = 0.0
        self.keys = keys

    def update(self, x: Sequence[Sequence[int]], weight: float, estimate: SpanningTreeDistribution | None) -> None:
        """Accumulate weighted edge appearances for one spanning tree."""
        edges = _canonical_edges(x, self.dim)
        self.edge_counts[edges[:, 0], edges[:, 1]] += weight
        self.edge_counts[edges[:, 1], edges[:, 0]] += weight
        self.count += weight

    def initialize(self, x: Sequence[Sequence[int]], weight: float, rng: RandomState | None) -> None:
        """Initialize edge-count statistics from one spanning tree."""
        self.update(x, weight, None)

    def seq_update(
        self, x: Sequence[np.ndarray], weights: np.ndarray, estimate: SpanningTreeDistribution | None
    ) -> None:
        """Accumulate edge appearances from encoded spanning trees."""
        for edges, w in zip(x, weights):
            self.edge_counts[edges[:, 0], edges[:, 1]] += w
            self.edge_counts[edges[:, 1], edges[:, 0]] += w
        self.count += float(np.sum(weights, dtype=np.float64))

    def seq_initialize(self, x: Sequence[np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize edge-count statistics from encoded spanning trees."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray]) -> "SpanningTreeAccumulator":
        """Merge another spanning-tree sufficient-statistic tuple."""
        self.count += suff_stat[0]
        self.edge_counts += suff_stat[1]
        return self

    def value(self) -> tuple[float, np.ndarray]:
        """Return total tree weight and symmetric edge-count matrix."""
        return self.count, self.edge_counts

    def from_value(self, x: tuple[float, np.ndarray]) -> "SpanningTreeAccumulator":
        """Replace accumulator contents from edge-count statistics."""
        self.count, self.edge_counts = x[0], np.asarray(x[1])
        self.dim = self.edge_counts.shape[0]
        return self

    def acc_to_encoder(self) -> "SpanningTreeDataEncoder":
        """Return the encoder used by this accumulator."""
        return SpanningTreeDataEncoder(dim=self.dim)


class SpanningTreeAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for SpanningTreeAccumulator."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = dim
        self.keys = keys

    def make(self) -> SpanningTreeAccumulator:
        """Create a fresh spanning-tree accumulator."""
        return SpanningTreeAccumulator(dim=self.dim, keys=self.keys)


class SpanningTreeEstimator(ParameterEstimator):
    """Estimate edge weights by matching empirical or smoothed tree edge marginals."""

    def __init__(
        self,
        dim: int,
        pseudo_count: float | None = None,
        max_steps: int = 500,
        learning_rate: float = 1.0,
        tol: float = 1.0e-7,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if dim is None or dim < 2:
            raise ValueError("SpanningTreeEstimator requires the number of nodes dim >= 2.")
        if pseudo_count is not None and pseudo_count < 0.0:
            raise ValueError("SpanningTreeEstimator requires a non-negative pseudo_count.")
        self.dim = int(dim)
        self.pseudo_count = pseudo_count
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.tol = tol
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> SpanningTreeAccumulatorFactory:
        """Return an accumulator factory for spanning-tree edge counts."""
        return SpanningTreeAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray]) -> SpanningTreeDistribution:
        """Estimate edge weights by matching target edge marginals."""
        count, edge_counts = suff_stat
        n = self.dim
        candidate = (edge_counts + edge_counts.T) > 0.0
        np.fill_diagonal(candidate, False)
        if count <= 0.0 or not np.any(candidate):
            return SpanningTreeDistribution(np.ones((n, n)) - np.eye(n), name=self.name, keys=self.keys)

        target = _smoothed_edge_target(edge_counts, count, candidate, self.pseudo_count)

        log_w = np.where(candidate, 0.0, -np.inf)
        weights = np.where(candidate, 1.0, 0.0)
        for _ in range(self.max_steps):
            marginals = _edge_marginals(weights)
            grad = (target - marginals) * candidate
            if np.max(np.abs(grad)) < self.tol:
                break
            log_w = np.where(
                candidate, np.clip(log_w + self.learning_rate * grad, _MIN_LOG_WEIGHT, _MAX_LOG_WEIGHT), -np.inf
            )
            # Fix the scale gauge (p(T) is invariant to a global weight rescale).
            log_w = np.where(candidate, log_w - np.mean(log_w[candidate]), -np.inf)
            weights = np.where(candidate, np.exp(log_w), 0.0)

        return SpanningTreeDistribution(weights, name=self.name, keys=self.keys)


class SpanningTreeDataEncoder(DataSequenceEncoder):
    """Encode a sequence of spanning trees (edge lists) into per-observation canonical edge arrays."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim

    def __str__(self) -> str:
        return "SpanningTreeDataEncoder"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SpanningTreeDataEncoder)

    def seq_encode(self, x: Sequence[Sequence[Sequence[int]]]) -> list[np.ndarray]:
        """Encode spanning trees as canonical sorted edge arrays."""
        dim = self.dim
        if dim is None:
            dim = max(int(np.max(np.asarray(tree))) for tree in x) + 1
        return [_canonical_edges(tree, dim) for tree in x]


def _canonical_edges(tree: Sequence[Sequence[int]], n: int) -> np.ndarray:
    """Validate that ``tree`` is a spanning tree of 0,...,n-1 and return its sorted (m, 2) edge array."""
    edges = np.asarray([(min(int(a), int(b)), max(int(a), int(b))) for a, b in tree], dtype=int)
    if edges.shape[0] != n - 1:
        raise ValueError("SpanningTreeDistribution requires exactly n-1 edges.")
    if np.any(edges[:, 0] == edges[:, 1]) or np.any(edges < 0) or np.any(edges >= n):
        raise ValueError("SpanningTreeDistribution edges must be valid node pairs without self-loops.")
    # Union-find connectivity / acyclicity check.
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seen = set()
    for a, b in edges:
        key = (int(a), int(b))
        if key in seen:
            raise ValueError("SpanningTreeDistribution edges must be distinct.")
        seen.add(key)
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            raise ValueError("SpanningTreeDistribution edges must form an acyclic spanning tree.")
        parent[ra] = rb
    if len({find(i) for i in range(n)}) != 1:
        raise ValueError("SpanningTreeDistribution edges must connect all n nodes.")
    return edges[np.lexsort((edges[:, 1], edges[:, 0]))]
