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

import operator
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
_COUNT_ATOL = 1.0e-8


class SpanningTreeFitError(RuntimeError):
    """Raised when spanning-tree statistics do not identify a valid fit."""


def _validated_dim(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Spanning-tree dimension must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError("Spanning-tree dimension must be an integer") from exc
    if result < 2:
        raise ValueError("Spanning-tree dimension must be at least two")
    return result


def _validated_nonnegative_scalar(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError(f"Spanning-tree {label} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Spanning-tree {label} must be a real scalar") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(
            f"Spanning-tree {label} must be finite and non-negative"
        )
    return result


def _validated_positive_scalar(value: object, *, label: str) -> float:
    result = _validated_nonnegative_scalar(value, label=label)
    if result == 0.0:
        raise ValueError(f"Spanning-tree {label} must be positive")
    return result


def _validated_positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"Spanning-tree {label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"Spanning-tree {label} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"Spanning-tree {label} must be positive")
    return result


def _validated_sample_size(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Spanning-tree sample size must be a non-negative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            "Spanning-tree sample size must be a non-negative integer"
        ) from exc
    if result < 0:
        raise ValueError("Spanning-tree sample size must be non-negative")
    return result


def _validated_weight_matrix(
    value: object,
    *,
    dim: int | None = None,
    label: str = "weight matrix",
) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Spanning-tree {label} must be numeric") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Spanning-tree {label} must be square")
    if dim is not None and matrix.shape != (dim, dim):
        raise ValueError(
            f"Spanning-tree {label} must have exact shape ({dim}, {dim})"
        )
    if matrix.shape[0] < 2:
        raise ValueError(f"Spanning-tree {label} must have dimension at least two")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(
            f"Spanning-tree {label} must contain finite non-negative values"
        )
    if not np.array_equal(matrix, matrix.T):
        raise ValueError(f"Spanning-tree {label} must be exactly symmetric")
    if np.any(np.diag(matrix) != 0.0):
        raise ValueError(f"Spanning-tree {label} must have an explicit zero diagonal")
    return matrix.copy()


def _validated_candidate_support(
    value: object,
    *,
    dim: int,
) -> np.ndarray:
    candidate = np.asarray(value)
    if candidate.shape != (dim, dim):
        raise ValueError(
            "Spanning-tree candidate support must have exact shape (%d, %d)"
            % (dim, dim)
        )
    if candidate.dtype.kind != "b":
        raise TypeError("Spanning-tree candidate support must be boolean")
    if not np.array_equal(candidate, candidate.T) or np.any(np.diag(candidate)):
        raise ValueError(
            "Spanning-tree candidate support must be symmetric with a false diagonal"
        )
    checked = candidate.copy()
    _log_partition(checked.astype(np.float64))
    return checked


def _validated_encoded_batch(
    value: object,
    *,
    dim: int,
) -> tuple[np.ndarray, ...]:
    if isinstance(value, np.ndarray) and value.ndim == 2:
        raise ValueError(
            "Spanning-tree batch scoring requires a sequence of tree edge arrays"
        )
    try:
        trees = tuple(value)
    except TypeError as exc:
        raise TypeError("Spanning-tree encoded batches must be iterable") from exc
    return tuple(_canonical_edges(tree, dim) for tree in trees)


def _validated_weights(value: object, rows: int) -> np.ndarray:
    try:
        weights = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Spanning-tree weights must be numeric") from exc
    if weights.shape != (rows,):
        raise ValueError(
            "Spanning-tree weights must have exact shape (%d,)" % rows
        )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(
            "Spanning-tree weights must be finite and non-negative"
        )
    return weights


def _validated_statistics(
    value: object,
    *,
    dim: int,
    candidate: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(
            "Spanning-tree sufficient statistics must contain two items"
        )
    count = _validated_nonnegative_scalar(value[0], label="total weight")
    try:
        edge_counts = np.asarray(value[1], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Spanning-tree edge counts must be numeric") from exc
    if edge_counts.shape != (dim, dim):
        raise ValueError(
            "Spanning-tree edge counts must have exact shape (%d, %d)"
            % (dim, dim)
        )
    if (
        np.any(~np.isfinite(edge_counts))
        or np.any(edge_counts < 0.0)
        or not np.array_equal(edge_counts, edge_counts.T)
        or np.any(np.diag(edge_counts) != 0.0)
    ):
        raise ValueError(
            "Spanning-tree edge counts must be finite, non-negative, symmetric, "
            "and have a zero diagonal"
        )
    tolerance = _COUNT_ATOL * max(1.0, count)
    if np.any(edge_counts > count + tolerance):
        raise ValueError(
            "Spanning-tree edge counts cannot exceed total tree weight"
        )
    expected_edge_mass = count * (dim - 1)
    if not np.isclose(
        float(edge_counts.sum() / 2.0),
        expected_edge_mass,
        rtol=0.0,
        atol=_COUNT_ATOL * max(1.0, expected_edge_mass),
    ):
        raise ValueError(
            "Spanning-tree edge counts do not contain exactly n-1 edges per tree"
        )
    if candidate is not None and np.any(edge_counts[~candidate] != 0.0):
        raise ValueError(
            "Spanning-tree edge counts include structurally forbidden edges"
        )
    return count, edge_counts.copy()


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
    marginals = weights * r_eff
    marginals = np.triu(marginals, k=1)
    return marginals + marginals.T


def _smoothed_edge_target(
    edge_counts: np.ndarray,
    count: float,
    candidate: np.ndarray,
    pseudo_count: float | None,
    prior_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return empirical edge marginals, optionally smoothed toward a candidate prior."""
    if count > 0.0:
        target = edge_counts / count
    else:
        target = np.zeros_like(edge_counts)
    if pseudo_count is not None and pseudo_count > 0.0:
        prior = (
            np.where(candidate, 1.0, 0.0)
            if prior_weights is None
            else prior_weights
        )
        prior_marginals = _edge_marginals(prior)
        target = (
            count * target + pseudo_count * prior_marginals
        ) / (count + pseudo_count)
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
        w = _validated_weight_matrix(weights)
        n = w.shape[0]
        self.weights = w
        self.dim = n
        with np.errstate(divide="ignore"):
            self.log_weights = np.log(w)
        self.log_z = _log_partition(w)
        self.name = name
        self.keys = keys
        self.weights.setflags(write=False)
        self.log_weights.setflags(write=False)

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
        """Return vectorized log-probabilities for a validated sequence of edge arrays."""
        trees = _validated_encoded_batch(x, dim=self.dim)
        return np.asarray(
            [
                self._edge_log_weight_sum(edges) - self.log_z
                for edges in trees
            ],
            dtype=float,
        )

    def sampler(self, seed: int | None = None) -> "SpanningTreeSampler":
        """Return a sampler for drawing spanning trees from this distribution."""
        return SpanningTreeSampler(self, seed)

    def enumerator(
        self,
        max_edge_subsets: int | None = _DEFAULT_MAX_ENUMERATION_SUBSETS,
    ) -> "SpanningTreeEnumerator":
        """Return a probability-ordered enumerator capped at ``max_edge_subsets`` returned trees."""
        return SpanningTreeEnumerator(self, max_edge_subsets=max_edge_subsets)

    def estimator(self, pseudo_count: float | None = None) -> "SpanningTreeEstimator":
        """Return an estimator that keeps the node count fixed at this distribution's n."""
        return SpanningTreeEstimator(
            dim=self.dim,
            pseudo_count=pseudo_count,
            candidate_support=self.weights > 0.0,
            prior_weights=self.weights,
            name=self.name,
            keys=self.keys,
        )

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
        super().__init__(dist)
        if max_edge_subsets is None:
            self.max_edge_subsets = None
        else:
            if isinstance(max_edge_subsets, (bool, np.bool_)):
                raise TypeError("max_edge_subsets must be a non-negative integer or None")
            try:
                checked_limit = operator.index(max_edge_subsets)
            except TypeError as exc:
                raise TypeError(
                    "max_edge_subsets must be a non-negative integer or None"
                ) from exc
            if checked_limit < 0:
                raise ValueError(
                    "max_edge_subsets must be a non-negative integer or None"
                )
            self.max_edge_subsets = checked_limit
        self.items_yielded = 0
        self.truncated = False
        self.termination_reason: str | None = None
        with np.errstate(divide="ignore"):
            cost = -dist.log_weights  # +inf where the edge weight is 0 (absent edge)
        self._gen = k_best_spanning_trees(cost)
        self._log_z = dist.log_z

    def __next__(self) -> tuple[list[tuple[int, int]], float]:
        if (
            self.max_edge_subsets is not None
            and self.items_yielded >= self.max_edge_subsets
        ):
            self.truncated = True
            self.termination_reason = "item_budget_exhausted"
            raise StopIteration
        try:
            total, tree = next(self._gen)
        except StopIteration:
            self.termination_reason = "support_exhausted"
            raise
        canon = _canonical_edges(tree, self.dist.dim)  # same canonical edge representation as log_density
        value = [(int(a), int(b)) for a, b in canon]
        self.items_yielded += 1
        return value, float(-total - self._log_z)

    @property
    def receipt(self) -> dict[str, object]:
        """Return enumeration progress and termination metadata."""
        return {
            "items_yielded": self.items_yielded,
            "item_budget": self.max_edge_subsets,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
        }


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
        return [self._sample_one() for _ in range(_validated_sample_size(size))]


class SpanningTreeAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted edge-appearance counts (the sufficient statistic for the tree weights)."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = _validated_dim(dim)
        self.edge_counts = np.zeros((self.dim, self.dim))
        self.count = 0.0
        self.keys = keys

    def update(self, x: Sequence[Sequence[int]], weight: float, estimate: SpanningTreeDistribution | None) -> None:
        """Accumulate weighted edge appearances for one spanning tree."""
        edges = _canonical_edges(x, self.dim)
        checked_weight = _validated_nonnegative_scalar(weight, label="weight")
        self.edge_counts[edges[:, 0], edges[:, 1]] += checked_weight
        self.edge_counts[edges[:, 1], edges[:, 0]] += checked_weight
        self.count += checked_weight

    def initialize(self, x: Sequence[Sequence[int]], weight: float, rng: RandomState | None) -> None:
        """Initialize edge-count statistics from one spanning tree."""
        self.update(x, weight, None)

    def seq_update(
        self, x: Sequence[np.ndarray], weights: np.ndarray, estimate: SpanningTreeDistribution | None
    ) -> None:
        """Accumulate edge appearances from encoded spanning trees."""
        trees = _validated_encoded_batch(x, dim=self.dim)
        checked_weights = _validated_weights(weights, len(trees))
        for edges, w in zip(trees, checked_weights):
            self.edge_counts[edges[:, 0], edges[:, 1]] += w
            self.edge_counts[edges[:, 1], edges[:, 0]] += w
        self.count += float(np.sum(checked_weights, dtype=np.float64))

    def seq_initialize(self, x: Sequence[np.ndarray], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize edge-count statistics from encoded spanning trees."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray]) -> "SpanningTreeAccumulator":
        """Merge another spanning-tree sufficient-statistic tuple."""
        count, edge_counts = _validated_statistics(suff_stat, dim=self.dim)
        combined_count, combined_edges = _validated_statistics(
            (self.count + count, self.edge_counts + edge_counts),
            dim=self.dim,
        )
        self.count = combined_count
        self.edge_counts = combined_edges
        return self

    def value(self) -> tuple[float, np.ndarray]:
        """Return total tree weight and symmetric edge-count matrix."""
        return self.count, self.edge_counts.copy()

    def from_value(self, x: tuple[float, np.ndarray]) -> "SpanningTreeAccumulator":
        """Replace accumulator contents from edge-count statistics."""
        self.count, self.edge_counts = _validated_statistics(x, dim=self.dim)
        return self

    def scale(self, c: float) -> "SpanningTreeAccumulator":
        """Scale total and edge-count sufficient statistics."""
        checked_scale = _validated_nonnegative_scalar(c, label="scale")
        self.count *= checked_scale
        self.edge_counts *= checked_scale
        return self

    def acc_to_encoder(self) -> "SpanningTreeDataEncoder":
        """Return the encoder used by this accumulator."""
        return SpanningTreeDataEncoder(dim=self.dim)


class SpanningTreeAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for SpanningTreeAccumulator."""

    def __init__(self, dim: int, keys: str | None = None) -> None:
        self.dim = _validated_dim(dim)
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
        candidate_support: np.ndarray | None = None,
        prior_weights: np.ndarray | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.dim = _validated_dim(dim)
        self.pseudo_count = (
            None
            if pseudo_count is None
            else _validated_nonnegative_scalar(
                pseudo_count,
                label="pseudo-count",
            )
        )
        self.max_steps = _validated_positive_integer(
            max_steps,
            label="max_steps",
        )
        self.learning_rate = _validated_positive_scalar(
            learning_rate,
            label="learning_rate",
        )
        self.tol = _validated_positive_scalar(tol, label="tolerance")
        self.candidate_support = (
            np.ones((self.dim, self.dim), dtype=bool)
            ^ np.eye(self.dim, dtype=bool)
            if candidate_support is None
            else _validated_candidate_support(
                candidate_support,
                dim=self.dim,
            )
        )
        if prior_weights is None:
            self.prior_weights = np.where(
                self.candidate_support,
                1.0,
                0.0,
            )
        else:
            checked_prior = _validated_weight_matrix(
                prior_weights,
                dim=self.dim,
                label="prior weight matrix",
            )
            if not np.array_equal(
                checked_prior > 0.0,
                self.candidate_support,
            ):
                raise ValueError(
                    "Spanning-tree prior weights must be positive exactly on candidate support"
                )
            _log_partition(checked_prior)
            self.prior_weights = checked_prior
        self.name = name
        self.keys = keys
        self.candidate_support.setflags(write=False)
        self.prior_weights.setflags(write=False)

    def accumulator_factory(self) -> SpanningTreeAccumulatorFactory:
        """Return an accumulator factory for spanning-tree edge counts."""
        return SpanningTreeAccumulatorFactory(dim=self.dim, keys=self.keys)

    def estimate(self, nobs: float | None, suff_stat: tuple[float, np.ndarray]) -> SpanningTreeDistribution:
        """Estimate edge weights by matching target edge marginals."""
        count, edge_counts = _validated_statistics(
            suff_stat,
            dim=self.dim,
            candidate=self.candidate_support,
        )
        prior_weight = 0.0 if self.pseudo_count is None else self.pseudo_count
        if count == 0.0 and prior_weight == 0.0:
            raise SpanningTreeFitError(
                "Spanning-tree fitting requires positive observation or prior weight"
            )

        candidate = self.candidate_support
        target = _smoothed_edge_target(
            edge_counts,
            count,
            candidate,
            self.pseudo_count,
            prior_weights=self.prior_weights,
        )

        with np.errstate(divide="ignore"):
            log_w = np.where(
                candidate,
                np.log(self.prior_weights),
                -np.inf,
            )
        log_w = np.where(
            candidate,
            log_w - np.mean(log_w[candidate]),
            -np.inf,
        )
        weights = np.where(candidate, np.exp(log_w), 0.0)
        converged = False
        residual = np.inf
        iterations = 0
        for step in range(1, self.max_steps + 1):
            marginals = _edge_marginals(weights)
            grad = (target - marginals) * candidate
            residual = float(np.max(np.abs(grad[candidate])))
            iterations = step
            if residual < self.tol:
                converged = True
                break
            log_w = np.where(
                candidate, np.clip(log_w + self.learning_rate * grad, _MIN_LOG_WEIGHT, _MAX_LOG_WEIGHT), -np.inf
            )
            # Fix the scale gauge (p(T) is invariant to a global weight rescale).
            log_w = np.where(candidate, log_w - np.mean(log_w[candidate]), -np.inf)
            weights = np.where(candidate, np.exp(log_w), 0.0)
        if not converged:
            final_grad = (target - _edge_marginals(weights)) * candidate
            residual = float(np.max(np.abs(final_grad[candidate])))
            converged = residual < self.tol

        result = SpanningTreeDistribution(
            weights,
            name=self.name,
            keys=self.keys,
        )
        result.fit_metadata = {
            "converged": converged,
            "solver": "projected-log-weight-gradient",
            "iterations": iterations,
            "max_steps": self.max_steps,
            "residual": residual,
            "tolerance": self.tol,
            "regularized": prior_weight > 0.0,
            "candidate_edges": int(candidate.sum() // 2),
            "repairs": (),
        }
        return result


class SpanningTreeDataEncoder(DataSequenceEncoder):
    """Encode a sequence of spanning trees (edge lists) into per-observation canonical edge arrays."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = None if dim is None else _validated_dim(dim)

    def __str__(self) -> str:
        return "SpanningTreeDataEncoder(dim=%r)" % self.dim

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SpanningTreeDataEncoder) and self.dim == other.dim

    def seq_encode(self, x: Sequence[Sequence[Sequence[int]]]) -> list[np.ndarray]:
        """Encode spanning trees as canonical sorted edge arrays."""
        try:
            trees = tuple(x)
        except TypeError as exc:
            raise TypeError("Spanning-tree batches must be iterable") from exc
        dim = self.dim
        if dim is None:
            if not trees:
                raise ValueError(
                    "Cannot infer spanning-tree dimension from an empty batch"
                )
            maximum = -1
            for tree in trees:
                for edge in tree:
                    if not isinstance(edge, (tuple, list, np.ndarray)) or len(edge) != 2:
                        raise ValueError(
                            "Spanning-tree edges must be endpoint pairs"
                        )
                    for endpoint in edge:
                        if isinstance(endpoint, (bool, np.bool_)):
                            raise TypeError(
                                "Spanning-tree endpoints must be exact integers"
                            )
                        try:
                            checked = operator.index(endpoint)
                        except TypeError as exc:
                            raise TypeError(
                                "Spanning-tree endpoints must be exact integers"
                            ) from exc
                        maximum = max(maximum, checked)
            dim = _validated_dim(maximum + 1)
        return list(_validated_encoded_batch(trees, dim=dim))

    def row_count(self, x: Sequence[np.ndarray]) -> int:
        """Return the number of validated encoded spanning trees."""
        if self.dim is None:
            return len(tuple(x))
        return len(_validated_encoded_batch(x, dim=self.dim))


def _canonical_edges(tree: Sequence[Sequence[int]], n: int) -> np.ndarray:
    """Validate that ``tree`` is a spanning tree of 0,...,n-1 and return its sorted (m, 2) edge array."""
    checked_dim = _validated_dim(n)
    try:
        raw_edges = tuple(tree)
    except TypeError as exc:
        raise TypeError("Spanning-tree edges must be iterable") from exc
    canonical = []
    for edge in raw_edges:
        if not isinstance(edge, (tuple, list, np.ndarray)) or len(edge) != 2:
            raise ValueError("Spanning-tree edges must be endpoint pairs")
        endpoints = []
        for endpoint in edge:
            if isinstance(endpoint, (bool, np.bool_)):
                raise TypeError("Spanning-tree endpoints must be exact integers")
            try:
                endpoints.append(operator.index(endpoint))
            except TypeError as exc:
                raise TypeError(
                    "Spanning-tree endpoints must be exact integers"
                ) from exc
        canonical.append(
            (min(endpoints[0], endpoints[1]), max(endpoints[0], endpoints[1]))
        )
    edges = np.asarray(canonical, dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Spanning-tree edges must be endpoint pairs")
    if edges.shape[0] != checked_dim - 1:
        raise ValueError("SpanningTreeDistribution requires exactly n-1 edges.")
    if (
        np.any(edges[:, 0] == edges[:, 1])
        or np.any(edges < 0)
        or np.any(edges >= checked_dim)
    ):
        raise ValueError("SpanningTreeDistribution edges must be valid node pairs without self-loops.")
    # Union-find connectivity / acyclicity check.
    parent = list(range(checked_dim))

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
    if len({find(i) for i in range(checked_dim)}) != 1:
        raise ValueError("SpanningTreeDistribution edges must connect all n nodes.")
    return edges[np.lexsort((edges[:, 1], edges[:, 0]))]
