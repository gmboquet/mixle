"""Random dot-product graph distributions for binary undirected graphs.

Data type: a binary undirected graph on n nodes (a square adjacency matrix, a NetworkX-like graph, or
any mapping accepted by ``GraphDataEncoder``).

The RDPG is a latent-position graph model: each node i carries a latent vector ``x_i`` in R^d, and
edges are independent Bernoulli draws with probability equal to the dot product of the endpoints'
positions,

    P(A_ij = 1) = clip(<x_i, x_j>, 0, 1).

This generalizes Erdos-Renyi (rank-1, constant positions) and captures community / homophily structure
through the geometry of the positions. Sampling draws independent Bernoulli edges from the probability
matrix ``X X^T``. Estimation uses Adjacency Spectral Embedding (ASE): the latent positions are the top-d
scaled eigenvectors of the mean adjacency matrix, the standard consistent RDPG estimator.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

if TYPE_CHECKING:
    from mixle.data.sources.graph_source import GraphDataEncoder, GraphObservation

# mixle.data.sources.graph_source's own top-level import of mixle.stats.compute.pdist.DataSequenceEncoder
# forces mixle.stats's package __init__ to run first, which eagerly imports this very module -- so a
# module-level `from mixle.data.sources.graph_source import ...` here is circular whenever
# mixle.data.sources.graph_source is the entry point (e.g. `import mixle.data.sources.graph_source`
# directly, before mixle.stats has been warmed by any other path): graph_source would still be
# mid-import (paused inside its own DataSequenceEncoder import, above this module in the same chain),
# so none of GraphDataEncoder/GraphObservation/_extract_observation would exist as attributes yet.
# Every usage below is either purely a type annotation (deferred to a string by the `from __future__
# import annotations` above) or a call inside a function/method body, so the imports are deferred to
# call time, once every module in the cycle has finished loading normally -- mirroring the
# erdos_renyi_graph.py fix for the same shape of cycle.

class RandomDotProductGraphDistribution(SequenceEncodableProbabilityDistribution):
    """Random Dot Product Graph over n nodes with d-dimensional latent positions X (edge prob X X^T)."""

    @classmethod
    def compute_capabilities(cls):
        """Return compute-backend metadata for RDPG log-density evaluation."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic_object")

    def __init__(
        self,
        positions: Sequence[Sequence[float]] | np.ndarray,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """Create a random dot-product graph distribution.

        Args:
            positions (Union[Sequence[Sequence[float]], np.ndarray]): n-by-d latent positions; node i
                is row i. Edge probability between i and j is clip(<x_i, x_j>, 0, 1).
            name (Optional[str]): Optional distribution name.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            positions (np.ndarray): n-by-d latent-position matrix.
            num_nodes (int): Number of nodes n.
            dim (int): Latent dimension d.
            probs (np.ndarray): n-by-n edge probability matrix (clipped, zero diagonal).

        """
        x = np.array(positions, dtype=float, copy=True)
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("RandomDotProductGraphDistribution requires an n-by-d position matrix.")
        if not np.all(np.isfinite(x)):
            raise ValueError("RandomDotProductGraphDistribution requires finite latent positions.")
        x.setflags(write=False)
        self.positions = x
        self.num_nodes = x.shape[0]
        self.dim = x.shape[1]
        probs = np.clip(x @ x.T, 0.0, 1.0)
        np.fill_diagonal(probs, 0.0)
        probs.setflags(write=False)
        self.probs = probs
        self._log_p = np.full_like(probs, -np.inf)
        self._log_1mp = np.full_like(probs, -np.inf)
        positive = probs > 0.0
        below_one = probs < 1.0
        self._log_p[positive] = np.log(probs[positive])
        self._log_1mp[below_one] = np.log1p(-probs[below_one])
        self._log_p.setflags(write=False)
        self._log_1mp.setflags(write=False)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return a constructor-style representation of the random dot-product graph distribution."""
        return "RandomDotProductGraphDistribution(%s, name=%s, keys=%s)" % (
            repr([[float(v) for v in row] for row in self.positions]),
            repr(self.name),
            repr(self.keys),
        )

    def edge_marginals(self) -> np.ndarray:
        """Return the n-by-n matrix of edge probabilities P(A_ij = 1)."""
        return self.probs.copy()

    def _graph_log_density(self, adjacency: np.ndarray) -> float:
        from mixle.data.sources.graph_source import _validate_graph_constraints

        a = np.asarray(adjacency, dtype=float)
        if a.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("RandomDotProductGraphDistribution observation size does not match the positions.")
        # RDPG is always undirected with no self-loops (no directed/self_loops constructor flags), and
        # the mask below only ever reads the strict upper triangle -- an asymmetric entry or a
        # nonzero diagonal in `a` would otherwise be silently ignored rather than rejected.
        _validate_graph_constraints(a, directed=False, self_loops=False)
        mask = np.triu(np.ones_like(a, dtype=bool), 1)  # undirected, no self-loops
        present = a[mask] == 1.0
        terms = np.where(present, self._log_p[mask], self._log_1mp[mask])
        return float(np.sum(terms))

    def density(self, x: Any) -> float:
        """Return the probability of a graph x."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Any) -> float:
        """Return the log-probability of a binary undirected graph x."""
        from mixle.data.sources.graph_source import _extract_observation

        return self._graph_log_density(_extract_observation(x).adjacency)

    def seq_log_density(self, x: Sequence[GraphObservation]) -> np.ndarray:
        """Return vectorized log-probabilities for a sequence of graph observations."""
        from mixle.data.sources.graph_source import _extract_observation

        return np.asarray([self._graph_log_density(_extract_observation(o).adjacency) for o in x], dtype=np.float64)

    def backend_seq_log_density(self, x: Sequence[GraphObservation], engine: Any) -> Any:
        """Engine-routed RDPG edge log-likelihood (reduction runs on the active engine)."""
        from mixle.data.sources.graph_source import _extract_observation, _validate_graph_constraints

        mask = np.triu(np.ones((self.num_nodes, self.num_nodes), dtype=bool), 1)
        log_p = engine.asarray(self._log_p[mask])
        log_1mp = engine.asarray(self._log_1mp[mask])
        adjacencies = []
        for o in x:
            adj = _extract_observation(o).adjacency
            _validate_graph_constraints(adj, directed=False, self_loops=False)
            if adj.shape != (self.num_nodes, self.num_nodes):
                raise ValueError("RandomDotProductGraphDistribution observation size does not match the positions.")
            adjacencies.append(adj[mask])
        rows = np.asarray(adjacencies, dtype=np.float64)
        a = engine.asarray(rows)
        return engine.sum(engine.where(a == engine.asarray(1.0), log_p[None, :], log_1mp[None, :]), axis=1)

    def sampler(self, seed: int | None = None) -> RandomDotProductGraphSampler:
        """Return a sampler for drawing graphs from this distribution."""
        return RandomDotProductGraphSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> RandomDotProductGraphEstimator:
        """Return an ASE estimator that keeps the latent dimension fixed at this distribution's d."""
        if pseudo_count is not None:
            raise ValueError("RDPG estimation does not support pseudo_count.")
        return RandomDotProductGraphEstimator(
            dim=self.dim,
            num_nodes=self.num_nodes,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> GraphDataEncoder:
        """Return the shared graph data encoder."""
        from mixle.data.sources.graph_source import GraphDataEncoder

        return GraphDataEncoder(directed=False)


class RandomDotProductGraphSampler(DistributionSampler):
    """Sample binary undirected graphs from an RDPG (independent Bernoulli edges with prob X X^T)."""

    def __init__(self, dist: RandomDotProductGraphDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample_graph(self) -> np.ndarray:
        """Draw one symmetric binary adjacency matrix with no self-loops."""
        n = self.dist.num_nodes
        draws = (self.rng.rand(n, n) < self.dist.probs).astype(np.int8)
        upper = np.triu(draws, 1)
        return upper + upper.T

    def sample(self, size: int | None = None, *, batched: bool = True) -> np.ndarray | list[np.ndarray]:
        """Draw graphs (adjacency matrices); a single matrix when size is None."""
        if size is None:
            return self.sample_graph()
        from mixle.data.sources.graph_source import _require_exact_int

        sample_size = _require_exact_int(size, "size")
        if sample_size < 0:
            raise ValueError("size must be non-negative.")
        return [self.sample_graph() for _ in range(sample_size)]


class RandomDotProductGraphAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted sum of adjacency matrices (the sufficient statistic for ASE)."""

    def __init__(self, num_nodes: int | None = None, keys: str | None = None) -> None:
        from mixle.data.sources.graph_source import _require_exact_int

        self.num_nodes = None if num_nodes is None else _require_exact_int(num_nodes, "num_nodes")
        if self.num_nodes is not None and self.num_nodes < 1:
            raise ValueError("num_nodes must be positive when specified.")
        self.adj_sum: np.ndarray | None = None
        self.count = 0.0
        self.keys = keys

    def _add(self, adjacency: np.ndarray, weight: float) -> None:
        from mixle.data.sources.graph_source import _validate_graph_constraints

        a = np.asarray(adjacency, dtype=float)
        _validate_graph_constraints(a, directed=False, self_loops=False)
        if self.num_nodes is not None and a.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("RDPG observation size does not match the configured number of nodes.")
        checked_weight = float(weight)
        if not np.isfinite(checked_weight) or checked_weight < 0.0:
            raise ValueError("weight must be finite and non-negative.")
        if self.adj_sum is None:
            self.adj_sum = np.zeros_like(a)
            if self.num_nodes is None:
                self.num_nodes = a.shape[0]
        elif a.shape != self.adj_sum.shape:
            raise ValueError("all RDPG observations must have the same shape.")
        self.adj_sum += checked_weight * a
        self.count += checked_weight

    def update(self, x: Any, weight: float, estimate: RandomDotProductGraphDistribution | None) -> None:
        """Accumulate the weighted adjacency matrix for one graph observation."""
        from mixle.data.sources.graph_source import _extract_observation

        self._add(_extract_observation(x).adjacency, weight)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics with one weighted graph."""
        self.update(x, weight, None)

    def seq_update(
        self, x: Sequence[GraphObservation], weights: np.ndarray, estimate: RandomDotProductGraphDistribution | None
    ) -> None:
        """Accumulate weighted adjacency matrices for encoded graph observations."""
        from mixle.data.sources.graph_source import _extract_observation

        checked_weights = np.asarray(weights, dtype=np.float64)
        if checked_weights.ndim != 1 or len(checked_weights) != len(x):
            raise ValueError("weights must be a one-dimensional array aligned with the graph batch.")
        if np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        observations = [_extract_observation(obs).adjacency for obs in x]
        # Validate the whole batch before mutating state.
        expected = self.num_nodes
        for adjacency in observations:
            from mixle.data.sources.graph_source import _validate_graph_constraints

            _validate_graph_constraints(adjacency, directed=False, self_loops=False)
            if expected is None:
                expected = adjacency.shape[0]
            if adjacency.shape != (expected, expected):
                raise ValueError("all RDPG observations must have the same shape.")
        for adjacency, weight in zip(observations, checked_weights):
            self._add(adjacency, float(weight))

    def seq_initialize(self, x: Sequence[GraphObservation], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize the sufficient statistics from encoded graph observations."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray | None]) -> RandomDotProductGraphAccumulator:
        """Merge serialized adjacency-sum statistics into this accumulator."""
        count, adj_sum = _validate_rdpg_statistics(suff_stat, num_nodes=self.num_nodes)
        if self.adj_sum is not None and adj_sum is not None and self.adj_sum.shape != adj_sum.shape:
            raise ValueError("cannot combine RDPG statistics with different graph sizes.")
        self.count += count
        if adj_sum is not None:
            if self.adj_sum is None:
                self.adj_sum = adj_sum.copy()
                self.num_nodes = adj_sum.shape[0]
            else:
                self.adj_sum += adj_sum
        return self

    def value(self) -> tuple[float, np.ndarray | None]:
        """Return the total weight and weighted adjacency-matrix sum."""
        return self.count, None if self.adj_sum is None else self.adj_sum.copy()

    def from_value(self, x: tuple[float, np.ndarray | None]) -> RandomDotProductGraphAccumulator:
        """Restore the accumulator from serialized adjacency-sum statistics."""
        self.count, self.adj_sum = _validate_rdpg_statistics(x, num_nodes=self.num_nodes)
        if self.adj_sum is not None:
            self.adj_sum = self.adj_sum.copy()
            self.num_nodes = self.adj_sum.shape[0]
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator under its configured sharing key."""
        if self.keys is None:
            return
        if self.keys in stats_dict:
            other = stats_dict[self.keys]
            if not isinstance(other, RandomDotProductGraphAccumulator):
                raise ValueError("shared RDPG key is bound to incompatible statistics.")
            other.combine(self.value())
        else:
            stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator from its configured sharing key."""
        if self.keys is not None and self.keys in stats_dict:
            other = stats_dict[self.keys]
            if not isinstance(other, RandomDotProductGraphAccumulator):
                raise ValueError("shared RDPG key is bound to incompatible statistics.")
            self.from_value(other.value())

    def acc_to_encoder(self) -> GraphDataEncoder:
        """Return the undirected graph encoder used by the accumulator."""
        from mixle.data.sources.graph_source import GraphDataEncoder

        return GraphDataEncoder(directed=False)


class RandomDotProductGraphAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for RandomDotProductGraphAccumulator."""

    def __init__(self, num_nodes: int | None = None, keys: str | None = None) -> None:
        self.num_nodes = num_nodes
        self.keys = keys

    def make(self) -> RandomDotProductGraphAccumulator:
        """Create an empty RDPG accumulator."""
        return RandomDotProductGraphAccumulator(num_nodes=self.num_nodes, keys=self.keys)


class RandomDotProductGraphEstimator(ParameterEstimator):
    """Adjacency Spectral Embedding estimator for the RDPG latent positions."""

    def __init__(
        self,
        dim: int,
        num_nodes: int | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        from mixle.data.sources.graph_source import _require_exact_int

        if isinstance(dim, (bool, np.bool_)):
            raise ValueError("RandomDotProductGraphEstimator requires an exact integer dim >= 1.")
        checked_dim = _require_exact_int(dim, "dim")
        if checked_dim < 1:
            raise ValueError("RandomDotProductGraphEstimator requires the latent dimension dim >= 1.")
        self.dim = checked_dim
        self.num_nodes = None if num_nodes is None else _require_exact_int(num_nodes, "num_nodes")
        if self.num_nodes is not None and self.num_nodes < 1:
            raise ValueError("num_nodes must be positive when specified.")
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> RandomDotProductGraphAccumulatorFactory:
        """Return a factory for RDPG sufficient-statistic accumulators."""
        return RandomDotProductGraphAccumulatorFactory(num_nodes=self.num_nodes, keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, np.ndarray | None]
    ) -> RandomDotProductGraphDistribution:
        """Estimate latent positions from the mean adjacency matrix using ASE."""
        count, adj_sum = _validate_rdpg_statistics(suff_stat, num_nodes=self.num_nodes)
        if adj_sum is None or count <= 0.0:
            raise ValueError("cannot estimate an RDPG without positive-weight graph evidence.")

        mean_adj = 0.5 * (adj_sum + adj_sum.T) / count  # symmetric mean adjacency
        n = mean_adj.shape[0]
        d = min(self.dim, n)
        # Diagonal augmentation (Scheinerman): the diagonal of X X^T is unobserved (no self-loops), so
        # impute it from the off-diagonal row means before the spectral embedding to remove ASE bias.
        np.fill_diagonal(mean_adj, 0.0)
        if n > 1:
            np.fill_diagonal(mean_adj, mean_adj.sum(axis=1) / (n - 1))
        # PSD ASE: use the largest positive eigenvalues. Negative eigenvalues belong to an
        # indefinite latent-space model and must not consume dimensions in an RDPG embedding.
        eigvals, eigvecs = np.linalg.eigh(mean_adj)
        positive = np.flatnonzero(eigvals > 0.0)
        order = positive[np.argsort(eigvals[positive])[::-1]][:d]
        positions = eigvecs[:, order] * np.sqrt(eigvals[order])[None, :]
        if positions.shape[1] < self.dim:
            positions = np.hstack([positions, np.zeros((n, self.dim - positions.shape[1]))])
        return RandomDotProductGraphDistribution(positions, name=self.name, keys=self.keys)


def _validate_rdpg_statistics(
    suff_stat: Any,
    *,
    num_nodes: int | None,
) -> tuple[float, np.ndarray | None]:
    """Own and validate an RDPG ``(weight, weighted_adjacency_sum)`` record."""
    if not isinstance(suff_stat, (tuple, list)) or len(suff_stat) != 2:
        raise ValueError("RDPG statistics must be (count, adjacency_sum).")
    count = float(suff_stat[0])
    if not np.isfinite(count) or count < 0.0:
        raise ValueError("RDPG statistic count must be finite and non-negative.")
    if suff_stat[1] is None:
        if count != 0.0:
            raise ValueError("positive RDPG statistic count requires an adjacency sum.")
        return count, None
    adjacency = np.array(suff_stat[1], dtype=np.float64, copy=True)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("RDPG adjacency-sum statistic must be square.")
    if num_nodes is not None and adjacency.shape != (num_nodes, num_nodes):
        raise ValueError("RDPG adjacency-sum size does not match configured num_nodes.")
    if np.any(~np.isfinite(adjacency)) or np.any(adjacency < 0.0) or np.any(adjacency > count):
        raise ValueError("RDPG adjacency sums must be finite values in [0, count].")
    if not np.array_equal(adjacency, adjacency.T) or np.any(np.diag(adjacency) != 0.0):
        raise ValueError("RDPG adjacency sums must be symmetric with a zero diagonal.")
    if count == 0.0 and np.any(adjacency != 0.0):
        raise ValueError("zero-weight RDPG statistics cannot contain edges.")
    return count, adjacency
