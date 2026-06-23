"""Create, estimate, and sample from a Random Dot Product Graph (RDPG) distribution.

Defines the RandomDotProductGraphDistribution, RandomDotProductGraphSampler,
RandomDotProductGraphAccumulatorFactory, RandomDotProductGraphAccumulator,
RandomDotProductGraphEstimator, and reuses the shared GraphDataEncoder for use with pysparkplug.

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

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from pysp.data.sources.graph_source import GraphDataEncoder, GraphObservation, _extract_observation
from pysp.stats.compute.pdist import (
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)

_EPS = 1.0e-12


class RandomDotProductGraphDistribution(SequenceEncodableProbabilityDistribution):
    """Random Dot Product Graph over n nodes with d-dimensional latent positions X (edge prob X X^T)."""

    @classmethod
    def compute_capabilities(cls):
        from pysp.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic_object")

    def __init__(
        self,
        positions: Sequence[Sequence[float]] | np.ndarray,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        """RandomDotProductGraphDistribution object.

        Args:
            positions (Union[Sequence[Sequence[float]], np.ndarray]): n-by-d latent positions; node i
                is row i. Edge probability between i and j is clip(<x_i, x_j>, 0, 1).
            name (Optional[str]): Optional name for object instance.
            keys (Optional[str]): Optional key for merging sufficient statistics.

        Attributes:
            positions (np.ndarray): n-by-d latent-position matrix.
            num_nodes (int): Number of nodes n.
            dim (int): Latent dimension d.
            probs (np.ndarray): n-by-n edge probability matrix (clipped, zero diagonal).

        """
        x = np.asarray(positions, dtype=float)
        if x.ndim != 2 or x.shape[0] < 1:
            raise ValueError("RandomDotProductGraphDistribution requires an n-by-d position matrix.")
        if not np.all(np.isfinite(x)):
            raise ValueError("RandomDotProductGraphDistribution requires finite latent positions.")
        self.positions = x
        self.num_nodes = x.shape[0]
        self.dim = x.shape[1]
        probs = np.clip(x @ x.T, _EPS, 1.0 - _EPS)
        np.fill_diagonal(probs, 0.0)
        self.probs = probs
        self._log_p = np.log(np.clip(probs, _EPS, 1.0 - _EPS))
        self._log_1mp = np.log(np.clip(1.0 - probs, _EPS, 1.0 - _EPS))
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        """Return string representation of RandomDotProductGraphDistribution object."""
        return "RandomDotProductGraphDistribution(%s, name=%s, keys=%s)" % (
            repr([[float(v) for v in row] for row in self.positions]),
            repr(self.name),
            repr(self.keys),
        )

    def edge_marginals(self) -> np.ndarray:
        """Return the n-by-n matrix of edge probabilities P(A_ij = 1)."""
        return self.probs

    def _graph_log_density(self, adjacency: np.ndarray) -> float:
        a = np.asarray(adjacency, dtype=float)
        if a.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("RandomDotProductGraphDistribution observation size does not match the positions.")
        mask = np.triu(np.ones_like(a, dtype=bool), 1)  # undirected, no self-loops
        return float(np.sum(a[mask] * self._log_p[mask] + (1.0 - a[mask]) * self._log_1mp[mask]))

    def density(self, x: Any) -> float:
        """Return the probability of a graph x."""
        return float(np.exp(self.log_density(x)))

    def log_density(self, x: Any) -> float:
        """Return the log-probability of a binary undirected graph x."""
        return self._graph_log_density(_extract_observation(x).adjacency)

    def seq_log_density(self, x: Sequence[GraphObservation]) -> np.ndarray:
        """Return vectorized log-probabilities for a sequence of graph observations."""
        return np.asarray([self._graph_log_density(_extract_observation(o).adjacency) for o in x], dtype=np.float64)

    def backend_seq_log_density(self, x: Sequence[GraphObservation], engine: Any) -> Any:
        """Engine-routed RDPG edge log-likelihood (reduction runs on the active engine)."""
        mask = np.triu(np.ones((self.num_nodes, self.num_nodes), dtype=bool), 1)
        log_p = engine.asarray(self._log_p[mask])
        log_1mp = engine.asarray(self._log_1mp[mask])
        rows = np.asarray(
            [_extract_observation(o).adjacency[mask] for o in x],
            dtype=np.float64,
        )
        a = engine.asarray(rows)
        return engine.sum(a * log_p[None, :] + (engine.asarray(1.0) - a) * log_1mp[None, :], axis=1)

    def sampler(self, seed: int | None = None) -> "RandomDotProductGraphSampler":
        """Return a sampler for drawing graphs from this distribution."""
        return RandomDotProductGraphSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "RandomDotProductGraphEstimator":
        """Return an ASE estimator that keeps the latent dimension fixed at this distribution's d."""
        return RandomDotProductGraphEstimator(dim=self.dim, name=self.name, keys=self.keys)

    def dist_to_encoder(self) -> GraphDataEncoder:
        """Return the shared graph data encoder."""
        return GraphDataEncoder(directed=False)


class RandomDotProductGraphSampler(DistributionSampler):
    """Sample binary undirected graphs from an RDPG (independent Bernoulli edges with prob X X^T)."""

    def __init__(self, dist: RandomDotProductGraphDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample_graph(self) -> np.ndarray:
        n = self.dist.num_nodes
        draws = (self.rng.rand(n, n) < self.dist.probs).astype(np.int8)
        upper = np.triu(draws, 1)
        return upper + upper.T

    def sample(self, size: int | None = None) -> np.ndarray | list[np.ndarray]:
        """Draw graphs (adjacency matrices); a single matrix when size is None."""
        if size is None:
            return self.sample_graph()
        return [self.sample_graph() for _ in range(int(size))]


class RandomDotProductGraphAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate the weighted sum of adjacency matrices (the sufficient statistic for ASE)."""

    def __init__(self, keys: str | None = None) -> None:
        self.adj_sum: np.ndarray | None = None
        self.count = 0.0
        self.keys = keys

    def _add(self, adjacency: np.ndarray, weight: float) -> None:
        a = np.asarray(adjacency, dtype=float)
        if self.adj_sum is None:
            self.adj_sum = np.zeros_like(a)
        self.adj_sum += weight * a
        self.count += weight

    def update(self, x: Any, weight: float, estimate: RandomDotProductGraphDistribution | None) -> None:
        self._add(_extract_observation(x).adjacency, weight)

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(
        self, x: Sequence[GraphObservation], weights: np.ndarray, estimate: RandomDotProductGraphDistribution | None
    ) -> None:
        for obs, w in zip(x, weights):
            self._add(_extract_observation(obs).adjacency, float(w))

    def seq_initialize(self, x: Sequence[GraphObservation], weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, np.ndarray | None]) -> "RandomDotProductGraphAccumulator":
        count, adj_sum = suff_stat
        self.count += count
        if adj_sum is not None:
            if self.adj_sum is None:
                self.adj_sum = np.asarray(adj_sum, dtype=float).copy()
            else:
                self.adj_sum += adj_sum
        return self

    def value(self) -> tuple[float, np.ndarray | None]:
        return self.count, self.adj_sum

    def from_value(self, x: tuple[float, np.ndarray | None]) -> "RandomDotProductGraphAccumulator":
        self.count, self.adj_sum = x[0], (None if x[1] is None else np.asarray(x[1], dtype=float))
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> GraphDataEncoder:
        return GraphDataEncoder(directed=False)


class RandomDotProductGraphAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for RandomDotProductGraphAccumulator."""

    def __init__(self, keys: str | None = None) -> None:
        self.keys = keys

    def make(self) -> RandomDotProductGraphAccumulator:
        return RandomDotProductGraphAccumulator(keys=self.keys)


class RandomDotProductGraphEstimator(ParameterEstimator):
    """Adjacency Spectral Embedding estimator for the RDPG latent positions."""

    def __init__(
        self,
        dim: int,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        if dim is None or dim < 1:
            raise ValueError("RandomDotProductGraphEstimator requires the latent dimension dim >= 1.")
        self.dim = int(dim)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> RandomDotProductGraphAccumulatorFactory:
        return RandomDotProductGraphAccumulatorFactory(keys=self.keys)

    def estimate(
        self, nobs: float | None, suff_stat: tuple[float, np.ndarray | None]
    ) -> RandomDotProductGraphDistribution:
        count, adj_sum = suff_stat
        if adj_sum is None or count <= 0.0:
            return RandomDotProductGraphDistribution(np.zeros((1, self.dim)), name=self.name, keys=self.keys)

        mean_adj = 0.5 * (adj_sum + adj_sum.T) / count  # symmetric mean adjacency
        n = mean_adj.shape[0]
        d = min(self.dim, n)
        # Diagonal augmentation (Scheinerman): the diagonal of X X^T is unobserved (no self-loops), so
        # impute it from the off-diagonal row means before the spectral embedding to remove ASE bias.
        np.fill_diagonal(mean_adj, 0.0)
        if n > 1:
            np.fill_diagonal(mean_adj, mean_adj.sum(axis=1) / (n - 1))
        # ASE: latent positions are the top-d (by |eigenvalue|) scaled eigenvectors of the mean adjacency.
        eigvals, eigvecs = np.linalg.eigh(mean_adj)
        order = np.argsort(np.abs(eigvals))[::-1][:d]
        scale = np.sqrt(np.clip(eigvals[order], 0.0, None))
        positions = eigvecs[:, order] * scale[None, :]
        if d < self.dim:
            positions = np.hstack([positions, np.zeros((n, self.dim - d))])
        return RandomDotProductGraphDistribution(positions, name=self.name, keys=self.keys)
