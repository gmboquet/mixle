"""Erdos-Renyi graph distributions for binary graph observations.

Data type: a binary graph observation represented as a square adjacency matrix,
a NetworkX-like graph, or a mapping accepted by ``GraphDataEncoder``.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.data.sources.graph_source import (
    GraphDataEncoder,
    GraphObservation,
    _bernoulli_log_likelihood,
    _clip_prob,
    _edge_counts,
    _edge_indices,
    _extract_observation,
)
from mixle.enumeration.algorithms import BufferedStream, ProductEnumerator
from mixle.stats.compute.pdist import (
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class ErdosRenyiGraphDistribution(SequenceEncodableProbabilityDistribution):
    """Independent Bernoulli distribution over binary graph edges."""

    @classmethod
    def compute_capabilities(cls):
        """Return backend capabilities for Bernoulli graph scoring."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic_object")

    @classmethod
    def compute_declaration(cls):
        """Return the structured declaration for the Erdos-Renyi graph family."""
        from mixle.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

        return DistributionDeclaration(
            name="erdos_renyi_graph",
            distribution_type=cls,
            parameters=(
                ParameterSpec("p", constraint="unit_interval"),
                ParameterSpec("directed", constraint="fixed", differentiable=False),
                ParameterSpec("self_loops", constraint="fixed", differentiable=False),
                ParameterSpec("num_nodes", constraint="optional_integer", differentiable=False),
            ),
            statistics=(
                StatisticSpec("edge_opportunities"),
                StatisticSpec("edge_count"),
            ),
            support="binary_graph",
        )

    def __init__(
        self,
        p: float,
        directed: bool = False,
        self_loops: bool = False,
        num_nodes: int | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.p = _clip_prob(p)
        self.log_p = math.log(self.p)
        self.log_1p = math.log1p(-self.p)
        self.directed = bool(directed)
        self.self_loops = bool(self_loops)
        self.num_nodes = None if num_nodes is None else int(num_nodes)
        if self.num_nodes is not None and self.num_nodes < 0:
            raise ValueError("num_nodes must be non-negative.")
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "ErdosRenyiGraphDistribution(p=%s, directed=%s, self_loops=%s, num_nodes=%s, name=%s, keys=%s)" % (
            repr(self.p),
            repr(self.directed),
            repr(self.self_loops),
            repr(self.num_nodes),
            repr(self.name),
            repr(self.keys),
        )

    @classmethod
    def from_model(cls, model: Any) -> "ErdosRenyiGraphDistribution":
        """Create a distribution wrapper from an Erdos-Renyi model."""
        return cls(model.p, directed=model.directed, self_loops=model.self_loops, name=getattr(model, "name", None))

    def to_model(self) -> Any:
        """Convert this distribution to the corresponding random-graph model."""
        from mixle.models.random_graph import ErdosRenyiGraphModel

        return ErdosRenyiGraphModel(self.p, directed=self.directed, self_loops=self.self_loops, name=self.name)

    def density(self, x: Any) -> float:
        """Return the probability mass of one graph observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Return the Bernoulli edge log probability of one graph."""
        obs = _extract_observation(x, directed=self.directed)
        total, successes = _edge_counts(obs.adjacency, self.directed, self.self_loops)
        return _bernoulli_log_likelihood(successes, total, self.p)

    def seq_log_density(self, x: Sequence[GraphObservation]) -> np.ndarray:
        """Score a batch of graph observations."""
        # Extract (opportunities, successes) per graph once (ragged adjacency forces the per-graph
        # extraction), then score the whole batch with one vectorized Bernoulli log-likelihood instead
        # of a Python log_density call per graph.
        if len(x) == 0:
            return np.zeros(0, dtype=np.float64)
        counts = np.asarray(
            [
                _edge_counts(_extract_observation(o, directed=self.directed).adjacency, self.directed, self.self_loops)
                for o in x
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        total, successes = counts[:, 0], counts[:, 1]
        return successes * self.log_p + (total - successes) * self.log_1p

    def backend_seq_log_density(self, x: Sequence[GraphObservation], engine: Any) -> Any:
        """Engine-routed Bernoulli edge log-likelihood.

        Per-graph edge opportunities/successes are extracted host-side (the graphs are ragged
        object data), but the Bernoulli reduction runs on the active engine, so the model's scoring
        math is engine-native (and differentiable in ``p`` on torch).
        """
        p = _clip_prob(self.p)
        counts = np.asarray(
            [
                _edge_counts(_extract_observation(o, directed=self.directed).adjacency, self.directed, self.self_loops)
                for o in x
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        total = engine.asarray(counts[:, 0])
        successes = engine.asarray(counts[:, 1])
        return successes * engine.asarray(math.log(p)) + (total - successes) * engine.asarray(math.log1p(-p))

    def edge_probability(self, i: int | None = None, j: int | None = None, context: Any | None = None) -> float:
        """Return the common edge probability ``p``."""
        return self.p

    def edge_marginals(self, num_nodes: int | None = None) -> np.ndarray:
        """Return the matrix of marginal edge probabilities for ``num_nodes``."""
        n = self.num_nodes if num_nodes is None else int(num_nodes)
        if n is None:
            raise ValueError("num_nodes is required when distribution.num_nodes is None.")
        mat = np.full((n, n), self.p, dtype=np.float64)
        if not self.self_loops:
            np.fill_diagonal(mat, 0.0)
        return mat

    def posterior(self, x: Any) -> dict[str, float]:
        """Return edge opportunities, observed edge count, and fitted probability for ``x``."""
        total, successes = _edge_counts(
            _extract_observation(x, directed=self.directed).adjacency, self.directed, self.self_loops
        )
        return {"edge_opportunities": total, "edge_count": successes, "p": self.p}

    def sampler(self, seed: int | None = None) -> "ErdosRenyiGraphSampler":
        """Return a sampler for Erdos-Renyi graph observations."""
        return ErdosRenyiGraphSampler(self, seed)

    def enumerator(self) -> DistributionEnumerator:
        """Enumerate binary graphs in descending probability order (requires ``num_nodes``).

        The edges are independent Bernoulli(p) over the free positions ``_edge_indices(n, directed,
        self_loops)``, so the graph distribution is a product of edge factors and enumerates by
        best-first over the per-edge supports -- exactly like a composite of Bernoullis, with the
        combined value assembled into an adjacency matrix (mirrored for the undirected case). Each
        graph carries its exact ``log_density``. ``num_nodes`` must be set so the edge set is finite.
        """
        if self.num_nodes is None:
            raise EnumerationError(self, reason="num_nodes must be set to enumerate graphs")
        return ErdosRenyiGraphEnumerator(self)

    def estimator(self, pseudo_count: float | None = None) -> "ErdosRenyiGraphEstimator":
        """Return the edge-count estimator for this graph family."""
        return ErdosRenyiGraphEstimator(
            directed=self.directed,
            self_loops=self.self_loops,
            pseudo_count=pseudo_count,
            prior_p=self.p,
            num_nodes=self.num_nodes,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> GraphDataEncoder:
        """Return the graph encoder used by vectorized scoring and fitting."""
        return GraphDataEncoder(directed=self.directed)


class ErdosRenyiGraphEnumerator(DistributionEnumerator):
    """Enumerator over finite Erdos-Renyi binary graph support."""

    def __init__(self, dist: ErdosRenyiGraphDistribution) -> None:
        """Best-first enumeration of binary graphs over independent edge factors.

        Args:
            dist (ErdosRenyiGraphDistribution): Distribution whose graphs are enumerated (its
                ``num_nodes`` must be set).
        """
        super().__init__(dist)
        n = dist.num_nodes
        edges = list(_edge_indices(n, dist.directed, dist.self_loops))
        directed = dist.directed
        # Each free edge is an independent Bernoulli(p): present (1) at log_p, absent (0) at log_1p,
        # ordered by descending probability so the per-edge stream is sorted.
        present, absent = (1, 0)
        edge_pair = (
            [(present, dist.log_p), (absent, dist.log_1p)]
            if dist.log_p >= dist.log_1p
            else [(absent, dist.log_1p), (present, dist.log_p)]
        )

        def combine(edge_values: tuple[int, ...]) -> np.ndarray:
            adj = np.zeros((n, n), dtype=np.int8)
            for (i, j), v in zip(edges, edge_values):
                adj[i, j] = v
                if not directed:
                    adj[j, i] = v
            return adj

        streams = [BufferedStream(iter(list(edge_pair))) for _ in edges]
        self._product = ProductEnumerator(streams, combine=combine)

    def __next__(self) -> tuple[np.ndarray, float]:
        """Return the next adjacency matrix and its log probability."""
        return next(self._product)


class ErdosRenyiGraphSampler(DistributionSampler):
    """Sample binary graphs from an Erdos-Renyi distribution."""

    def __init__(self, dist: ErdosRenyiGraphDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample_graph(self, num_nodes: int | None = None) -> np.ndarray:
        """Draw one binary graph adjacency matrix."""
        n = self.dist.num_nodes if num_nodes is None else int(num_nodes)
        if n is None:
            raise ValueError("num_nodes is required when distribution.num_nodes is None.")
        if n < 0:
            raise ValueError("num_nodes must be non-negative.")
        mat = (self.rng.rand(n, n) < self.dist.p).astype(np.int8)
        if self.dist.directed:
            if not self.dist.self_loops:
                np.fill_diagonal(mat, 0)
            return mat

        upper = np.triu(mat, k=0 if self.dist.self_loops else 1)
        mat = upper + upper.T
        if self.dist.self_loops:
            diag = (self.rng.rand(n) < self.dist.p).astype(np.int8)
            np.fill_diagonal(mat, diag)
        return mat

    def sample(
        self, size: int | None = None, num_nodes: int | None = None, *, batched: bool = True
    ) -> np.ndarray | list[np.ndarray]:
        """Draw one graph or a list of graphs."""
        if size is None:
            return self.sample_graph(num_nodes=num_nodes)
        return [self.sample_graph(num_nodes=num_nodes) for _ in range(int(size))]


class ErdosRenyiGraphAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate edge counts for Erdos-Renyi graph fitting."""

    def __init__(
        self, directed: bool = False, self_loops: bool = False, name: str | None = None, keys: str | None = None
    ) -> None:
        self.edge_opportunities = 0.0
        self.edge_count = 0.0
        self.directed = bool(directed)
        self.self_loops = bool(self_loops)
        self.name = name
        self.keys = keys

    def update(self, x: Any, weight: float, estimate: ErdosRenyiGraphDistribution | None) -> None:
        """Accumulate weighted edge opportunities and successes from one graph."""
        obs = _extract_observation(x, directed=self.directed)
        total, successes = _edge_counts(obs.adjacency, self.directed, self.self_loops)
        self.edge_opportunities += float(weight) * total
        self.edge_count += float(weight) * successes

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from one weighted graph."""
        self.update(x, weight, None)

    def seq_update(
        self, x: Sequence[GraphObservation], weights: np.ndarray, estimate: ErdosRenyiGraphDistribution | None
    ) -> None:
        """Accumulate weighted edge counts from a batch of graphs."""
        for obs, weight in zip(x, weights):
            total, successes = _edge_counts(obs.adjacency, self.directed, self.self_loops)
            self.edge_opportunities += float(weight) * total
            self.edge_count += float(weight) * successes

    def seq_initialize(self, x: Sequence[GraphObservation], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted graph batch."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float]) -> "ErdosRenyiGraphAccumulator":
        """Merge serialized edge-count sufficient statistics."""
        self.edge_opportunities += suff_stat[0]
        self.edge_count += suff_stat[1]
        return self

    def value(self) -> tuple[float, float]:
        """Return serialized edge-count sufficient statistics."""
        return self.edge_opportunities, self.edge_count

    def from_value(self, x: tuple[float, float]) -> "ErdosRenyiGraphAccumulator":
        """Restore accumulator state from serialized edge counts."""
        self.edge_opportunities = float(x[0])
        self.edge_count = float(x[1])
        return self

    def acc_to_encoder(self) -> GraphDataEncoder:
        """Return the encoder associated with this accumulator."""
        return GraphDataEncoder(directed=self.directed)


class ErdosRenyiGraphAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for ErdosRenyiGraphAccumulator."""

    def __init__(
        self, directed: bool = False, self_loops: bool = False, name: str | None = None, keys: str | None = None
    ) -> None:
        self.directed = bool(directed)
        self.self_loops = bool(self_loops)
        self.name = name
        self.keys = keys

    def make(self) -> ErdosRenyiGraphAccumulator:
        """Create a fresh Erdos-Renyi graph accumulator."""
        return ErdosRenyiGraphAccumulator(
            directed=self.directed, self_loops=self.self_loops, name=self.name, keys=self.keys
        )


class ErdosRenyiGraphEstimator(ParameterEstimator):
    """Estimate an Erdos-Renyi graph distribution from edge counts."""

    def __init__(
        self,
        directed: bool = False,
        self_loops: bool = False,
        pseudo_count: float | None = None,
        prior_p: float = 0.5,
        num_nodes: int | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.directed = bool(directed)
        self.self_loops = bool(self_loops)
        self.pseudo_count = pseudo_count
        self.prior_p = float(prior_p)
        self.num_nodes = None if num_nodes is None else int(num_nodes)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> ErdosRenyiGraphAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return ErdosRenyiGraphAccumulatorFactory(
            directed=self.directed, self_loops=self.self_loops, name=self.name, keys=self.keys
        )

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> ErdosRenyiGraphDistribution:
        """Estimate the Bernoulli edge probability from edge-count statistics."""
        total, successes = suff_stat
        if self.pseudo_count is not None:
            successes += float(self.pseudo_count) * float(self.prior_p)
            total += float(self.pseudo_count)
        p = 0.5 if total <= 0.0 else successes / total
        return ErdosRenyiGraphDistribution(
            p,
            directed=self.directed,
            self_loops=self.self_loops,
            num_nodes=self.num_nodes,
            name=self.name,
            keys=self.keys,
        )
