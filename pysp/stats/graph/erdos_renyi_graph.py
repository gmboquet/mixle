"""Create, estimate, and sample from an Erdos-Renyi graph distribution.

Data type: a binary graph observation represented as a square adjacency matrix,
a NetworkX-like graph, or a mapping accepted by ``GraphDataEncoder``.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from pysp.data.graph_data import (
    GraphDataEncoder,
    GraphObservation,
    _bernoulli_log_likelihood,
    _clip_prob,
    _edge_counts,
    _extract_observation,
)
from pysp.stats.compute.pdist import (
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
)


class ErdosRenyiGraphDistribution(SequenceEncodableProbabilityDistribution):
    """Independent Bernoulli distribution over binary graph edges."""

    @classmethod
    def compute_capabilities(cls):
        from pysp.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic_object")

    @classmethod
    def compute_declaration(cls):
        from pysp.stats.compute.declarations import DistributionDeclaration, ParameterSpec, StatisticSpec

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
        return cls(model.p, directed=model.directed, self_loops=model.self_loops, name=getattr(model, "name", None))

    def to_model(self) -> Any:
        from pysp.models.random_graph import ErdosRenyiGraphModel

        return ErdosRenyiGraphModel(self.p, directed=self.directed, self_loops=self.self_loops, name=self.name)

    def density(self, x: Any) -> float:
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        obs = _extract_observation(x, directed=self.directed)
        total, successes = _edge_counts(obs.adjacency, self.directed, self.self_loops)
        return _bernoulli_log_likelihood(successes, total, self.p)

    def seq_log_density(self, x: Sequence[GraphObservation]) -> np.ndarray:
        return np.asarray([self.log_density(obs) for obs in x], dtype=np.float64)

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
        return self.p

    def edge_marginals(self, num_nodes: int | None = None) -> np.ndarray:
        n = self.num_nodes if num_nodes is None else int(num_nodes)
        if n is None:
            raise ValueError("num_nodes is required when distribution.num_nodes is None.")
        mat = np.full((n, n), self.p, dtype=np.float64)
        if not self.self_loops:
            np.fill_diagonal(mat, 0.0)
        return mat

    def posterior(self, x: Any) -> dict[str, float]:
        total, successes = _edge_counts(
            _extract_observation(x, directed=self.directed).adjacency, self.directed, self.self_loops
        )
        return {"edge_opportunities": total, "edge_count": successes, "p": self.p}

    def sampler(self, seed: int | None = None) -> "ErdosRenyiGraphSampler":
        return ErdosRenyiGraphSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> "ErdosRenyiGraphEstimator":
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
        return GraphDataEncoder(directed=self.directed)


class ErdosRenyiGraphSampler(DistributionSampler):
    """Sample binary graphs from an Erdos-Renyi distribution."""

    def __init__(self, dist: ErdosRenyiGraphDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample_graph(self, num_nodes: int | None = None) -> np.ndarray:
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

    def sample(self, size: int | None = None, num_nodes: int | None = None) -> np.ndarray | list[np.ndarray]:
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
        self.key = keys

    def update(self, x: Any, weight: float, estimate: ErdosRenyiGraphDistribution | None) -> None:
        obs = _extract_observation(x, directed=self.directed)
        total, successes = _edge_counts(obs.adjacency, self.directed, self.self_loops)
        self.edge_opportunities += float(weight) * total
        self.edge_count += float(weight) * successes

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        self.update(x, weight, None)

    def seq_update(
        self, x: Sequence[GraphObservation], weights: np.ndarray, estimate: ErdosRenyiGraphDistribution | None
    ) -> None:
        for obs, weight in zip(x, weights):
            total, successes = _edge_counts(obs.adjacency, self.directed, self.self_loops)
            self.edge_opportunities += float(weight) * total
            self.edge_count += float(weight) * successes

    def seq_initialize(self, x: Sequence[GraphObservation], weights: np.ndarray, rng: RandomState | None) -> None:
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: tuple[float, float]) -> "ErdosRenyiGraphAccumulator":
        self.edge_opportunities += suff_stat[0]
        self.edge_count += suff_stat[1]
        return self

    def value(self) -> tuple[float, float]:
        return self.edge_opportunities, self.edge_count

    def from_value(self, x: tuple[float, float]) -> "ErdosRenyiGraphAccumulator":
        self.edge_opportunities = float(x[0])
        self.edge_count = float(x[1])
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        if self.key is not None:
            if self.key in stats_dict:
                stats_dict[self.key].combine(self.value())
            else:
                stats_dict[self.key] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        if self.key is not None and self.key in stats_dict:
            self.from_value(stats_dict[self.key].value())

    def acc_to_encoder(self) -> GraphDataEncoder:
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
        return ErdosRenyiGraphAccumulatorFactory(
            directed=self.directed, self_loops=self.self_loops, name=self.name, keys=self.keys
        )

    def estimate(self, nobs: float | None, suff_stat: tuple[float, float]) -> ErdosRenyiGraphDistribution:
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
