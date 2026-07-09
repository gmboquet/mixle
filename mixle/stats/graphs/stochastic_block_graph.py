"""Stochastic block graph distributions with observed or fixed block assignments.

This module handles Bernoulli edges conditional on observed or fixed node block
assignments. It does not marginalize over unknown block assignments.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.data.sources.graph_source import (
    _EPS,
    GraphDataEncoder,
    GraphObservation,
    _as_assignments,
    _bernoulli_log_likelihood,
    _edge_indices,
    _extract_observation,
    _normalize_prior,
    _validate_block_indices,
    _validate_block_probs,
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


class StochasticBlockGraphDistribution(SequenceEncodableProbabilityDistribution):
    """Bernoulli stochastic block graph distribution.

    The distribution can be used conditionally on observed block assignments, or as a
    population model that samples assignments from ``block_prior`` for new graphs.
    Exact marginal likelihood over unknown assignments is intentionally not implied.
    """

    @classmethod
    def compute_capabilities(cls):
        """Return backend capabilities for block-structured Bernoulli graph scoring."""
        from mixle.stats.compute.capabilities import DistributionCapabilities

        return DistributionCapabilities(engine_ready=("numpy", "torch"), kernel_status="generic_object")

    def __init__(
        self,
        block_probs: Any,
        block_assignments: Any | None = None,
        block_prior: Any | None = None,
        directed: bool = False,
        self_loops: bool = False,
        include_assignment_prior: bool = False,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        probs = _validate_block_probs(block_probs)
        if not directed and not np.allclose(probs, probs.T):
            raise ValueError("undirected block_probs must be symmetric.")
        self.block_probs = probs
        self.num_blocks = int(probs.shape[0])
        self.block_assignments = (
            None if block_assignments is None else _as_assignments(block_assignments, len(block_assignments))
        )
        if self.block_assignments is not None:
            _validate_block_indices(self.block_assignments, self.num_blocks)
        self.block_prior = _normalize_prior(block_prior, self.num_blocks)
        self.log_block_prior = np.log(np.clip(self.block_prior, _EPS, 1.0))
        self.directed = bool(directed)
        self.self_loops = bool(self_loops)
        self.include_assignment_prior = bool(include_assignment_prior)
        self.name = name
        self.keys = keys

    def __str__(self) -> str:
        return "StochasticBlockGraphDistribution(num_blocks=%d, directed=%s, self_loops=%s, name=%s, keys=%s)" % (
            self.num_blocks,
            repr(self.directed),
            repr(self.self_loops),
            repr(self.name),
            repr(self.keys),
        )

    @classmethod
    def from_model(
        cls, model: Any, block_prior: Any | None = None, include_assignment_prior: bool = False
    ) -> "StochasticBlockGraphDistribution":
        """Create a distribution wrapper from a random-graph SBM model."""
        return cls(
            model.block_probs,
            block_assignments=model.block_assignments,
            block_prior=block_prior,
            directed=model.directed,
            self_loops=model.self_loops,
            include_assignment_prior=include_assignment_prior,
            name=getattr(model, "name", None),
        )

    def to_model(self) -> Any:
        """Convert this distribution to the corresponding random-graph model."""
        if self.block_assignments is None:
            raise ValueError("fixed block_assignments are required to convert to StochasticBlockGraphModel.")
        from mixle.models.random_graph import StochasticBlockGraphModel

        return StochasticBlockGraphModel(
            self.block_probs, self.block_assignments, directed=self.directed, self_loops=self.self_loops, name=self.name
        )

    def _obs_with_assignments(self, x: Any) -> GraphObservation:
        obs = _extract_observation(x, directed=self.directed, fallback_assignments=self.block_assignments)
        if obs.block_assignments is None:
            raise ValueError("block assignments are required for SBM log-density.")
        _validate_block_indices(obs.block_assignments, self.num_blocks)
        return obs

    def density(self, x: Any) -> float:
        """Return the probability mass of one graph observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Return the conditional block-model log probability of one graph."""
        obs = self._obs_with_assignments(x)
        adj = obs.adjacency
        assignments = obs.block_assignments
        ll = 0.0
        for i, j in _edge_indices(adj.shape[0], directed=self.directed, self_loops=self.self_loops):
            p = self.block_probs[assignments[i], assignments[j]]
            ll += _bernoulli_log_likelihood(adj[i, j], 1.0, p)
        if self.include_assignment_prior:
            ll += float(np.sum(self.log_block_prior[assignments]))
        return float(ll)

    def seq_log_density(self, x: Sequence[GraphObservation]) -> np.ndarray:
        """Score a batch of graph observations."""
        return np.asarray([self.log_density(obs) for obs in x], dtype=np.float64)

    def backend_seq_log_density(self, x: Sequence[GraphObservation], engine: Any) -> Any:
        """Engine-routed block-structured Bernoulli edge log-likelihood.

        Each graph's edges are flattened host-side into per-edge (value, block-pair probability)
        arrays with a graph-segment id; the Bernoulli terms and the segment reduction run on the
        active engine (differentiable in ``block_probs`` on torch). The optional assignment prior is
        added per graph.
        """
        n = len(x)
        seg, adj_vals, p_vals = [], [], []
        priors = np.zeros(n, dtype=np.float64)
        for gi, obs in enumerate(x):
            obs = self._obs_with_assignments(obs)
            adj = obs.adjacency
            assignments = obs.block_assignments
            for i, j in _edge_indices(adj.shape[0], directed=self.directed, self_loops=self.self_loops):
                seg.append(gi)
                adj_vals.append(float(adj[i, j]))
                p_vals.append(float(self.block_probs[assignments[i], assignments[j]]))
            if self.include_assignment_prior:
                priors[gi] = float(np.sum(self.log_block_prior[assignments]))

        out = engine.zeros(n)
        if seg:
            p_arr = np.clip(np.asarray(p_vals, dtype=np.float64), _EPS, 1.0 - _EPS)
            av = engine.asarray(np.asarray(adj_vals, dtype=np.float64))
            bern = av * engine.asarray(np.log(p_arr)) + (1.0 - av) * engine.asarray(np.log1p(-p_arr))
            out = engine.index_add(out, engine.asarray(np.asarray(seg, dtype=np.int64)), bern)
        if self.include_assignment_prior:
            out = out + engine.asarray(priors)
        return out

    def _prior_predictive_link_probability(self, same_node: bool = False) -> float:
        if same_node:
            return float(np.sum(self.block_prior * np.diag(self.block_probs)))
        return float(self.block_prior @ self.block_probs @ self.block_prior)

    def link_probability(self, i: int, j: int, block_assignments: Any | None = None) -> float:
        """Return the marginal or assignment-conditional edge probability for node pair ``(i, j)``."""
        if i == j and not self.self_loops:
            return 0.0
        assignments = (
            self.block_assignments if block_assignments is None else np.asarray(block_assignments, dtype=np.int64)
        )
        if assignments is None:
            return self._prior_predictive_link_probability(same_node=(i == j))
        _validate_block_indices(assignments, self.num_blocks)
        return float(self.block_probs[int(assignments[i]), int(assignments[j])])

    def edge_marginals(self, block_assignments: Any | None = None, num_nodes: int | None = None) -> np.ndarray:
        """Return the matrix of edge probabilities under fixed or prior-predictive assignments."""
        if block_assignments is None:
            if self.block_assignments is None:
                if num_nodes is None:
                    raise ValueError("block_assignments or num_nodes is required.")
                n = int(num_nodes)
                edge_p = self._prior_predictive_link_probability(same_node=False)
                mat = np.full((n, n), edge_p, dtype=np.float64)
                if self.self_loops:
                    np.fill_diagonal(mat, self._prior_predictive_link_probability(same_node=True))
                else:
                    np.fill_diagonal(mat, 0.0)
                return mat
            else:
                assignments = self.block_assignments
        else:
            assignments = np.asarray(block_assignments, dtype=np.int64)
        _validate_block_indices(assignments, self.num_blocks)
        n = len(assignments)
        mat = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                mat[i, j] = self.block_probs[assignments[i], assignments[j]]
        if not self.self_loops:
            np.fill_diagonal(mat, 0.0)
        return mat

    def block_marginals(self, x: Any = None) -> np.ndarray:
        """Return empirical block proportions for ``x`` or the model block prior."""
        if x is not None:
            obs = self._obs_with_assignments(x)
            counts = np.bincount(obs.block_assignments, minlength=self.num_blocks).astype(np.float64)
            return counts / counts.sum() if counts.sum() > 0.0 else self.block_prior.copy()
        return self.block_prior.copy()

    def posterior(self, x: Any) -> dict[str, Any]:
        """Return block counts, block proportions, and edge marginals for an observed graph."""
        obs = self._obs_with_assignments(x)
        counts = np.bincount(obs.block_assignments, minlength=self.num_blocks).astype(np.float64)
        return {
            "block_counts": counts,
            "block_marginals": counts / counts.sum() if counts.sum() > 0.0 else self.block_prior.copy(),
            "edge_marginals": self.edge_marginals(obs.block_assignments),
        }

    def sampler(self, seed: int | None = None) -> "StochasticBlockGraphSampler":
        """Return a sampler for SBM graph observations."""
        return StochasticBlockGraphSampler(self, seed)

    def enumerator(self) -> DistributionEnumerator:
        """Enumerate binary graphs in descending probability order, for FIXED block assignments.

        With the node block assignments fixed (``block_assignments`` set on the distribution), each
        edge ``(i, j)`` is an independent Bernoulli with its own probability ``block_probs[a_i, a_j]``,
        so the graph distribution is a product of edge factors -- enumerated by best-first over the
        per-edge supports and assembled into an adjacency matrix (mirrored when undirected). The
        constant assignment-prior term (when ``include_assignment_prior``) enters as a score offset so
        each graph carries its exact ``log_density``.

        Marginalizing over UNKNOWN assignments is intentionally not modeled by this family, so
        enumeration requires fixed assignments; otherwise EnumerationError.
        """
        if self.block_assignments is None:
            raise EnumerationError(
                self,
                reason="enumeration requires fixed block_assignments (the family does not marginalize "
                "over unknown assignments)",
            )
        return StochasticBlockGraphEnumerator(self)

    def estimator(self, pseudo_count: float | None = None) -> "StochasticBlockGraphEstimator":
        """Return an estimator for observed-assignment SBM fitting."""
        return StochasticBlockGraphEstimator(
            num_blocks=self.num_blocks,
            block_assignments=self.block_assignments,
            directed=self.directed,
            self_loops=self.self_loops,
            pseudo_count=pseudo_count,
            prior_p=0.5,
            block_prior=self.block_prior,
            include_assignment_prior=self.include_assignment_prior,
            name=self.name,
            keys=self.keys,
        )

    def dist_to_encoder(self) -> GraphDataEncoder:
        """Return the graph encoder used by vectorized scoring and fitting."""
        return GraphDataEncoder(directed=self.directed, fallback_assignments=self.block_assignments)


class StochasticBlockGraphEnumerator(DistributionEnumerator):
    """Enumerator over binary graphs with fixed block assignments."""

    def __init__(self, dist: StochasticBlockGraphDistribution) -> None:
        """Best-first enumeration of binary graphs over independent block-dependent edge factors.

        Args:
            dist (StochasticBlockGraphDistribution): Distribution whose graphs are enumerated (its
                ``block_assignments`` must be fixed).
        """
        super().__init__(dist)
        assignments = np.asarray(dist.block_assignments)
        n = len(assignments)
        edges = list(_edge_indices(n, dist.directed, dist.self_loops))
        directed = dist.directed
        with np.errstate(divide="ignore"):
            offset = float(np.sum(dist.log_block_prior[assignments])) if dist.include_assignment_prior else 0.0

        streams = []
        for i, j in edges:
            p = float(dist.block_probs[assignments[i], assignments[j]])
            lp1, lp0 = math.log(max(p, _EPS)), math.log(max(1.0 - p, _EPS))
            pair = [(1, lp1), (0, lp0)] if lp1 >= lp0 else [(0, lp0), (1, lp1)]
            streams.append(BufferedStream(iter(pair)))

        def combine(edge_values: tuple[int, ...]) -> np.ndarray:
            adj = np.zeros((n, n), dtype=np.int8)
            for (i, j), v in zip(edges, edge_values):
                adj[i, j] = v
                if not directed:
                    adj[j, i] = v
            return adj

        self._product = ProductEnumerator(streams, combine=combine, offset=offset)

    def __next__(self) -> tuple[np.ndarray, float]:
        """Return the next adjacency matrix and its log probability."""
        return next(self._product)


class StochasticBlockGraphSampler(DistributionSampler):
    """Sample binary graphs from an SBM."""

    def __init__(self, dist: StochasticBlockGraphDistribution, seed: int | None = None) -> None:
        self.dist = dist
        self.rng = RandomState(seed)

    def sample_assignments(self, num_nodes: int) -> np.ndarray:
        """Draw node block assignments from the block prior."""
        n = int(num_nodes)
        if n < 0:
            raise ValueError("num_nodes must be non-negative.")
        return self.rng.choice(self.dist.num_blocks, size=n, p=self.dist.block_prior).astype(np.int64)

    def sample_graph(
        self, num_nodes: int | None = None, block_assignments: Any | None = None, return_assignments: bool = False
    ) -> Any:
        """Draw one graph, optionally returning the assignments used."""
        if block_assignments is None:
            if self.dist.block_assignments is not None and num_nodes is None:
                assignments = self.dist.block_assignments
            else:
                if num_nodes is None:
                    raise ValueError("num_nodes is required when block_assignments are not fixed.")
                assignments = self.sample_assignments(int(num_nodes))
        else:
            assignments = np.asarray(block_assignments, dtype=np.int64)
        _validate_block_indices(assignments, self.dist.num_blocks)

        n = len(assignments)
        mat = np.zeros((n, n), dtype=np.int8)
        for i, j in _edge_indices(n, directed=self.dist.directed, self_loops=self.dist.self_loops):
            p = self.dist.block_probs[assignments[i], assignments[j]]
            edge = int(self.rng.rand() < p)
            mat[i, j] = edge
            if not self.dist.directed and i != j:
                mat[j, i] = edge
        return (mat, assignments.copy()) if return_assignments else mat

    def sample(
        self,
        size: int | None = None,
        num_nodes: int | None = None,
        block_assignments: Any | None = None,
        return_assignments: bool = False,
    ) -> Any:
        """Draw one graph or a list of graphs from the SBM."""
        if size is None:
            return self.sample_graph(
                num_nodes=num_nodes, block_assignments=block_assignments, return_assignments=return_assignments
            )
        return [
            self.sample_graph(
                num_nodes=num_nodes, block_assignments=block_assignments, return_assignments=return_assignments
            )
            for _ in range(int(size))
        ]


class StochasticBlockGraphAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate block-pair edge counts for SBM fitting."""

    def __init__(
        self,
        num_blocks: int | None = None,
        block_assignments: Any | None = None,
        directed: bool = False,
        self_loops: bool = False,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.num_blocks = None if num_blocks is None else int(num_blocks)
        self.block_assignments = None if block_assignments is None else np.asarray(block_assignments, dtype=np.int64)
        if self.num_blocks is None and self.block_assignments is not None and self.block_assignments.size:
            self.num_blocks = int(self.block_assignments.max()) + 1
        k = 0 if self.num_blocks is None else self.num_blocks
        self.successes = np.zeros((k, k), dtype=np.float64)
        self.totals = np.zeros((k, k), dtype=np.float64)
        self.block_counts = np.zeros(k, dtype=np.float64)
        self.total_nodes = 0.0
        self.num_graphs = 0.0
        self.directed = bool(directed)
        self.self_loops = bool(self_loops)
        self.name = name
        self.keys = keys

    def _ensure_capacity(self, num_blocks: int) -> None:
        if self.num_blocks is None:
            self.num_blocks = int(num_blocks)
        if int(num_blocks) <= self.successes.shape[0]:
            return
        old = self.successes.shape[0]
        new = int(num_blocks)
        s = np.zeros((new, new), dtype=np.float64)
        t = np.zeros((new, new), dtype=np.float64)
        c = np.zeros(new, dtype=np.float64)
        s[:old, :old] = self.successes
        t[:old, :old] = self.totals
        c[:old] = self.block_counts
        self.successes = s
        self.totals = t
        self.block_counts = c
        self.num_blocks = new

    def update(self, x: Any, weight: float, estimate: StochasticBlockGraphDistribution | None) -> None:
        """Accumulate weighted block-pair edge counts from one graph."""
        fallback = self.block_assignments
        if fallback is None and estimate is not None:
            fallback = estimate.block_assignments
        obs = _extract_observation(x, directed=self.directed, fallback_assignments=fallback)
        if obs.block_assignments is None:
            raise ValueError("block assignments are required for SBM accumulation.")
        assignments = obs.block_assignments
        needed = int(assignments.max()) + 1 if assignments.size else 0
        self._ensure_capacity(max(needed, 0 if self.num_blocks is None else self.num_blocks))
        w = float(weight)
        self.block_counts[:needed] += w * np.bincount(assignments, minlength=needed)
        self.total_nodes += w * len(assignments)
        self.num_graphs += w
        for i, j in _edge_indices(obs.adjacency.shape[0], directed=self.directed, self_loops=self.self_loops):
            a = int(assignments[i])
            b = int(assignments[j])
            self.successes[a, b] += w * obs.adjacency[i, j]
            self.totals[a, b] += w
            if not self.directed and a != b:
                self.successes[b, a] += w * obs.adjacency[i, j]
                self.totals[b, a] += w

    def initialize(self, x: Any, weight: float, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from one weighted graph."""
        self.update(x, weight, None)

    def seq_update(
        self, x: Sequence[GraphObservation], weights: np.ndarray, estimate: StochasticBlockGraphDistribution | None
    ) -> None:
        """Accumulate weighted block-pair edge counts from a batch."""
        for obs, weight in zip(x, weights):
            self.update(obs, float(weight), estimate)

    def seq_initialize(self, x: Sequence[GraphObservation], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted graph batch."""
        self.seq_update(x, weights, None)

    def combine(
        self, suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray, float, float]
    ) -> "StochasticBlockGraphAccumulator":
        """Merge serialized SBM sufficient statistics."""
        successes, totals, block_counts, total_nodes, num_graphs = suff_stat
        self._ensure_capacity(successes.shape[0])
        k = successes.shape[0]
        self.successes[:k, :k] += successes
        self.totals[:k, :k] += totals
        self.block_counts[:k] += block_counts
        self.total_nodes += float(total_nodes)
        self.num_graphs += float(num_graphs)
        return self

    def value(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Return serialized SBM sufficient statistics."""
        return (self.successes.copy(), self.totals.copy(), self.block_counts.copy(), self.total_nodes, self.num_graphs)

    def from_value(
        self, x: tuple[np.ndarray, np.ndarray, np.ndarray, float, float]
    ) -> "StochasticBlockGraphAccumulator":
        """Restore accumulator state from serialized SBM sufficient statistics."""
        successes, totals, block_counts, total_nodes, num_graphs = x
        self.successes = np.asarray(successes, dtype=np.float64).copy()
        self.totals = np.asarray(totals, dtype=np.float64).copy()
        self.block_counts = np.asarray(block_counts, dtype=np.float64).copy()
        self.num_blocks = int(self.successes.shape[0])
        self.total_nodes = float(total_nodes)
        self.num_graphs = float(num_graphs)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge tied SBM sufficient statistics into ``stats_dict``."""
        if self.keys is not None:
            if self.keys in stats_dict:
                stats_dict[self.keys].combine(self.value())
            else:
                stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace tied SBM sufficient statistics from ``stats_dict``."""
        if self.keys is not None and self.keys in stats_dict:
            self.from_value(stats_dict[self.keys].value())

    def acc_to_encoder(self) -> GraphDataEncoder:
        """Return the encoder associated with this accumulator."""
        return GraphDataEncoder(directed=self.directed, fallback_assignments=self.block_assignments)


class StochasticBlockGraphAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory for StochasticBlockGraphAccumulator."""

    def __init__(
        self,
        num_blocks: int | None = None,
        block_assignments: Any | None = None,
        directed: bool = False,
        self_loops: bool = False,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.num_blocks = None if num_blocks is None else int(num_blocks)
        self.block_assignments = None if block_assignments is None else np.asarray(block_assignments, dtype=np.int64)
        self.directed = bool(directed)
        self.self_loops = bool(self_loops)
        self.name = name
        self.keys = keys

    def make(self) -> StochasticBlockGraphAccumulator:
        """Create a fresh SBM accumulator."""
        return StochasticBlockGraphAccumulator(
            num_blocks=self.num_blocks,
            block_assignments=self.block_assignments,
            directed=self.directed,
            self_loops=self.self_loops,
            name=self.name,
            keys=self.keys,
        )


class StochasticBlockGraphEstimator(ParameterEstimator):
    """Estimate an SBM from graphs with observed block assignments."""

    def __init__(
        self,
        num_blocks: int | None = None,
        block_assignments: Any | None = None,
        directed: bool = False,
        self_loops: bool = False,
        pseudo_count: float | None = None,
        prior_p: float = 0.5,
        block_prior: Any | None = None,
        estimate_block_prior: bool = True,
        include_assignment_prior: bool = False,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        self.num_blocks = None if num_blocks is None else int(num_blocks)
        self.block_assignments = None if block_assignments is None else np.asarray(block_assignments, dtype=np.int64)
        if self.num_blocks is None and self.block_assignments is not None and self.block_assignments.size:
            self.num_blocks = int(self.block_assignments.max()) + 1
        self.directed = bool(directed)
        self.self_loops = bool(self_loops)
        self.pseudo_count = pseudo_count
        self.prior_p = float(prior_p)
        self.block_prior = (
            None if block_prior is None or self.num_blocks is None else _normalize_prior(block_prior, self.num_blocks)
        )
        self.estimate_block_prior = bool(estimate_block_prior)
        self.include_assignment_prior = bool(include_assignment_prior)
        self.name = name
        self.keys = keys

    def accumulator_factory(self) -> StochasticBlockGraphAccumulatorFactory:
        """Return the accumulator factory used by this estimator."""
        return StochasticBlockGraphAccumulatorFactory(
            num_blocks=self.num_blocks,
            block_assignments=self.block_assignments,
            directed=self.directed,
            self_loops=self.self_loops,
            name=self.name,
            keys=self.keys,
        )

    def estimate(
        self, nobs: float | None, suff_stat: tuple[np.ndarray, np.ndarray, np.ndarray, float, float]
    ) -> StochasticBlockGraphDistribution:
        """Estimate block probabilities and the block prior from sufficient statistics."""
        successes, totals, block_counts, total_nodes, num_graphs = suff_stat
        successes = np.asarray(successes, dtype=np.float64).copy()
        totals = np.asarray(totals, dtype=np.float64).copy()
        k = successes.shape[0]
        if k == 0:
            k = 1 if self.num_blocks is None else max(1, int(self.num_blocks))
            successes = np.zeros((k, k), dtype=np.float64)
            totals = np.zeros((k, k), dtype=np.float64)
            block_counts = np.zeros(k, dtype=np.float64)
        if self.pseudo_count is not None:
            successes += float(self.pseudo_count) * float(self.prior_p)
            totals += float(self.pseudo_count)
        probs = np.divide(successes, totals, out=np.full_like(successes, float(self.prior_p)), where=totals > 0.0)
        probs = np.clip(probs, _EPS, 1.0 - _EPS)
        if not self.directed:
            probs = 0.5 * (probs + probs.T)

        if self.estimate_block_prior and np.sum(block_counts) > 0.0:
            block_prior = np.asarray(block_counts, dtype=np.float64) / float(np.sum(block_counts))
        elif self.block_prior is not None:
            block_prior = self.block_prior
        else:
            block_prior = np.full(k, 1.0 / float(k), dtype=np.float64)

        return StochasticBlockGraphDistribution(
            probs,
            block_assignments=self.block_assignments,
            block_prior=block_prior,
            directed=self.directed,
            self_loops=self.self_loops,
            include_assignment_prior=self.include_assignment_prior,
            name=self.name,
            keys=self.keys,
        )
