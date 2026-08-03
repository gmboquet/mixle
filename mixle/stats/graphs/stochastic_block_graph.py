"""Stochastic block graph distributions with explicit conditional or joint laws.

Fixed assignments define ``p(adjacency | assignments)``. Without fixed
assignments, observations are ``(adjacency, assignments)`` pairs scored under
the joint law ``p(assignments) p(adjacency | assignments)``. This family does
not claim to marginalize assignments out of a bare adjacency observation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.random import RandomState

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
from mixle.utils.exact import require_exact_bool

if TYPE_CHECKING:
    from mixle.data.sources.graph_source import GraphDataEncoder, GraphObservation

# mixle.data.sources.graph_source's own top-level import of mixle.stats.compute.pdist.DataSequenceEncoder
# forces mixle.stats's package __init__ to run first, which eagerly imports this very module -- so a
# module-level `from mixle.data.sources.graph_source import ...` here is circular whenever
# mixle.data.sources.graph_source is the entry point (e.g. `import mixle.data.sources.graph_source`
# directly, before mixle.stats has been warmed by any other path): graph_source would still be
# mid-import (paused inside its own DataSequenceEncoder import, above this module in the same chain),
# so none of GraphDataEncoder/GraphObservation/the coercion+validation helpers would exist as
# attributes yet. Every usage below is either purely a type annotation (deferred to a string by the
# `from __future__ import annotations` above) or a call inside a function/method body, so the imports
# are deferred to call time, once every module in the cycle has finished loading normally -- mirroring
# the erdos_renyi_graph.py fix for the same shape of cycle.


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
        include_assignment_prior: bool | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        from mixle.data.sources.graph_source import (
            _as_assignments,
            _normalize_prior,
            _validate_block_indices,
            _validate_block_probs,
        )

        probs = np.array(_validate_block_probs(block_probs), dtype=np.float64, copy=True)
        if not directed and not np.array_equal(probs, probs.T):
            raise ValueError("undirected block_probs must be symmetric.")
        probs.setflags(write=False)
        self.block_probs = probs
        self.num_blocks = int(probs.shape[0])
        assignments = None if block_assignments is None else _as_assignments(block_assignments, len(block_assignments))
        if assignments is not None:
            assignments = np.array(assignments, dtype=np.int64, copy=True)
            assignments.setflags(write=False)
        self.block_assignments = assignments
        if self.block_assignments is not None:
            _validate_block_indices(self.block_assignments, self.num_blocks)
        prior = np.array(_normalize_prior(block_prior, self.num_blocks), dtype=np.float64, copy=True)
        prior.setflags(write=False)
        self.block_prior = prior
        self.log_block_prior = np.full(self.num_blocks, -math.inf, dtype=np.float64)
        positive_prior = prior > 0.0
        self.log_block_prior[positive_prior] = np.log(prior[positive_prior])
        self.log_block_prior.setflags(write=False)
        self.directed = require_exact_bool(directed, "directed")
        self.self_loops = require_exact_bool(self_loops, "self_loops")
        expected_joint = self.block_assignments is None
        if include_assignment_prior is not None and not isinstance(include_assignment_prior, (bool, np.bool_)):
            raise ValueError("include_assignment_prior must be Boolean or None.")
        if (
            include_assignment_prior is not None
            and require_exact_bool(include_assignment_prior, "include_assignment_prior") != expected_joint
        ):
            mode = "population" if expected_joint else "fixed-assignment"
            required = expected_joint
            raise ValueError(
                f"{mode} SBM requires include_assignment_prior={required}; "
                "conditional and joint sample spaces cannot be mixed."
            )
        self.include_assignment_prior = expected_joint
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
    ) -> StochasticBlockGraphDistribution:
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
        from mixle.data.sources.graph_source import (
            _extract_observation,
            _validate_block_indices,
            _validate_graph_constraints,
        )

        obs = _extract_observation(x, directed=self.directed, fallback_assignments=self.block_assignments)
        if obs.block_assignments is None:
            raise ValueError("block assignments are required for SBM log-density.")
        _validate_block_indices(obs.block_assignments, self.num_blocks)
        if self.block_assignments is not None and not np.array_equal(obs.block_assignments, self.block_assignments):
            raise ValueError("observation assignments conflict with this fixed-assignment SBM.")
        # log_density/backend_seq_log_density (and posterior/block_marginals, which share this same
        # observation-consuming choke point) only ever read the edge positions _edge_indices yields for
        # (self.directed, self.self_loops) -- e.g. the strict upper triangle when undirected with no
        # self-loops -- so a self-loop or asymmetric entry outside that read set would otherwise be
        # silently ignored instead of rejected.
        _validate_graph_constraints(obs.adjacency, self.directed, self.self_loops)
        return obs

    def density(self, x: Any) -> float:
        """Return the probability mass of one graph observation."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Any) -> float:
        """Return the conditional block-model log probability of one graph."""
        from mixle.data.sources.graph_source import _bernoulli_log_likelihood, _edge_indices

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
        # Parameters are immutable host values, while observations are ragged object
        # records. Compute with the same branch-safe scalar contract as log_density and
        # transfer the result; this preserves exact p=0/1 support on every backend.
        return engine.asarray(np.asarray([self.log_density(obs) for obs in x], dtype=np.float64))

    def _prior_predictive_link_probability(self, same_node: bool = False) -> float:
        if same_node:
            return float(np.sum(self.block_prior * np.diag(self.block_probs)))
        return float(self.block_prior @ self.block_probs @ self.block_prior)

    def link_probability(self, i: int, j: int, block_assignments: Any | None = None) -> float:
        """Return the marginal or assignment-conditional edge probability for node pair ``(i, j)``."""
        from mixle.data.sources.graph_source import _as_assignments, _require_exact_int, _validate_block_indices

        ii = _require_exact_int(i, "i")
        jj = _require_exact_int(j, "j")

        if ii == jj and not self.self_loops:
            return 0.0
        assignments = (
            self.block_assignments
            if block_assignments is None
            else _as_assignments(block_assignments, len(block_assignments))
        )
        if assignments is None:
            return self._prior_predictive_link_probability(same_node=(ii == jj))
        _validate_block_indices(assignments, self.num_blocks)
        if ii < 0 or jj < 0 or ii >= len(assignments) or jj >= len(assignments):
            raise ValueError("node indices must be within the block-assignment vector.")
        return float(self.block_probs[int(assignments[ii]), int(assignments[jj])])

    def edge_marginals(self, block_assignments: Any | None = None, num_nodes: int | None = None) -> np.ndarray:
        """Return the matrix of edge probabilities under fixed or prior-predictive assignments."""
        from mixle.data.sources.graph_source import _validate_block_indices

        if block_assignments is None:
            if self.block_assignments is None:
                if num_nodes is None:
                    raise ValueError("block_assignments or num_nodes is required.")
                from mixle.data.sources.graph_source import _require_exact_int

                n = _require_exact_int(num_nodes, "num_nodes")
                if n < 0:
                    raise ValueError("num_nodes must be non-negative.")
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
            from mixle.data.sources.graph_source import _as_assignments

            assignments = _as_assignments(block_assignments, len(block_assignments))
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

    def sampler(self, seed: int | None = None) -> StochasticBlockGraphSampler:
        """Return a sampler for SBM graph observations."""
        return StochasticBlockGraphSampler(self, seed)

    def enumerator(self) -> DistributionEnumerator:
        """Enumerate binary graphs in descending probability order, for FIXED block assignments.

        With the node block assignments fixed (``block_assignments`` set on the distribution), each
        edge ``(i, j)`` is an independent Bernoulli with its own probability ``block_probs[a_i, a_j]``,
        so the graph distribution is a product of edge factors -- enumerated by best-first over the
        per-edge supports and assembled into an adjacency matrix (mirrored when undirected).

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

    def estimator(self, pseudo_count: float | None = None) -> StochasticBlockGraphEstimator:
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
        from mixle.data.sources.graph_source import GraphDataEncoder

        return GraphDataEncoder(directed=self.directed, fallback_assignments=self.block_assignments)


class StochasticBlockGraphEnumerator(DistributionEnumerator):
    """Enumerator over binary graphs with fixed block assignments."""

    def __init__(self, dist: StochasticBlockGraphDistribution) -> None:
        """Best-first enumeration of binary graphs over independent block-dependent edge factors.

        Args:
            dist (StochasticBlockGraphDistribution): Distribution whose graphs are enumerated (its
                ``block_assignments`` must be fixed).
        """
        from mixle.data.sources.graph_source import _edge_indices

        super().__init__(dist)
        assignments = np.asarray(dist.block_assignments)
        n = len(assignments)
        edges = list(_edge_indices(n, dist.directed, dist.self_loops))
        directed = dist.directed
        streams = []
        for i, j in edges:
            p = float(dist.block_probs[assignments[i], assignments[j]])
            if p == 0.0:
                pair = [(0, 0.0)]
            elif p == 1.0:
                pair = [(1, 0.0)]
            else:
                lp1, lp0 = math.log(p), math.log1p(-p)
                pair = [(1, lp1), (0, lp0)] if lp1 >= lp0 else [(0, lp0), (1, lp1)]
            streams.append(BufferedStream(iter(pair)))

        def combine(edge_values: tuple[int, ...]) -> np.ndarray:
            adj = np.zeros((n, n), dtype=np.int8)
            for (i, j), v in zip(edges, edge_values):
                adj[i, j] = v
                if not directed:
                    adj[j, i] = v
            return adj

        self._product = ProductEnumerator(streams, combine=combine)

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
        from mixle.data.sources.graph_source import _require_exact_int

        n = _require_exact_int(num_nodes, "num_nodes")
        if n < 0:
            raise ValueError("num_nodes must be non-negative.")
        return self.rng.choice(self.dist.num_blocks, size=n, p=self.dist.block_prior).astype(np.int64)

    def sample_graph(
        self,
        num_nodes: int | None = None,
        block_assignments: Any | None = None,
        return_assignments: bool | None = None,
    ) -> Any:
        """Draw one graph, optionally returning the assignments used.

        Population models always return the joint ``(adjacency, assignments)``
        outcome. Fixed-assignment models return the conditional adjacency by
        default; callers may request the redundant fixed assignments as well.
        """
        from mixle.data.sources.graph_source import _as_assignments, _edge_indices, _validate_block_indices

        sampled_from_prior = False
        if block_assignments is None:
            if self.dist.block_assignments is not None and num_nodes is None:
                assignments = self.dist.block_assignments
            else:
                if num_nodes is None:
                    raise ValueError("num_nodes is required when block_assignments are not fixed.")
                assignments = self.sample_assignments(num_nodes)
                sampled_from_prior = True
        else:
            assignments = _as_assignments(block_assignments, len(block_assignments))
        _validate_block_indices(assignments, self.dist.num_blocks)
        if self.dist.block_assignments is not None and not np.array_equal(assignments, self.dist.block_assignments):
            raise ValueError("explicit assignments conflict with this fixed-assignment SBM.")

        n = len(assignments)
        mat = np.zeros((n, n), dtype=np.int8)
        for i, j in _edge_indices(n, directed=self.dist.directed, self_loops=self.dist.self_loops):
            p = self.dist.block_probs[assignments[i], assignments[j]]
            edge = int(self.rng.rand() < p)
            mat[i, j] = edge
            if not self.dist.directed and i != j:
                mat[j, i] = edge
        if sampled_from_prior and return_assignments is False:
            raise ValueError("population SBM samples must retain assignments so the joint outcome is scoreable.")
        if return_assignments is not None and not isinstance(return_assignments, (bool, np.bool_)):
            raise ValueError("return_assignments must be Boolean or None.")
        include_assignments = (
            sampled_from_prior
            if return_assignments is None
            else require_exact_bool(return_assignments, "return_assignments")
        )
        return (mat, assignments.copy()) if include_assignments else mat

    def sample(
        self,
        size: int | None = None,
        num_nodes: int | None = None,
        block_assignments: Any | None = None,
        return_assignments: bool | None = None,
        *,
        batched: bool = True,
    ) -> Any:
        """Draw one graph or a list of graphs from the SBM.

        See :meth:`sample_graph` for how ``return_assignments=None`` (the default) auto-includes
        assignments only when they are not otherwise recoverable for scoring.
        """
        if size is None:
            return self.sample_graph(
                num_nodes=num_nodes, block_assignments=block_assignments, return_assignments=return_assignments
            )
        from mixle.data.sources.graph_source import _require_exact_int

        sample_size = _require_exact_int(size, "size")
        if sample_size < 0:
            raise ValueError("size must be non-negative.")
        return [
            self.sample_graph(
                num_nodes=num_nodes, block_assignments=block_assignments, return_assignments=return_assignments
            )
            for _ in range(sample_size)
        ]


class StochasticBlockGraphStatistics(NamedTuple):
    """Versioned, configuration-bound sufficient statistics for an SBM."""

    schema_version: int
    successes: np.ndarray
    totals: np.ndarray
    block_counts: np.ndarray
    total_nodes: float
    num_graphs: float
    directed: bool
    self_loops: bool


def _validate_sbm_statistics(
    value: Any,
    *,
    num_blocks: int | None,
    directed: bool,
    self_loops: bool,
) -> StochasticBlockGraphStatistics:
    if not isinstance(value, StochasticBlockGraphStatistics) or value.schema_version != 1:
        raise ValueError("SBM statistics must use StochasticBlockGraphStatistics schema version 1.")
    if value.directed != directed or value.self_loops != self_loops:
        raise ValueError("SBM statistics were produced for an incompatible graph configuration.")
    successes = np.array(value.successes, dtype=np.float64, copy=True)
    totals = np.array(value.totals, dtype=np.float64, copy=True)
    counts = np.array(value.block_counts, dtype=np.float64, copy=True)
    if (
        successes.ndim != 2
        or successes.shape[0] != successes.shape[1]
        or totals.shape != successes.shape
        or counts.ndim != 1
        or counts.shape[0] != successes.shape[0]
    ):
        raise ValueError("SBM statistics require aligned K-by-K count matrices and a length-K block vector.")
    if num_blocks is not None and successes.shape != (num_blocks, num_blocks):
        raise ValueError("SBM statistic dimension does not match configured num_blocks.")
    if (
        np.any(~np.isfinite(successes))
        or np.any(~np.isfinite(totals))
        or np.any(~np.isfinite(counts))
        or np.any(successes < 0.0)
        or np.any(totals < 0.0)
        or np.any(counts < 0.0)
        or np.any(successes > totals)
    ):
        raise ValueError("SBM counts must be finite, non-negative, and satisfy successes <= totals.")
    total_nodes = float(value.total_nodes)
    num_graphs = float(value.num_graphs)
    if not np.isfinite(total_nodes) or total_nodes < 0.0 or not np.isfinite(num_graphs) or num_graphs < 0.0:
        raise ValueError("SBM total_nodes and num_graphs must be finite and non-negative.")
    if not np.isclose(float(np.sum(counts)), total_nodes, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("SBM block_counts must sum to total_nodes.")
    if not directed and (not np.array_equal(successes, successes.T) or not np.array_equal(totals, totals.T)):
        raise ValueError("undirected SBM edge statistics must be symmetric.")
    return StochasticBlockGraphStatistics(
        1,
        successes,
        totals,
        counts,
        total_nodes,
        num_graphs,
        directed,
        self_loops,
    )


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
        from mixle.data.sources.graph_source import _as_assignments, _require_exact_int, _validate_block_indices

        self.num_blocks = None if num_blocks is None else _require_exact_int(num_blocks, "num_blocks")
        if self.num_blocks is not None and self.num_blocks < 1:
            raise ValueError("num_blocks must be positive when specified.")
        self.block_assignments = (
            None if block_assignments is None else _as_assignments(block_assignments, len(block_assignments))
        )
        if self.num_blocks is None and self.block_assignments is not None and self.block_assignments.size:
            self.num_blocks = int(self.block_assignments.max()) + 1
        if self.block_assignments is not None and self.num_blocks is not None:
            _validate_block_indices(self.block_assignments, self.num_blocks)
        k = 0 if self.num_blocks is None else self.num_blocks
        self.successes = np.zeros((k, k), dtype=np.float64)
        self.totals = np.zeros((k, k), dtype=np.float64)
        self.block_counts = np.zeros(k, dtype=np.float64)
        self.total_nodes = 0.0
        self.num_graphs = 0.0
        self.directed = require_exact_bool(directed, "directed")
        self.self_loops = require_exact_bool(self_loops, "self_loops")
        self.name = name
        self.keys = keys

    def _ensure_capacity(self, num_blocks: int) -> None:
        if self.num_blocks is None:
            self.num_blocks = int(num_blocks)
            self.successes = np.zeros((num_blocks, num_blocks), dtype=np.float64)
            self.totals = np.zeros((num_blocks, num_blocks), dtype=np.float64)
            self.block_counts = np.zeros(num_blocks, dtype=np.float64)
            return
        if int(num_blocks) > self.num_blocks:
            raise ValueError("observed assignments exceed configured num_blocks.")

    def update(self, x: Any, weight: float, estimate: StochasticBlockGraphDistribution | None) -> None:
        """Accumulate weighted block-pair edge counts from one graph."""
        from mixle.data.sources.graph_source import (
            _edge_indices,
            _extract_observation,
            _validate_block_indices,
            _validate_graph_constraints,
        )

        fallback = self.block_assignments
        if fallback is None and estimate is not None:
            fallback = estimate.block_assignments
        obs = _extract_observation(x, directed=self.directed, fallback_assignments=fallback)
        if obs.block_assignments is None:
            raise ValueError("block assignments are required for SBM accumulation.")
        # Mirrors StochasticBlockGraphDistribution._obs_with_assignments: _edge_indices only ever reads
        # the edge positions implied by (self.directed, self.self_loops), so a self-loop or asymmetric
        # entry outside that read set would otherwise be silently folded into (or excluded from) the
        # sufficient statistics instead of being rejected. Validated before any state mutation below so
        # a rejected graph leaves the accumulator untouched.
        _validate_graph_constraints(obs.adjacency, self.directed, self.self_loops)
        assignments = obs.block_assignments
        needed = int(assignments.max()) + 1 if assignments.size else 0
        if self.num_blocks is None and needed == 0:
            raise ValueError("num_blocks is required to accumulate an empty-assignment graph.")
        if self.num_blocks is not None:
            _validate_block_indices(assignments, self.num_blocks)
        self._ensure_capacity(needed if self.num_blocks is None else self.num_blocks)
        w = float(weight)
        if not np.isfinite(w) or w < 0.0:
            raise ValueError("weight must be finite and non-negative.")
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
        checked_weights = np.asarray(weights, dtype=np.float64)
        if checked_weights.ndim != 1 or len(checked_weights) != len(x):
            raise ValueError("weights must be a one-dimensional array aligned with the graph batch.")
        if np.any(~np.isfinite(checked_weights)) or np.any(checked_weights < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        pending = StochasticBlockGraphAccumulator(
            num_blocks=self.num_blocks,
            block_assignments=self.block_assignments,
            directed=self.directed,
            self_loops=self.self_loops,
        )
        for obs, weight in zip(x, checked_weights):
            pending.update(obs, float(weight), estimate)
        self.combine(pending.value())

    def seq_initialize(self, x: Sequence[GraphObservation], weights: np.ndarray, rng: RandomState | None) -> None:
        """Initialize sufficient statistics from a weighted graph batch."""
        self.seq_update(x, weights, None)

    def combine(self, suff_stat: StochasticBlockGraphStatistics) -> StochasticBlockGraphAccumulator:
        """Merge serialized SBM sufficient statistics."""
        checked = _validate_sbm_statistics(
            suff_stat,
            num_blocks=self.num_blocks,
            directed=self.directed,
            self_loops=self.self_loops,
        )
        successes, totals, block_counts = checked.successes, checked.totals, checked.block_counts
        self._ensure_capacity(successes.shape[0])
        k = successes.shape[0]
        self.successes[:k, :k] += successes
        self.totals[:k, :k] += totals
        self.block_counts[:k] += block_counts
        self.total_nodes += checked.total_nodes
        self.num_graphs += checked.num_graphs
        return self

    def value(self) -> StochasticBlockGraphStatistics:
        """Return serialized SBM sufficient statistics."""
        return StochasticBlockGraphStatistics(
            1,
            self.successes.copy(),
            self.totals.copy(),
            self.block_counts.copy(),
            self.total_nodes,
            self.num_graphs,
            self.directed,
            self.self_loops,
        )

    def from_value(self, x: StochasticBlockGraphStatistics) -> StochasticBlockGraphAccumulator:
        """Restore accumulator state from serialized SBM sufficient statistics."""
        checked = _validate_sbm_statistics(
            x,
            num_blocks=self.num_blocks,
            directed=self.directed,
            self_loops=self.self_loops,
        )
        self.successes = checked.successes
        self.totals = checked.totals
        self.block_counts = checked.block_counts
        self.num_blocks = int(self.successes.shape[0])
        self.total_nodes = checked.total_nodes
        self.num_graphs = checked.num_graphs
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge this accumulator under its configured sharing key."""
        if self.keys is None:
            return
        if self.keys in stats_dict:
            other = stats_dict[self.keys]
            if not isinstance(other, StochasticBlockGraphAccumulator):
                raise ValueError("shared SBM key is bound to incompatible statistics.")
            other.combine(self.value())
        else:
            stats_dict[self.keys] = self

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace this accumulator from its configured sharing key."""
        if self.keys is not None and self.keys in stats_dict:
            other = stats_dict[self.keys]
            if not isinstance(other, StochasticBlockGraphAccumulator):
                raise ValueError("shared SBM key is bound to incompatible statistics.")
            self.from_value(other.value())

    def acc_to_encoder(self) -> GraphDataEncoder:
        """Return the encoder associated with this accumulator."""
        from mixle.data.sources.graph_source import GraphDataEncoder

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
        from mixle.data.sources.graph_source import _as_assignments, _require_exact_int

        self.num_blocks = None if num_blocks is None else _require_exact_int(num_blocks, "num_blocks")
        self.block_assignments = (
            None if block_assignments is None else _as_assignments(block_assignments, len(block_assignments))
        )
        self.directed = require_exact_bool(directed, "directed")
        self.self_loops = require_exact_bool(self_loops, "self_loops")
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
        include_assignment_prior: bool | None = None,
        name: str | None = None,
        keys: str | None = None,
    ) -> None:
        from mixle.data.sources.graph_source import (
            _as_assignments,
            _clip_prob,
            _normalize_prior,
            _require_exact_int,
            _validate_block_indices,
        )

        self.num_blocks = None if num_blocks is None else _require_exact_int(num_blocks, "num_blocks")
        if self.num_blocks is not None and self.num_blocks < 1:
            raise ValueError("num_blocks must be positive when specified.")
        self.block_assignments = (
            None if block_assignments is None else _as_assignments(block_assignments, len(block_assignments))
        )
        if self.num_blocks is None and self.block_assignments is not None and self.block_assignments.size:
            self.num_blocks = int(self.block_assignments.max()) + 1
        if self.block_assignments is not None and self.num_blocks is not None:
            _validate_block_indices(self.block_assignments, self.num_blocks)
        self.directed = require_exact_bool(directed, "directed")
        self.self_loops = require_exact_bool(self_loops, "self_loops")
        self.pseudo_count = None if pseudo_count is None else float(pseudo_count)
        if self.pseudo_count is not None and (not np.isfinite(self.pseudo_count) or self.pseudo_count < 0.0):
            raise ValueError("pseudo_count must be finite and non-negative.")
        self.prior_p = _clip_prob(prior_p)
        self.block_prior = (
            None if block_prior is None or self.num_blocks is None else _normalize_prior(block_prior, self.num_blocks)
        )
        self.estimate_block_prior = require_exact_bool(estimate_block_prior, "estimate_block_prior")
        expected_joint = self.block_assignments is None
        if include_assignment_prior is not None and not isinstance(include_assignment_prior, (bool, np.bool_)):
            raise ValueError("include_assignment_prior must be Boolean or None.")
        if (
            include_assignment_prior is not None
            and require_exact_bool(include_assignment_prior, "include_assignment_prior") != expected_joint
        ):
            raise ValueError(
                f"include_assignment_prior must be {expected_joint} for this SBM estimator's sample space."
            )
        self.include_assignment_prior = expected_joint
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
        self, nobs: float | None, suff_stat: StochasticBlockGraphStatistics
    ) -> StochasticBlockGraphDistribution:
        """Estimate block probabilities and the block prior from sufficient statistics."""
        checked = _validate_sbm_statistics(
            suff_stat,
            num_blocks=self.num_blocks,
            directed=self.directed,
            self_loops=self.self_loops,
        )
        successes, totals, block_counts = checked.successes, checked.totals, checked.block_counts
        k = successes.shape[0]
        if k == 0:
            raise ValueError("cannot estimate an SBM without a declared or observed block dimension.")
        if self.pseudo_count is not None:
            successes += self.pseudo_count * self.prior_p
            totals += self.pseudo_count
        if not np.any(totals > 0.0):
            raise ValueError("cannot estimate an SBM without edge evidence or positive pseudo-count.")
        probs = np.divide(successes, totals, out=np.full_like(successes, float(self.prior_p)), where=totals > 0.0)
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
