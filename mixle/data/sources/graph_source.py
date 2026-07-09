"""Shared graph observation encoding helpers for stats graph distributions.

Graph observations may be square binary adjacency matrices, NetworkX-like graph
objects, ``(adjacency, block_assignments)`` pairs, or mappings with
``adjacency``/``adj`` and optional ``block_assignments``/``blocks``.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - scipy is a package dependency in normal use.
    sp = None

from mixle.stats.compute.pdist import DataSequenceEncoder

_EPS = 1.0e-12


@dataclass(frozen=True)
class GraphObservation:
    """Canonical binary graph observation used by graph encoders."""

    adjacency: np.ndarray
    block_assignments: np.ndarray | None = None


def _clip_prob(p: float) -> float:
    pp = float(p)
    if not np.isfinite(pp) or pp < 0.0 or pp > 1.0:
        raise ValueError("probabilities must be finite values in [0, 1].")
    return float(np.clip(pp, _EPS, 1.0 - _EPS))


def _bernoulli_log_likelihood(successes: float, total: float, p: float) -> float:
    pp = _clip_prob(p)
    return float(successes * math.log(pp) + (total - successes) * math.log1p(-pp))


def _edge_indices(n: int, directed: bool, self_loops: bool):
    if directed:
        for i in range(n):
            for j in range(n):
                if self_loops or i != j:
                    yield i, j
    else:
        start = 0 if self_loops else 1
        for i in range(n):
            for j in range(i + start, n):
                yield i, j


def _networkx_like_to_adjacency(graph: Any) -> tuple[np.ndarray, np.ndarray | None]:
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    adj = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    directed = bool(graph.is_directed()) if hasattr(graph, "is_directed") else False

    for edge in graph.edges(data=True):
        if len(edge) == 3:
            u, v, data = edge
        else:
            u, v = edge[:2]
            data = {}
        weight = 1.0
        if isinstance(data, Mapping):
            weight = data.get("weight", 1.0)
        adj[index[u], index[v]] = weight
        if not directed and u != v:
            adj[index[v], index[u]] = weight

    assignments = []
    found_assignment = False
    for node in nodes:
        value = None
        try:
            attrs = graph.nodes[node]
            if isinstance(attrs, Mapping):
                if "block" in attrs:
                    value = attrs["block"]
                elif "block_assignment" in attrs:
                    value = attrs["block_assignment"]
        except Exception:
            value = None
        assignments.append(value)
        found_assignment = found_assignment or value is not None

    if found_assignment:
        if any(value is None for value in assignments):
            raise ValueError("all graph nodes must have block labels when any node has one.")
        return adj, np.asarray(assignments, dtype=np.int64)
    return adj, None


def _edge_list_to_adjacency(edges: Sequence[Any], num_nodes: int, directed: bool) -> np.ndarray:
    n = int(num_nodes)
    if n < 0:
        raise ValueError("num_nodes must be non-negative.")
    adj = np.zeros((n, n), dtype=np.float64)
    for edge in edges:
        if len(edge) < 2:
            raise ValueError("edge entries must contain at least two node indices.")
        i = int(edge[0])
        j = int(edge[1])
        if i < 0 or i >= n or j < 0 or j >= n:
            raise ValueError("edge node indices must be in [0, num_nodes).")
        weight = float(edge[2]) if len(edge) >= 3 else 1.0
        adj[i, j] = weight
        if not directed and i != j:
            adj[j, i] = weight
    return adj


def _as_adjacency(adjacency: Any) -> np.ndarray:
    if hasattr(adjacency, "nodes") and hasattr(adjacency, "edges"):
        adjacency, _ = _networkx_like_to_adjacency(adjacency)
    elif sp is not None and sp.issparse(adjacency):
        adjacency = adjacency.toarray()

    adj = np.asarray(adjacency, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("graph adjacency must be a square matrix.")
    if np.any(~np.isfinite(adj)):
        raise ValueError("graph adjacency must be finite.")
    if np.any((adj != 0.0) & (adj != 1.0)):
        raise ValueError("graph adjacency must contain binary values 0/1.")
    return adj


def _as_assignments(assignments: Any | None, n: int) -> np.ndarray | None:
    if assignments is None:
        return None
    rv = np.asarray(assignments, dtype=np.int64)
    if rv.ndim != 1 or rv.shape[0] != n:
        raise ValueError("block assignments must be a length-%d one-dimensional sequence." % n)
    if rv.size and rv.min() < 0:
        raise ValueError("block assignments must be non-negative integers.")
    return rv


def _coerce_graph_observation(x: Any, directed: bool, fallback_assignments: Any | None) -> GraphObservation:
    adj = _as_adjacency(x.adjacency)
    return GraphObservation(adj, _as_assignments(x.block_assignments, adj.shape[0]))


def _coerce_mapping(x: Any, directed: bool, fallback_assignments: Any | None) -> GraphObservation:
    assignments = x.get("block_assignments", x.get("blocks", fallback_assignments))
    if "adjacency" in x:
        adj = _as_adjacency(x["adjacency"])
    elif "adj" in x:
        adj = _as_adjacency(x["adj"])
    elif "graph" in x:
        adj, graph_assignments = _networkx_like_to_adjacency(x["graph"])
        if assignments is None:
            assignments = graph_assignments
        adj = _as_adjacency(adj)
    elif "edges" in x and "num_nodes" in x:
        adj = _edge_list_to_adjacency(x["edges"], int(x["num_nodes"]), directed=directed)
    else:
        raise ValueError("graph mapping must contain adjacency, adj, graph, or edges+num_nodes.")
    return GraphObservation(adj, _as_assignments(assignments, adj.shape[0]))


def _coerce_pair_tuple(x: Any, directed: bool, fallback_assignments: Any | None) -> GraphObservation:
    # Mirror _coerce_pair_list's fallback: a 2-tuple is not ALWAYS an (adjacency, assignments) pair --
    # a plain 2-node adjacency matrix given as nested tuples, e.g. ((0, 1), (1, 0)), also matches this
    # predicate. Without the fallback, x[0] (a 1-D row) fails _as_adjacency's square-matrix check and
    # this raises, while the identical input as nested LISTS was already handled correctly.
    try:
        adj = _as_adjacency(x[0])
        assignments = x[1] if x[1] is not None else fallback_assignments
    except Exception:
        adj = _as_adjacency(x)
        assignments = fallback_assignments
    return GraphObservation(adj, _as_assignments(assignments, adj.shape[0]))


def _coerce_pair_list(x: Any, directed: bool, fallback_assignments: Any | None) -> GraphObservation:
    try:
        adj = _as_adjacency(x[0])
        assignments = x[1] if x[1] is not None else fallback_assignments
    except Exception:
        adj = _as_adjacency(x)
        assignments = fallback_assignments
    return GraphObservation(adj, _as_assignments(assignments, adj.shape[0]))


def _coerce_networkx_like(x: Any, directed: bool, fallback_assignments: Any | None) -> GraphObservation:
    adj, graph_assignments = _networkx_like_to_adjacency(x)
    assignments = graph_assignments if graph_assignments is not None else fallback_assignments
    adj = _as_adjacency(adj)
    return GraphObservation(adj, _as_assignments(assignments, adj.shape[0]))


def _coerce_adjacency_like(x: Any, directed: bool, fallback_assignments: Any | None) -> GraphObservation:
    adj = _as_adjacency(x)
    return GraphObservation(adj, _as_assignments(fallback_assignments, adj.shape[0]))


#: Ordered coercer registry: the first ``(predicate, handler)`` whose predicate matches ``x`` wins.
#: New input types register here instead of extending an ``isinstance`` ladder. Order matters --
#: it preserves the original precedence (GraphObservation -> Mapping -> 2-tuple -> 2-list ->
#: networkx-like -> raw adjacency fallback).
_OBSERVATION_COERCERS: list[tuple[Callable[[Any], bool], Callable[..., GraphObservation]]] = [
    (lambda x: isinstance(x, GraphObservation), _coerce_graph_observation),
    (lambda x: isinstance(x, Mapping), _coerce_mapping),
    (lambda x: isinstance(x, tuple) and len(x) == 2, _coerce_pair_tuple),
    (lambda x: isinstance(x, list) and len(x) == 2 and not np.isscalar(x[0]), _coerce_pair_list),
    (lambda x: hasattr(x, "nodes") and hasattr(x, "edges"), _coerce_networkx_like),
]


def _extract_observation(x: Any, directed: bool = False, fallback_assignments: Any | None = None) -> GraphObservation:
    for predicate, handler in _OBSERVATION_COERCERS:
        if predicate(x):
            return handler(x, directed, fallback_assignments)
    return _coerce_adjacency_like(x, directed, fallback_assignments)


def _edge_counts(adj: np.ndarray, directed: bool, self_loops: bool) -> tuple[float, float]:
    n = adj.shape[0]
    diag = np.trace(adj)
    if directed:
        if self_loops:
            total = float(n * n)
            successes = float(adj.sum())
        else:
            total = float(n * (n - 1))
            successes = float(adj.sum() - diag)
    else:
        if self_loops:
            total = float(n * (n + 1) // 2)
            successes = float(adj[np.triu_indices(n, k=0)].sum())
        else:
            total = float(n * (n - 1) // 2)
            successes = float(adj[np.triu_indices(n, k=1)].sum())
    return total, successes


def _validate_block_probs(block_probs: Any) -> np.ndarray:
    probs = np.asarray(block_probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != probs.shape[1]:
        raise ValueError("block_probs must be a square matrix.")
    if probs.shape[0] == 0:
        raise ValueError("block_probs must contain at least one block.")
    if np.any(~np.isfinite(probs)) or np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("block probabilities must be finite and in [0, 1].")
    return probs


def _validate_block_indices(assignments: np.ndarray, num_blocks: int) -> None:
    if assignments.ndim != 1:
        raise ValueError("block assignments must be a one-dimensional sequence.")
    if assignments.size and (assignments.min() < 0 or assignments.max() >= num_blocks):
        raise ValueError("block assignments must index block_probs.")


def _normalize_prior(block_prior: Any | None, num_blocks: int) -> np.ndarray:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive.")
    if block_prior is None:
        return np.full(int(num_blocks), 1.0 / float(num_blocks), dtype=np.float64)
    prior = np.asarray(block_prior, dtype=np.float64)
    if prior.ndim != 1 or prior.shape[0] != num_blocks:
        raise ValueError("block_prior must be a length-num_blocks vector.")
    if np.any(~np.isfinite(prior)) or np.any(prior < 0.0) or prior.sum() <= 0.0:
        raise ValueError("block_prior must contain non-negative finite values with positive sum.")
    return prior / prior.sum()


class GraphDataEncoder(DataSequenceEncoder):
    """Encode graph observations as canonical adjacency/assignment objects."""

    def __init__(self, directed: bool = False, fallback_assignments: Any | None = None) -> None:
        self.directed = bool(directed)
        self.fallback_assignments = (
            None
            if fallback_assignments is None
            else tuple(int(u) for u in np.asarray(fallback_assignments, dtype=np.int64))
        )

    def __str__(self) -> str:
        return "GraphDataEncoder(directed=%s)" % repr(self.directed)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GraphDataEncoder)
            and self.directed == other.directed
            and self.fallback_assignments == other.fallback_assignments
        )

    def seq_encode(self, x: Sequence[Any]) -> tuple[GraphObservation, ...]:
        """Encode graph-like observations into graph observation records."""
        fallback = None if self.fallback_assignments is None else np.asarray(self.fallback_assignments, dtype=np.int64)
        return tuple(_extract_observation(u, directed=self.directed, fallback_assignments=fallback) for u in x)

    def nbytes(self, x: Any) -> int:
        """Return approximate encoded byte size for graph observations."""
        total = 0
        for obs in x:
            total += int(obs.adjacency.nbytes)
            if obs.block_assignments is not None:
                total += int(obs.block_assignments.nbytes)
        return total
