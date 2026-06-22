"""Dependency-free random graph models.

These model helpers deliberately keep graph likelihood math in the model layer.
They do not add graph-specific code to compute engines.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

_EPS = 1.0e-12


@dataclass
class HardEMResult:
    """Result from hard-EM fitting of a stochastic block model."""

    model: StochasticBlockGraphModel
    history: list[float]


class ErdosRenyiGraphModel:
    """Independent Bernoulli edge model for directed or undirected graphs."""

    def __init__(self, p: float, directed: bool = False, self_loops: bool = False, name: str | None = None) -> None:
        if p < 0.0 or p > 1.0 or not np.isfinite(p):
            raise ValueError("ErdosRenyiGraphModel requires p in [0, 1].")
        self.p = float(p)
        self.directed = bool(directed)
        self.self_loops = bool(self_loops)
        self.name = name

    def __str__(self) -> str:
        return "ErdosRenyiGraphModel(p=%r, directed=%r, self_loops=%r, name=%r)" % (
            self.p,
            self.directed,
            self.self_loops,
            self.name,
        )

    @classmethod
    def fit_mle(
        cls,
        adjacency: Any,
        directed: bool = False,
        self_loops: bool = False,
        pseudo_count: float = 0.0,
        prior_p: float = 0.5,
        name: str | None = None,
    ) -> ErdosRenyiGraphModel:
        """Thin shim delegating to ``fit_erdos_renyi_mle`` (kept for the classmethod-fit call API)."""
        return fit_erdos_renyi_mle(
            adjacency,
            directed=directed,
            self_loops=self_loops,
            pseudo_count=pseudo_count,
            prior_p=prior_p,
            name=name,
        )

    def log_likelihood(self, adjacency: Any) -> float:
        """Return the Bernoulli graph log likelihood."""
        adj = _as_adjacency(adjacency)
        values = _edge_values(adj, directed=self.directed, self_loops=self.self_loops)
        return _bernoulli_log_likelihood(values, self.p)

    def sample(self, num_nodes: int, seed: int | None = None) -> np.ndarray:
        """Draw one binary adjacency matrix."""
        if num_nodes < 0:
            raise ValueError("num_nodes must be non-negative.")
        rng = np.random.RandomState(seed)
        mat = (rng.rand(num_nodes, num_nodes) < self.p).astype(np.int8)
        if not self.directed:
            upper = np.triu(mat, k=1 if not self.self_loops else 0)
            mat = upper + upper.T
            if self.self_loops:
                diag = (rng.rand(num_nodes) < self.p).astype(np.int8)
                np.fill_diagonal(mat, diag)
        elif not self.self_loops:
            np.fill_diagonal(mat, 0)
        return mat

    def bic(self, adjacency: Any) -> float:
        """Bayesian information criterion with one free parameter."""
        n_edges = _edge_values(_as_adjacency(adjacency), self.directed, self.self_loops).size
        return -2.0 * self.log_likelihood(adjacency) + np.log(max(1, n_edges))


class StochasticBlockGraphModel:
    """Bernoulli stochastic block model with fixed node assignments."""

    def __init__(
        self,
        block_probs: Any,
        block_assignments: Sequence[int],
        directed: bool = False,
        self_loops: bool = False,
        name: str | None = None,
    ) -> None:
        probs = np.asarray(block_probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[0] != probs.shape[1]:
            raise ValueError("block_probs must be a square matrix.")
        if np.any(~np.isfinite(probs)) or np.any(probs < 0.0) or np.any(probs > 1.0):
            raise ValueError("block probabilities must be finite and in [0, 1].")
        assignments = np.asarray(block_assignments, dtype=np.int64)
        if assignments.ndim != 1:
            raise ValueError("block_assignments must be a one-dimensional sequence.")
        if assignments.size and (assignments.min() < 0 or assignments.max() >= probs.shape[0]):
            raise ValueError("block assignments must index block_probs.")
        if not directed and not np.allclose(probs, probs.T):
            raise ValueError("undirected block_probs must be symmetric.")
        self.block_probs = probs
        self.block_assignments = assignments
        self.num_blocks = int(probs.shape[0])
        self.directed = bool(directed)
        self.self_loops = bool(self_loops)
        self.name = name

    def __str__(self) -> str:
        return "StochasticBlockGraphModel(num_blocks=%d, directed=%r, self_loops=%r, name=%r)" % (
            self.num_blocks,
            self.directed,
            self.self_loops,
            self.name,
        )

    @classmethod
    def fit_mle(
        cls,
        adjacency: Any,
        block_assignments: Sequence[int],
        num_blocks: int | None = None,
        directed: bool = False,
        self_loops: bool = False,
        pseudo_count: float = 0.0,
        prior_p: float = 0.5,
        name: str | None = None,
    ) -> StochasticBlockGraphModel:
        """Thin shim delegating to ``fit_stochastic_block_mle`` (kept for the classmethod-fit call API)."""
        return fit_stochastic_block_mle(
            adjacency,
            block_assignments,
            num_blocks=num_blocks,
            directed=directed,
            self_loops=self_loops,
            pseudo_count=pseudo_count,
            prior_p=prior_p,
            name=name,
        )

    def log_likelihood(self, adjacency: Any) -> float:
        """Return the Bernoulli SBM log likelihood."""
        adj = _as_adjacency(adjacency)
        if adj.shape[0] != self.block_assignments.shape[0]:
            raise ValueError("adjacency size must match block assignments.")
        ll = 0.0
        for i, j in _edge_indices(adj.shape[0], self.directed, self.self_loops):
            p = self.block_probs[self.block_assignments[i], self.block_assignments[j]]
            ll += _bernoulli_log_likelihood(np.asarray([adj[i, j]]), p)
        return float(ll)

    def sample(self, seed: int | None = None) -> np.ndarray:
        """Draw one graph from the block model."""
        rng = np.random.RandomState(seed)
        n = self.block_assignments.shape[0]
        mat = np.zeros((n, n), dtype=np.int8)
        for i, j in _edge_indices(n, self.directed, self.self_loops):
            p = self.block_probs[self.block_assignments[i], self.block_assignments[j]]
            edge = int(rng.rand() < p)
            mat[i, j] = edge
            if not self.directed and i != j:
                mat[j, i] = edge
        return mat

    def bic(self, adjacency: Any) -> float:
        """BIC using the number of identifiable block edge probabilities."""
        n_edges = _edge_values(_as_adjacency(adjacency), self.directed, self.self_loops).size
        k = self.num_blocks * self.num_blocks if self.directed else self.num_blocks * (self.num_blocks + 1) / 2
        return -2.0 * self.log_likelihood(adjacency) + float(k) * np.log(max(1, n_edges))


def fit_erdos_renyi_mle(
    adjacency: Any,
    directed: bool = False,
    self_loops: bool = False,
    pseudo_count: float = 0.0,
    prior_p: float = 0.5,
    name: str | None = None,
) -> ErdosRenyiGraphModel:
    """Conjugate-Bernoulli MLE of the edge probability (module-level estimation, not a classmethod-fit)."""
    adj = _as_adjacency(adjacency)
    values = _edge_values(adj, directed=directed, self_loops=self_loops)
    successes = float(values.sum())
    total = float(values.size)
    if pseudo_count > 0.0:
        successes += float(pseudo_count) * float(prior_p)
        total += float(pseudo_count)
    p = 0.5 if total == 0.0 else successes / total
    return ErdosRenyiGraphModel(p, directed=directed, self_loops=self_loops, name=name)


def fit_stochastic_block_mle(
    adjacency: Any,
    block_assignments: Sequence[int],
    num_blocks: int | None = None,
    directed: bool = False,
    self_loops: bool = False,
    pseudo_count: float = 0.0,
    prior_p: float = 0.5,
    name: str | None = None,
) -> StochasticBlockGraphModel:
    """Conjugate-Bernoulli MLE of block edge probabilities for fixed assignments (module-level estimation)."""
    adj = _as_adjacency(adjacency)
    assignments = np.asarray(block_assignments, dtype=np.int64)
    if assignments.shape[0] != adj.shape[0]:
        raise ValueError("block_assignments length must equal the number of nodes.")
    if num_blocks is None:
        num_blocks = 0 if assignments.size == 0 else int(assignments.max()) + 1
    successes = np.zeros((num_blocks, num_blocks), dtype=np.float64)
    totals = np.zeros((num_blocks, num_blocks), dtype=np.float64)
    for i, j in _edge_indices(adj.shape[0], directed=directed, self_loops=self_loops):
        a = assignments[i]
        b = assignments[j]
        successes[a, b] += adj[i, j]
        totals[a, b] += 1.0
        if not directed and a != b:
            successes[b, a] += adj[i, j]
            totals[b, a] += 1.0
    if pseudo_count > 0.0:
        successes += float(pseudo_count) * float(prior_p)
        totals += float(pseudo_count)
    probs = np.divide(successes, totals, out=np.full_like(successes, float(prior_p)), where=totals > 0.0)
    if not directed:
        probs = 0.5 * (probs + probs.T)
    return StochasticBlockGraphModel(probs, assignments, directed=directed, self_loops=self_loops, name=name)


def hard_em_stochastic_block_model(
    adjacency: Any,
    num_blocks: int,
    max_its: int = 20,
    restarts: int = 1,
    seed: int | None = None,
    directed: bool = False,
    self_loops: bool = False,
    pseudo_count: float = 1.0,
    prior_p: float = 0.5,
) -> HardEMResult:
    """Classification/hard-EM fit for a stochastic block model."""
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive.")
    adj = _as_adjacency(adjacency)
    rng = np.random.RandomState(seed)
    best_model = None
    best_history: list[float] = []
    best_ll = -np.inf

    for _ in range(max(1, int(restarts))):
        assignments = _initial_assignments(adj.shape[0], num_blocks, rng)
        history: list[float] = []
        model = fit_stochastic_block_mle(
            adj,
            assignments,
            num_blocks=num_blocks,
            directed=directed,
            self_loops=self_loops,
            pseudo_count=pseudo_count,
            prior_p=prior_p,
        )
        ll = model.log_likelihood(adj)
        history.append(ll)
        for _ in range(max(1, int(max_its))):
            candidate_assignments = _hard_reassign(adj, model)
            candidate_model = fit_stochastic_block_mle(
                adj,
                candidate_assignments,
                num_blocks=num_blocks,
                directed=directed,
                self_loops=self_loops,
                pseudo_count=pseudo_count,
                prior_p=prior_p,
            )
            candidate_ll = candidate_model.log_likelihood(adj)
            if candidate_ll < ll - 1.0e-12:
                break
            assignments = candidate_assignments
            model = candidate_model
            history.append(candidate_ll)
            if abs(candidate_ll - ll) < 1.0e-12:
                break
            ll = candidate_ll
        if history and history[-1] > best_ll:
            best_ll = history[-1]
            best_model = model
            best_history = history
    return HardEMResult(best_model, best_history)


def _as_adjacency(adjacency: Any) -> np.ndarray:
    adj = np.asarray(adjacency, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency must be a square matrix.")
    if np.any(~np.isfinite(adj)):
        raise ValueError("adjacency must be finite.")
    if np.any((adj != 0.0) & (adj != 1.0)):
        raise ValueError("adjacency must contain binary values 0/1.")
    return adj


def _edge_indices(n: int, directed: bool, self_loops: bool):
    if directed:
        for i in range(n):
            for j in range(n):
                if self_loops or i != j:
                    yield i, j
    else:
        start_offset = 0 if self_loops else 1
        for i in range(n):
            for j in range(i + start_offset, n):
                yield i, j


def _edge_values(adj: np.ndarray, directed: bool, self_loops: bool) -> np.ndarray:
    return np.asarray([adj[i, j] for i, j in _edge_indices(adj.shape[0], directed, self_loops)], dtype=np.float64)


def _bernoulli_log_likelihood(values: np.ndarray, p: float) -> float:
    pp = float(np.clip(p, _EPS, 1.0 - _EPS))
    return float(values.sum() * np.log(pp) + (values.size - values.sum()) * np.log1p(-pp))


def _initial_assignments(n: int, num_blocks: int, rng: np.random.RandomState) -> np.ndarray:
    assignments = rng.randint(0, num_blocks, size=n)
    for k in range(min(n, num_blocks)):
        assignments[k] = k
    return assignments


def _hard_reassign(adj: np.ndarray, model: StochasticBlockGraphModel) -> np.ndarray:
    assignments = model.block_assignments.copy()
    for i in range(adj.shape[0]):
        scores = np.asarray([_node_block_score(adj, model, i, k) for k in range(model.num_blocks)])
        assignments[i] = int(np.argmax(scores))
    return assignments


def _node_block_score(adj: np.ndarray, model: StochasticBlockGraphModel, node: int, block: int) -> float:
    score = 0.0
    for j in range(adj.shape[0]):
        if not model.self_loops and j == node:
            continue
        bj = model.block_assignments[j]
        p = model.block_probs[block, bj]
        score += _bernoulli_log_likelihood(np.asarray([adj[node, j]]), p)
        if model.directed:
            p_in = model.block_probs[bj, block]
            score += _bernoulli_log_likelihood(np.asarray([adj[j, node]]), p_in)
    return float(score)
