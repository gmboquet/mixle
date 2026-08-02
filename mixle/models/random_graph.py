"""Dependency-free random graph models.

These model helpers deliberately keep graph likelihood math in the model layer.
They do not add graph-specific code to compute engines.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from mixle.models._result import FitResult

_EPS = 1.0e-12


@dataclass
class HardEMResult(FitResult["StochasticBlockGraphModel"]):
    """Result from hard-EM fitting of a stochastic block model."""


class ErdosRenyiGraphModel:
    """Independent Bernoulli edge model for directed or undirected graphs."""

    def __init__(self, p: float, directed: bool = False, self_loops: bool = False, name: str | None = None) -> None:
        self.p = _probability(p, "p")
        self.directed = _exact_bool(directed, "directed")
        self.self_loops = _exact_bool(self_loops, "self_loops")
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
        _validate_adjacency_structure(adj, directed=self.directed, self_loops=self.self_loops)
        values = _edge_values(adj, directed=self.directed, self_loops=self.self_loops)
        return _bernoulli_log_likelihood(values, self.p)

    def sample(self, num_nodes: int, seed: int | None = None) -> np.ndarray:
        """Draw one binary adjacency matrix."""
        num_nodes = _exact_int(num_nodes, "num_nodes", minimum=0)
        seed = _seed(seed)
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
        # np.asarray does NOT copy an array that is already float64, so the model aliased the
        # caller's matrix: every validation below passed, and the caller then wrote a different
        # probability into their own array and changed what this model scores (MXR-080-1889). The
        # copy is taken before validation so what is checked is what is kept.
        probs = np.array(block_probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[0] != probs.shape[1] or probs.shape[0] == 0:
            raise ValueError("block_probs must be a non-empty square matrix.")
        if np.any(~np.isfinite(probs)) or np.any(probs < 0.0) or np.any(probs > 1.0):
            raise ValueError("block probabilities must be finite and in [0, 1].")
        assignments = _assignments(block_assignments, probs.shape[0])
        directed = _exact_bool(directed, "directed")
        self_loops = _exact_bool(self_loops, "self_loops")
        if not directed and not np.array_equal(probs, probs.T):
            raise ValueError("undirected block_probs must be symmetric.")
        self.block_probs = probs
        self.block_assignments = assignments
        self.num_blocks = int(probs.shape[0])
        self.directed = directed
        self.self_loops = self_loops
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
        _validate_adjacency_structure(adj, directed=self.directed, self_loops=self.self_loops)
        ll = 0.0
        for i, j in _edge_indices(adj.shape[0], self.directed, self.self_loops):
            p = self.block_probs[self.block_assignments[i], self.block_assignments[j]]
            ll += _bernoulli_log_likelihood(np.asarray([adj[i, j]]), p)
        return float(ll)

    def sample(self, seed: int | None = None) -> np.ndarray:
        """Draw one graph from the block model."""
        # Handed straight to RandomState, which accepts True as the seed 1: this method did not use
        # the module's own _seed validator that ErdosRenyiGraphModel.sample has always used, so the
        # two public samplers in one module disagreed about what a seed is (MXR-080-1889).
        seed = _seed(seed)
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
    directed = _exact_bool(directed, "directed")
    self_loops = _exact_bool(self_loops, "self_loops")
    pseudo_count = _nonnegative_finite(pseudo_count, "pseudo_count")
    prior_p = _probability(prior_p, "prior_p")
    adj = _as_adjacency(adjacency)
    _validate_adjacency_structure(adj, directed=directed, self_loops=self_loops)
    values = _edge_values(adj, directed=directed, self_loops=self_loops)
    successes = float(values.sum())
    total = float(values.size)
    if pseudo_count > 0.0:
        successes += pseudo_count * prior_p
        total += pseudo_count
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
    directed = _exact_bool(directed, "directed")
    self_loops = _exact_bool(self_loops, "self_loops")
    pseudo_count = _nonnegative_finite(pseudo_count, "pseudo_count")
    prior_p = _probability(prior_p, "prior_p")
    adj = _as_adjacency(adjacency)
    _validate_adjacency_structure(adj, directed=directed, self_loops=self_loops)
    if num_blocks is None:
        assignments = _assignments(block_assignments)
        if assignments.size == 0:
            raise ValueError("num_blocks is required when block_assignments is empty")
        num_blocks = int(assignments.max()) + 1
    else:
        num_blocks = _exact_int(num_blocks, "num_blocks", minimum=1)
        assignments = _assignments(block_assignments, num_blocks)
    if assignments.shape[0] != adj.shape[0]:
        raise ValueError("block_assignments length must equal the number of nodes.")
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
        successes += pseudo_count * prior_p
        totals += pseudo_count
    probs = np.divide(successes, totals, out=np.full_like(successes, prior_p), where=totals > 0.0)
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
    num_blocks = _exact_int(num_blocks, "num_blocks", minimum=1)
    max_its = _exact_int(max_its, "max_its", minimum=1)
    restarts = _exact_int(restarts, "restarts", minimum=1)
    seed = _seed(seed)
    directed = _exact_bool(directed, "directed")
    self_loops = _exact_bool(self_loops, "self_loops")
    pseudo_count = _nonnegative_finite(pseudo_count, "pseudo_count")
    prior_p = _probability(prior_p, "prior_p")
    adj = _as_adjacency(adjacency)
    _validate_adjacency_structure(adj, directed=directed, self_loops=self_loops)
    rng = np.random.RandomState(seed)
    best_model = None
    best_history: list[float] = []
    best_ll = -np.inf

    for _ in range(restarts):
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
        for _ in range(max_its):
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


def _exact_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _exact_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _seed(value: Any) -> int | None:
    if value is None:
        return None
    result = _exact_int(value, "seed", minimum=0)
    if result > np.iinfo(np.uint32).max:
        raise ValueError("seed must fit in an unsigned 32-bit integer")
    return result


def _nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real number")
    return result


def _probability(value: Any, name: str) -> float:
    result = _nonnegative_finite(value, name)
    if result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _assignments(value: Any, num_blocks: int | None = None) -> np.ndarray:
    assignments = np.asarray(value)
    if assignments.ndim != 1:
        raise ValueError("block_assignments must be a one-dimensional sequence")
    if assignments.dtype.kind not in {"i", "u"}:
        raise ValueError("block_assignments must contain exact integer indices")
    if np.any(assignments < 0) or np.any(assignments > np.iinfo(np.intp).max):
        raise ValueError("block_assignments must contain supported non-negative indices")
    assignments = assignments.astype(np.int64, copy=False)
    if num_blocks is not None and assignments.size and np.any(assignments >= num_blocks):
        raise ValueError(f"block_assignments must index the {num_blocks} declared blocks")
    return assignments


def _validate_adjacency_structure(adj: np.ndarray, directed: bool, self_loops: bool) -> None:
    """Reject adjacency matrices that violate the model's directed/self-loop assumptions.

    ``_edge_indices`` only ever iterates the index pairs implied by ``directed``/``self_loops``;
    it never inspects the matrix itself, so an asymmetric matrix scored by an undirected model
    (or a nonzero diagonal scored by a self_loops=False model) would otherwise be read only on
    the subset of entries the iteration happens to visit, silently ignoring the rest.
    """
    if not directed and not np.array_equal(adj, adj.T):
        raise ValueError("undirected models require a symmetric adjacency matrix.")
    if not self_loops and np.any(np.diagonal(adj) != 0.0):
        raise ValueError("self_loops=False models require an all-zero diagonal (no self-loops).")


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
    """Exact Bernoulli log-likelihood, including at ``p == 0`` and ``p == 1`` (MXR-080-1889).

    Clipping ``p`` into ``[eps, 1 - eps]`` gave every declared endpoint a finite score: a ``p = 0``
    model scored a PRESENT edge at ``log(1e-12) = -27.63`` -- finite probability for evidence the
    model declares impossible -- and scored a graph with no edges at ``-1e-12`` rather than the exact
    ``0`` that a certain event has. Both directions matter: the first lets impossible evidence enter a
    likelihood ratio, a BIC, or an EM responsibility as though it were merely unlikely, and the second
    means a model that predicts its data perfectly does not say so exactly.

    The clip was presumably there for ``0 * log(0)``, which is ``0 * -inf = nan`` computed naively.
    The convention is that an outcome observed zero times contributes nothing, so the endpoints are
    handled by counting instead: at ``p == 0`` the likelihood is ``-inf`` if any edge is present and
    exactly ``0`` otherwise, and symmetrically at ``p == 1``.
    """
    present = float(values.sum())
    absent = float(values.size) - present
    if p <= 0.0:
        return float("-inf") if present > 0.0 else 0.0
    if p >= 1.0:
        return float("-inf") if absent > 0.0 else 0.0
    return float(present * np.log(p) + absent * np.log1p(-p))


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
