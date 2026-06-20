"""Combinatorial optimization problems with best-first solution enumeration.

The pysp idiom for enumeration: you *specify* a problem (its data + objective), then ask it for an
``enumerator()`` of solutions in best-first order -- the same shape as a distribution yielding a
``sampler()`` / ``estimator()`` / ``enumerator()``. Every problem shares one surface:

    problem.best()        -> the single optimal ``(solution, objective)`` (or ``None`` if infeasible)
    problem.top(k)        -> the ``k`` best solutions as a list
    problem.enumerator()  -> a lazy iterator over ``(solution, objective)``, best-first
    for solution, objective in problem: ...

So assignment, spanning tree, weighted edit distance, k-best Viterbi, shortest path, and best-subset
regression are all *specified and consumed the same way*, each delegating to whatever engine fits
(Murty for assignment, Gabow for spanning trees, A* / :func:`best_first_paths` for paths/edit/Viterbi,
exhaustive ranking for best-subset). ``sense`` records whether the objective is minimized
(increasing cost out) or maximized (decreasing score out).

:func:`best_first_paths` is the shared low-level engine: lazy best-first / A* over an arbitrary state
graph. :class:`ShortestPath` is its problem-object wrapper; the other problems reduce onto it or onto
the dedicated k-best engines.
"""

from __future__ import annotations

import heapq
import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

from pysp.utils.assignment import k_best_assignments
from pysp.utils.spanning import k_best_spanning_trees

__all__ = [
    "Assignment",
    "BestSubsetRegression",
    "EditDistance",
    "OptimizationProblem",
    "ShortestPath",
    "SpanningTree",
    "ViterbiPath",
    "best_first_paths",
]


# ---------------------------------------------------------------------------
# Shared low-level engine: lazy best-first / A* over an arbitrary state graph
# ---------------------------------------------------------------------------
def best_first_paths(
    start: Any,
    successors: Callable[[Any], Iterable[tuple[Any, float]]],
    is_goal: Callable[[Any], bool],
    *,
    sense: str = "min",
    heuristic: Callable[[Any], float] | None = None,
    max_results: int | None = None,
    return_paths: bool = True,
) -> Iterator[tuple[Any, float]]:
    """Lazily enumerate goal states in monotone order of total additive cost/score.

    Args:
        start: The initial state.
        successors: ``state -> iterable of (next_state, step)`` where ``step`` is the edge cost
            (``sense="min"``) or edge score (``sense="max"``).
        is_goal: Predicate; when true for a popped state it is emitted (and not expanded further).
        sense: ``"min"`` to minimise total cost (increasing order out) or ``"max"`` to maximise total
            score (decreasing order out).
        heuristic: Optional admissible estimate of the *remaining* cost (a lower bound, ``min``) or
            remaining score (an upper bound, ``max``); ``0`` at goals. ``None`` is the always-admissible
            zero heuristic (uniform-cost search).
        max_results: Stop after yielding this many goals (``None`` = exhaust the graph).
        return_paths: Yield ``(path_list, total)`` when true, else ``(goal_state, total)``.

    Yields:
        ``(path_or_state, total)`` in best-first order; with an admissible heuristic the order is
        exact, and on a DAG/trellis the search enumerates k-best paths.
    """
    if sense not in ("min", "max"):
        raise ValueError("sense must be 'min' or 'max'")
    flip = 1.0 if sense == "min" else -1.0
    h = heuristic or (lambda _s: 0.0)
    cnt = itertools.count()

    def priority(g: float, state: Any) -> float:
        return flip * (g + h(state))

    heap: list[tuple[float, int, float, Any, tuple]] = [(priority(0.0, start), next(cnt), 0.0, start, (start,))]
    emitted = 0
    while heap:
        _, _, g, state, path = heapq.heappop(heap)
        if is_goal(state):
            yield (list(path) if return_paths else state, g)
            emitted += 1
            if max_results is not None and emitted >= max_results:
                return
            continue
        for nxt, step in successors(state):
            g2 = g + step
            heapq.heappush(heap, (priority(g2, nxt), next(cnt), g2, nxt, path + (nxt,)))


# ---------------------------------------------------------------------------
# The shared problem interface
# ---------------------------------------------------------------------------
class OptimizationProblem(ABC):
    """A combinatorial optimization problem whose solutions can be enumerated best-first.

    Subclasses implement :meth:`enumerator`; :meth:`best`, :meth:`top` and iteration come for free.
    ``sense`` is ``"min"`` (objective minimized, solutions out in increasing cost) or ``"max"``
    (objective maximized, solutions out in decreasing score).
    """

    sense: str = "min"

    @abstractmethod
    def enumerator(self, k: int | None = None) -> Iterator[tuple[Any, float]]:
        """Lazily yield ``(solution, objective)`` best-first; at most ``k`` if given (``None`` = all)."""

    def best(self) -> tuple[Any, float] | None:
        """The single optimal ``(solution, objective)``, or ``None`` if the problem is infeasible."""
        return next(self.enumerator(k=1), None)

    def top(self, k: int) -> list[tuple[Any, float]]:
        """The ``k`` best ``(solution, objective)`` pairs as a list."""
        return list(self.enumerator(k=k))

    def __iter__(self) -> Iterator[tuple[Any, float]]:
        return self.enumerator()


# ---------------------------------------------------------------------------
# General graph problem (direct wrapper of the engine)
# ---------------------------------------------------------------------------
class ShortestPath(OptimizationProblem):
    """k-shortest-path / best-first search over an arbitrary state graph.

    Specify the graph by callables; the solution is the list of states from ``start`` to a goal.
    """

    def __init__(
        self,
        start: Any,
        successors: Callable[[Any], Iterable[tuple[Any, float]]],
        is_goal: Callable[[Any], bool],
        *,
        sense: str = "min",
        heuristic: Callable[[Any], float] | None = None,
    ) -> None:
        self.start = start
        self.successors = successors
        self.is_goal = is_goal
        self.sense = sense
        self.heuristic = heuristic

    def enumerator(self, k: int | None = None) -> Iterator[tuple[Any, float]]:
        return best_first_paths(
            self.start, self.successors, self.is_goal, sense=self.sense, heuristic=self.heuristic, max_results=k
        )


# ---------------------------------------------------------------------------
# Linear assignment (Murty)
# ---------------------------------------------------------------------------
class Assignment(OptimizationProblem):
    """Linear assignment / bipartite matching: match rows to columns at extremal total cost.

    The solution is ``col_ind`` -- ``col_ind[i]`` is the column assigned to row ``i``.
    """

    def __init__(self, cost: np.ndarray, maximize: bool = False) -> None:
        self.cost = np.asarray(cost, dtype=np.float64)
        if self.cost.ndim != 2:
            raise ValueError("cost must be a 2-D matrix")
        self.maximize = bool(maximize)
        self.sense = "max" if maximize else "min"

    def enumerator(self, k: int | None = None) -> Iterator[tuple[np.ndarray, float]]:
        for total, _rows, cols in k_best_assignments(self.cost, k=k, maximize=self.maximize):
            yield cols, float(total)


# ---------------------------------------------------------------------------
# Minimum spanning tree (Gabow)
# ---------------------------------------------------------------------------
class SpanningTree(OptimizationProblem):
    """Spanning trees of a weighted undirected graph, enumerated in increasing total edge weight.

    The solution is the list of ``(i, j)`` edges. Non-finite weights are forbidden edges.
    """

    sense = "min"

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.weights.ndim != 2 or self.weights.shape[0] != self.weights.shape[1]:
            raise ValueError("weights must be a square matrix")

    def enumerator(self, k: int | None = None) -> Iterator[tuple[list[tuple[int, int]], float]]:
        for total, edges in k_best_spanning_trees(self.weights, k=k):
            yield edges, float(total)


# ---------------------------------------------------------------------------
# Non-uniform (weighted) edit distance / alignment
# ---------------------------------------------------------------------------
class EditDistance(OptimizationProblem):
    """Non-uniform (weighted) edit distance: enumerate edit scripts in increasing total cost.

    Each operation carries its own (possibly symbol/position-dependent) cost, so this is the general
    weighted edit distance / alignment. It reduces to a shortest path in the edit DAG over cursor
    states ``(i, j)``: a diagonal step matches/substitutes, a vertical step deletes from ``source``,
    a horizontal step inserts from ``target``. The solution is a list of ``(kind, a, b)`` operations
    with ``kind`` in ``{"match", "sub", "del", "ins"}``; the best objective is the edit distance.
    """

    sense = "min"

    def __init__(
        self,
        source: Iterable[Any],
        target: Iterable[Any],
        *,
        sub_cost: Callable[[Any, Any], float] | None = None,
        ins_cost: Callable[[Any], float] | None = None,
        del_cost: Callable[[Any], float] | None = None,
    ) -> None:
        self.source = list(source)
        self.target = list(target)
        self.sub_cost = sub_cost or (lambda a, b: 0.0 if a == b else 1.0)
        self.ins_cost = ins_cost or (lambda b: 1.0)
        self.del_cost = del_cost or (lambda a: 1.0)

    def enumerator(self, k: int | None = None) -> Iterator[tuple[list[tuple[str, Any, Any]], float]]:
        ns, nt = len(self.source), len(self.target)
        goal = (ns, nt)

        def successors(node):
            i, j = node
            out = []
            if i < ns and j < nt:
                out.append(((i + 1, j + 1), self.sub_cost(self.source[i], self.target[j])))
            if i < ns:
                out.append(((i + 1, j), self.del_cost(self.source[i])))
            if j < nt:
                out.append(((i, j + 1), self.ins_cost(self.target[j])))
            return out

        for path, cost in best_first_paths((0, 0), successors, lambda n: n == goal, sense="min", max_results=k):
            yield self._path_to_ops(path), cost

    def _path_to_ops(self, path):
        ops = []
        for (i0, j0), (i1, j1) in zip(path, path[1:]):
            if i1 == i0 + 1 and j1 == j0 + 1:
                a, b = self.source[i0], self.target[j0]
                ops.append(("match" if a == b else "sub", a, b))
            elif i1 == i0 + 1:
                ops.append(("del", self.source[i0], None))
            else:
                ops.append(("ins", None, self.target[j0]))
        return ops


# ---------------------------------------------------------------------------
# k-best Viterbi (most-likely hidden-state sequences of an HMM)
# ---------------------------------------------------------------------------
class ViterbiPath(OptimizationProblem):
    """k most-likely hidden-state sequences of an HMM, enumerated in decreasing joint log-probability.

    Standard Viterbi returns only the single best path; this reduces the trellis (nodes ``(t, s)``)
    to a longest-log-prob path and yields the top ``k``. The solution is a length-``T`` list of state
    indices.

    Args:
        log_init: ``log p(state s at t=0)``, length ``S``.
        log_trans: ``log p(s' | s)``, shape ``(S, S)``.
        log_obs: ``log p(observation_t | state s)``, shape ``(T, S)`` (emission log-likelihoods).
    """

    sense = "max"

    def __init__(self, log_init: Any, log_trans: Any, log_obs: Any) -> None:
        self.log_init = np.asarray(log_init, dtype=np.float64)
        self.log_trans = np.asarray(log_trans, dtype=np.float64)
        self.log_obs = [list(row) for row in log_obs]
        self.n_states = self.log_init.shape[0]
        self.n_steps = len(self.log_obs)

    def enumerator(self, k: int | None = None) -> Iterator[tuple[list[int], float]]:
        if self.n_steps == 0:
            return
        t_last, s = self.n_steps - 1, self.n_states
        log_init, log_trans, log_obs = self.log_init, self.log_trans, self.log_obs

        def successors(node):
            t, state = node
            if t == -1:
                return [((0, sp), float(log_init[sp]) + float(log_obs[0][sp])) for sp in range(s)]
            if t >= t_last:
                return []
            return [((t + 1, sp), float(log_trans[state][sp]) + float(log_obs[t + 1][sp])) for sp in range(s)]

        for path, score in best_first_paths((-1, -1), successors, lambda n: n[0] == t_last, sense="max", max_results=k):
            yield [st for (_t, st) in path[1:]], score


# ---------------------------------------------------------------------------
# Best-subset regression (least squares)
# ---------------------------------------------------------------------------
class BestSubsetRegression(OptimizationProblem):
    """Best-subset feature selection for least squares, enumerated in increasing selection criterion.

    Solutions are feature-index tuples ranked by ``criterion``: residual sum of squares (``"rss"``),
    Akaike (``"aic"``) or Bayesian (``"bic"``) information criterion (Gaussian form). Best-subset
    selection is inherently exponential, so this scores subsets exhaustively up to ``max_size``
    features -- cap ``max_size`` (and/or the number of features) for large ``p``.

    Args:
        X: Design matrix, shape ``(n, p)``.
        y: Response vector, length ``n``.
        criterion: ``"aic"`` (default), ``"bic"``, or ``"rss"``.
        max_size: Largest subset size to consider (``None`` = all ``p`` features).
        intercept: Fit an (unpenalized, always-included) intercept column.
    """

    sense = "min"

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        criterion: str = "aic",
        max_size: int | None = None,
        intercept: bool = True,
    ) -> None:
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if self.X.ndim != 2 or self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X must be (n, p) and y must have length n")
        if criterion not in ("aic", "bic", "rss"):
            raise ValueError("criterion must be 'aic', 'bic', or 'rss'")
        self.criterion = criterion
        self.n, self.p = self.X.shape
        self.max_size = self.p if max_size is None else int(max_size)
        self.intercept = bool(intercept)

    def _score(self, subset: tuple[int, ...]) -> float:
        cols = [self.X[:, j] for j in subset]
        if self.intercept:
            cols = [np.ones(self.n)] + cols
        design = np.column_stack(cols) if cols else np.zeros((self.n, 0))
        if design.shape[1] == 0:
            rss = float(np.dot(self.y, self.y))
        else:
            beta, _res, _rank, _sv = np.linalg.lstsq(design, self.y, rcond=None)
            resid = self.y - design @ beta
            rss = float(np.dot(resid, resid))
        if self.criterion == "rss":
            return rss
        k = len(subset) + (1 if self.intercept else 0)
        rss = max(rss, 1e-300)  # guard log(0) for a perfect fit
        ll_term = self.n * np.log(rss / self.n)
        penalty = 2.0 * k if self.criterion == "aic" else np.log(self.n) * k
        return float(ll_term + penalty)

    def enumerator(self, k: int | None = None) -> Iterator[tuple[tuple[int, ...], float]]:
        scored = []
        for size in range(0, self.max_size + 1):
            for subset in itertools.combinations(range(self.p), size):
                scored.append((self._score(subset), subset))
        scored.sort(key=lambda t: t[0])
        for score, subset in scored if k is None else scored[:k]:
            yield subset, score
