"""Relations over structured spaces, enumerated in order of a residual.

A :class:`Relation` is not an optimization problem -- it is a constraint imposed on a structured
space (matchings, spanning trees, strings near a center, hidden-state sequences, feature subsets),
whose members are enumerated ranked by a residual/cost. Finding the single best member is incidental;
the value is the *whole ranked set*. You specify the relation, then ask it for an ``enumerator()`` --
the same shape as a distribution yielding a ``sampler()`` / ``estimator()`` / ``enumerator()``. Every
relation shares one surface::

    relation.solve()       -> the minimal-residual Solution (or None if the relation is empty)
    relation.top(k)        -> the k smallest-residual members as a list
    relation.enumerator()  -> a lazy iterator over members, smallest residual first
    for solution in relation: ...

Each item is a :class:`Solution` namedtuple ``(value, objective)`` -- it reads as ``sol.value`` /
``sol.objective`` and still unpacks as ``value, objective = sol``. ``value`` is the member itself
(an assignment, a nearby string, a state sequence, a feature subset, ...) and ``objective`` is its
residual: a cost (minimized) or score (maximized); ``sense`` records which.

Assignment, spanning tree, the edit-distance ball, k-best Viterbi, shortest path, and best-subset
regression are all specified and consumed the same way, each delegating to whatever engine fits
(Murty for assignment, Gabow for spanning trees, A* / :func:`best_first_paths` for paths and Viterbi,
Dijkstra / :func:`nearest_first` for the edit-distance ball, exhaustive ranking for best-subset).
The two shared low-level engines are :func:`best_first_paths` (k-best *paths to a goal*) and
:func:`nearest_first` (distinct *states outward* from a center -- an expanding metric ball).

    >>> from pysp.relations import Assignment
    >>> sol = Assignment([[1, 9], [9, 1]]).solve()
    >>> sol.value, sol.objective          # the column assignment and its total cost
    (array([0, 1]), 2.0)
"""

from __future__ import annotations

import heapq
import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import Any, NamedTuple

import numpy as np

from pysp.utils.assignment import k_best_assignments
from pysp.utils.spanning import k_best_spanning_trees

__all__ = [
    "Assignment",
    "BestSubsetRegression",
    "EditDistance",
    "Relation",
    "ShortestPath",
    "Solution",
    "SpanningTree",
    "ViterbiPath",
    "best_first_paths",
    "nearest_first",
]


class Solution(NamedTuple):
    """One enumerated solution: ``value`` (the solution itself) and ``objective`` (its cost/score)."""

    value: Any
    objective: float


# ---------------------------------------------------------------------------
# Shared low-level engine: lazy best-first / A* over an arbitrary state graph
# ---------------------------------------------------------------------------
def best_first_paths(
    start: Any,
    successors: Callable[[Any], Iterable[tuple[Any, float]]],
    is_goal: Callable[[Any], bool] | None = None,
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
        is_goal: Predicate; when true for a popped state it is emitted (and not expanded). ``None``
            (the default) treats any *sink* -- a state with no successors -- as a goal, which covers
            DAG/trellis/edit-graph searches without a separate goal test.
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
        if is_goal is not None:
            if is_goal(state):
                yield (list(path) if return_paths else state, g)
                emitted += 1
                if max_results is not None and emitted >= max_results:
                    return
                continue
            succ = successors(state)
        else:  # sink == goal
            succ = list(successors(state))
            if not succ:
                yield (list(path) if return_paths else state, g)
                emitted += 1
                if max_results is not None and emitted >= max_results:
                    return
                continue
        for nxt, step in succ:
            g2 = g + step
            heapq.heappush(heap, (priority(g2, nxt), next(cnt), g2, nxt, path + (nxt,)))


def nearest_first(
    start: Any,
    neighbors: Callable[[Any], Iterable[tuple[Any, float]]],
    *,
    key: Callable[[Any], Any] | None = None,
    max_distance: float | None = None,
    max_results: int | None = None,
) -> Iterator[tuple[Any, float]]:
    """Enumerate distinct states outward from ``start`` in increasing distance (Dijkstra).

    The dual of :func:`best_first_paths`: instead of enumerating *paths to goal states*, this
    enumerates the reachable *states themselves*, each once, nearest first -- an expanding metric
    "ball" around ``start``. ``neighbors(state) -> iterable of (next_state, step_cost)`` with
    non-negative steps; ``key(state)`` gives a hashable identity for de-duplication (default: the
    state itself). The space may be infinite, so bound it with ``max_distance`` and/or ``max_results``
    (or just consume the lazy iterator finitely).

    Yields:
        ``(state, distance)`` where ``distance`` is the shortest total cost from ``start``, in
        nondecreasing order.
    """
    key = key or (lambda s: s)
    cnt = itertools.count()
    heap: list[tuple[float, int, Any]] = [(0.0, next(cnt), start)]
    seen: set = set()
    emitted = 0
    while heap:
        dist, _, state = heapq.heappop(heap)
        sk = key(state)
        if sk in seen:
            continue
        seen.add(sk)
        yield state, dist
        emitted += 1
        if max_results is not None and emitted >= max_results:
            return
        for nxt, step in neighbors(state):
            nd = dist + step
            if (max_distance is None or nd <= max_distance) and key(nxt) not in seen:
                heapq.heappush(heap, (nd, next(cnt), nxt))


# ---------------------------------------------------------------------------
# The shared problem interface
# ---------------------------------------------------------------------------
class Relation(ABC):
    """A constraint over a structured space whose members are enumerated ranked by a residual.

    Subclasses implement :meth:`enumerator` (yielding :class:`Solution` items); :meth:`solve`,
    :meth:`top` and iteration come for free. ``sense`` is ``"min"`` (residual minimized, members out
    in increasing cost) or ``"max"`` (residual maximized, members out in decreasing score).
    """

    sense: str = "min"

    @abstractmethod
    def enumerator(self, k: int | None = None) -> Iterator[Solution]:
        """Lazily yield :class:`Solution` items best-first; at most ``k`` if given (``None`` = all)."""

    def solve(self) -> Solution | None:
        """The single optimal :class:`Solution`, or ``None`` if the problem is infeasible."""
        return next(self.enumerator(k=1), None)

    def top(self, k: int) -> list[Solution]:
        """The ``k`` best solutions as a list."""
        return list(self.enumerator(k=k))

    def sample(
        self,
        size: int | None = None,
        *,
        rng=None,
        temperature: float = 1.0,
        k: int | None = None,
        uniform: bool = False,
    ) -> Any:
        """Draw member value(s) of the relation by a Gibbs weight over the objective.

        Each enumerated member is weighted ``exp(-objective / temperature)`` when ``sense == "min"``
        (low cost favoured) or ``exp(objective / temperature)`` when ``sense == "max"`` (high score
        favoured). ``temperature -> 0`` concentrates on the optimum; ``-> inf`` (or ``uniform=True``)
        is uniform over the enumerated members.

        Exactness: the weights are normalized over the members actually enumerated, so this is an
        *exact* draw from the Gibbs distribution only when the relation is finite and fully enumerated
        (``k=None``). For an infinite or very large relation pass ``k`` to truncate to the ``k`` best --
        the dropped tail is the lowest-weight mass (for ``sense="min"`` the ``k`` best are the highest
        weight), so it is a good approximation at low temperature and degrades as ``temperature`` grows.
        ``k`` is required for an infinite relation (otherwise enumeration does not terminate).

        Args:
            size: ``None`` returns a single member value; an int returns a list of that many draws.
            rng: a ``numpy.random.RandomState``, an integer seed, or ``None``.
            temperature: Gibbs temperature (default 1.0).
            k: enumerate at most this many members before sampling (``None`` = all; required if infinite).
            uniform: ignore objectives and sample uniformly over the enumerated members.

        Returns:
            A member ``value`` (``size=None``) or a list of member values (``size=int``).

        Raises:
            ValueError: if the relation enumerates no members (infeasible).
        """
        if rng is None or isinstance(rng, (int, np.integer)):
            rng = np.random.RandomState(None if rng is None else int(rng))
        sols = list(self.enumerator(k=k))
        if not sols:
            raise ValueError("relation is infeasible: no members to sample.")
        obj = np.array([s.objective for s in sols], dtype=float)
        if uniform or not np.isfinite(temperature):  # infinite temperature -> uniform
            log_w = np.zeros(len(sols))
        elif temperature <= 0.0:  # zero temperature -> point mass on the best enumerated member
            best = int(np.argmin(obj) if self.sense == "min" else np.argmax(obj))
            return sols[best].value if size is None else [sols[best].value] * size
        else:
            sign = -1.0 if self.sense == "min" else 1.0
            log_w = sign * obj / float(temperature)
        log_w = log_w - log_w.max()
        p = np.exp(log_w)
        p /= p.sum()
        idx = rng.choice(len(sols), size=size, p=p)
        if size is None:
            return sols[int(idx)].value
        return [sols[int(i)].value for i in idx]

    def __iter__(self) -> Iterator[Solution]:
        return self.enumerator()


# ---------------------------------------------------------------------------
# General graph problem (direct wrapper of the engine)
# ---------------------------------------------------------------------------
class ShortestPath(Relation):
    """k-shortest-path / best-first search over an arbitrary state graph.

    Specify the graph by ``start`` and ``successors``; the solution value is the list of states from
    ``start`` to a goal. By default a *sink* (a state with no successors) is a goal, so a finite
    DAG search needs no goal test; pass ``is_goal`` for infinite graphs or early goals. Use
    ``sense="max"`` for highest-score paths and an admissible ``heuristic`` for A*.
    """

    def __init__(
        self,
        start: Any,
        successors: Callable[[Any], Iterable[tuple[Any, float]]],
        is_goal: Callable[[Any], bool] | None = None,
        *,
        sense: str = "min",
        heuristic: Callable[[Any], float] | None = None,
    ) -> None:
        self.start = start
        self.successors = successors
        self.is_goal = is_goal
        self.sense = sense
        self.heuristic = heuristic

    def enumerator(self, k: int | None = None) -> Iterator[Solution]:
        for path, cost in best_first_paths(
            self.start, self.successors, self.is_goal, sense=self.sense, heuristic=self.heuristic, max_results=k
        ):
            yield Solution(path, float(cost))


# ---------------------------------------------------------------------------
# Linear assignment (Murty)
# ---------------------------------------------------------------------------
class Assignment(Relation):
    """Linear assignment / bipartite matching: match rows to columns at extremal total cost.

    The solution value is ``col_ind`` -- ``col_ind[i]`` is the column assigned to row ``i``.
    """

    def __init__(self, cost: np.ndarray, maximize: bool = False) -> None:
        self.cost = np.asarray(cost, dtype=np.float64)
        if self.cost.ndim != 2:
            raise ValueError("cost must be a 2-D matrix")
        self.maximize = bool(maximize)
        self.sense = "max" if maximize else "min"

    def enumerator(self, k: int | None = None) -> Iterator[Solution]:
        for total, _rows, cols in k_best_assignments(self.cost, k=k, maximize=self.maximize):
            yield Solution(cols, float(total))


# ---------------------------------------------------------------------------
# Minimum spanning tree (Gabow)
# ---------------------------------------------------------------------------
class SpanningTree(Relation):
    """Spanning trees of a weighted undirected graph, enumerated in increasing total edge weight.

    The solution value is the list of ``(i, j)`` edges. Non-finite weights are forbidden edges.
    """

    sense = "min"

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.weights.ndim != 2 or self.weights.shape[0] != self.weights.shape[1]:
            raise ValueError("weights must be a square matrix")

    def enumerator(self, k: int | None = None) -> Iterator[Solution]:
        for total, edges in k_best_spanning_trees(self.weights, k=k):
            yield Solution(edges, float(total))


# ---------------------------------------------------------------------------
# Non-uniform (weighted) edit distance / alignment
# ---------------------------------------------------------------------------
class EditDistance(Relation):
    """Enumerate strings outward from a center by (non-uniform) edit distance -- an edit-distance ball.

    You give a single center string and an alphabet, *not* two endpoints (the distance between two
    fixed strings is just one number). The enumerator yields strings in increasing edit distance from
    the center: the center itself at distance 0, then its 1-edit neighbours, then 2-edit, and so on --
    a Dijkstra expansion over string space with per-operation costs (:func:`nearest_first`). The ball
    is infinite (insertions grow strings without bound), so bound it with ``max_distance`` or by
    taking ``top(k)`` / ``enumerator(k)``. Solution values are strings (or symbol tuples, matching the
    center's type); the objective is the edit distance from the center.

    Args:
        center: The center string (or sequence of symbols).
        alphabet: The symbols available for substitution and insertion.
        sub_cost: ``(a, b) -> cost`` of substituting ``a`` with ``b`` (default unit; 0 if equal).
        ins_cost: ``c -> cost`` of inserting symbol ``c`` (default 1).
        del_cost: ``a -> cost`` of deleting symbol ``a`` (default 1).
        max_distance: Only enumerate strings within this edit distance (``None`` = unbounded/lazy).
    """

    sense = "min"

    def __init__(
        self,
        center: Iterable[Any],
        alphabet: Iterable[Any],
        *,
        sub_cost: Callable[[Any, Any], float] | None = None,
        ins_cost: Callable[[Any], float] | None = None,
        del_cost: Callable[[Any], float] | None = None,
        max_distance: float | None = None,
    ) -> None:
        self._as_str = isinstance(center, str)
        self.center = tuple(center)
        self.alphabet = tuple(alphabet)
        self.sub_cost = sub_cost or (lambda a, b: 0.0 if a == b else 1.0)
        self.ins_cost = ins_cost or (lambda b: 1.0)
        self.del_cost = del_cost or (lambda a: 1.0)
        self.max_distance = max_distance

    def _neighbors(self, s: tuple) -> list[tuple[tuple, float]]:
        out = []
        n = len(s)
        for i in range(n):  # substitutions
            si = s[i]
            for c in self.alphabet:
                if c != si:
                    out.append((s[:i] + (c,) + s[i + 1 :], self.sub_cost(si, c)))
        for i in range(n):  # deletions
            out.append((s[:i] + s[i + 1 :], self.del_cost(s[i])))
        for i in range(n + 1):  # insertions
            for c in self.alphabet:
                out.append((s[:i] + (c,) + s[i:], self.ins_cost(c)))
        return out

    def _format(self, state: tuple):
        return "".join(state) if self._as_str else state

    def enumerator(self, k: int | None = None) -> Iterator[Solution]:
        for state, dist in nearest_first(self.center, self._neighbors, max_distance=self.max_distance, max_results=k):
            yield Solution(self._format(state), float(dist))


# ---------------------------------------------------------------------------
# k-best Viterbi (most-likely hidden-state sequences of an HMM)
# ---------------------------------------------------------------------------
class ViterbiPath(Relation):
    """k most-likely hidden-state sequences of an HMM, enumerated in decreasing joint log-probability.

    Standard Viterbi returns only the single best path; this reduces the trellis (nodes ``(t, s)``)
    to a longest-log-prob path and yields the top ``k``. The solution value is a length-``T`` list of
    state indices.

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

    def enumerator(self, k: int | None = None) -> Iterator[Solution]:
        if self.n_steps == 0:
            return
        t_last, s = self.n_steps - 1, self.n_states
        log_init, log_trans, log_obs = self.log_init, self.log_trans, self.log_obs

        def successors(node):
            t, state = node
            if t == -1:
                return [((0, sp), float(log_init[sp]) + float(log_obs[0][sp])) for sp in range(s)]
            if t >= t_last:
                return []  # states at the final step are sinks -> goals
            return [((t + 1, sp), float(log_trans[state][sp]) + float(log_obs[t + 1][sp])) for sp in range(s)]

        for path, score in best_first_paths((-1, -1), successors, sense="max", max_results=k):
            yield Solution([st for (_t, st) in path[1:]], float(score))


# ---------------------------------------------------------------------------
# Best-subset regression (least squares)
# ---------------------------------------------------------------------------
class BestSubsetRegression(Relation):
    """Best-subset feature selection for least squares, enumerated in increasing selection criterion.

    Solution values are feature-index tuples ranked by ``criterion``: residual sum of squares
    (``"rss"``), Akaike (``"aic"``) or Bayesian (``"bic"``) information criterion (Gaussian form).
    Best-subset selection is inherently exponential, so this scores subsets exhaustively up to
    ``max_size`` features -- cap ``max_size`` (and/or the number of features) for large ``p``.

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

    def enumerator(self, k: int | None = None) -> Iterator[Solution]:
        scored = []
        for size in range(0, self.max_size + 1):
            for subset in itertools.combinations(range(self.p), size):
                scored.append((self._score(subset), subset))
        scored.sort(key=lambda t: t[0])
        for score, subset in scored if k is None else scored[:k]:
            yield Solution(subset, float(score))
