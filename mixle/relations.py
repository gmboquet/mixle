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

    >>> from mixle.relations import Assignment
    >>> sol = Assignment([[1, 9], [9, 1]]).solve()
    >>> sol.value, sol.objective          # the column assignment and its total cost
    (array([0, 1]), 2.0)
"""

from __future__ import annotations

import heapq
import itertools
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, NamedTuple

import numpy as np

from mixle.enumeration.assignment import k_best_assignments
from mixle.enumeration.spanning import k_best_spanning_trees

__all__ = [
    "admm_bounded_least_squares",
    "Assignment",
    "BestSubsetRegression",
    "branch_and_bound_milp",
    "cardinality_constrained_milp",
    "EditDistance",
    "graph_coloring",
    "irreducible_infeasible_subset",
    "Relation",
    "RelationSampler",
    "ShortestPath",
    "max_clique",
    "max_independent_set",
    "is_stable_matching",
    "max_flow",
    "min_arborescence",
    "min_cut",
    "stable_matching",
    "tsp_held_karp",
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
# Stable matching (Gale-Shapley)
# ---------------------------------------------------------------------------
def stable_matching(proposer_prefs: Sequence[Sequence[int]], receiver_prefs: Sequence[Sequence[int]]) -> list[int]:
    """Proposer-optimal stable matching via Gale-Shapley.

    ``proposer_prefs[i]`` is proposer ``i``'s receivers in descending preference; ``receiver_prefs[j]``
    likewise for receiver ``j``. Preference lists may be partial (an unlisted partner is unacceptable)
    and the two sides may differ in size. Returns ``match`` with ``match[i]`` the receiver assigned to
    proposer ``i`` (or ``-1`` if unmatched). The result is the proposer-optimal stable matching: it is
    stable (no blocking pair) and every proposer gets the best partner achievable in any stable matching.

    Reference: Gale & Shapley, "College admissions and the stability of marriage", *Amer. Math. Monthly*
    (1962).
    """
    n, m = len(proposer_prefs), len(receiver_prefs)
    rank = [{p: r for r, p in enumerate(receiver_prefs[j])} for j in range(m)]
    next_choice = [0] * n
    match_p = [-1] * n
    match_r = [-1] * m
    free = deque(range(n))
    while free:
        i = free.popleft()
        while next_choice[i] < len(proposer_prefs[i]):
            j = proposer_prefs[i][next_choice[i]]
            next_choice[i] += 1
            if i not in rank[j]:
                continue  # receiver j finds proposer i unacceptable
            cur = match_r[j]
            if cur == -1:
                match_r[j], match_p[i] = i, j
                break
            if rank[j][i] < rank[j][cur]:  # j prefers i to its current partner
                match_p[cur] = -1
                free.append(cur)
                match_r[j], match_p[i] = i, j
                break
            # else j rejects i; i keeps proposing down its list
        # if i exhausts its list it stays unmatched (not re-queued)
    return match_p


def is_stable_matching(
    match: Sequence[int], proposer_prefs: Sequence[Sequence[int]], receiver_prefs: Sequence[Sequence[int]]
) -> bool:
    """Return ``True`` iff ``match`` has no blocking pair (a mutually-preferred unmatched proposer/receiver)."""
    m = len(receiver_prefs)
    p_rank = [{r: k for k, r in enumerate(proposer_prefs[i])} for i in range(len(proposer_prefs))]
    r_rank = [{p: k for k, p in enumerate(receiver_prefs[j])} for j in range(m)]
    receiver_of = match
    proposer_of = [-1] * m
    for i, j in enumerate(match):
        if j != -1:
            proposer_of[j] = i
    for i, prefs in enumerate(proposer_prefs):
        for j in prefs:  # receivers i prefers, best-first
            if receiver_of[i] == j:
                break  # i is matched to j or someone it prefers more; no blocking pair beyond here
            if i not in r_rank[j]:
                continue  # j won't accept i anyway
            cur = proposer_of[j]
            # blocking iff j is unmatched, or j prefers i to its current partner
            if cur == -1 or r_rank[j][i] < r_rank[j][cur]:
                if receiver_of[i] == -1 or p_rank[i][j] < p_rank[i][receiver_of[i]]:
                    return False
    return True


# ---------------------------------------------------------------------------
# Maximum flow / minimum cut (Edmonds-Karp)
# ---------------------------------------------------------------------------
def max_flow(capacity: Any, source: int, sink: int) -> tuple[float, np.ndarray]:
    """Maximum ``source -> sink`` flow in a directed network (Edmonds-Karp).

    ``capacity`` is an ``n x n`` non-negative matrix of arc capacities. Returns ``(value, flow)`` where
    ``flow[u, v]`` is the flow on arc ``u -> v`` (conserved at every node but the source/sink) and
    ``value`` is the total flow out of ``source``. Edmonds-Karp augments along BFS shortest paths in the
    residual network, so it runs in ``O(V E^2)`` and terminates on real-valued capacities.
    """
    cap = np.asarray(capacity, dtype=np.float64)
    n = cap.shape[0]
    residual = cap.copy()
    value = 0.0
    while True:
        parent = [-1] * n
        parent[source] = source
        q = deque([source])
        while q and parent[sink] == -1:
            u = q.popleft()
            for v in range(n):
                if parent[v] == -1 and residual[u, v] > 1.0e-12:
                    parent[v] = u
                    q.append(v)
        if parent[sink] == -1:
            break  # no augmenting path
        bottleneck = np.inf
        v = sink
        while v != source:
            bottleneck = min(bottleneck, residual[parent[v], v])
            v = parent[v]
        v = sink
        while v != source:
            u = parent[v]
            residual[u, v] -= bottleneck
            residual[v, u] += bottleneck
            v = u
        value += float(bottleneck)
    flow = np.where(cap > 0.0, np.maximum(cap - residual, 0.0), 0.0)
    return value, flow


def min_cut(capacity: Any, source: int, sink: int) -> tuple[float, list[int], list[tuple[int, int]]]:
    """Minimum ``source/sink`` cut of a directed network (via max-flow; the max-flow min-cut theorem).

    Returns ``(capacity, source_side, cut_edges)``: the cut capacity (equal to the max-flow value), the
    set of nodes on the source side (reachable from ``source`` in the final residual graph), and the
    saturated arcs crossing from the source side to the sink side.
    """
    cap = np.asarray(capacity, dtype=np.float64)
    n = cap.shape[0]
    value, flow = max_flow(cap, source, sink)
    residual = cap - flow + flow.T  # residual of the optimal flow
    reachable = {source}
    q = deque([source])
    while q:
        u = q.popleft()
        for v in range(n):
            if v not in reachable and residual[u, v] > 1.0e-12:
                reachable.add(v)
                q.append(v)
    cut_edges = [(u, v) for u in reachable for v in range(n) if v not in reachable and cap[u, v] > 0.0]
    cut_capacity = float(sum(cap[u, v] for u, v in cut_edges))
    return cut_capacity, sorted(reachable), cut_edges


# ---------------------------------------------------------------------------
# Travelling salesman (exact, Held-Karp dynamic program)
# ---------------------------------------------------------------------------
def tsp_held_karp(distance: Any) -> tuple[float, list[int]]:
    """Exact minimum-cost Hamiltonian cycle through all nodes (Held-Karp).

    ``distance`` is an ``n x n`` matrix of arc costs (may be asymmetric). Returns ``(cost, tour)`` where
    ``tour`` starts at node 0, visits every node once, and the cost includes the closing arc back to 0.
    The Held-Karp bitmask DP is exact in ``O(2^n n^2)`` time / ``O(2^n n)`` memory, so it is intended for
    small ``n`` (roughly <= 15-18); beyond that use a heuristic.
    """
    d = np.asarray(distance, dtype=np.float64)
    n = d.shape[0]
    if n <= 1:
        return 0.0, list(range(n))
    if n == 2:
        return float(d[0, 1] + d[1, 0]), [0, 1]
    full = (1 << (n - 1)) - 1
    # C[(mask, j)] = (min cost of a path 0 -> ... -> j visiting exactly the nodes in mask, predecessor k)
    # where mask is a bitmask over nodes 1..n-1.
    cost_to: dict[tuple[int, int], tuple[float, int]] = {(1 << (j - 1), j): (float(d[0, j]), 0) for j in range(1, n)}
    for mask in range(1, full + 1):
        for j in range(1, n):
            bj = 1 << (j - 1)
            if not (mask & bj) or mask == bj:
                continue  # j not in mask, or the singleton already seeded above
            prev_mask = mask ^ bj
            best: tuple[float, int] | None = None
            for k in range(1, n):
                if k == j or not (prev_mask & (1 << (k - 1))):
                    continue
                pc = cost_to.get((prev_mask, k))
                if pc is None:
                    continue
                cand = pc[0] + float(d[k, j])
                if best is None or cand < best[0]:
                    best = (cand, k)
            if best is not None:
                cost_to[(mask, j)] = best
    # close each full path back to node 0 and take the best
    end: tuple[float, int] | None = None
    for j in range(1, n):
        c = cost_to.get((full, j))
        if c is None:
            continue
        cand = c[0] + float(d[j, 0])
        if end is None or cand < end[0]:
            end = (cand, j)
    cost, last = end  # type: ignore[misc]
    rev = []
    mask, j = full, last
    while j != 0:
        rev.append(j)
        _, k = cost_to[(mask, j)]
        mask ^= 1 << (j - 1)
        j = k
    return float(cost), [0, *rev[::-1]]


# ---------------------------------------------------------------------------
# Graph coloring (exact chromatic number)
# ---------------------------------------------------------------------------
def graph_coloring(adjacency: Any) -> tuple[int, list[int]]:
    """Exact minimum proper vertex coloring of an undirected graph.

    ``adjacency`` is an ``n x n`` symmetric 0/1 (or boolean) matrix with a zero diagonal. Returns
    ``(k, coloring)`` where ``k`` is the chromatic number and ``coloring[v]`` in ``0..k-1`` gives no two
    adjacent vertices the same color. Solved by backtracking with the standard symmetry break (a vertex
    may introduce at most one new color), trying ``k = 1, 2, ...`` until colorable -- exact, but
    worst-case exponential, so intended for small/medium graphs.
    """
    a = np.asarray(adjacency)
    n = a.shape[0]
    if n == 0:
        return 0, []
    nb = [[j for j in range(n) if j != i and a[i, j]] for i in range(n)]

    def colorable(k: int) -> list[int] | None:
        coloring = [-1] * n

        def rec(v: int) -> bool:
            if v == n:
                return True
            cap = min(k, max(coloring[:v], default=-1) + 2)  # symmetry break: <= 1 new color per vertex
            used = {coloring[u] for u in nb[v] if coloring[u] != -1}
            for c in range(cap):
                if c not in used:
                    coloring[v] = c
                    if rec(v + 1):
                        return True
                    coloring[v] = -1
            return False

        return coloring if rec(0) else None

    for k in range(1, n + 1):
        col = colorable(k)
        if col is not None:
            return k, col
    return n, list(range(n))  # unreachable (a graph of n vertices is always n-colorable)


# ---------------------------------------------------------------------------
# Maximum clique / maximum independent set
# ---------------------------------------------------------------------------
def max_clique(adjacency: Any) -> list[int]:
    """A maximum clique (largest mutually-adjacent vertex set) of an undirected graph.

    ``adjacency`` is an ``n x n`` symmetric 0/1 (or boolean) matrix with a zero diagonal. Returns the
    sorted vertices of one maximum clique via Carraghan-Pardalos branch-and-bound (prune when the
    current clique plus the remaining candidates cannot beat the incumbent) -- exact, worst-case
    exponential, intended for small/medium graphs.
    """
    a = np.asarray(adjacency)
    n = a.shape[0]
    nb = [{j for j in range(n) if j != i and a[i, j]} for i in range(n)]
    best: list[int] = []

    def expand(clique: list[int], cands: list[int]) -> None:
        nonlocal best
        if not cands:
            if len(clique) > len(best):
                best = clique[:]
            return
        cands = list(cands)
        while cands:
            if len(clique) + len(cands) <= len(best):
                return  # cannot beat the incumbent even taking every candidate
            v = cands.pop()
            expand([*clique, v], [u for u in cands if u in nb[v]])

    expand([], list(range(n)))
    return sorted(best)


def max_independent_set(adjacency: Any) -> list[int]:
    """A maximum independent set (largest pairwise-non-adjacent vertex set) -- a max clique of the complement."""
    a = np.asarray(adjacency)
    n = a.shape[0]
    complement = 1 - np.asarray(a, dtype=int)
    if n:
        np.fill_diagonal(complement, 0)
    return max_clique(complement)


# ---------------------------------------------------------------------------
# Minimum spanning arborescence (Chu-Liu / Edmonds)
# ---------------------------------------------------------------------------
def _edmonds(nodes: set[int], edges: list[tuple[int, int, float]], root: int) -> list[tuple[int, int, float]] | None:
    """Recursive Chu-Liu/Edmonds: return the chosen original edges, or ``None`` if no arborescence."""
    min_in: dict[int, tuple[int, int, float]] = {}
    for v in nodes:
        if v == root:
            continue
        cands = [e for e in edges if e[1] == v]
        if not cands:
            return None  # node v is unreachable -> no spanning arborescence
        min_in[v] = min(cands, key=lambda e: e[2])
    cycle = None
    for start in nodes:
        if start == root:
            continue
        seen: list[int] = []
        v = start
        while v != root and v not in seen:
            seen.append(v)
            v = min_in[v][0]
        if v != root and v in seen:
            cycle = seen[seen.index(v) :]
            break
    if cycle is None:
        return list(min_in.values())
    cyc = set(cycle)
    super_node = max(nodes) + 1
    new_nodes = {x for x in nodes if x not in cyc} | {super_node}
    new_edges: list[tuple[int, int, float]] = []
    origin: dict[tuple[int, int], list[tuple[tuple[int, int, float], float]]] = {}
    for u, v, w in edges:
        if u in cyc and v in cyc:
            continue
        if v in cyc:  # edge into the cycle: discount by the in-edge it would replace
            key = (super_node if u in cyc else u, super_node)
            adj = w - min_in[v][2]
        elif u in cyc:  # edge leaving the cycle
            key = (super_node, v)
            adj = w
        else:
            key = (u, v)
            adj = w
        new_edges.append((key[0], key[1], adj))
        origin.setdefault(key, []).append(((u, v, w), adj))
    sub = _edmonds(new_nodes, new_edges, root)
    if sub is None:
        return None
    result: list[tuple[int, int, float]] = []
    entered = None
    for u, v, _w in sub:
        orig_edge, _adj = min(origin[(u, v)], key=lambda oa: oa[1])
        result.append(orig_edge)
        if v == super_node:
            entered = orig_edge[1]  # the cycle vertex actually entered from outside
    for v in cyc:
        if v != entered:
            result.append(min_in[v])
    return result


def min_arborescence(weight: Any, root: int = 0) -> tuple[float, list[int]] | None:
    """Minimum-weight spanning arborescence rooted at ``root`` (directed MST; Chu-Liu/Edmonds).

    ``weight`` is an ``n x n`` matrix of directed arc costs with ``inf`` for absent arcs. Returns
    ``(total, parent)`` where ``parent[v]`` is the chosen in-arc tail for each non-root ``v`` (and
    ``parent[root] = -1``), forming the minimum-cost arborescence in which every node is reachable from
    ``root``; returns ``None`` if no such arborescence exists.
    """
    w = np.asarray(weight, dtype=np.float64)
    n = w.shape[0]
    edges = [
        (u, v, float(w[u, v])) for u in range(n) for v in range(n) if u != v and v != root and np.isfinite(w[u, v])
    ]
    chosen = _edmonds(set(range(n)), edges, root)
    if chosen is None:
        return None
    parent = [-1] * n
    total = 0.0
    for u, v, ew in chosen:
        parent[v] = u
        total += ew
    return total, parent


# ---------------------------------------------------------------------------
# Mixed-integer linear program (branch-and-bound over the LP relaxation)
# ---------------------------------------------------------------------------
def branch_and_bound_milp(
    c: Any,
    a_ub: Any | None = None,
    b_ub: Any | None = None,
    integer: Sequence[int] | None = None,
    bounds: Sequence[tuple[float, float]] | None = None,
    *,
    sense: str = "min",
    tol: float = 1.0e-6,
) -> tuple[float, np.ndarray] | None:
    """Solve a mixed-integer linear program by branch-and-bound over the LP relaxation.

    Minimizes (``sense="min"``) or maximizes (``sense="max"``) ``c @ x`` subject to ``a_ub @ x <= b_ub``
    and per-variable ``bounds`` ``(lo, hi)``, with the variables indexed by ``integer`` constrained to
    integers (default: all). Returns ``(objective, x)`` or ``None`` if infeasible. Each node solves the
    continuous relaxation with ``scipy.optimize.linprog`` (HiGHS) and, if an integer variable is
    fractional, branches into ``x_i <= floor`` and ``x_i >= ceil``; best-bound search prunes nodes that
    cannot beat the incumbent. Exact for bounded integer feasible regions.
    """
    from scipy.optimize import linprog

    cvec = np.asarray(c, dtype=np.float64)
    n = cvec.size
    obj = -cvec if sense == "max" else cvec
    if sense not in ("min", "max"):
        raise ValueError("sense must be 'min' or 'max'")
    integer = range(n) if integer is None else integer
    lo0 = [(-np.inf if bounds is None else bounds[i][0]) for i in range(n)]
    hi0 = [(np.inf if bounds is None else bounds[i][1]) for i in range(n)]

    def relax(lo: list[float], hi: list[float]) -> tuple[float, np.ndarray] | None:
        res = linprog(obj, A_ub=a_ub, b_ub=b_ub, bounds=list(zip(lo, hi, strict=False)), method="highs")
        return (float(res.fun), res.x) if res.success else None

    root = relax(lo0, hi0)
    if root is None:
        return None
    counter = itertools.count()
    incumbent: list[Any] = [np.inf, None]
    heap: list[tuple[float, int, list[float], list[float], np.ndarray]] = [(root[0], next(counter), lo0, hi0, root[1])]
    while heap:
        f, _, lo, hi, x = heapq.heappop(heap)
        if f >= incumbent[0] - tol:
            continue  # bound: cannot improve on the incumbent
        frac = next((i for i in integer if abs(x[i] - round(x[i])) > tol), None)
        if frac is None:
            if f < incumbent[0]:
                incumbent = [f, np.array(x)]
            continue
        floor_hi = [hi[j] if j != frac else float(np.floor(x[frac])) for j in range(n)]
        ceil_lo = [lo[j] if j != frac else float(np.ceil(x[frac])) for j in range(n)]
        for nlo, nhi in ((lo, floor_hi), (ceil_lo, hi)):
            if nlo[frac] > nhi[frac]:
                continue
            child = relax(nlo, nhi)
            if child is not None and child[0] < incumbent[0] - tol:
                heapq.heappush(heap, (child[0], next(counter), nlo, nhi, child[1]))
    if incumbent[1] is None:
        return None
    value = -incumbent[0] if sense == "max" else incumbent[0]
    return value, incumbent[1]


def cardinality_constrained_milp(
    c: Any,
    a_ub: Any | None,
    b_ub: Any | None,
    max_nonzero: int,
    bounds: Sequence[tuple[float, float]],
    *,
    sense: str = "min",
) -> tuple[float, np.ndarray] | None:
    """Minimize/maximize ``c @ x`` with at most ``max_nonzero`` of the variables nonzero.

    Adds a cardinality (sparsity) constraint to the linear program ``a_ub @ x <= b_ub`` with per-variable
    ``bounds`` via the standard big-M indicator formulation: a binary ``z_i`` gates each variable
    (``lower_i z_i <= x_i <= upper_i z_i``, so ``z_i = 0`` forces ``x_i = 0``) and ``sum z_i <=
    max_nonzero``; the extended mixed-integer program is solved by :func:`branch_and_bound_milp`. Returns
    ``(objective, x)`` (the sparse optimizer) or ``None`` if infeasible. This is the indicator/
    set-membership/cardinality constraint primitive (best-subset selection, sparse design).
    """
    c = np.asarray(c, dtype=np.float64)
    n = c.size
    a = np.asarray(a_ub, dtype=np.float64) if a_ub is not None else np.zeros((0, n))
    b = np.asarray(b_ub, dtype=np.float64) if b_ub is not None else np.zeros(0)
    lo = np.array([bd[0] for bd in bounds], dtype=np.float64)
    hi = np.array([bd[1] for bd in bounds], dtype=np.float64)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for i in range(a.shape[0]):  # original constraints, padded for the z block
        rows.append(np.concatenate([a[i], np.zeros(n)]))
        rhs.append(float(b[i]))
    for i in range(n):  # x_i <= hi_i z_i  and  x_i >= lo_i z_i
        r = np.zeros(2 * n)
        r[i], r[n + i] = 1.0, -hi[i]
        rows.append(r)
        rhs.append(0.0)
        r = np.zeros(2 * n)
        r[i], r[n + i] = -1.0, lo[i]
        rows.append(r)
        rhs.append(0.0)
    z_row = np.zeros(2 * n)  # sum z_i <= max_nonzero
    z_row[n:] = 1.0
    rows.append(z_row)
    rhs.append(float(max_nonzero))
    ext_c = np.concatenate([c, np.zeros(n)])
    ext_bounds = [*list(bounds), *([(0.0, 1.0)] * n)]
    res = branch_and_bound_milp(
        ext_c, np.array(rows), np.array(rhs), integer=range(n, 2 * n), bounds=ext_bounds, sense=sense
    )
    if res is None:
        return None
    value, x_ext = res
    return value, x_ext[:n]


# ---------------------------------------------------------------------------
# Infeasibility diagnostics (irreducible infeasible subset of linear constraints)
# ---------------------------------------------------------------------------
def _lp_feasible(a_ub: np.ndarray, b_ub: np.ndarray, bounds: Sequence[tuple[float, float]]) -> bool:
    """True iff ``{x in bounds : a_ub @ x <= b_ub}`` is non-empty (a zero-objective LP feasibility check)."""
    from scipy.optimize import linprog

    if len(b_ub) == 0:
        return True
    res = linprog(np.zeros(a_ub.shape[1]), A_ub=a_ub, b_ub=b_ub, bounds=list(bounds), method="highs")
    return bool(res.success)


def irreducible_infeasible_subset(
    a_ub: Any, b_ub: Any, bounds: Sequence[tuple[float, float]] | None = None
) -> list[int] | None:
    """Find an irreducible infeasible subset (IIS) of the linear constraints ``a_ub @ x <= b_ub``.

    Returns the row indices of a minimal infeasible subset: the subsystem is itself infeasible, yet
    dropping any single one of its rows makes it feasible (within the variable ``bounds``, default
    unbounded). Returns ``None`` if the full system is already feasible. Uses the deletion filter --
    tentatively remove each constraint and keep it removed whenever the remainder stays infeasible --
    so the result certifies *which* constraints conflict, the standard infeasibility diagnostic.
    """
    a = np.asarray(a_ub, dtype=np.float64)
    b = np.asarray(b_ub, dtype=np.float64)
    n = a.shape[1]
    bnds = [(-np.inf, np.inf)] * n if bounds is None else list(bounds)
    if _lp_feasible(a, b, bnds):
        return None  # feasible system has no infeasible subset
    rows = list(range(len(b)))
    for i in list(rows):
        trial = [r for r in rows if r != i]
        if not _lp_feasible(a[trial], b[trial], bnds):
            rows = trial  # constraint i is not needed for infeasibility -> drop it
    return rows


# ---------------------------------------------------------------------------
# ADMM for box-constrained least squares (augmented-Lagrangian splitting)
# ---------------------------------------------------------------------------
def admm_bounded_least_squares(
    a: Any,
    b: Any,
    lower: Any = 0.0,
    upper: Any = np.inf,
    *,
    rho: float = 1.0,
    max_iter: int = 5000,
    tol: float = 1.0e-8,
) -> np.ndarray:
    """Solve ``min_x ||A x - b||^2`` subject to ``lower <= x <= upper`` by ADMM.

    The alternating-direction method of multipliers splits the problem as ``f(x) = ||A x - b||^2`` plus
    the box indicator ``g(z)``, with ``x = z``, and alternates: an ``x``-update (the ridge solve
    ``(A^T A + rho I) x = A^T b + rho (z - u)``, factorized once), a ``z``-update (project ``x + u`` onto
    the box), and the scaled dual update ``u += x - z``. This is the augmented-Lagrangian path "beyond
    pure penalty": it converges to the exact constrained optimum (``lower=0, upper=inf`` recovers
    non-negative least squares). Returns the bounded solution ``x``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.shape[1]
    lo = np.broadcast_to(np.asarray(lower, dtype=np.float64), (n,))
    hi = np.broadcast_to(np.asarray(upper, dtype=np.float64), (n,))
    chol = np.linalg.cholesky(a.T @ a + rho * np.eye(n))  # SPD for rho > 0; factor once
    atb = a.T @ b
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)
    for _ in range(max_iter):
        x = np.linalg.solve(chol.T, np.linalg.solve(chol, atb + rho * (z - u)))
        z_old = z
        z = np.clip(x + u, lo, hi)
        u = u + x - z
        if np.linalg.norm(x - z) < tol and rho * np.linalg.norm(z - z_old) < tol:
            break  # primal + dual residuals small
    return z


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

    def sampler(
        self,
        seed: int | None = None,
        *,
        temperature: float = 1.0,
        k: int | None = None,
        uniform: bool = False,
        rng=None,
    ) -> RelationSampler:
        """Return a :class:`RelationSampler` that draws members under a Gibbs measure over the objective.

        A relation is a *specification* of a structured space, not itself a random object; sampling it
        means imposing a distribution over its members, which needs an RNG and a temperature. So -- like
        every other mixle object -- it hands back a sampler (``relation.sampler(seed).sample(size)``) that
        owns the stream and the Gibbs measure, rather than being sampled directly.

        Each enumerated member is weighted ``exp(-objective / temperature)`` when ``sense == "min"``
        (low cost favoured) or ``exp(objective / temperature)`` when ``sense == "max"``. ``temperature
        -> 0`` concentrates on the optimum; ``-> inf`` (or ``uniform=True``) is uniform. The draw is an
        *exact* Gibbs sample only when the relation is finite and fully enumerated (``k=None``); pass
        ``k`` to truncate an infinite/large relation to its ``k`` best (the dropped tail is the
        lowest-weight mass -- a good low-temperature approximation, and ``k`` is required if infinite).

        Args:
            seed: scalar seed for the sampler's RandomState (ignored if ``rng`` is given).
            temperature: Gibbs temperature (default 1.0).
            k: enumerate at most this many members (``None`` = all; required if infinite).
            uniform: ignore objectives and sample uniformly over the enumerated members.
            rng: a shared ``numpy.random.RandomState`` (takes precedence over ``seed``).
        """
        return RelationSampler(self, seed, temperature=temperature, k=k, uniform=uniform, rng=rng)

    def __iter__(self) -> Iterator[Solution]:
        return self.enumerator()


class RelationSampler:
    """Draws members of a :class:`Relation` under a Gibbs measure ``exp(-objective / temperature)``.

    Constructed via :meth:`Relation.sampler`. It enumerates the relation's members once (lazily, on the
    first draw) and caches the resulting categorical, so repeated ``sample`` calls are low-overhead. ``size=None``
    returns one member value; ``size=int`` returns a list of that many draws.
    """

    def __init__(
        self,
        relation: Relation,
        seed: int | None = None,
        *,
        temperature: float = 1.0,
        k: int | None = None,
        uniform: bool = False,
        rng=None,
    ) -> None:
        self.relation = relation
        self.rng = rng if rng is not None else np.random.RandomState(seed)
        self.temperature = temperature
        self.k = k
        self.uniform = uniform
        self._values: list[Any] | None = None
        self._p: np.ndarray | None = None  # categorical over members; None at zero temperature
        self._point: int | None = None  # index of the optimum when temperature == 0

    def _prepare(self) -> None:
        if self._values is not None:
            return
        sols = list(self.relation.enumerator(k=self.k))
        if not sols:
            raise ValueError("relation is infeasible: no members to sample.")
        self._values = [s.value for s in sols]
        obj = np.array([s.objective for s in sols], dtype=float)
        if self.uniform or not np.isfinite(self.temperature):  # infinite temperature -> uniform
            self._p = np.full(len(sols), 1.0 / len(sols))
        elif self.temperature <= 0.0:  # zero temperature -> point mass on the best enumerated member
            self._point = int(np.argmin(obj) if self.relation.sense == "min" else np.argmax(obj))
        else:
            sign = -1.0 if self.relation.sense == "min" else 1.0
            log_w = sign * obj / float(self.temperature)
            log_w -= log_w.max()
            p = np.exp(log_w)
            self._p = p / p.sum()

    def sample(self, size: int | None = None) -> Any:
        """Draw a member value (``size=None``) or a list of ``size`` member values."""
        self._prepare()
        if self._p is None:  # zero temperature: the optimum, deterministically
            return self._values[self._point] if size is None else [self._values[self._point]] * size
        idx = self.rng.choice(len(self._values), size=size, p=self._p)
        if size is None:
            return self._values[int(idx)]
        return [self._values[int(i)] for i in idx]


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
        """Yield up to ``k`` shortest or highest-scoring paths."""
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
        """Yield up to ``k`` assignments ordered by total assignment cost."""
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
        """Yield up to ``k`` spanning trees ordered by total edge weight."""
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
        """Yield edit-distance neighbors ordered outward from the center."""
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
        """Yield up to ``k`` hidden-state paths ordered by joint log probability."""
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
        """Yield up to ``k`` feature subsets ordered by the configured criterion."""
        scored = []
        for size in range(0, self.max_size + 1):
            for subset in itertools.combinations(range(self.p), size):
                scored.append((self._score(subset), subset))
        scored.sort(key=lambda t: t[0])
        for score, subset in scored if k is None else scored[:k]:
            yield Solution(subset, float(score))
