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
(Murty for assignment, Gabow for spanning trees, A* / :func:`best_first_paths` for paths,
:func:`mixle.enumeration.hmm_paths.hmm_best_paths` for Viterbi, Dijkstra / :func:`nearest_first`
for the edit-distance ball, exhaustive ranking for best-subset).
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
from mixle.enumeration.hmm_paths import hmm_best_paths
from mixle.enumeration.spanning import k_best_spanning_trees
from mixle.utils.exact import require_exact_bool

__all__ = [
    "admm_bounded_least_squares",
    "Assignment",
    "BestSubsetRegression",
    "branch_and_bound_milp",
    "cardinality_constrained_milp",
    "Design",
    "EditDistance",
    "Flow",
    "graph_coloring",
    "irreducible_infeasible_subset",
    "min_cost_flow",
    "multicommodity_flow",
    "network_design",
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
# Shared input contracts (MXR-080-1903)
#
# These APIs answer combinatorial questions, and the answers are consumed as certificates: a flow, a
# cut, an optimum, a k-best list. That makes silent coercion at the boundary worse here than in a
# reporting function -- the caller cannot tell a certificate for the problem they posed from a
# certificate for a different one. The recurring shapes are collected here so every entry point
# refuses the same way.
# ---------------------------------------------------------------------------
def _require_count(value: Any, name: str, *, minimum: int = 0, allow_none: bool = False) -> int | None:
    """An EXACT integer count -- never silently truncated or rounded (MXR-080-1903).

    Sizes, caps and result budgets were read with a bare comparison or ``int()``, so ``max_size=2.9``
    became ``2``, ``max_size="3"`` became ``3``, and ``max_size=True`` became a size-one search --
    each answering a smaller question than the caller asked, with a confident optimum to show for it.

    Accepts a Python/numpy integer or an exactly-integer-valued float (``3.0``), matching the contract
    :mod:`mixle.enumeration` already uses. ``bool`` is refused: it is a yes/no answer, not a count.
    """
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or value != np.floor(value):
            raise ValueError(f"{name} must be a whole number, not a fractional value, got {value!r}")
    ivalue = int(value)
    if ivalue < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {ivalue}")
    return ivalue


def _require_square(value: Any, name: str) -> np.ndarray:
    """A square 2-D float matrix, refusing a rectangular one rather than reading its leading block.

    ``n = matrix.shape[0]`` with no shape check silently DISCARDED every column past ``n``
    (MXR-080-1903): a 2x3 capacity matrix answered ``max_flow`` for its 2x2 corner, a 3x4 distance
    matrix produced a "tour" over three of four cities. :func:`min_cost_flow` and
    :func:`multicommodity_flow` already checked this; the rest did not.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square (n, n) matrix, got shape {array.shape}")
    return array


def _reject_nan(array: np.ndarray, name: str, *, also_negative_inf: bool = False) -> None:
    """Refuse NaN (and optionally ``-inf``) in a matrix whose entries gate a comparison.

    Every one of these algorithms decides arc presence with a comparison -- ``residual[u, v] > tol``,
    ``cap[u, v] > 0.0``, ``np.isfinite(d[i, j])``. NaN fails ALL of them, so a NaN entry silently
    means "this arc does not exist" and the answer is a confident certificate for a different graph:
    ``min_cut`` returned capacity ``0.0`` with an EMPTY cut-edge list on a graph that has arcs
    (MXR-080-1903). ``-inf`` gets the same treatment where ``+inf`` legitimately marks a missing arc:
    a ``-inf`` cost means an unbounded objective, not an absent one, and collapsing the two hides it.

    ``+inf`` is deliberately NOT rejected -- :func:`min_cost_flow`, :func:`min_arborescence` and
    :func:`tsp_held_karp` all document it as "unbounded capacity" / "missing arc", and the suite
    exercises exactly that.
    """
    if np.isnan(array).any():
        raise ValueError(
            f"{name} must not contain NaN: every arc test here is a comparison, and NaN fails all of "
            "them, so a NaN entry would silently read as an absent arc"
        )
    if also_negative_inf and np.isneginf(array).any():
        raise ValueError(f"{name} must not contain -inf: a -inf entry means an unbounded objective, not a missing arc")


def _require_adjacency(value: Any, name: str) -> np.ndarray:
    """A square, symmetric, 0/1 (or Boolean) adjacency matrix -- the contract the docstrings state.

    These three graph routines all documented "symmetric 0/1 (or boolean) matrix" and none of them
    checked it, so they returned certificates that are false on their face (MXR-080-1903):

    * asymmetry -- ``graph_coloring([[0, 1], [0, 0]])`` returned chromatic number 1 with both vertices
      the SAME color despite ``a[0, 1] = 1``, because the neighbour lists are built per-row and vertex
      1's row never mentions vertex 0. ``max_clique([[0, 0], [1, 0]])`` likewise returned ``[0, 1]``,
      which is not a clique.
    * truthiness -- adjacency was read as ``if a[i, j]``, so a 0.4 weight, a NaN and the STRING
      ``"false"`` all meant "adjacent", while :func:`max_independent_set` read the same matrix through
      ``dtype=int`` where 0.4 truncates to 0. The same 0.4 matrix therefore reported ``{0, 1}`` as both
      a maximum clique and a maximum independent set -- two mutually exclusive claims, from one input.

    The DIAGONAL is deliberately not checked. Every loop here already skips ``j == i``, so a self-loop
    changes no answer, and refusing it would reject inputs that are only cosmetically off-contract.
    """
    array = np.asarray(value)
    if array.dtype == bool:
        array = array.astype(np.int64)
    else:
        try:
            array = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a Boolean or numeric 0/1 matrix, got dtype {np.asarray(value).dtype}"
            ) from exc
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square (n, n) matrix, got shape {array.shape}")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(
            f"{name} must contain only 0/1 (or Boolean) entries: adjacency is read as a yes/no fact, "
            "so a weight, a NaN or a non-empty string would silently mean 'adjacent'"
        )
    if not np.array_equal(array, array.T):
        raise ValueError(
            f"{name} must be symmetric (these routines model an UNDIRECTED graph); an asymmetric "
            "matrix yields per-row neighbour lists that disagree, and the returned coloring/clique is "
            "then not one -- symmetrize explicitly to say which reading you meant"
        )
    return array.astype(np.int64)


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
    # MXR-080-1903. `emitted += 1` ran BEFORE `emitted >= max_results`, so a budget of 0 still yielded
    # one goal -- and `top(0)`, which routes here, returned a one-element list. A float budget was
    # effectively ceiled (2.1 gave 3 results, because `1 >= 2.1` is False but `3 >= 2.1` is True), and
    # a negative one behaved like 1. A result budget is an exact count, so it is validated as one; 0
    # now means what it says.
    max_results = _require_count(max_results, "max_results", minimum=0, allow_none=True)
    if max_results == 0:
        return
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
    # Same off-by-one and same float/negative coercion as :func:`best_first_paths` (MXR-080-1903).
    max_results = _require_count(max_results, "max_results", minimum=0, allow_none=True)
    if max_results == 0:
        return
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

    ``capacity`` must be square and NaN-free, and ``source``/``sink`` must be distinct in-range nodes;
    see :func:`_require_square`, :func:`_reject_nan` and the note below for what each of those used to
    do instead (MXR-080-1903).
    """
    cap = _require_square(capacity, "capacity")
    _reject_nan(cap, "capacity")
    n = cap.shape[0]
    # `source == sink` never cleared `parent[sink] == -1`, so the augmenting loop ran forever with an
    # infinite bottleneck: the call HUNG rather than answering or refusing (MXR-080-1903).
    source = _require_count(source, "source", minimum=0)
    sink = _require_count(sink, "sink", minimum=0)
    if source >= n or sink >= n:
        raise ValueError(f"source/sink must be nodes in 0..{n - 1}, got source={source}, sink={sink}")
    if source == sink:
        raise ValueError("source and sink must be distinct nodes; a self-flow has no maximum")
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

    Input contract as :func:`max_flow`'s, and enforced there: a NaN capacity used to make this return
    capacity ``0.0`` with an EMPTY cut-edge list on a graph that plainly has arcs (MXR-080-1903).
    """
    cap = np.asarray(capacity, dtype=np.float64)
    value, flow = max_flow(cap, source, sink)
    n = cap.shape[0]
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
# Minimum-cost flow, multicommodity flow, and capacitated network design
# ---------------------------------------------------------------------------
class Flow(NamedTuple):
    """A resolved flow: ``value`` (total/objective cost) and ``flow[u, v]`` per-arc, like `max_flow`'s output."""

    value: float
    flow: np.ndarray


class Design(NamedTuple):
    """A network-design solution: ``cost`` (total), ``open`` (opened-arc/facility mask), and the induced ``flow``."""

    cost: float
    open: np.ndarray
    flow: np.ndarray


def _find_negative_cycle(
    big_cap: np.ndarray, big_cost: np.ndarray, flow: np.ndarray, m: int
) -> list[tuple[int, int, bool]] | None:
    """Bellman-Ford search for a negative-cost cycle in the residual graph implied by ``flow``.

    Seeds every node's distance at 0, as if a virtual zero-cost source fed every node directly, so a
    cycle is found wherever it sits in the graph rather than only along paths from one specific node.
    Relaxes for ``m`` rounds; if a round still improves some node's distance, a negative cycle is
    reachable from it (the standard "one round past convergence" test), and walking ``pred`` pointers
    back ``m`` times from that node is guaranteed to land on the cycle itself -- only ``m`` distinct
    nodes exist, so ``m`` hops must re-enter it. Uses ``big_cost`` (true arc costs) throughout, never
    the main solve's Johnson's-algorithm reduced costs -- those are only kept non-negative along a
    shortest-path tree from a single source and do not reliably preserve a cycle's sign in general.

    Each ordered pair ``(u, v)`` can offer *two* distinct residual edges, and they must be costed
    separately here rather than folded into one ``residual[u, v]`` capacity the way the main SSP solve
    does: genuine unused capacity on a real arc ``u -> v`` costs ``big_cost[u, v]``, while "undo credit"
    against flow already pushed the other way (``flow[v, u] > 0``) costs ``-big_cost[v, u]`` -- the
    negation of what *that* arc charges, not whatever ``big_cost[u, v]`` happens to hold. Collapsing both
    into a single scalar is harmless for Dijkstra/SSP, which only ever consumes capacity forward, but it
    breaks cycle-canceling on a genuine antiparallel arc pair (``cap[u, v]`` and ``cap[v, u]`` both
    positive): once both directions are pushed to their joint optimum, the *combined* residual capacity
    in each direction (``cap[u, v] - flow[u, v] + flow[v, u]``) is exactly what it was before either was
    pushed, so a detector reading a single ``big_cost[u, v]`` for that capacity keeps "seeing" the same
    negative cycle forever and oscillates instead of converging (verified empirically while building
    this: the naive single-cost version cancels the cycle, then un-cancels it, forever).

    Returns the cycle as a list of ``(u, v, is_undo)`` edges in traversal order (``is_undo`` True means
    the edge spends undo-credit against ``flow[v, u]``, False means it spends genuine forward capacity),
    or ``None`` if no negative cycle is reachable via positive capacity anywhere in the graph.
    """
    dist = np.zeros(m)
    pred = np.full(m, -1, dtype=int)
    pred_is_undo = np.zeros(m, dtype=bool)
    last_relaxed = -1
    for _ in range(m):
        last_relaxed = -1
        for u in range(m):
            du = dist[u]
            for v in range(m):
                if u == v:
                    continue
                fwd_cap = big_cap[u, v] - flow[u, v]
                if fwd_cap > 1.0e-12:
                    nd = du + big_cost[u, v]
                    if nd < dist[v] - 1.0e-9:
                        dist[v], pred[v], pred_is_undo[v] = nd, u, False
                        last_relaxed = v
                undo_cap = flow[v, u]
                if undo_cap > 1.0e-12:
                    nd = du - big_cost[v, u]
                    if nd < dist[v] - 1.0e-9:
                        dist[v], pred[v], pred_is_undo[v] = nd, u, True
                        last_relaxed = v
        if last_relaxed == -1:
            return None  # converged before the m-th sweep: no negative cycle reachable

    # The m-th sweep still relaxed something -> a negative cycle is reachable from `last_relaxed`.
    x = last_relaxed
    for _ in range(m):
        x = pred[x]
    cycle_nodes = [x]
    v = pred[x]
    while v != x:
        cycle_nodes.append(v)
        v = pred[v]
    cycle_nodes.reverse()
    k = len(cycle_nodes)
    return [(cycle_nodes[i], cycle_nodes[(i + 1) % k], bool(pred_is_undo[cycle_nodes[(i + 1) % k]])) for i in range(k)]


def _cancel_negative_cycles(
    big_cap: np.ndarray, big_cost: np.ndarray, residual: np.ndarray, flow: np.ndarray, m: int
) -> None:
    """Post-process an already-feasible flow to true minimality by canceling residual negative cycles.

    Successive-shortest-path (:func:`min_cost_flow`'s main loop) finds a min-cost flow among
    source-to-sink augmenting paths, but stops the instant supply is fully routed -- it never checks
    whether the residual graph *at that final flow* still has a negative-cost cycle disjoint from the
    super-source/super-sink. If one exists, pushing flow around it (up to its bottleneck residual
    capacity) strictly lowers total cost while leaving every node's supply/demand balance untouched (a
    cycle has zero net flow at every node it passes through, by definition), so canceling can only ever
    improve an already-feasible flow and is safe to run once feasibility is reached, with no need to
    re-check feasibility afterward.

    Mutates ``residual`` and ``flow`` in place. Raises :class:`ValueError` if a reachable negative cycle
    has unbounded (infinite) capacity -- cost is then unbounded below, so no finite minimum exists -- or
    if cancellation fails to converge within a generous, graph-size-scaled iteration budget (an
    internal-consistency guard against float-precision pathologies; a well-posed instance should never
    come close to it).
    """
    max_iterations = max(1000, 100 * m * m)
    iterations = 0
    while True:
        cycle = _find_negative_cycle(big_cap, big_cost, flow, m)
        if cycle is None:
            return
        bottleneck = min((flow[v, u] if is_undo else big_cap[u, v] - flow[u, v]) for u, v, is_undo in cycle)
        if not np.isfinite(bottleneck):
            raise ValueError(
                "min_cost_flow: cost structure admits an unbounded negative cycle, no finite minimum exists"
            )
        for u, v, is_undo in cycle:
            if is_undo:
                flow[v, u] -= bottleneck
            else:
                flow[u, v] += bottleneck
            residual[u, v] -= bottleneck
            residual[v, u] += bottleneck
        iterations += 1
        if iterations > max_iterations:
            raise ValueError(
                f"min_cost_flow: negative-cycle canceling failed to converge after {max_iterations} "
                "iterations (internal consistency error)"
            )


def min_cost_flow(cap: Any, cost: Any, supply: Any) -> Flow:
    """Min-cost feasible flow meeting node ``supply`` (positive = source) under arc ``cap``/``cost``.

    Successive-shortest-path with node potentials: a super-source feeds every ``supply > 0`` node and
    every ``supply < 0`` node feeds a super-sink (arc capacity ``|supply|``, cost 0), then the residual
    super-source -> super-sink path of least *reduced* cost is repeatedly augmented (Dijkstra, since the
    potentials keep every reduced cost non-negative) until the super-sink is unreachable. Potentials are
    seeded with a Bellman-Ford pass so arbitrary-sign ``cost`` entries are handled, then updated by the
    per-round Dijkstra distances (the standard Johnson's-algorithm maintenance).

    Successive-shortest-path alone only guarantees minimality *among source-to-sink augmenting paths* --
    it says nothing about a profitable cycle that never touches the super-source/super-sink at all. Once
    feasibility is reached, :func:`_cancel_negative_cycles` runs a Bellman-Ford negative-cycle-canceling
    pass over the resulting residual graph and pushes flow around any negative-cost cycle it finds (up to
    its bottleneck capacity) -- which strictly lowers cost without disturbing any node's supply/demand
    balance, since a cycle has zero net flow at every node it passes through. This is what makes the
    returned flow truly minimum rather than merely feasible.

    ``cap``/``cost`` are ``n x n`` arc matrices; ``supply`` is length ``n`` and must sum to
    (approximately) zero. Returns `Flow(value=total cost, flow=(n, n) arc flows)`; raises
    :class:`ValueError` if the supply cannot be fully routed under the given capacities, or if the cost
    structure admits an unbounded negative cycle (infinite capacity available to an improving cycle, so
    no finite minimum cost exists).
    """
    cap = np.asarray(cap, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    supply = np.asarray(supply, dtype=np.float64)
    n = supply.shape[0]
    if cap.shape != (n, n) or cost.shape != (n, n):
        raise ValueError("cap/cost must be (n, n), matching supply's length n")
    # `abs(nan) >= 1e-6` is False, so a NaN supply entry sailed through the balance check that exists
    # to catch exactly this -- and then `supply[i] > 1e-12` and `supply[i] < -1e-12` were BOTH False,
    # so the node was wired up as balanced. The result was a confident "min-cost feasible flow" for an
    # instance that has none (MXR-080-1903). `cap`/`cost` reject NaN for the same reason arc tests are
    # comparisons; +inf capacity stays legal (it is how an uncapacitated arc is spelled here).
    if not np.isfinite(supply).all():
        raise ValueError("supply must be finite at every node; a non-finite entry has no balance to check")
    _reject_nan(cap, "cap")
    _reject_nan(cost, "cost")
    if abs(float(supply.sum())) >= 1.0e-6:
        raise ValueError("supply must sum to (approximately) zero")

    ss, tt, m = n, n + 1, n + 2  # super-source, super-sink, extended node count
    big_cap = np.zeros((m, m))
    big_cap[:n, :n] = cap
    big_cost = np.zeros((m, m))
    big_cost[:n, :n] = cost
    total_supply = 0.0
    for i in range(n):
        if supply[i] > 1.0e-12:
            big_cap[ss, i] = supply[i]
            total_supply += float(supply[i])
        elif supply[i] < -1.0e-12:
            big_cap[i, tt] = -float(supply[i])

    residual = big_cap.copy()
    flow = np.zeros((m, m))

    def bellman_ford_from_source() -> np.ndarray:
        dist = np.full(m, np.inf)
        dist[ss] = 0.0
        for _ in range(m - 1):
            changed = False
            for u in range(m):
                if not np.isfinite(dist[u]):
                    continue
                for v in range(m):
                    if residual[u, v] > 1.0e-12 and dist[u] + big_cost[u, v] < dist[v] - 1.0e-9:
                        dist[v] = dist[u] + big_cost[u, v]
                        changed = True
            if not changed:
                break
        return np.where(np.isfinite(dist), dist, 0.0)

    pot = bellman_ford_from_source()  # seeds potentials so Dijkstra sees non-negative reduced costs
    routed = 0.0
    while routed < total_supply - 1.0e-9:
        reduced = big_cost + pot[:, None] - pot[None, :]
        dist = np.full(m, np.inf)
        dist[ss] = 0.0
        prev = np.full(m, -1, dtype=int)
        visited = np.zeros(m, dtype=bool)
        heap: list[tuple[float, int]] = [(0.0, ss)]
        while heap:
            d, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            for v in range(m):
                if not visited[v] and residual[u, v] > 1.0e-12:
                    nd = d + reduced[u, v]
                    if nd < dist[v] - 1.0e-12:
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(heap, (nd, v))
        if not visited[tt]:
            break  # super-sink unreachable: no more augmenting paths
        for v in range(m):
            if visited[v] and np.isfinite(dist[v]):
                pot[v] += dist[v]  # Johnson's-algorithm potential maintenance

        bottleneck = total_supply - routed
        path: list[tuple[int, int]] = []
        v = tt
        while v != ss:
            u = prev[v]
            bottleneck = min(bottleneck, residual[u, v])
            path.append((u, v))
            v = u
        for u, v in path:
            residual[u, v] -= bottleneck
            residual[v, u] += bottleneck
            cancel = min(bottleneck, flow[v, u])  # undo previously-pushed flow on the reverse arc first
            flow[v, u] -= cancel
            if bottleneck - cancel > 1.0e-15:
                flow[u, v] += bottleneck - cancel
        routed += bottleneck

    if routed < total_supply - 1.0e-6:
        raise ValueError("min_cost_flow: infeasible -- supply cannot be fully routed under the given capacities")

    # SSP alone is only minimal among source-to-sink augmenting paths; cancel any profitable cycle left
    # in the residual graph (see _cancel_negative_cycles) so the returned flow is truly minimum-cost.
    _cancel_negative_cycles(big_cap, big_cost, residual, flow, m)

    value = float((flow[:n, :n] * cost).sum())
    return Flow(value=value, flow=flow[:n, :n])


def multicommodity_flow(cap: Any, cost: Any, demands: Any) -> Flow:
    """Multi-commodity min-cost flow (different products/grades share arc ``cap``) via LP relaxation.

    ``cap``/``cost`` are ``n x n`` arc matrices (an arc exists where ``cap > 0``); ``demands`` lists
    ``(source, sink, amount)`` rows, one per commodity. Each commodity gets its own flow variables with
    its own conservation constraints (per-commodity supply at its source, demand at its sink, balance
    elsewhere); every arc's *combined* commodity flow is capped at ``cap[u, v]``. Solved as a single LP
    (``scipy.optimize.linprog``, HiGHS) over the stacked per-commodity arc variables. Returns `Flow` with
    the aggregate cost and the summed arc flows (``flow = sum_k x_k``); raises :class:`ValueError` if the
    demands cannot be met under the shared capacities.
    """
    from scipy.optimize import linprog

    cap = np.asarray(cap, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    demands = np.atleast_2d(np.asarray(demands, dtype=np.float64))
    n = cap.shape[0]
    if cap.shape != (n, n) or cost.shape != (n, n):
        raise ValueError("cap/cost must be square (n, n) arc matrices")
    k = demands.shape[0]
    arcs = [(u, v) for u in range(n) for v in range(n) if cap[u, v] > 0.0]
    n_arcs = len(arcs)

    def var(kk: int, ai: int) -> int:
        return kk * n_arcs + ai

    n_vars = k * n_arcs
    c = np.zeros(n_vars)
    for kk in range(k):
        for ai, (u, v) in enumerate(arcs):
            c[var(kk, ai)] = cost[u, v]

    a_eq_rows: list[np.ndarray] = []
    b_eq: list[float] = []
    for kk in range(k):
        # `int(demands[kk, 0])` truncated: a demand row of `[0.9, 2.7, 4.0]` routed commodity kk from
        # node 0 to node 2 without a word (MXR-080-1903). A node index is an exact identity, not a
        # measurement, so a fractional one is refused rather than rounded to a neighbour.
        src = _require_count(demands[kk, 0], f"demands[{kk}] source node", minimum=0)
        snk = _require_count(demands[kk, 1], f"demands[{kk}] sink node", minimum=0)
        if src >= n or snk >= n:
            raise ValueError(f"demands[{kk}] references a node outside 0..{n - 1}: source={src}, sink={snk}")
        qty = float(demands[kk, 2])
        if not np.isfinite(qty):
            raise ValueError(f"demands[{kk}] quantity must be finite, got {qty!r}")
        for node in range(n):
            row = np.zeros(n_vars)
            for ai, (u, v) in enumerate(arcs):
                if u == node:
                    row[var(kk, ai)] += 1.0
                if v == node:
                    row[var(kk, ai)] -= 1.0
            rhs = qty if node == src else (-qty if node == snk else 0.0)
            a_eq_rows.append(row)
            b_eq.append(rhs)

    a_ub_rows: list[np.ndarray] = []
    b_ub: list[float] = []
    for ai, (u, v) in enumerate(arcs):
        row = np.zeros(n_vars)
        for kk in range(k):
            row[var(kk, ai)] = 1.0
        a_ub_rows.append(row)
        b_ub.append(cap[u, v])

    res = linprog(
        c,
        A_ub=np.array(a_ub_rows) if a_ub_rows else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(a_eq_rows) if a_eq_rows else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=[(0.0, None)] * n_vars,
        method="highs",
    )
    if not res.success:
        raise ValueError("multicommodity_flow: infeasible -- demands cannot be met under the shared capacities")

    flow = np.zeros((n, n))
    for kk in range(k):
        for ai, (u, v) in enumerate(arcs):
            flow[u, v] += res.x[var(kk, ai)]
    return Flow(value=float(res.fun), flow=flow)


def network_design(nodes: Sequence[int], arcs: Sequence[tuple[int, int]], fixed_costs: Any, demands: Any) -> Design:
    """Capacitated fixed-charge network design (which arcs/facilities to open) via big-M MILP.

    ``nodes`` lists node ids; ``arcs`` lists candidate directed arcs (``fixed_costs[i]`` is the cost of
    opening ``arcs[i]``); ``demands`` is a length-``len(nodes)`` net supply/demand vector aligned with
    ``nodes`` (positive = supply, negative = demand, summing to zero) -- the same convention as
    :func:`min_cost_flow`'s ``supply``. Continuous flow ``x[arc] >= 0`` is big-M-linked to the binary
    open/closed decision ``y[arc]`` (``x[arc] <= M * y[arc]``, ``M`` sized from the total demand so it
    never binds a genuinely open arc); flow balance ``A x = demands`` is encoded as the paired
    inequalities ``A x <= demands`` and ``-A x <= -demands`` (:func:`branch_and_bound_milp` exposes only
    ``a_ub``/``b_ub``). The frozen signature carries no per-unit routing cost, so the objective is the
    total opening cost ``fixed_costs @ y`` (the classic uncapacitated-routing fixed-charge network design:
    which arcs to build so the demand vector is routable at minimum build cost). Returns
    ``Design(cost, open, flow)`` where ``open`` is the boolean opened-arc mask and ``flow`` is the induced
    ``(n, n)`` arc flow; raises :class:`ValueError` if no arc-opening choice can route the demands.
    """
    node_list = list(nodes)
    n = len(node_list)
    index = {node: i for i, node in enumerate(node_list)}
    arc_list = list(arcs)
    n_arcs = len(arc_list)
    fixed = np.asarray(fixed_costs, dtype=np.float64)
    demand = np.asarray(demands, dtype=np.float64)
    if fixed.shape != (n_arcs,):
        raise ValueError("fixed_costs must align with arcs, one entry per candidate arc")
    if demand.shape != (n,):
        raise ValueError("demands must align with nodes, one net supply/demand entry per node")
    # `abs(nan) > 1e-6` is False, so the balance guard below could not see a non-finite demand; the
    # NaN then reached linprog and surfaced as a solver complaint about the constraint matrix rather
    # than as the input error it is (MXR-080-1903).
    if not np.isfinite(demand).all():
        raise ValueError("network_design: demands must be finite at every node")
    if not np.isfinite(fixed).all():
        raise ValueError("network_design: fixed_costs must be finite for every candidate arc")
    if abs(float(demand.sum())) > 1.0e-6:
        raise ValueError("network_design: demands must sum to (approximately) zero")

    big_m = float(np.abs(demand).sum()) or 1.0  # a non-binding capacity once an arc is opened
    n_vars = 2 * n_arcs  # [x_0..x_{n_arcs-1}, y_0..y_{n_arcs-1}]

    def x_idx(a: int) -> int:
        return a

    def y_idx(a: int) -> int:
        return n_arcs + a

    c = np.zeros(n_vars)
    c[n_arcs:] = fixed  # objective: total opening cost (no per-unit routing cost in the frozen signature)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for a in range(n_arcs):  # big-M link: x[a] - M * y[a] <= 0
        row = np.zeros(n_vars)
        row[x_idx(a)] = 1.0
        row[y_idx(a)] = -big_m
        rows.append(row)
        rhs.append(0.0)

    balance = np.zeros((n, n_vars))
    for a, (u, v) in enumerate(arc_list):
        balance[index[u], x_idx(a)] += 1.0
        balance[index[v], x_idx(a)] -= 1.0
    for node_row, b in zip(balance, demand, strict=True):  # A x = b as paired A x <= b, -A x <= -b
        rows.append(node_row.copy())
        rhs.append(float(b))
        rows.append(-node_row)
        rhs.append(-float(b))

    bounds = [(0.0, big_m)] * n_arcs + [(0.0, 1.0)] * n_arcs
    result = branch_and_bound_milp(
        c, np.array(rows), np.array(rhs), integer=list(range(n_arcs, n_vars)), bounds=bounds, sense="min"
    )
    if result is None:
        raise ValueError("network_design: infeasible -- no arc-opening choice routes the given demands")
    value, sol = result
    opened = np.round(sol[n_arcs:]).astype(bool)
    x = sol[:n_arcs]
    flow_matrix = np.zeros((n, n))
    for a, (u, v) in enumerate(arc_list):
        # `arcs` may list more than one candidate for the same (u, v) pair (e.g. comparing a cheap vs.
        # premium design for the same physical link); each gets its own decision variable, but they all
        # accumulate into the same aggregate matrix cell rather than the last one clobbering the rest --
        # two candidates on one node pair are parallel capacity, not alternatives that overwrite in-place.
        flow_matrix[index[u], index[v]] += x[a]
    return Design(cost=float(value), open=opened, flow=flow_matrix)


# ---------------------------------------------------------------------------
# Travelling salesman (exact, Held-Karp dynamic program)
# ---------------------------------------------------------------------------
def tsp_held_karp(distance: Any) -> tuple[float, list[int]]:
    """Exact minimum-cost Hamiltonian cycle through all nodes (Held-Karp).

    ``distance`` is an ``n x n`` matrix of arc costs (may be asymmetric); ``+inf`` marks a missing arc.
    NaN and ``-inf`` are refused (MXR-080-1903): the DP decides arc presence with ``np.isfinite``, so
    both used to read as "missing", silently answering for a smaller graph -- and a ``-inf`` arc means
    an unboundedly profitable one, which is not the same fact as an absent one at all.
    Returns ``(cost, tour)`` where ``tour`` starts at node 0, visits every node once, and
    the cost includes the closing arc back to 0. Raises :class:`ValueError` when the finite arcs admit
    no Hamiltonian cycle. The Held-Karp bitmask DP is exact in ``O(2^n n^2)`` time / ``O(2^n n)``
    memory, so it is intended for small ``n`` (roughly <= 15-18); beyond that use a heuristic.
    """
    d = _require_square(distance, "distance")
    _reject_nan(d, "distance", also_negative_inf=True)
    n = d.shape[0]
    no_cycle = "no Hamiltonian cycle: the finite arcs admit no tour visiting every node once"
    if n <= 1:
        return 0.0, list(range(n))
    if n == 2:
        if not (np.isfinite(d[0, 1]) and np.isfinite(d[1, 0])):
            raise ValueError(no_cycle)
        return float(d[0, 1] + d[1, 0]), [0, 1]
    full = (1 << (n - 1)) - 1
    # C[(mask, j)] = (min cost of a path 0 -> ... -> j visiting exactly the nodes in mask, predecessor k)
    # where mask is a bitmask over nodes 1..n-1; unreachable (mask, j) simply have no entry.
    cost_to: dict[tuple[int, int], tuple[float, int]] = {
        (1 << (j - 1), j): (float(d[0, j]), 0) for j in range(1, n) if np.isfinite(d[0, j])
    }
    for mask in range(1, full + 1):
        for j in range(1, n):
            bj = 1 << (j - 1)
            if not (mask & bj) or mask == bj:
                continue  # j not in mask, or the singleton already seeded above
            prev_mask = mask ^ bj
            best: tuple[float, int] | None = None
            for k in range(1, n):
                if k == j or not (prev_mask & (1 << (k - 1))) or not np.isfinite(d[k, j]):
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
        if c is None or not np.isfinite(d[j, 0]):
            continue
        cand = c[0] + float(d[j, 0])
        if end is None or cand < end[0]:
            end = (cand, j)
    if end is None:
        raise ValueError(no_cycle)
    cost, last = end
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
    a = _require_adjacency(adjacency, "adjacency")
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
    a = _require_adjacency(adjacency, "adjacency")
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
    a = _require_adjacency(adjacency, "adjacency")
    n = a.shape[0]
    complement = 1 - a
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

    ``weight`` is an ``n x n`` matrix of directed arc costs with ``+inf`` for absent arcs. Returns
    ``(total, parent)`` where ``parent[v]`` is the chosen in-arc tail for each non-root ``v`` (and
    ``parent[root] = -1``), forming the minimum-cost arborescence in which every node is reachable from
    ``root``; returns ``None`` if no such arborescence exists.

    NaN and ``-inf`` are refused for the same reason as in :func:`tsp_held_karp`: arc presence is
    decided by ``np.isfinite``, so both used to read as "absent arc" (MXR-080-1903).
    """
    w = _require_square(weight, "weight")
    _reject_nan(w, "weight", also_negative_inf=True)
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

    ``tol`` is the integrality/pruning tolerance and must be finite and strictly below ``0.5``: it is
    the width within which a relaxation coordinate counts as already integral, and half a unit is the
    most that can ever mean. See the note at its validation below for what a wider one returned.
    """
    from scipy.optimize import linprog

    cvec = np.asarray(c, dtype=np.float64)
    n = cvec.size
    obj = -cvec if sense == "max" else cvec
    if sense not in ("min", "max"):
        raise ValueError("sense must be 'min' or 'max'")
    # MXR-080-1903. `tol` was unvalidated, and it is compared against, never computed with -- so a NaN
    # made `abs(x[i] - round(x[i])) > tol` False for EVERY coordinate, declaring the LP relaxation
    # integral. The relaxation point was then snapped by np.round while the incumbent kept the
    # UN-rounded LP objective, so the call returned an x that violates `a_ub @ x <= b_ub` together
    # with an objective that is not even `c @ x`. Any tol >= 0.5 does the same thing deliberately.
    # This is the shared solver behind network_design, cardinality_constrained_milp, blending,
    # stochastic_opt and precedence_scheduling; none of them passes `tol`, so nothing legitimate in
    # the repo produces the state now refused.
    if isinstance(tol, bool) or not isinstance(tol, (int, float, np.integer, np.floating)):
        raise ValueError(f"tol must be a finite number in (0, 0.5), got {tol!r}")
    tol = float(tol)
    if not np.isfinite(tol) or not 0.0 < tol < 0.5:
        raise ValueError(
            f"tol must be finite and in (0, 0.5), got {tol!r}: it is the width within which a "
            "relaxation coordinate counts as integral, so anything at or past half a unit accepts a "
            "fractional point as an integer solution"
        )
    integer = list(range(n)) if integer is None else list(integer)  # materialize once: consumed at every node
    # A negative index silently addressed a DIFFERENT variable (-1 read as x[n-1]), so the variable the
    # caller named as integral came back fractional; duplicates were accepted and re-branched.
    seen: set[int] = set()
    for position, i in enumerate(integer):
        index = _require_count(i, f"integer[{position}]", minimum=0)
        if index >= n:
            raise ValueError(f"integer[{position}]={index} is outside the variable range 0..{n - 1}")
        if index in seen:
            raise ValueError(f"integer lists variable {index} more than once")
        seen.add(index)
    integer = [int(i) for i in integer]
    if bounds is not None and len(bounds) != n:
        raise ValueError(f"bounds must have one (lo, hi) pair per variable: expected {n}, got {len(bounds)}")
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
                x_int = np.array(x, dtype=np.float64)
                x_int[integer] = np.round(x_int[integer])  # snap within-tol coordinates to exact integers
                incumbent = [f, x_int]
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
    max_nonzero``; the extended mixed-integer program is solved by :func:`branch_and_bound_milp`. Every
    bound must be finite -- the bounds are the big-M constants of the indicator rows. Returns
    ``(objective, x)`` (the sparse optimizer) or ``None`` if infeasible. This is the indicator/
    set-membership/cardinality constraint primitive (best-subset selection, sparse design).
    """
    c = np.asarray(c, dtype=np.float64)
    n = c.size
    # A cardinality bound counts variables, so it is an exact integer: `max_nonzero=2.9` used to be
    # floored into the `sum z_i <= 2.9` row and answer the max-2 problem instead (MXR-080-1903).
    max_nonzero = _require_count(max_nonzero, "max_nonzero", minimum=0)
    a = np.asarray(a_ub, dtype=np.float64) if a_ub is not None else np.zeros((0, n))
    b = np.asarray(b_ub, dtype=np.float64) if b_ub is not None else np.zeros(0)
    lo = np.array([bd[0] for bd in bounds], dtype=np.float64)
    hi = np.array([bd[1] for bd in bounds], dtype=np.float64)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError(
            "cardinality_constrained_milp needs finite bounds for every variable: the big-M indicator "
            "rows (lo_i z_i <= x_i <= hi_i z_i) are built from them, and a non-finite bound would put "
            "inf/nan into the constraint matrix"
        )
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

    ``lower <= upper`` elementwise, neither NaN; ``rho > 0`` (the ridge term is SPD only then);
    ``max_iter >= 1``. An infinite ``upper`` stays legal -- it is how the NNLS case is spelled.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.shape[1]
    lo = np.broadcast_to(np.asarray(lower, dtype=np.float64), (n,))
    hi = np.broadcast_to(np.asarray(upper, dtype=np.float64), (n,))
    # MXR-080-1903. Each of these returned a plausible vector for a problem the caller did not pose:
    #   * `lower=5, upper=0` -- np.clip applies its bounds in order, so the upper one wins and the
    #     returned point VIOLATES `lower <= x` with no complaint at all.
    #   * NaN in a bound -- np.clip propagates it, so every coordinate came back NaN as "the solution".
    #   * `max_iter=0` -- `range(0)` runs no iterations, so the initial all-zero z was returned as the
    #     converged answer, indistinguishable from a real one.
    #   * `rho <= 0` -- the docstring's own SPD precondition; at rho=0 the Cholesky is of A^T A alone.
    # `upper=np.inf` is deliberately still accepted: admm_test.py's NNLS case depends on it.
    if np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError("admm_bounded_least_squares: lower/upper must not contain NaN")
    if (lo > hi).any():
        raise ValueError("admm_bounded_least_squares: every lower bound must be <= its upper bound")
    if isinstance(rho, bool) or not isinstance(rho, (int, float, np.integer, np.floating)):
        raise ValueError(f"admm_bounded_least_squares: rho must be a positive finite number, got {rho!r}")
    rho = float(rho)
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError(
            f"admm_bounded_least_squares: rho must be positive and finite, got {rho!r}; the "
            "augmented-Lagrangian term A^T A + rho I is symmetric positive definite only for rho > 0"
        )
    max_iter = _require_count(max_iter, "max_iter", minimum=1)
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
        """The ``k`` best solutions as a list.

        ``k`` is an exact non-negative count: ``top(0)`` used to return ONE solution and ``top(2.9)``
        three, because the engines counted emissions after yielding and compared against the raw float
        (MXR-080-1903).
        """
        return list(self.enumerator(k=_require_count(k, "k", minimum=0)))

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
        # MXR-080-1903. `uniform` was read with a bare `if self.uniform or ...`, so `uniform="false"`
        # -- the shape a flag arrives in from serialized configuration -- produced the UNIFORM stream,
        # byte-for-byte identical to `uniform=True`. Assignment.maximize and
        # BestSubsetRegression.intercept were already converted to the exact contract; this one was
        # missed. `temperature` was equally unguarded: NaN fell into the `not np.isfinite` branch and
        # silently sampled uniformly (only +inf is documented to do that), and a NEGATIVE temperature
        # fell into `<= 0.0` and silently became a point mass on the optimum -- the opposite of what a
        # negative Gibbs temperature would mean.
        self.uniform = require_exact_bool(uniform, "uniform")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float, np.integer, np.floating)):
            raise ValueError(f"temperature must be a non-negative number (inf = uniform), got {temperature!r}")
        temperature = float(temperature)
        if np.isnan(temperature) or temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {temperature!r}; 0 is the zero-temperature "
                "point mass on the optimum and +inf is the uniform measure, but NaN and a negative "
                "temperature name no measure at all"
            )
        self.temperature = temperature
        self.k = _require_count(k, "k", minimum=1, allow_none=True)
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
        self.maximize = require_exact_bool(maximize, "maximize")
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
    state indices. Delegates to :func:`mixle.enumeration.hmm_paths.hmm_best_paths`, whose backward
    Viterbi completion value is a tight admissible heuristic: emission log-*densities* may be positive
    (any density > 1), so a zero-heuristic ``sense="max"`` search would pop suboptimal goals first.

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
        log_b = np.asarray(self.log_obs, dtype=np.float64)
        for path, score in hmm_best_paths(self.log_init, self.log_trans, log_b, k=k):
            yield Solution(list(path), float(score))


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
        # A non-finite response makes EVERY subset score NaN, and NaN comparisons in the ranking sort
        # are all False -- so the enumeration came out in input order and `solve()` returned the first
        # subset it happened to build as "the optimum", with objective NaN (MXR-080-1903). A ranking
        # over an unordered set is not a ranking; refuse the input rather than rank it.
        if not np.isfinite(self.y).all():
            raise ValueError("BestSubsetRegression: y must be finite; a non-finite response scores every subset NaN")
        if criterion not in ("aic", "bic", "rss"):
            raise ValueError("criterion must be 'aic', 'bic', or 'rss'")
        self.criterion = criterion
        self.n, self.p = self.X.shape
        # `int(max_size)` truncated: max_size=2.9 searched subsets up to size 2, max_size="3" was
        # accepted outright, max_size=True searched size 1, and max_size=-1 made solve() return None
        # as though the problem were infeasible (MXR-080-1903).
        self.max_size = self.p if max_size is None else _require_count(max_size, "max_size", minimum=0)
        self.intercept = require_exact_bool(intercept, "intercept")

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
