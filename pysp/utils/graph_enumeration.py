"""Unified lazy best-first / k-shortest-path enumeration over arbitrary state graphs.

A single primitive underlies many of pysp's enumeration tasks: enumerating assignments in
increasing cost, ranking permutations in decreasing probability, decoding sequences, and finding
k-best paths in a DAG are all *"explore a state graph in monotone order of an additive cost (or
descending additive score), lazily, optionally with an admissible heuristic (A*)."*

:func:`best_first_paths` is that primitive. The rest of the module shows how new problem types map
onto it as one-screen reductions -- :func:`k_best_edit_scripts` (non-uniform / weighted edit
distance) and :func:`k_best_viterbi_paths` (k-best HMM hidden-state sequences). Both were missing
from the per-problem enumerators; here they are corollaries of the same core, which is the point of
unifying the framework.

Cost conventions are made explicit with ``sense``:

* ``sense="min"`` -- ``successors`` return *step costs* (>= 0 for Dijkstra-correctness, or any reals
  with an admissible lower-bound heuristic); goals come out in *increasing* total cost. Use for
  edit distance, assignment, shortest paths.
* ``sense="max"`` -- ``successors`` return *step scores* (e.g. log-probabilities <= 0); goals come
  out in *decreasing* total score. Use for ranking / decoding / Viterbi.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Iterable, Iterator
from typing import Any

__all__ = [
    "best_first_paths",
    "k_best_edit_scripts",
    "k_best_viterbi_paths",
]


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
        start: The initial state (any hashable-free object; states need not be hashable).
        successors: ``state -> iterable of (next_state, step)`` where ``step`` is the edge cost
            (``sense="min"``) or edge score (``sense="max"``).
        is_goal: Predicate; when true for a popped state it is emitted (and not expanded further).
        sense: ``"min"`` to minimise total cost (increasing order out) or ``"max"`` to maximise
            total score (decreasing order out).
        heuristic: Optional ``state -> float`` admissible estimate of the *remaining* cost
            (a lower bound, ``sense="min"``) or remaining score (an upper bound, ``sense="max"``).
            Must be 0 at goals. ``None`` is the always-admissible zero heuristic (uniform-cost).
        max_results: Stop after yielding this many goals (``None`` = exhaust the graph).
        return_paths: Yield ``(path_list, total)`` when true, else ``(goal_state, total)``.

    Yields:
        ``(path_or_state, total)`` pairs in best-first order. With an admissible heuristic the order
        is exact; the search is a path-based A*, so on a DAG/trellis it enumerates k-best paths.
    """
    if sense not in ("min", "max"):
        raise ValueError("sense must be 'min' or 'max'")
    flip = 1.0 if sense == "min" else -1.0
    h = heuristic or (lambda _s: 0.0)
    cnt = itertools.count()

    def priority(g: float, state: Any) -> float:
        return flip * (g + h(state))

    # heap entries: (priority, tie_break, g_so_far, state, path)
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
# Reduction 1: non-uniform (weighted) edit distance / sequence alignment
# ---------------------------------------------------------------------------
def k_best_edit_scripts(
    source: list,
    target: list,
    *,
    sub_cost: Callable[[Any, Any], float] | None = None,
    ins_cost: Callable[[Any], float] | None = None,
    del_cost: Callable[[Any], float] | None = None,
    k: int | None = None,
) -> Iterator[tuple[list[tuple[str, Any, Any]], float]]:
    """Enumerate the k lowest-cost edit scripts turning ``source`` into ``target``.

    Each operation carries its own (possibly position/symbol-dependent) cost, so this is the
    *non-uniform* edit distance. It reduces to a shortest path in the edit DAG whose nodes are
    ``(i, j)`` cursor positions: a diagonal step matches/substitutes, a vertical step deletes from
    ``source``, a horizontal step inserts from ``target``.

    Args:
        source, target: The two sequences (lists of comparable tokens).
        sub_cost: ``(a, b) -> cost`` of aligning ``a`` (source) with ``b`` (target); defaults to the
            unit cost (0 if equal else 1).
        ins_cost: ``b -> cost`` of inserting target token ``b`` (default 1).
        del_cost: ``a -> cost`` of deleting source token ``a`` (default 1).
        k: Number of scripts to return (``None`` = all, in increasing cost). The first is the
            (weighted) edit distance and its optimal alignment.

    Yields:
        ``(ops, total_cost)`` where ``ops`` is a list of ``(kind, a, b)`` with ``kind`` in
        ``{"match", "sub", "del", "ins"}`` (``a``/``b`` are the involved tokens or ``None``).
    """
    sub_cost = sub_cost or (lambda a, b: 0.0 if a == b else 1.0)
    ins_cost = ins_cost or (lambda b: 1.0)
    del_cost = del_cost or (lambda a: 1.0)
    ns, nt = len(source), len(target)
    goal = (ns, nt)

    def successors(node):
        i, j = node
        out = []
        if i < ns and j < nt:
            out.append(((i + 1, j + 1), sub_cost(source[i], target[j])))
        if i < ns:
            out.append(((i + 1, j), del_cost(source[i])))
        if j < nt:
            out.append(((i, j + 1), ins_cost(target[j])))
        return out

    # admissible lower bound: every remaining unmatched length difference needs >=0 cost; 0 is safe.
    for path, cost in best_first_paths((0, 0), successors, lambda n: n == goal, sense="min", max_results=k):
        yield (_path_to_ops(path, source, target), cost)


def _path_to_ops(path, source, target):
    ops = []
    for (i0, j0), (i1, j1) in zip(path, path[1:]):
        if i1 == i0 + 1 and j1 == j0 + 1:
            a, b = source[i0], target[j0]
            ops.append(("match" if a == b else "sub", a, b))
        elif i1 == i0 + 1:
            ops.append(("del", source[i0], None))
        else:
            ops.append(("ins", None, target[j0]))
    return ops


# ---------------------------------------------------------------------------
# Reduction 2: k-best Viterbi (k most-likely hidden state sequences)
# ---------------------------------------------------------------------------
def k_best_viterbi_paths(
    log_init: Any,
    log_trans: Any,
    log_obs: Any,
    k: int | None = None,
) -> Iterator[tuple[list[int], float]]:
    """Enumerate the k most-likely hidden-state sequences of an HMM (k-best Viterbi).

    Reduces to a longest-(log-prob)-path enumeration in the trellis whose nodes are ``(t, s)``.
    The standard Viterbi algorithm returns only the single best path; this yields the top ``k`` in
    decreasing joint log-probability, which the per-problem enumerators could not do for HMMs.

    Args:
        log_init: ``log p(state s at t=0)``, length ``S``.
        log_trans: ``log p(s' | s)``, shape ``(S, S)``.
        log_obs: ``log p(observation_t | state s)``, shape ``(T, S)`` -- already conditioned on the
            observed sequence (so the caller supplies emission log-likelihoods per timestep).
        k: Number of state sequences to return (``None`` = all, in decreasing log-prob).

    Yields:
        ``(states, total_log_prob)`` where ``states`` is a length-``T`` list of state indices.
    """
    log_obs = [list(row) for row in log_obs]
    T = len(log_obs)
    S = len(log_init)
    if T == 0:
        return
    start = (-1, -1)  # virtual start before choosing the t=0 state

    def successors(node):
        t, s = node
        if t == -1:
            return [((0, sp), float(log_init[sp]) + float(log_obs[0][sp])) for sp in range(S)]
        if t >= T - 1:
            return []
        return [((t + 1, sp), float(log_trans[s][sp]) + float(log_obs[t + 1][sp])) for sp in range(S)]

    for path, score in best_first_paths(start, successors, lambda n: n[0] == T - 1, sense="max", max_results=k):
        yield ([s for (_t, s) in path[1:]], score)
