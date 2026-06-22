"""k-best assignment enumeration (Murty's algorithm).

Enumerate the assignments of a cost matrix in increasing total cost -- the optimal Hungarian assignment first,
then the second-best, and so on -- without materializing the factorial space of permutations. This is Murty's
algorithm (1968): pop the best assignment from a priority queue, then partition the remaining solution space into
subproblems (each forcing some edges in and one edge out) and solve each with one Hungarian call.

The single Hungarian solve is scipy's ``linear_sum_assignment``; forbidden edges are marked ``inf`` (scipy avoids
them and raises when a subproblem is infeasible). For an n-by-n cost matrix the optimum is O(n^3) and each of the
k yielded assignments costs O(n) further Hungarian solves, so top-k is polynomial rather than O(n!).

Use ``k_best_assignments(cost, k)`` for a cost matrix directly; pass ``maximize=True`` to rank by decreasing
total weight instead. ``MatchingDistribution`` consumes this to enumerate matchings in decreasing probability.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Iterator

import numpy as np
from scipy.optimize import linear_sum_assignment


def best_assignment(cost: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the minimum-cost assignment as ``(total_cost, row_ind, col_ind)`` (scipy Hungarian)."""
    cost = np.asarray(cost, dtype=float)
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum()), rows, cols


def _solve_constrained(
    cost: np.ndarray, forced_in: tuple[tuple[int, int], ...], forced_out: frozenset[tuple[int, int]]
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Optimal assignment subject to a set of forced-in and forced-out edges, or None if infeasible.

    A forced-out edge is blocked with ``inf``. A forced-in edge ``(r, c)`` pins row r to column c by blocking the
    rest of row r and column c. If the resulting optimum still routes through a blocked cell (no feasible
    alternative), the subproblem is infeasible.
    """
    work = np.array(cost, dtype=float, copy=True)
    for r, c in forced_out:
        work[r, c] = np.inf
    for r, c in forced_in:
        keep = work[r, c]
        work[r, :] = np.inf
        work[:, c] = np.inf
        work[r, c] = keep
    try:
        rows, cols = linear_sum_assignment(work)
    except ValueError:
        return None  # no feasible assignment under these constraints
    if not np.isfinite(work[rows, cols]).all():
        return None  # optimum forced through a blocked cell -> infeasible
    return float(cost[rows, cols].sum()), rows, cols


def k_best_assignments(
    cost: np.ndarray, k: int | None = None, maximize: bool = False
) -> Iterator[tuple[float, np.ndarray, np.ndarray]]:
    """Yield assignments of ``cost`` in increasing total cost (Murty's algorithm).

    Each item is ``(total_cost, row_ind, col_ind)`` with the original-cost total (not the internal sign-flipped
    one when ``maximize=True``). Edges with non-finite cost are forbidden, so assignments that can only be
    completed through them are skipped. Enumeration is lazy; with ``k=None`` it runs until the (finite-cost)
    solution space is exhausted.

    Args:
        cost: 2-D cost matrix. Rectangular is allowed (scipy matches ``min(n, m)`` edges).
        k: maximum number of assignments to yield; ``None`` for all.
        maximize: rank by decreasing total weight instead of increasing cost (negates internally).

    Yields:
        ``(total_cost, row_ind, col_ind)`` in nondecreasing total cost.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost must be a 2-D matrix")
    work_cost = -cost if maximize else cost

    root = _solve_constrained(work_cost, (), frozenset())
    if root is None:
        return

    counter = itertools.count()  # tiebreaker so heap never compares arrays
    # heap items: (cost, tiebreak, forced_in, forced_out, row_ind, col_ind)
    heap: list = [(root[0], next(counter), (), frozenset(), root[1], root[2])]
    emitted = 0
    while heap and (k is None or emitted < k):
        node_cost, _, forced_in, forced_out, rows, cols = heapq.heappop(heap)
        total = -node_cost if maximize else node_cost
        yield total, rows, cols
        emitted += 1

        # Murty partition: the solution's edges not already forced in, in row order.
        forced_set = set(forced_in)
        free = [(int(r), int(c)) for r, c in zip(rows, cols) if (int(r), int(c)) not in forced_set]
        for t in range(len(free)):
            child_in = forced_in + tuple(free[:t])
            child_out = forced_out | {free[t]}
            child = _solve_constrained(work_cost, child_in, child_out)
            if child is not None:
                heapq.heappush(heap, (child[0], next(counter), child_in, child_out, child[1], child[2]))
