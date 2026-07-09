"""k-best spanning tree enumeration (Gabow's partition algorithm).

Enumerate the spanning trees of an undirected weighted graph in increasing total edge cost -- the minimum
spanning tree first, then the next-lowest-cost tree, and so on -- without materializing the (often exponential) set of
trees. This is the spanning-tree analogue of Murty's k-best assignment: pop the best tree from a priority queue,
then partition the remaining trees into subproblems (each forcing some tree edges in and one tree edge out) and
solve each with one constrained-MST call.

The MST oracle is a small Kruskal with union-find, so forcing edges in (add them first, fail on a cycle) and out
(skip them) is direct, and infeasibility (a forced-out edge disconnecting the graph) is detected by the tree
having fewer than n-1 edges. Edges with non-finite cost are absent. ``SpanningTreeDistribution`` consumes this to
enumerate trees in decreasing probability via ``cost = -log(weights)``.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Iterator

import numpy as np


def _kruskal(
    n: int,
    cost: np.ndarray,
    sorted_edges: list[tuple[float, int, int]],
    required: tuple[tuple[int, int], ...],
    forbidden: frozenset[tuple[int, int]],
) -> tuple[float, list[tuple[int, int]]] | None:
    """Minimum spanning tree containing every ``required`` edge and no ``forbidden`` edge, or None if infeasible."""
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    total = 0.0
    tree: list[tuple[int, int]] = []
    required_set = set(required)
    for i, j in required:
        c = cost[i, j]
        if not np.isfinite(c) or not union(i, j):  # forced edge is absent, or closes a cycle -> infeasible
            return None
        tree.append((i, j))
        total += float(c)
    for c, i, j in sorted_edges:
        if (i, j) in forbidden or (i, j) in required_set:
            continue
        if union(i, j):
            tree.append((i, j))
            total += c
    if len(tree) != n - 1:  # forced-out edges disconnected the graph
        return None
    return total, tree


def k_best_spanning_trees(cost: np.ndarray, k: int | None = None) -> Iterator[tuple[float, list[tuple[int, int]]]]:
    """Yield spanning trees of a symmetric cost matrix in increasing total cost (Gabow's algorithm).

    Each item is ``(total_cost, edges)`` with ``edges`` a list of ``(i, j)`` pairs (``i < j``). Non-finite cost
    entries are treated as absent edges. Enumeration is lazy; ``k=None`` runs until the trees are exhausted.

    Args:
        cost: symmetric n-by-n edge-cost matrix (diagonal ignored).
        k: maximum number of trees to yield; ``None`` for all.

    Yields:
        ``(total_cost, edges)`` in nondecreasing total cost.
    """
    cost = np.asarray(cost, dtype=float)
    n = cost.shape[0]
    sorted_edges = sorted(
        (float(cost[i, j]), i, j) for i in range(n) for j in range(i + 1, n) if np.isfinite(cost[i, j])
    )

    root = _kruskal(n, cost, sorted_edges, (), frozenset())
    if root is None:
        return

    counter = itertools.count()
    heap: list = [(root[0], next(counter), (), frozenset(), root[1])]
    emitted = 0
    while heap and (k is None or emitted < k):
        total, _, required, forbidden, tree = heapq.heappop(heap)
        yield total, tree
        emitted += 1

        required_set = set(required)
        free = [e for e in tree if e not in required_set]
        for t in range(len(free)):
            child_required = required + tuple(free[:t])
            child_forbidden = forbidden | {free[t]}
            child = _kruskal(n, cost, sorted_edges, child_required, child_forbidden)
            if child is not None:
                heapq.heappush(heap, (child[0], next(counter), child_required, child_forbidden, child[1]))
