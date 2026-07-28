"""Structural genotype for evolutionary structure search — a model's compositional tree + a distance on it.

Grammar/structure induction is not open research: it is genetic programming over structures (Koza 1992) with a
tree-edit genotype distance (Zhang & Shasha 1989) and selection by fitness. This module supplies the genotype
(``model_signature``) and the distance (``structural_distance``); the :class:`~mixle.evolve.operators.Mutate`
operator supplies the mutations and the :class:`~mixle.evolve.population.Population` + verify gate supply selection.

A signature is a nested ``(type_label, [child_signatures])`` tree — a mixture recurses into its ``.components``,
a leaf is childless. The distance is an (unordered) tree-edit distance: exact relabel cost plus a **minimum-cost
assignment** between the two nodes' children, with unmatched subtrees fully inserted/deleted, normalized to
``[0, 1]``.

Child matching used to be greedy — take each of ``a``'s children in order and pair it with whichever of ``b``'s
remaining children is currently closest. That is order-dependent, and therefore not symmetric: ``A(A)`` vs
``A(B,A)`` measured 1, and the same pair with the operands swapped measured 2, because the greedy walk over
``[B, A]`` spent its only partner on ``B`` before reaching ``A``. Normalization cannot repair that. A "distance"
that changes when you swap its arguments is not one, and :class:`~mixle.evolve.population.Population`'s diversity
selection consumes it as if it were: two members' dissimilarity depended on which came first in the list."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

Signature = tuple[str, list]


def model_signature(model: Any) -> Signature:
    """The compositional structure of ``model`` as a ``(type_label, [child signatures])`` tree."""
    label = type(model).__name__
    components = getattr(model, "components", None)
    if isinstance(components, (list, tuple)) and len(components) > 0:
        return (label, [model_signature(c) for c in components])
    return (label, [])


def _size(sig: Signature) -> int:
    return 1 + sum(_size(child) for child in sig[1])


def tree_edit_distance(a: Signature, b: Signature) -> int:
    """Unordered tree-edit distance: relabel cost (0/1) + minimum-cost matching of children, with unmatched
    subtrees fully inserted/deleted. Symmetric, zero exactly on identical trees, and non-negative."""
    relabel = 0 if a[0] == b[0] else 1
    return relabel + _match_children(list(a[1]), list(b[1]))


def _match_children(kids_a: list[Signature], kids_b: list[Signature]) -> int:
    """Cheapest way to turn ``kids_a`` into ``kids_b``: an optimal assignment, not a greedy walk.

    Pairing is always at least as cheap as deleting one subtree and inserting the other -- by induction
    ``tree_edit_distance(x, y) <= size(x) + size(y) - 1`` -- so an optimal solution matches exactly
    ``min(len(a), len(b))`` pairs and inserts/deletes the surplus. The cost matrix is therefore padded to
    a square with the surplus side's subtree sizes and solved exactly (Jonker-Volgenant); the matrix
    simply transposes when the operands swap, which is what makes the result order-independent.
    """
    if not kids_a and not kids_b:
        return 0
    if not kids_a:
        return sum(_size(t) for t in kids_b)  # insert all of b's children
    if not kids_b:
        return sum(_size(t) for t in kids_a)  # delete all of a's children
    n, m = len(kids_a), len(kids_b)
    side = max(n, m)
    cost = np.zeros((side, side), dtype=np.int64)
    for i in range(side):
        for j in range(side):
            if i < n and j < m:
                cost[i, j] = tree_edit_distance(kids_a[i], kids_b[j])
            elif i < n:
                cost[i, j] = _size(kids_a[i])  # surplus a-child: deleted
            elif j < m:
                cost[i, j] = _size(kids_b[j])  # surplus b-child: inserted
    rows, cols = linear_sum_assignment(cost)
    return int(cost[rows, cols].sum())


def structural_distance(a: Any, b: Any) -> float:
    """A symmetric ``[0, 1]`` genotype distance between two models' structures (tree-edit distance, size-normalized)."""
    sig_a, sig_b = model_signature(a), model_signature(b)
    denom = max(_size(sig_a) + _size(sig_b), 1)
    return tree_edit_distance(sig_a, sig_b) / denom


__all__ = ["model_signature", "tree_edit_distance", "structural_distance"]
