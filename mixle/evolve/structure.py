"""Structural genotype for evolutionary structure search — a model's compositional tree + a distance on it.

Grammar/structure induction is not open research: it is genetic programming over structures (Koza 1992) with a
tree-edit genotype distance (Zhang & Shasha 1989) and selection by fitness. This module supplies the genotype
(``model_signature``) and the distance (``structural_distance``); the :class:`~mixle.evolve.operators.Mutate`
operator supplies the mutations and the :class:`~mixle.evolve.population.Population` + verify gate supply selection.

A signature is a nested ``(type_label, [child_signatures])`` tree — a mixture recurses into its ``.components``,
a leaf is childless. The distance is an (unordered) tree-edit distance with greedy child matching: exact relabel
cost plus insert/delete of unmatched subtrees, normalized to ``[0, 1]``. Greedy matching is a standard, adequate
approximation for the shallow trees model structures produce."""

from __future__ import annotations

from typing import Any

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
    """Unordered tree-edit distance: relabel cost (0/1) + greedy min-cost matching of children, with unmatched
    subtrees fully inserted/deleted. Exact for identical trees; a standard greedy approximation otherwise."""
    relabel = 0 if a[0] == b[0] else 1
    return relabel + _match_children(list(a[1]), list(b[1]))


def _match_children(kids_a: list[Signature], kids_b: list[Signature]) -> int:
    if not kids_a and not kids_b:
        return 0
    if not kids_a:
        return sum(_size(t) for t in kids_b)  # insert all of b's children
    if not kids_b:
        return sum(_size(t) for t in kids_a)  # delete all of a's children
    remaining = list(kids_b)
    total = 0
    for child in kids_a:
        if not remaining:
            total += _size(child)  # nothing left to match -> delete
            continue
        dists = [tree_edit_distance(child, other) for other in remaining]
        j = min(range(len(remaining)), key=lambda i: dists[i])
        total += dists[j]
        remaining.pop(j)
    total += sum(_size(t) for t in remaining)  # leftover b children -> insert
    return total


def structural_distance(a: Any, b: Any) -> float:
    """A ``[0, 1]`` genotype distance between two models' structures (tree-edit distance, size-normalized)."""
    sig_a, sig_b = model_signature(a), model_signature(b)
    denom = max(_size(sig_a) + _size(sig_b), 1)
    return tree_edit_distance(sig_a, sig_b) / denom


__all__ = ["model_signature", "tree_edit_distance", "structural_distance"]
