"""Unambiguous value types for the two common permutation representations."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from mixle.stats.rankings._contracts import permutation


@dataclass(frozen=True)
class ItemOrdering(Sequence[int]):
    """A permutation whose entry at each rank is the item occupying that rank."""

    values: tuple[int, ...]

    def __init__(self, values: Sequence[int] | np.ndarray) -> None:
        raw = np.asarray(values)
        checked = permutation(raw, len(raw), label="item ordering")
        object.__setattr__(self, "values", tuple(int(value) for value in checked))

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[int]:
        return iter(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def to_rank_vector(self) -> RankVector:
        """Return the inverse permutation, indexed by item."""
        inverse = np.empty(len(self), dtype=np.int64)
        inverse[np.asarray(self.values, dtype=np.int64)] = np.arange(len(self), dtype=np.int64)
        return RankVector(inverse)


@dataclass(frozen=True)
class RankVector(Sequence[int]):
    """A permutation whose entry for each item is that item's rank."""

    values: tuple[int, ...]

    def __init__(self, values: Sequence[int] | np.ndarray) -> None:
        raw = np.asarray(values)
        checked = permutation(raw, len(raw), label="rank vector")
        object.__setattr__(self, "values", tuple(int(value) for value in checked))

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[int]:
        return iter(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def to_item_ordering(self) -> ItemOrdering:
        """Return the inverse permutation, indexed by rank."""
        return ItemOrdering(np.argsort(np.asarray(self.values, dtype=np.int64)))


def item_ordering_to_rank_vector(value: Sequence[int] | np.ndarray | ItemOrdering) -> np.ndarray:
    """Convert ``x[rank] = item`` to owned ``rank[item] = rank`` data."""
    ordering = value if isinstance(value, ItemOrdering) else ItemOrdering(value)
    return np.asarray(ordering.to_rank_vector().values, dtype=np.int64)


def rank_vector_to_item_ordering(value: Sequence[int] | np.ndarray | RankVector) -> np.ndarray:
    """Convert ``rank[item] = rank`` to owned ``x[rank] = item`` data."""
    ranks = value if isinstance(value, RankVector) else RankVector(value)
    return np.asarray(ranks.to_item_ordering().values, dtype=np.int64)


__all__ = [
    "ItemOrdering",
    "RankVector",
    "item_ordering_to_rank_vector",
    "rank_vector_to_item_ordering",
]
