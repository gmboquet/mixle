"""Sample structure -- the exchangeability tag carried by every :class:`~pysp.data.core.DataSource`.

``seq_encode(data, num_chunks=C)`` partitions a dataset by *striding* -- chunk ``i`` is ``data[i::C]`` --
which silently reorders observations. That is correct only when the records are exchangeable. This module
makes the intended joint structure explicit so partitioning is *justified* rather than assumed, and so a
model can be checked against the data it is handed:

* ``IID``                   -- independent & identically distributed records.
* ``EXCHANGEABLE``          -- the joint law is permutation-invariant (de Finetti): order is irrelevant
                              but latent coupling is allowed (mixtures, Dirichlet-process models, ...).
* ``PARTIALLY_EXCHANGEABLE`` (``by``) -- exchangeable *within* groups keyed by ``by`` (hierarchical /
                              grouped / panel data): groups must stay intact on a partition.
* ``SEQUENTIAL``            -- each record is a whole ordered sequence (HMM / Markov / Hawkes / AR); the
                              records are mutually exchangeable, so they may be strided, but a record is
                              never split internally (the encoder owns the within-record order).

The first three (and ``SEQUENTIAL``, whose records are atomic) may stride records freely; only
``PARTIALLY_EXCHANGEABLE`` constrains partitioning -- groups are distributed whole. The default for an
un-annotated dataset is ``EXCHANGEABLE``, which is exactly today's striding behavior, so nothing changes
until a user opts in by tagging a source.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SampleStructure:
    """The joint structure of a dataset's records (an exchangeability class)."""

    kind: str  # "iid" | "exchangeable" | "partially_exchangeable" | "sequential"
    by: str | Callable[[Any], Any] | None = None  # grouping key for partial exchangeability

    @property
    def strides_records(self) -> bool:
        """True if records may be strided/shuffled across partitions (everything but grouped data)."""
        return self.kind != "partially_exchangeable"

    def group_key(self, record: Any) -> Any:
        """Return the group key of ``record`` for partial exchangeability (else ``None``)."""
        if self.by is None:
            return None
        if callable(self.by):
            return self.by(record)
        if isinstance(record, dict):
            return record[self.by]
        return getattr(record, self.by)

    def __str__(self) -> str:
        return self.kind if self.by is None else "%s(by=%r)" % (self.kind, self.by)


IID = SampleStructure("iid")
EXCHANGEABLE = SampleStructure("exchangeable")
SEQUENTIAL = SampleStructure("sequential")


def partially_exchangeable(by: str | Callable[[Any], Any]) -> SampleStructure:
    """Return a ``PARTIALLY_EXCHANGEABLE`` structure grouped by field name or key function ``by``."""
    return SampleStructure("partially_exchangeable", by)
