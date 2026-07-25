"""``DataSource`` -- a lazy, typed, structured reference to data that yields encoder-ready records.

This is the single concept that replaces the "is it a list? a DataFrame? an RDD? a SQL cursor?" branching
scattered across call sites. A ``DataSource`` carries a :class:`~mixle.data.schema.Schema` (logical field
types) and a :class:`~mixle.data.structure.SampleStructure` (its exchangeability class), and knows how to
yield ``records()`` and ``partition`` itself safely.

It is purely additive: ``seq_encode(list)`` and ``seq_encode(rdd)`` are untouched fast paths;
``seq_encode`` gains one branch that recognizes a ``DataSource`` and routes through its structure-aware
encoder, returning the same ``[(count, payload)]`` shape consumers already expect.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

from mixle.data.partition import encode_partitions, num_chunks_for, partition_records
from mixle.data.schema import Schema
from mixle.data.structure import EXCHANGEABLE, SampleStructure


@runtime_checkable
class DataSource(Protocol):
    """A lazy, typed, structured source of encoder-ready records."""

    schema: Schema | None
    structure: SampleStructure

    def records(self) -> Iterable[Any]:
        """Yield raw records compatible with an encoder's input type."""
        ...

    def encode(self, encoder: Any, num_chunks: int = 1, chunk_size: int | None = None) -> Any:
        """Partition (structure-aware) and ``seq_encode`` -> the same ``[(count, payload)]`` shape."""
        ...


class MaterializedSource:
    """An in-memory :class:`DataSource` wrapping a ``Sequence`` -- what a bare list becomes."""

    def __init__(
        self, data: Sequence[Any], structure: SampleStructure = EXCHANGEABLE, schema: Schema | None = None
    ) -> None:
        # This class owns one replayable snapshot.  ``__len__`` is not evidence that an object is
        # replayable: custom iterators commonly expose a remaining-length hint while still returning
        # themselves from ``iter()``.  Snapshotting once also prevents later caller mutation of a list
        # from changing the source between validation, hashing and fitting.
        raw = list(data)
        self._data = tuple(schema.conform(raw) if schema is not None else raw)
        self.structure = structure
        self.schema = schema

    def records(self) -> Iterable[Any]:
        """Return conformed records from the owned replayable snapshot."""
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def materialize(self) -> list[Any]:
        """Return a caller-owned list copy of the conformed snapshot."""
        return list(self._data)

    def partition(self, n: int, *, by: Any = None) -> list[MaterializedSource]:
        """Split into ``n`` structure-safe sub-sources (group-aware for partially-exchangeable data)."""
        structure = self.structure if by is None else SampleStructure("partially_exchangeable", by)
        return [
            MaterializedSource(p, structure, self.schema) for p in partition_records(self.materialize(), structure, n)
        ]

    def encode(self, encoder: Any, num_chunks: int = 1, chunk_size: int | None = None) -> list[tuple[int, Any]]:
        """Encode the materialized records through structure-aware partitioning."""
        return encode_partitions(self.materialize(), encoder, self.structure, num_chunks, chunk_size)


class LazySource:
    """A :class:`DataSource` that defers reading to a records *factory* and materializes on demand.

    Connectors (Parquet, SQL, CSV, ...) return one of these so ``open(...)`` does no I/O until the data
    is actually encoded; the records are read (and schema-coerced) once and cached.
    """

    def __init__(
        self,
        factory: Any,
        structure: SampleStructure = EXCHANGEABLE,
        schema: Schema | None = None,
        length: int | None = None,
    ) -> None:
        self._factory = factory
        self.structure = structure
        self.schema = schema
        if isinstance(length, bool) or (length is not None and (not isinstance(length, int) or length < 0)):
            raise ValueError(f"length must be a non-negative integer or None, got {length!r}")
        self._length = length
        self._cache: tuple[Any, ...] | None = None

    def _realize(self) -> tuple[Any, ...]:
        """Build and validate the immutable cache exactly once."""
        if self._cache is None:
            records = list(self._factory())
            conformed = self.schema.conform(records) if self.schema is not None else records
            realized = tuple(conformed)
            if self._length is not None and self._length != len(realized):
                raise ValueError(
                    f"declared length {self._length} does not match the realized record count {len(realized)}"
                )
            self._cache = realized
            self._length = len(realized)
        return self._cache

    def materialize(self) -> list[Any]:
        """Return a caller-owned list copy of the immutable, conformed cache."""
        return list(self._realize())

    def records(self) -> Iterable[Any]:
        """Return an iterator over the cached materialized records."""
        return iter(self._realize())

    def __len__(self) -> int:
        return self._length if self._length is not None and self._cache is None else len(self._realize())

    def partition(self, n: int, *, by: Any = None) -> list[MaterializedSource]:
        """Materialize and split into ``n`` structure-aware in-memory sources."""
        return MaterializedSource(self.materialize(), self.structure, self.schema).partition(n, by=by)

    def encode(self, encoder: Any, num_chunks: int = 1, chunk_size: int | None = None) -> list[tuple[int, Any]]:
        """Materialize records and encode them through structure-aware partitioning."""
        return encode_partitions(self.materialize(), encoder, self.structure, num_chunks, chunk_size)


def as_source(data: Any, structure: SampleStructure = EXCHANGEABLE, schema: Schema | None = None) -> DataSource:
    """Coerce ``data`` to a :class:`DataSource` (pass a source through; wrap a sequence as materialized)."""
    if isinstance(data, DataSource):
        return data
    return MaterializedSource(data, structure, schema)


__all__ = ["DataSource", "MaterializedSource", "LazySource", "as_source", "num_chunks_for"]
