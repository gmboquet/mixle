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
        # materialize()/records() re-derive from self._data on every call (cheap and correct for a
        # real Sequence, which supports repeated iteration by definition) rather than caching like
        # LazySource does -- but that's only safe if data really IS re-iterable. A one-shot iterator
        # or generator passed here despite the type hint would silently materialize empty/partial on
        # the SECOND call, far from this constructor. Checking __len__ (not isinstance(..., Sequence))
        # is deliberate: numpy.ndarray is a perfectly safe, re-iterable container here but does NOT
        # register as collections.abc.Sequence, while every one-shot iterator/generator lacks __len__.
        if not hasattr(data, "__len__"):
            raise TypeError(
                f"MaterializedSource requires a re-iterable container (with __len__), got "
                f"{type(data).__name__}; wrap a one-shot iterable with list(...) first."
            )
        self._data = data
        self.structure = structure
        self.schema = schema

    def records(self) -> Iterable[Any]:
        """Return an iterator over the in-memory records without copying the underlying sequence."""
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def materialize(self) -> list[Any]:
        """Return the records as a list, coerced to the schema if one is set."""
        return self.schema.conform(self._data) if self.schema is not None else list(self._data)

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
        self._length = length
        self._cache: list[Any] | None = None

    def materialize(self) -> list[Any]:
        """Read records from the factory once, apply the schema if present, and cache the list."""
        if self._cache is None:
            records = list(self._factory())
            self._cache = self.schema.conform(records) if self.schema is not None else records
        return self._cache

    def records(self) -> Iterable[Any]:
        """Return an iterator over the cached materialized records."""
        return iter(self.materialize())

    def __len__(self) -> int:
        return self._length if self._length is not None else len(self.materialize())

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
