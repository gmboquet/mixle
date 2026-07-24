"""SQL connector -- any SQLAlchemy URL (Postgres / MySQL / SQLite / ...) into a DataSource.

Optional: requires ``sqlalchemy`` (``pip install mixle[sql]``). One connector covers every RDBMS. Unlike
most connectors here, a SQL result set is not a cheap thing to hold in memory -- a large query must not
have to fit in the driver, or in mixle, before the first record reaches an encoder. See
:class:`SqlCursorSource` for the streaming, disposal, and replay contract (MXR-080-0064).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from mixle.data.partition import num_chunks_for
from mixle.data.schema import Schema
from mixle.data.structure import EXCHANGEABLE, SampleStructure

try:  # optional dependency
    import sqlalchemy as _sa
except ImportError:  # pragma: no cover - exercised only without sqlalchemy
    _sa = None


# Bounds two related but distinct things, both by default tied to this one constructor knob (override
# per call via encode(..., batch_size=...) if the two ever need to differ): (1) SQLAlchemy's
# `Result.yield_per` -- how many rows the DBAPI driver buffers per fetch round-trip, which is also what
# makes `stream_results=True` actually use a server-side cursor instead of the driver's default
# full-result client-side buffering (psycopg2 / mysqlclient do this unless told otherwise; SQLite has no
# separate server process but honors the same fetch-batching path); and (2) how many records `encode()`
# accumulates per partition before schema-conforming and flushing them through the encoder. Mirrors
# `mixle.data.sources.array_source._ENCODE_BATCH_ROWS`.
_STREAM_BATCH_ROWS = 1024

# Sentinel for "no group seen yet" in encode()'s grouped-streaming loop -- distinct from any real group
# key, including None (None is a legitimate group_key() return for an ungrouped record).
_NO_GROUP = object()


def _check_driver() -> None:
    if _sa is None:
        from mixle.utils.optional_deps import require

        require("sqlalchemy", "sql")


class SqlCursorSource:
    """A cursor-backed, bounded-memory :class:`~mixle.data.core.DataSource` over a SQL query.

    Unlike :class:`~mixle.data.core.LazySource` (read a factory once, cache the result as a list -- the
    right contract for a connector that is cheap to hold in memory), this source never builds a Python
    list of the full result set internally. ``records()``/``encode()`` open a fresh connection and pull
    rows off a streaming server-side cursor (``stream_results=True``, fetched ``batch_size`` at a time via
    SQLAlchemy's ``yield_per``) straight into the caller/encoder, so a query with more rows than fit in
    memory can still be consumed one bounded batch at a time (MXR-080-0064). ``.materialize()`` remains
    available as an explicit, honestly-named opt-in for callers who really do want the whole result set
    as a list (e.g. a small lookup table) -- it is no longer what ``records()``/``encode()`` do by default.

    **Replay policy**: each call to ``records()``/``encode()``/``materialize()`` re-executes the query
    from scratch, on a fresh connection. A live server-side cursor cannot be rewound, so "replay" here
    means "run the query again" (as re-running it by hand would), not "replay a cached snapshot" -- if the
    underlying table changes between calls, later calls see the new data. Two calls are fully independent:
    nothing about this object is safe to share a single in-flight iterator across two consumers, but
    taking two separate ``records()`` calls is always fine, including concurrently.

    **Disposal**: every connection/engine this object opens is disposed in a ``finally``, deterministically
    -- on normal exhaustion, on an exception raised while a row is being processed downstream (the
    exception unwinds the consuming frame, which drops the last reference to the generator; CPython
    finalizes it synchronously, running the pending ``finally``), and on early abandonment (explicitly
    call ``.close()`` on the iterator returned by ``records()`` for an immediate, GC-independent close;
    letting it fall out of scope also works under CPython's refcounting but is not a language guarantee).
    There is deliberately no object-level ``close()``/context-manager pair here (contrast
    :class:`~mixle.data.sources.array_source.HDF5ArraySource`): the replay-by-re-execute design means no
    connection is held between calls, so there is nothing to close while the object is idle.

    **Grouped (partially-exchangeable) structure**: assigning whole groups to partitions needs to know a
    group is complete before it can be flushed, which a single forward pass can only bound in memory if
    the query already yields rows contiguously by group key (e.g. ``ORDER BY`` the group column) -- this
    is required and enforced, not assumed: a group key reappearing after the stream has moved on to a
    different group raises ``ValueError`` rather than silently splitting the group across partitions.
    """

    def __init__(
        self,
        url: str,
        query: str,
        columns: list[str] | None = None,
        *,
        structure: SampleStructure = EXCHANGEABLE,
        schema: Schema | None = None,
        batch_size: int = _STREAM_BATCH_ROWS,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        self._url = url
        self._query = query
        self._columns = columns
        self.structure = structure
        self.schema = schema
        self._batch_size = batch_size

    def _stream(self) -> Iterator[Any]:
        """A fresh generator per call: connect, stream rows, and deterministically dispose on every exit
        path (normal completion, early close, or an exception raised while a row is being consumed)."""
        _check_driver()
        engine = _sa.create_engine(self._url)  # not yet in the try/finally: nothing to dispose if this fails
        try:
            with engine.connect().execution_options(stream_results=True, yield_per=self._batch_size) as conn:
                result = conn.execute(_sa.text(self._query))
                keys = list(result.keys())
                idx = [keys.index(c) for c in self._columns] if self._columns is not None else list(range(len(keys)))
                for row in result:
                    picked = [row[i] for i in idx]
                    yield picked[0] if len(picked) == 1 else tuple(picked)
        finally:
            engine.dispose()

    def records(self) -> Iterable[Any]:
        """Stream one full query execution's rows. Call again to re-run the query (see class docstring)."""
        _check_driver()
        return self._stream()

    def materialize(self) -> list[Any]:
        """Run the query and return every row as a list -- an explicit opt-in that bypasses the
        bounded-memory path; use only when the result is known to be small enough to hold in memory."""
        _check_driver()
        records = list(self._stream())
        return self.schema.conform(records) if self.schema is not None else records

    def _count_rows(self) -> int:
        """Resolve the query's row count via a ``COUNT(*)`` subquery -- bounded to a single integer,
        never the row data -- needed only to turn a ``chunk_size`` target into a partition count."""
        _check_driver()
        wrapped = self._query.strip().rstrip(";")
        engine = _sa.create_engine(self._url)
        try:
            with engine.connect() as conn:
                try:
                    stmt = _sa.text(f"SELECT COUNT(*) FROM ({wrapped}) AS _mixle_count")
                    n = conn.execute(stmt).scalar_one()
                except _sa.exc.SQLAlchemyError as e:
                    raise ValueError(
                        "could not resolve chunk_size: wrapping the query in a COUNT(*) subquery to "
                        f"find its row count failed ({e}). Pass num_chunks explicitly instead (it needs "
                        "no row count), or call .materialize() and partition the list yourself."
                    ) from e
        finally:
            engine.dispose()
        return int(n)

    def encode(
        self,
        encoder: Any,
        num_chunks: int = 1,
        chunk_size: int | None = None,
        batch_size: int | None = None,
    ) -> list[tuple[int, Any]]:
        """Partition and encode straight off the cursor, in bounded batches -- never a full record list.

        Strideable structures (everything but partially-exchangeable) round-robin record ``i`` into
        partition ``i % n_parts`` as it streams by, matching the ``records[k::n]`` striding
        :func:`~mixle.data.partition.partition_records` uses on a materialized list -- bit-identical
        partitioning, just computed without ever holding ``records`` itself. Partially-exchangeable
        (grouped) structure streams whole groups round-robin, in first-encountered order, exactly as
        :func:`~mixle.data.partition.partition_records` does, but requires the query to already yield
        rows contiguous by group key (see class docstring).

        Like :func:`~mixle.data.sources.array_source._encode_indexable` (MXR-080-0056's sibling fix),
        a partition with no rows contributes nothing to the returned list rather than an explicit
        ``(0, ...)`` entry, and a partition may legitimately contribute more than one entry (one per
        flushed batch, in read order) instead of exactly one -- every existing caller sums/iterates/
        concatenates over however many entries the list contains rather than assuming exactly
        ``num_chunks`` of them, so this is not a behavior change from their point of view.
        """
        _check_driver()
        bs = self._batch_size if batch_size is None else batch_size
        if isinstance(bs, bool) or not isinstance(bs, int) or bs < 1:
            raise ValueError(f"batch_size must be a positive integer, got {bs!r}")
        n_parts = num_chunks_for(self._count_rows() if chunk_size is not None else 0, num_chunks, chunk_size)
        results: list[tuple[int, Any]] = []

        def flush(buf: list[Any]) -> None:
            if not buf:
                return
            conformed = self.schema.conform(buf) if self.schema is not None else list(buf)
            results.append((len(conformed), encoder.seq_encode(conformed)))

        if self.structure.strides_records:
            buffers: list[list[Any]] = [[] for _ in range(n_parts)]
            for i, rec in enumerate(self._stream()):
                part = i % n_parts
                buffers[part].append(rec)
                if len(buffers[part]) >= bs:
                    flush(buffers[part])
                    buffers[part] = []
            for buf in buffers:
                flush(buf)
            return results

        # Grouped: requires rows contiguous by group key (query must ORDER BY the group column). A new
        # group's partition is fixed the moment it starts (round-robin over distinct groups seen so
        # far) and its rows are appended to that partition's own buffer -- which may already hold an
        # earlier, different group's unflushed tail, exactly like partition_records's grouped branch
        # lets several groups share a partition -- flushing whenever that partition's buffer reaches
        # `bs`. Memory is bounded by `n_parts * bs`, never by the whole dataset or even one whole group.
        buffers = [[] for _ in range(n_parts)]
        current_part = 0
        current_key: Any = _NO_GROUP
        closed_keys: set[Any] = set()
        group_count = 0
        for rec in self._stream():
            key = self.structure.group_key(rec)
            if key != current_key:
                if key in closed_keys:
                    raise ValueError(
                        "partially-exchangeable SQL streaming requires rows contiguous by group key "
                        f"(add ORDER BY the group column to the query); group key {key!r} reappeared "
                        "after the stream had already moved on to a different group."
                    )
                if current_key is not _NO_GROUP:
                    closed_keys.add(current_key)
                current_key = key
                current_part = group_count % n_parts
                group_count += 1
            buffers[current_part].append(rec)
            if len(buffers[current_part]) >= bs:
                flush(buffers[current_part])
                buffers[current_part] = []
        for buf in buffers:
            flush(buf)
        return results


def read_sql(
    url: str,
    query: str,
    columns: list[str] | None = None,
    *,
    structure: SampleStructure = EXCHANGEABLE,
    schema: Schema | None = None,
    batch_size: int = _STREAM_BATCH_ROWS,
) -> SqlCursorSource:
    """Read the rows of ``query`` against the database at ``url`` into a cursor-backed, bounded-memory
    :class:`SqlCursorSource` -- see its docstring for the streaming, disposal, and replay contract."""
    return SqlCursorSource(url, query, columns, structure=structure, schema=schema, batch_size=batch_size)
