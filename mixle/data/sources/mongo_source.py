"""MongoDB connector -- a collection query into a :class:`~mixle.data.core.DataSource`.

Optional: requires ``pymongo`` (``pip install mixle[mongo]``). The ``_id`` field is dropped; supply a
:class:`~mixle.data.schema.Schema` to coerce the loosely-typed BSON documents. Like a SQL result set (see
:mod:`mixle.data.sources.sql_source`), a MongoDB query result is not a cheap thing to hold in memory -- a
large collection query must not have to fit in the driver, or in mixle, before the first document reaches
an encoder. See :class:`MongoCursorSource` for the streaming, disposal, replay, and grouped-ordering
contract (MXR-080-1666: "Mongo reads exhaust the cursor before yielding and never close the client" --
independently numbered from, but the same architectural gap as, the SQL connector's MXR-080-0064 fix: a
different backend, the same "materialize the whole result before the first record is usable" defect).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from mixle.data.partition import num_chunks_for
from mixle.data.schema import Schema
from mixle.data.structure import EXCHANGEABLE, SampleStructure

try:  # optional dependency
    import pymongo as _pymongo
except ImportError:  # pragma: no cover - exercised only without pymongo
    _pymongo = None


# Bounds two related but distinct things, both by default tied to this one constructor knob (override
# per call via encode(..., batch_size=...) if the two ever need to differ): (1) pymongo's
# ``Cursor.batch_size`` -- how many documents the driver fetches per ``getMore`` round-trip to the
# server, the direct Mongo analogue of SQLAlchemy's ``yield_per``/a server-side cursor's fetch size
# (pymongo cursors are batch-fetching by design; this just bounds the batch instead of leaving it at the
# driver's own default); and (2) how many records ``encode()`` accumulates per partition before
# schema-conforming and flushing them through the encoder. Mirrors
# ``mixle.data.sources.sql_source._STREAM_BATCH_ROWS`` / ``array_source._ENCODE_BATCH_ROWS``.
_STREAM_BATCH_DOCS = 1024

# Sentinel for "no group seen yet" in encode()'s grouped-streaming loop -- distinct from any real group
# key, including None (None is a legitimate group_key() return for an ungrouped record).
_NO_GROUP = object()

# pymongo.ASCENDING's actual value (1; pymongo.DESCENDING is -1) -- hardcoded rather than read off
# ``_pymongo.ASCENDING`` so that a test can monkeypatch just ``_pymongo.MongoClient`` (e.g. to
# ``mongomock.MongoClient``) without also having to fake this constant, and so this module never
# touches ``_pymongo`` at import time (it may be ``None``).
_ASCENDING = 1


def _check_driver() -> None:
    if _pymongo is None:
        from mixle.utils.optional_deps import require

        require("pymongo", "mongo")


class MongoCursorSource:
    """A cursor-backed, bounded-memory :class:`~mixle.data.core.DataSource` over a MongoDB query.

    Unlike :class:`~mixle.data.core.LazySource` (read a factory once, cache the result as a list -- the
    right contract for a connector that is cheap to hold in memory), this source never builds a Python
    list of the full result set internally. ``records()``/``encode()`` open a fresh client and pull
    documents off a batched cursor (``find(...).batch_size(batch_size)``) straight into the
    caller/encoder, so a collection query with more documents than fit in memory can still be consumed
    one bounded batch at a time (MXR-080-1666). ``.materialize()`` remains available as an explicit,
    honestly-named opt-in for callers who really do want the whole result set as a list (e.g. a small
    lookup collection) -- it is no longer what ``records()``/``encode()`` do by default.

    **Replay policy**: each call to ``records()``/``encode()``/``materialize()`` re-runs the query from
    scratch, on a fresh ``MongoClient``. A live cursor cannot be rewound, so "replay" here means "run the
    query again" (as re-running it by hand would), not "replay a cached snapshot" -- if the underlying
    collection changes between calls, later calls see the new data. Two calls are fully independent:
    nothing about this object is safe to share a single in-flight iterator across two consumers, but
    taking two separate ``records()`` calls is always fine, including concurrently.

    **Disposal**: every client this object opens is closed in a ``finally``, deterministically -- on
    normal exhaustion, on an exception raised while a document is being processed downstream (the
    exception unwinds the consuming frame, which drops the last reference to the generator; CPython
    finalizes it synchronously, running the pending ``finally``), and on early abandonment (explicitly
    call ``.close()`` on the iterator returned by ``records()`` for an immediate, GC-independent close;
    letting it fall out of scope also works under CPython's refcounting but is not a language guarantee).
    The live cursor itself is also closed explicitly (telling the server to free it immediately) before
    the client that owns it. There is deliberately no object-level ``close()``/context-manager pair here
    (contrast :class:`~mixle.data.sources.array_source.HDF5ArraySource`): the replay-by-re-execute design
    means no connection is held between calls, so there is nothing to close while the object is idle.

    **Count resolution**: a streaming cursor does not know its size up front, so a ``chunk_size`` target
    is resolved via ``count_documents()`` -- one bounded integer, never the document data. Simpler than
    :class:`~mixle.data.sources.sql_source.SqlCursorSource`'s ``COUNT(*)`` subquery: ``count_documents``
    takes the same filter dict ``_stream()`` already builds directly, with no query-text wrapping/parsing
    involved (there is no raw query string here to manipulate in the first place).

    **Projection**: ``columns=None`` fetches whole documents (minus ``_id``, always dropped); an
    explicitly empty ``columns=[]`` projects to zero columns (every record becomes ``()``), matching
    ``text_source.read_csv``'s and the fixed ``hadoop_source.read_remote``'s ``columns=[]`` semantics
    (MXR-080-0065's sibling truthiness bug: ``if columns`` treats ``[]`` and ``None`` alike, which they
    are not) -- implemented as a ``{"_id": 1}`` projection (the cheapest possible per-document payload),
    since an empty Mongo projection document ``{}`` means "no restriction" (returns full documents), not
    "zero fields": the two only look alike under Python's ``if columns`` truthiness, not in what either
    actually asks the server for. Requesting ``'_id'`` itself via ``columns`` is rejected at construction
    (it is unconditionally dropped, so honoring it would either silently do nothing or raise a confusing
    ``KeyError`` once the query actually runs).

    **Grouped (partially-exchangeable) structure**: assigning whole groups to partitions needs to know a
    group is complete before it can be flushed, which a single forward pass can only bound in memory if
    the query already yields documents contiguously by group key -- this is required and enforced, not
    assumed: a group key reappearing after the stream has moved on to a different group raises
    ``ValueError`` rather than silently splitting the group across partitions. Unlike
    :class:`~mixle.data.sources.sql_source.SqlCursorSource` (which cannot inject an ``ORDER BY`` into a
    caller's raw SQL text), this connector builds its query from structured pieces, so a plain string
    ``by`` is auto-sorted server-side (see ``_resolved_sort``) -- contiguous by construction, no caller
    action needed. A callable ``by`` has no server-side sort equivalent (Mongo cannot sort by an
    arbitrary Python function), so it still needs the caller to pass an explicit ``sort`` (or otherwise
    guarantee contiguous order) -- exactly SQL's ``ORDER BY``-by-hand requirement, just for a narrower
    case. Either way, a string ``by`` also requires ``columns=None``: a projected record is a
    tuple/scalar (positional), not a document dict, so ``SampleStructure.group_key`` cannot look a field
    up by name on it (mirrors ``SqlCursorSource``, whose rows are always tuples and so always need a
    callable ``by``) -- rejected at construction rather than raising a confusing ``AttributeError`` deep
    in ``encode()``.
    """

    def __init__(
        self,
        uri: str,
        database: str,
        collection: str,
        query: dict | None = None,
        columns: list[str] | None = None,
        *,
        structure: SampleStructure = EXCHANGEABLE,
        schema: Schema | None = None,
        batch_size: int = _STREAM_BATCH_DOCS,
        sort: Any = None,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if columns is not None and "_id" in columns:
            raise ValueError(
                f"columns={columns!r} requests '_id', but this connector always drops '_id' from every "
                "document before yielding it -- remove '_id' from columns (it would otherwise raise a "
                "confusing KeyError once the query actually runs)."
            )
        if columns is not None and not structure.strides_records and isinstance(structure.by, str):
            raise ValueError(
                f"partially_exchangeable(by={structure.by!r}) needs a callable `by`, not a field name, "
                "when columns is given: a projected record is a tuple/scalar (positional), not a "
                "document dict, so group_key() cannot look the field up by name (mirrors "
                "SqlCursorSource, whose rows are always tuples). Pass a callable `by` instead (e.g. "
                "``lambda record: record[i]``), or drop columns and group full documents."
            )
        self._uri = uri
        self._database = database
        self._collection = collection
        self._query = query
        self._columns = columns
        self.structure = structure
        self.schema = schema
        self._batch_size = batch_size
        self._sort = sort

    def _projection(self) -> dict[str, int] | None:
        """Build the Mongo projection document for ``self._columns`` (see class docstring)."""
        if self._columns is None:
            return None
        if not self._columns:
            return {"_id": 1}
        return {c: 1 for c in self._columns}

    def _resolved_sort(self) -> Any:
        """Resolve the sort spec actually passed to ``cursor.sort()`` (see class docstring: an explicit
        ``sort`` always wins, used verbatim; otherwise a string-keyed grouped structure is auto-sorted)."""
        if self._sort is not None:
            return self._sort
        if not self.structure.strides_records and isinstance(self.structure.by, str):
            return [(self.structure.by, _ASCENDING)]
        return None

    def _stream(self) -> Iterator[Any]:
        """A fresh generator per call: connect, stream documents, and deterministically dispose on every
        exit path (normal completion, early close, or an exception raised while a document is being
        consumed)."""
        _check_driver()
        client = _pymongo.MongoClient(self._uri)  # lazy; not yet in the try/finally -- nothing to
        try:  # dispose if construction itself fails (e.g. a malformed URI, see MongoCursorSourceDisposalTest)
            coll = client[self._database][self._collection]
            cursor = coll.find(self._query or {}, self._projection()).batch_size(self._batch_size)
            sort = self._resolved_sort()
            if sort is not None:
                cursor = cursor.sort(sort)
            try:
                for doc in cursor:
                    doc.pop("_id", None)
                    if self._columns is None:
                        yield doc
                    else:
                        try:
                            picked = [doc[c] for c in self._columns]
                        except KeyError as e:
                            raise KeyError(
                                f"document is missing requested column {e.args[0]!r} (has: "
                                f"{sorted(doc)}); Mongo documents are schemaless, so a column present "
                                "in some documents can be absent in others."
                            ) from e
                        yield picked[0] if len(picked) == 1 else tuple(picked)
            finally:
                cursor.close()
        finally:
            client.close()

    def records(self) -> Iterable[Any]:
        """Stream one full query execution's documents. Call again to re-run the query (see class
        docstring)."""
        _check_driver()
        return self._stream()

    def materialize(self) -> list[Any]:
        """Run the query and return every document as a list -- an explicit opt-in that bypasses the
        bounded-memory path; use only when the result is known to be small enough to hold in memory."""
        _check_driver()
        records = list(self._stream())
        return self.schema.conform(records) if self.schema is not None else records

    def _count_documents(self) -> int:
        """Resolve the query's document count via ``count_documents()`` -- bounded to a single integer,
        never the document data -- needed only to turn a ``chunk_size`` target into a partition count."""
        _check_driver()
        client = _pymongo.MongoClient(self._uri)
        try:
            coll = client[self._database][self._collection]
            try:
                n = coll.count_documents(self._query or {})
            except _pymongo.errors.PyMongoError as e:
                raise ValueError(
                    "could not resolve chunk_size: count_documents() failed "
                    f"({e}). Pass num_chunks explicitly instead (it needs no document count), or call "
                    ".materialize() and partition the list yourself."
                ) from e
        finally:
            client.close()
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
        documents contiguous by group key (see class docstring).

        Like :func:`~mixle.data.sources.array_source._encode_indexable` (MXR-080-0056's sibling fix),
        a partition with no documents contributes nothing to the returned list rather than an explicit
        ``(0, ...)`` entry, and a partition may legitimately contribute more than one entry (one per
        flushed batch, in read order) instead of exactly one -- every existing caller sums/iterates/
        concatenates over however many entries the list contains rather than assuming exactly
        ``num_chunks`` of them, so this is not a behavior change from their point of view.
        """
        _check_driver()
        bs = self._batch_size if batch_size is None else batch_size
        if isinstance(bs, bool) or not isinstance(bs, int) or bs < 1:
            raise ValueError(f"batch_size must be a positive integer, got {bs!r}")
        n_parts = num_chunks_for(self._count_documents() if chunk_size is not None else 0, num_chunks, chunk_size)
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

        # Grouped: requires documents contiguous by group key (query must be sorted by the group field --
        # automatic for a string `by`, see _resolved_sort; the caller's own responsibility for a callable
        # `by`). A new group's partition is fixed the moment it starts (round-robin over distinct groups
        # seen so far) and its documents are appended to that partition's own buffer -- which may already
        # hold an earlier, different group's unflushed tail, exactly like partition_records's grouped
        # branch lets several groups share a partition -- flushing whenever that partition's buffer
        # reaches `bs`. Memory is bounded by `n_parts * bs`, never by the whole dataset or even one whole
        # group.
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
                        "partially-exchangeable Mongo streaming requires documents contiguous by group "
                        "key (pass an explicit sort=... ordering by the group field, or use a string "
                        "`by` with columns=None so MongoCursorSource can sort automatically -- see "
                        f"class docstring); group key {key!r} reappeared after the stream had already "
                        "moved on to a different group."
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


def read_mongo(
    uri: str,
    database: str,
    collection: str,
    query: dict | None = None,
    columns: list[str] | None = None,
    *,
    structure: SampleStructure = EXCHANGEABLE,
    schema: Schema | None = None,
    batch_size: int = _STREAM_BATCH_DOCS,
    sort: Any = None,
) -> MongoCursorSource:
    """Read documents matching ``query`` from ``database.collection`` at ``uri`` into a cursor-backed,
    bounded-memory :class:`MongoCursorSource` -- see its docstring for the streaming, disposal, replay,
    and grouped-ordering contract."""
    return MongoCursorSource(
        uri,
        database,
        collection,
        query,
        columns,
        structure=structure,
        schema=schema,
        batch_size=batch_size,
        sort=sort,
    )
