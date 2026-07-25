"""Tests for the Mongo connector (mixle.data.sources.mongo_source).

Covers the connector's basic read/columns/schema contract, plus the later external-review finding
against this module (docs/audits/0.8.0-exhaustive-code-review.md MXR-080-1666, "Mongo reads exhaust the
cursor before yielding and never close the client" -- independently numbered from, but the same
architectural gap as, the SQL connector's MXR-080-0064): ``read_mongo`` appended every document from a
pymongo cursor into a Python list before ``LazySource.materialize()`` (itself a second full
materialization) cached it as one, and the client it opened was never retained or closed -- a large
collection query had to fit completely in memory before any document reached an encoder, and every call
leaked a client/connection pool.

``MongoCursorSource`` replaces the ``LazySource``-backed implementation: ``records()``/``encode()``
stream off a real pymongo cursor (batched via ``Cursor.batch_size``) in bounded batches, every client
opened is closed deterministically on every exit path, and calling ``records()``/``encode()`` again
explicitly re-runs the query (the replay policy) rather than silently caching or raising. MXR-080-1666
also calls for "projection validation": this suite also covers the ``columns=[]`` truthiness bug (the
same class of bug just fixed in ``hadoop_source.read_remote``'s CSV branch, commit bf74faf9 -- ``if
columns`` treats ``[]`` and ``None`` alike, which they are not) and rejecting ``'_id'`` requested via
``columns`` at construction instead of a confusing ``KeyError`` deep in iteration.

**Verification level reached in this environment** (no live MongoDB server, ``mongod``, or Docker was
available -- see the parent task's report for the full account): ``pymongo`` itself is a real,
pip-installed driver here (``pip install mixle[mongo]``'s actual dependency), so every exception type,
``MongoClient``/``Cursor`` method signature, and synchronous-vs-lazy-failure behavior this module relies
on was verified against the genuine library, not guessed from documentation. What pymongo cannot do
without a real server is answer queries -- for that, most tests here point ``MongoCursorSource`` at
``mongomock`` (an in-memory, wire-API-compatible fake) via ``mock.patch.object`` on the module's
``_pymongo.MongoClient`` reference, the direct Mongo analogue of the SQL tests' disposable sqlite file:
a real, fast, dependency-light local target instead of a genuine network round trip. ``ServerStore``
(``mongomock.store``) is threaded through every client construction in a test so that two separate
``MongoCursorSource``-internal ``MongoClient(uri)`` opens see the same in-memory data -- plain
``mongomock.MongoClient(same_uri)`` does NOT do this by default (verified empirically: unlike a real
server, each construction gets its own private store even given an identical URI string).

This gives strong confidence that ``MongoCursorSource``'s OWN code is lazy/correct/disposes properly
(mongomock's ``Cursor.__next__`` is monkeypatched at the class level to count documents pulled through
it -- see ``_install_pull_counter`` -- proving our generator never accumulates the whole result before
yielding the first record). It does NOT prove that a real pymongo ``Cursor`` against a real MongoDB
server actually throttles its ``getMore`` network round-trips to ``batch_size`` documents each -- that
half of the streaming claim rests on pymongo's own documented/well-known batching behavior
(``Cursor.batch_size``), inspected against the real library (signatures, chaining, defaults) but not
exercised end-to-end over a live wire protocol here. The connect/operation-failure disposal tests are the
one place a real MongoDB-shaped failure matters and mongomock (a pure in-memory fake with no network
layer) cannot simulate it, so those specifically use the real ``pymongo.MongoClient`` pointed at an
unreachable port (never a real server -- just a refused TCP connection) with a short
``serverSelectionTimeoutMS`` so the failure surfaces in well under a second.

All tests skip cleanly when pymongo and/or mongomock are not installed.
"""

from __future__ import annotations

import functools
import unittest
from typing import Any
from unittest import mock

from mixle.data.core import DataSource
from mixle.data.partition import partition_records
from mixle.data.schema import Field, Real, Schema
from mixle.data.sources import mongo_source
from mixle.data.sources.mongo_source import MongoCursorSource, read_mongo
from mixle.data.structure import EXCHANGEABLE, grouping_policy, partially_exchangeable

_HAS_PYMONGO = getattr(mongo_source, "_pymongo", None) is not None
try:
    import mongomock
    from mongomock.store import ServerStore

    _HAS_MONGOMOCK = True
except ImportError:  # pragma: no cover - exercised only without mongomock
    _HAS_MONGOMOCK = False

if _HAS_PYMONGO:
    import pymongo

_MOCK_URI = "mongodb://mixle-mongomock-fake-host/"
_DB = "db"
_COLL = "coll"


def _seed(docs: list[dict], *, store: Any = None) -> Any:
    """Populate a fresh (or given) mongomock ``ServerStore`` with ``docs`` at ``_DB.{_COLL}``; returns
    the store so callers can keep writing to it (e.g. to simulate an intervening write for a replay
    test) or hand it to another client construction that must see the same data."""
    store = store if store is not None else ServerStore()
    client = mongomock.MongoClient(_MOCK_URI, _store=store)
    if docs:
        client[_DB][_COLL].insert_many(list(docs))
    client.close()
    return store


def _patch_mongo_client(store: Any) -> Any:
    """Return an ``unittest.mock`` patcher that routes ``mongo_source``'s ``_pymongo.MongoClient`` to a
    ``mongomock.MongoClient`` pinned to ``store`` -- so every ``MongoCursorSource``-internal "fresh
    client" open (matching real MongoDB, where reconnecting to the same server sees the same data)
    actually shares one in-memory backing store, rather than each getting its own empty one."""
    return mock.patch.object(
        mongo_source._pymongo, "MongoClient", functools.partial(mongomock.MongoClient, _store=store)
    )


def _install_pull_counter() -> tuple[list[Any], Any]:
    """Monkeypatch mongomock's ``Cursor.__next__`` at the class level to record each document pulled off
    ANY mongomock cursor; returns ``(pulled, remove_fn)``.

    Mongo has no equivalent of SQLite's UDF-in-SELECT trick (see sql_source_test.py's own
    ``_install_pull_counter``) -- there is no way to run a Python callback embedded in a server-side
    query -- so this wraps the Python iterator protocol instead. It faithfully measures how many
    documents *our* generator has pulled off the cursor so far (the property ``records()``/``encode()``
    must bound), which is what a "bounded memory" claim is actually about; see the module docstring for
    what it does not prove. Patched at the class level (not the instance) because each call our source
    makes creates its own ``Cursor`` (the replay/disposal contract this module tests), so a per-instance
    listener would miss later calls.
    """
    import mongomock.collection as _mc

    pulled: list[Any] = []
    orig_next = _mc.Cursor.__next__

    def _counting_next(self: Any) -> Any:
        doc = orig_next(self)
        pulled.append(doc)
        return doc

    _mc.Cursor.__next__ = _counting_next

    def _remove() -> None:
        _mc.Cursor.__next__ = orig_next

    return pulled, _remove


class _RecordingEncoder:
    """Records each batch handed to ``seq_encode``, plus how many documents the driver had produced so
    far (via a ``_install_pull_counter()`` list) -- proves ``encode()`` streams/flushes incrementally
    rather than an eager ``list(self.records())`` first. Mirrors ``_RecordingEncoder`` in
    sql_source_test.py / array_data_sources_test.py.
    """

    def __init__(self, pulled: list[Any]) -> None:
        self._pulled = pulled
        self.batches: list[list[Any]] = []
        self.pulled_before_each_call: list[int] = []

    def seq_encode(self, data: Any) -> list[Any]:
        batch = list(data)
        self.pulled_before_each_call.append(len(self._pulled))
        self.batches.append(batch)
        return batch


@unittest.skipUnless(
    _HAS_PYMONGO and _HAS_MONGOMOCK, "pymongo/mongomock not installed; pip install mixle[mongo] mongomock"
)
class MongoSourceBasicsTest(unittest.TestCase):
    """Smoke-tests the connector's ordinary read/columns/schema contract (unchanged by MXR-080-1666)."""

    def setUp(self) -> None:
        self.store = _seed([{"id": 0, "val": 0.5}, {"id": 1, "val": 1.5}, {"id": 2, "val": 2.5}])
        patcher = _patch_mongo_client(self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_read_mongo_returns_a_mongo_cursor_source(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL)
        self.assertIsInstance(src, MongoCursorSource)
        self.assertIsInstance(src, DataSource)  # structural: schema/structure/records()/encode()

    def test_full_documents_are_dicts_with_id_dropped(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL, sort=[("id", 1)])
        docs = list(src.records())
        self.assertEqual(docs, [{"id": 0, "val": 0.5}, {"id": 1, "val": 1.5}, {"id": 2, "val": 2.5}])
        self.assertTrue(all("_id" not in d for d in docs))

    def test_single_column_yields_scalars(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL, columns=["id"], sort=[("id", 1)])
        self.assertEqual(list(src.records()), [0, 1, 2])

    def test_multi_column_yields_tuples(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL, columns=["id", "val"], sort=[("id", 1)])
        self.assertEqual(list(src.records()), [(0, 0.5), (1, 1.5), (2, 2.5)])

    def test_columns_projects_and_reorders(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL, columns=["val", "id"], sort=[("id", 1)])
        self.assertEqual(list(src.records()), [(0.5, 0), (1.5, 1), (2.5, 2)])

    def test_query_filters(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL, query={"id": {"$gte": 1}}, columns=["id"], sort=[("id", 1)])
        self.assertEqual(list(src.records()), [1, 2])

    def test_schema_applied_by_materialize_tuple_shaped(self) -> None:
        # Mirrors sql_source_test.py's own test exactly (columns= projects to tuples, same as SQL rows).
        schema = Schema(fields=(Field(name="id", type=Real()), Field(name="val", type=Real())))
        src = read_mongo(_MOCK_URI, _DB, _COLL, columns=["id", "val"], schema=schema, sort=[("id", 1)])
        out = src.materialize()
        self.assertEqual(out, [(0.0, 0.5), (1.0, 1.5), (2.0, 2.5)])
        self.assertIsInstance(out[0][0], float)  # Real coerces the integer id column to float

    def test_schema_applied_by_materialize_dict_shaped(self) -> None:
        # Mongo's OTHER natural shape (columns=None -> dict records) -- Schema.conform_record handles
        # dict records natively (mixle/data/schema.py), unlike SQL, which never yields dicts.
        schema = Schema(fields=(Field(name="id", type=Real()), Field(name="val", type=Real())))
        src = read_mongo(_MOCK_URI, _DB, _COLL, schema=schema, sort=[("id", 1)])
        out = src.materialize()
        self.assertEqual(out, [{"id": 0.0, "val": 0.5}, {"id": 1.0, "val": 1.5}, {"id": 2.0, "val": 2.5}])
        self.assertIsInstance(out[0]["id"], float)

    def test_missing_driver_raises_import_error(self) -> None:
        # Mirrors sql_source_test.py's own guard: only meaningful in a driverless environment, a no-op
        # (but still importable/runnable) otherwise.
        if getattr(mongo_source, "_pymongo", None) is None:  # pragma: no cover - not exercised in this venv
            with self.assertRaises(ImportError):
                mongo_source.read_mongo("mongodb://", "db", "coll").materialize()


@unittest.skipUnless(
    _HAS_PYMONGO and _HAS_MONGOMOCK, "pymongo/mongomock not installed; pip install mixle[mongo] mongomock"
)
class MongoCursorSourceStreamingTest(unittest.TestCase):
    """Regression coverage (audit MXR-080-1666): ``records()``/``encode()`` must stream off the cursor
    in bounded batches, never ``list(...)`` the whole result set before the first record is usable.
    """

    def test_records_does_not_pull_whole_result_before_first_record(self) -> None:
        n = 2000
        store = _seed([{"id": i, "val": float(i)} for i in range(n)])
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        pulled, remove = _install_pull_counter()
        self.addCleanup(remove)

        src = read_mongo(_MOCK_URI, _DB, _COLL, columns=["id", "val"], batch_size=64, sort=[("id", 1)])
        it = iter(src.records())
        first = next(it)
        # The historical bug: LazySource.materialize()'s `list(self._factory())` (and factory() itself)
        # pulled all n documents before records() could yield even one.
        self.assertLess(len(pulled), n)
        self.assertEqual(first, (0, 0.0))
        it.close()

    def test_encode_flushes_before_entire_result_is_pulled(self) -> None:
        n = 5000
        store = _seed([{"id": i, "val": float(i)} for i in range(n)])
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        pulled, remove = _install_pull_counter()
        self.addCleanup(remove)

        src = read_mongo(_MOCK_URI, _DB, _COLL, columns=["id", "val"], batch_size=100, sort=[("id", 1)])
        encoder = _RecordingEncoder(pulled)
        results = src.encode(encoder, num_chunks=1)

        self.assertTrue(encoder.pulled_before_each_call)
        self.assertLess(encoder.pulled_before_each_call[0], n)
        self.assertLessEqual(encoder.pulled_before_each_call[0], 100 + 10)
        self.assertEqual(sum(count for count, _ in results), n)
        self.assertEqual(len(encoder.batches), n // 100)  # 50 flushes of 100 documents each

    def test_encode_matches_list_striding_reference(self) -> None:
        n = 37
        store = _seed([{"id": i, "val": float(i)} for i in range(n)])
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        ref_records = [(i, float(i)) for i in range(n)]

        for nc in (1, 3, 8):
            with self.subTest(num_chunks=nc):
                src = read_mongo(_MOCK_URI, _DB, _COLL, columns=["id", "val"], batch_size=5, sort=[("id", 1)])
                encoder = _RecordingEncoder([])
                result = src.encode(encoder, num_chunks=nc)
                self.assertEqual(sum(count for count, _ in result), n)
                seen = sorted(row for batch in encoder.batches for row in batch)
                ref_parts = partition_records(ref_records, EXCHANGEABLE, nc)
                expected = sorted(row for part in ref_parts for row in part)
                self.assertEqual(seen, expected)

    def test_num_chunks_one_preserves_sorted_order(self) -> None:
        n = 41
        store = _seed([{"id": i, "val": float(i)} for i in range(n)])
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        src = read_mongo(_MOCK_URI, _DB, _COLL, columns=["id", "val"], batch_size=7, sort=[("id", 1)])
        encoder = _RecordingEncoder([])
        src.encode(encoder, num_chunks=1)
        flat = [row for batch in encoder.batches for row in batch]
        self.assertEqual(flat, [(i, float(i)) for i in range(n)])

    def test_empty_result_set_does_not_crash(self) -> None:
        store = _seed([])
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        src = read_mongo(_MOCK_URI, _DB, _COLL)
        self.assertEqual(list(src.records()), [])
        result = src.encode(_RecordingEncoder([]), num_chunks=3)
        self.assertEqual(result, [])

    def test_more_partitions_than_records_does_not_crash(self) -> None:
        store = _seed([{"id": 0, "val": 0.0}, {"id": 1, "val": 1.0}])
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        src = read_mongo(_MOCK_URI, _DB, _COLL, columns=["id", "val"])
        result = src.encode(_RecordingEncoder([]), num_chunks=5)
        self.assertEqual(sum(count for count, _ in result), 2)

    def test_missing_column_in_some_documents_raises_clear_keyerror(self) -> None:
        # Mongo is schemaless: a column present in some documents can be absent in others. Unlike SQL
        # (whose fixed result columns let a bad `columns` name be caught once, up front -- see
        # SqlCursorSource._stream), this can only be discovered per-document, so it must at least fail
        # with a clear message instead of a bare, contextless KeyError('b').
        store = _seed([{"id": 0, "val": 0.0}, {"id": 1}])  # second document has no "val"
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        src = read_mongo(_MOCK_URI, _DB, _COLL, columns=["id", "val"], sort=[("id", 1)])
        with self.assertRaises(KeyError) as ctx:
            list(src.records())
        self.assertIn("val", str(ctx.exception))


@unittest.skipUnless(
    _HAS_PYMONGO and _HAS_MONGOMOCK, "pymongo/mongomock not installed; pip install mixle[mongo] mongomock"
)
class MongoCursorSourceDisposalTest(unittest.TestCase):
    """Regression coverage (audit MXR-080-1666): the client opened for a query must be closed
    deterministically -- on normal completion, on early abandonment, and on an exception raised while a
    document is being consumed downstream -- never left to leak a connection pool.
    """

    def setUp(self) -> None:
        self.store = _seed([{"id": i, "val": float(i)} for i in range(50)])
        client_patcher = _patch_mongo_client(self.store)
        client_patcher.start()
        self.addCleanup(client_patcher.stop)
        # autospec=True makes the mock a proper bound-method stand-in (correctly receives `self` as
        # each real mongomock MongoClient instance); it does not call through to the real close(), which
        # is fine here -- we only need to know it was *called*.
        close_patcher = mock.patch.object(mongomock.MongoClient, "close", autospec=True)
        self.mock_close = close_patcher.start()
        self.addCleanup(close_patcher.stop)

    def test_disposed_after_full_iteration(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL)
        list(src.records())
        self.assertEqual(self.mock_close.call_count, 1)

    def test_disposed_after_explicit_early_close(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL)
        it = iter(src.records())
        next(it)
        next(it)
        self.assertEqual(self.mock_close.call_count, 0)  # not yet -- still mid-stream
        it.close()
        self.assertEqual(self.mock_close.call_count, 1)

    def test_disposed_after_exception_during_consumption(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL)
        with self.assertRaises(RuntimeError):
            for i, _rec in enumerate(src.records()):
                if i == 3:
                    raise RuntimeError("boom")
        self.assertEqual(self.mock_close.call_count, 1)

    def test_disposed_after_materialize(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL)
        src.materialize()
        self.assertEqual(self.mock_close.call_count, 1)

    def test_disposed_after_encode(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL)
        src.encode(_RecordingEncoder([]), num_chunks=2, batch_size=5)
        self.assertEqual(self.mock_close.call_count, 1)

    def test_disposed_after_chunk_size_count_and_stream(self) -> None:
        # chunk_size resolution opens its OWN client for count_documents() (mirrors SqlCursorSource's
        # separate COUNT(*) engine) in addition to the one _stream() opens -- two independent opens,
        # two independent closes.
        src = read_mongo(_MOCK_URI, _DB, _COLL)
        src.encode(_RecordingEncoder([]), chunk_size=10)
        self.assertEqual(self.mock_close.call_count, 2)

    def test_two_independent_records_calls_each_dispose_their_own_client(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL)
        list(src.records())
        list(src.records())
        self.assertEqual(self.mock_close.call_count, 2)


@unittest.skipUnless(_HAS_PYMONGO, "pymongo not installed; pip install mixle[mongo]")
class MongoCursorSourceRealDriverDisposalTest(unittest.TestCase):
    """Disposal-on-failure needs a genuine connection attempt to fail, which mongomock (a pure in-memory
    fake with no network layer) cannot simulate -- unlike every other test in this module, these two talk
    to the REAL ``pymongo.MongoClient`` (never a real MongoDB server -- only a refused TCP connection /
    a malformed URI), with a short ``serverSelectionTimeoutMS`` so the failure surfaces in well under a
    second instead of pymongo's 30-second default.
    """

    def test_disposed_after_operation_failure_against_unreachable_host(self) -> None:
        # side_effect=the real close() -- not just autospec's default no-op recorder -- so pymongo's own
        # background topology-monitor thread actually tears down. Unlike SQL's sqlite engines (a
        # trivially-GC'd local file handle, see sql_source_test.py's own comment on this same tradeoff),
        # a real MongoClient that is never truly closed leaves a live thread running until GC finalizes
        # it, which prints a noisy (if harmless) ResourceWarning -- avoided by still calling through.
        real_close = pymongo.MongoClient.close
        patcher = mock.patch.object(pymongo.MongoClient, "close", autospec=True, side_effect=real_close)
        mock_close = patcher.start()
        self.addCleanup(patcher.stop)
        # Port 1 is a reserved/privileged port nothing listens on; serverSelectionTimeoutMS bounds how
        # long pymongo retries before giving up (default 30000ms -- far too slow for a test).
        bad_uri = "mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=300"
        src = read_mongo(bad_uri, "db", "coll")
        with self.assertRaises(pymongo.errors.PyMongoError):
            list(src.records())
        # MongoClient(uri) itself succeeds (lazy -- pymongo never dials out at construction); the
        # failure happens later, inside the try/finally, so the client that WAS constructed is still
        # closed -- mirrors sql_source's test_disposed_on_connect_failure and its own comment that
        # create_engine() is deliberately left outside the try/finally.
        self.assertEqual(mock_close.call_count, 1)

    def test_malformed_uri_raises_before_any_client_is_constructed(self) -> None:
        patcher = mock.patch.object(pymongo.MongoClient, "close", autospec=True)
        mock_close = patcher.start()
        self.addCleanup(patcher.stop)
        src = read_mongo("not-a-valid-uri-scheme://???", "db", "coll")
        with self.assertRaises(pymongo.errors.InvalidURI):
            list(src.records())
        self.assertEqual(mock_close.call_count, 0)  # construction itself failed; nothing to close


@unittest.skipUnless(
    _HAS_PYMONGO and _HAS_MONGOMOCK, "pymongo/mongomock not installed; pip install mixle[mongo] mongomock"
)
class MongoCursorSourceReplayTest(unittest.TestCase):
    """Regression coverage (audit MXR-080-1666): the "explicit replay policy" the finding asks for.

    A live cursor cannot be rewound, so ``MongoCursorSource`` chooses (and documents) replay by
    re-execution: every call to ``records()``/``encode()``/``materialize()`` re-runs the query from
    scratch on a fresh client, so changes to the underlying collection between calls are visible on the
    next call -- exactly as re-running the query by hand would be.
    """

    def setUp(self) -> None:
        self.store = _seed([{"id": i, "val": float(i)} for i in range(5)])
        patcher = _patch_mongo_client(self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_records_reexecutes_and_sees_intervening_writes(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL, columns=["id", "val"])
        first_pass = list(src.records())
        self.assertEqual(len(first_pass), 5)

        # A genuinely separate client, sharing only the backing store -- the same relationship two real
        # MongoDB client connections to the same server would have.
        writer = mongomock.MongoClient(_MOCK_URI, _store=self.store)
        writer[_DB][_COLL].insert_one({"id": 99, "val": 99.0})
        writer.close()

        second_pass = list(src.records())
        self.assertEqual(len(second_pass), 6)
        self.assertIn((99, 99.0), second_pass)

    def test_two_records_iterators_are_independent(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL, columns=["id", "val"], sort=[("id", 1)])
        it1 = iter(src.records())
        it2 = iter(src.records())
        self.assertEqual(next(it1), (0, 0.0))
        self.assertEqual(next(it2), (0, 0.0))  # it2's own fresh execution, unaffected by it1's position
        self.assertEqual(next(it1), (1, 1.0))
        rest1 = list(it1)
        rest2 = list(it2)
        self.assertEqual(rest1, [(2, 2.0), (3, 3.0), (4, 4.0)])
        self.assertEqual(rest2, [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)])


@unittest.skipUnless(
    _HAS_PYMONGO and _HAS_MONGOMOCK, "pymongo/mongomock not installed; pip install mixle[mongo] mongomock"
)
class MongoCursorSourceGroupedStreamingTest(unittest.TestCase):
    """Regression coverage (audit MXR-080-1666): partially-exchangeable (grouped) streaming assigns
    whole groups round-robin to partitions, matching ``partition_records``'s reference grouping, but
    (being a single forward pass over a live cursor) requires the query to already yield documents
    contiguous by group key -- enforced with a clear ``ValueError`` rather than silently splitting a
    group. Also covers this connector's specific advantage over SQL: a string ``by`` is auto-sorted
    server-side (no caller action needed), since this connector builds its query from structured pieces
    rather than parsing/rewriting the caller's raw text.
    """

    def _seed_grouped(self, group_sizes: list[int]) -> tuple[Any, list[dict], list[tuple[str, float]]]:
        """Returns ``(store, docs, tuples)`` -- the same data in both shapes a caller might see it in:
        ``docs`` (dicts, what ``columns=None`` yields -- what a string ``by`` needs, since
        ``group_key`` looks a dict up by key) and ``tuples`` (what ``columns=["grp", "x"]`` yields --
        what a callable ``by`` operates on positionally, mirroring SQL's rows)."""
        docs: list[dict] = []
        tuples: list[tuple[str, float]] = []
        for gi, size in enumerate(group_sizes):
            for _ in range(size):
                docs.append({"grp": "g%d" % gi, "x": float(gi)})
                tuples.append(("g%d" % gi, float(gi)))
        store = _seed(docs)
        return store, docs, tuples

    def test_contiguous_groups_match_reference_partitioning_via_auto_sort(self) -> None:
        # String `by` + columns=None (dict records): MongoCursorSource auto-sorts by "grp" itself --
        # the caller passes no `sort` at all and contiguity still holds, which SqlCursorSource cannot
        # offer (it never parses the caller's raw SQL text to inject an ORDER BY).
        store, docs, _tuples = self._seed_grouped([3, 2, 4, 1])
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        structure = partially_exchangeable("grp")
        src = read_mongo(_MOCK_URI, _DB, _COLL, structure=structure, batch_size=2)
        encoder = _RecordingEncoder([])

        result = src.encode(encoder, num_chunks=2)

        self.assertEqual(sum(count for count, _ in result), len(docs))
        seen = sorted((d["grp"], d["x"]) for batch in encoder.batches for d in batch)
        # partition_records's own group_key() needs dict records to look "grp" up by name (same
        # constraint MongoCursorSource itself enforces at construction -- see
        # test_string_by_with_columns_rejected_at_construction below), so the reference is built from
        # `docs`, not `tuples`, even though both hold the same underlying data.
        ref_parts = partition_records(docs, structure, 2)
        expected = sorted((d["grp"], d["x"]) for part in ref_parts for d in part)
        self.assertEqual(seen, expected)

    def test_contiguous_groups_match_reference_partitioning_via_callable_and_explicit_sort(self) -> None:
        # The SQL-equivalent path: callable `by` (required once columns= makes records tuples, not
        # dicts) plus an explicit sort the caller supplies themselves -- exactly ORDER BY's role for SQL.
        store, _docs, tuples = self._seed_grouped([3, 2, 4, 1])
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        structure = partially_exchangeable(grouping_policy("test.mongo.group", "1", lambda row: row[0]))
        src = read_mongo(
            _MOCK_URI, _DB, _COLL, columns=["grp", "x"], structure=structure, batch_size=2, sort=[("grp", 1)]
        )
        encoder = _RecordingEncoder([])

        result = src.encode(encoder, num_chunks=2)

        self.assertEqual(sum(count for count, _ in result), len(tuples))
        seen = sorted(row for batch in encoder.batches for row in batch)
        ref_parts = partition_records(tuples, structure, 2)
        expected = sorted(row for part in ref_parts for row in part)
        self.assertEqual(seen, expected)

    def test_non_contiguous_groups_raise_value_error(self) -> None:
        store = _seed([{"grp": "a", "x": 1.0}, {"grp": "b", "x": 2.0}, {"grp": "a", "x": 3.0}])
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        # No sort at all, natural insertion order: a, b, a -- "a" reappears after "b" has started.
        structure = partially_exchangeable(grouping_policy("test.mongo.group", "1", lambda row: row[0]))
        src = read_mongo(_MOCK_URI, _DB, _COLL, columns=["grp", "x"], structure=structure)
        with self.assertRaises(ValueError):
            src.encode(_RecordingEncoder([]), num_chunks=2)

    def test_single_group_num_chunks_one(self) -> None:
        store, docs, _tuples = self._seed_grouped([5])
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        structure = partially_exchangeable("grp")
        src = read_mongo(_MOCK_URI, _DB, _COLL, structure=structure, batch_size=2)
        result = src.encode(_RecordingEncoder([]), num_chunks=1)
        self.assertEqual(sum(count for count, _ in result), 5)

    def test_string_by_with_columns_rejected_at_construction(self) -> None:
        # Projected records are tuples/scalars, not dicts -- SampleStructure.group_key can't look a
        # field up by name on those (mirrors SqlCursorSource, whose rows are always tuples). Caught here
        # (construction) rather than as a confusing AttributeError deep inside encode().
        structure = partially_exchangeable("grp")
        with self.assertRaises(ValueError):
            read_mongo(_MOCK_URI, _DB, _COLL, columns=["grp", "x"], structure=structure)


@unittest.skipUnless(
    _HAS_PYMONGO and _HAS_MONGOMOCK, "pymongo/mongomock not installed; pip install mixle[mongo] mongomock"
)
class MongoCursorSourceChunkSizeTest(unittest.TestCase):
    """``chunk_size`` needs a document count to resolve into a partition count; a streaming cursor does
    not know that up front, so it is resolved via ``count_documents()`` (one bounded integer, never the
    document data) rather than by falling back to full materialization. Simpler than SQL's ``COUNT(*)``
    subquery: no query-text wrapping is involved.
    """

    def test_chunk_size_resolves_via_count_documents(self) -> None:
        n = 23
        store = _seed([{"id": i, "val": float(i)} for i in range(n)])
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        src = read_mongo(_MOCK_URI, _DB, _COLL)
        result = src.encode(_RecordingEncoder([]), chunk_size=5)  # ceil(23/5) = 5 partitions
        self.assertEqual(sum(count for count, _ in result), n)

    def test_chunk_size_respects_query_filter(self) -> None:
        store = _seed([{"id": i, "val": float(i)} for i in range(20)])
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        src = read_mongo(_MOCK_URI, _DB, _COLL, query={"id": {"$lt": 10}})
        result = src.encode(_RecordingEncoder([]), chunk_size=3)  # ceil(10/3) = 4 partitions of the 10 matches
        self.assertEqual(sum(count for count, _ in result), 10)


@unittest.skipUnless(
    _HAS_PYMONGO and _HAS_MONGOMOCK, "pymongo/mongomock not installed; pip install mixle[mongo] mongomock"
)
class MongoCursorSourceValidationTest(unittest.TestCase):
    """``batch_size`` and ``columns`` must be validated at construction (and, for ``batch_size``, again
    as an ``encode()`` override) instead of failing confusingly later -- MXR-080-1666's "projection
    validation" ask, plus the same ``batch_size`` validation ``SqlCursorSource`` already has.
    """

    def setUp(self) -> None:
        self.store = _seed([{"id": 0, "val": 0.0}])
        patcher = _patch_mongo_client(self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_zero_batch_size_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            read_mongo(_MOCK_URI, _DB, _COLL, batch_size=0)

    def test_negative_batch_size_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            read_mongo(_MOCK_URI, _DB, _COLL, batch_size=-5)

    def test_bool_batch_size_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            read_mongo(_MOCK_URI, _DB, _COLL, batch_size=True)

    def test_invalid_batch_size_override_rejected_in_encode(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL)
        with self.assertRaises(ValueError):
            src.encode(_RecordingEncoder([]), batch_size=0)

    def test_valid_batch_size_still_constructs_and_works(self) -> None:
        src = read_mongo(_MOCK_URI, _DB, _COLL, columns=["id"], batch_size=10)
        self.assertEqual(list(src.records()), [0])

    def test_id_in_columns_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            read_mongo(_MOCK_URI, _DB, _COLL, columns=["_id", "id"])

    def test_empty_columns_projects_to_zero_columns(self) -> None:
        # The columns=[] truthiness bug (same class as MXR-080-0065 / the hadoop_source.read_remote CSV
        # fix in bf74faf9): `if columns` treats an explicitly-empty list the same as None ("no filter"),
        # when it should mean "project to zero columns" -- every record becomes ().
        store = _seed([{"id": 0, "val": 0.0}, {"id": 1, "val": 1.0}])
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        src = read_mongo(_MOCK_URI, _DB, _COLL, columns=[])
        self.assertEqual(list(src.records()), [(), ()])

    def test_empty_columns_differs_from_columns_none(self) -> None:
        # Negative control pinning the exact symptom of the truthiness bug: columns=[] (zero columns)
        # must NOT collapse to the same result as columns=None (no filter, every field).
        store = _seed([{"id": 0, "val": 0.0}])
        patcher = _patch_mongo_client(store)
        patcher.start()
        self.addCleanup(patcher.stop)
        empty = list(read_mongo(_MOCK_URI, _DB, _COLL, columns=[]).records())
        none = list(read_mongo(_MOCK_URI, _DB, _COLL, columns=None).records())
        self.assertNotEqual(empty, none)
        self.assertEqual(empty, [()])
        self.assertEqual(none, [{"id": 0, "val": 0.0}])


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(unittest.main())
