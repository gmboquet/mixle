"""Tests for the SQL connector (mixle.data.sources.sql_source).

Covers the connector's basic read/columns/schema contract, plus the later external-review finding
against this module (docs/audits/0.8.0-exhaustive-code-review.md MXR-080-0064): ``read_sql`` claimed
server-side-cursor streaming and bounded driver memory but appended every row to a list before
``LazySource.materialize()`` (itself a second full materialization) cached it as one -- a large query had
to fit completely in memory before any record reached an encoder, and the engine was never disposed.

``SqlCursorSource`` replaces the ``LazySource``-backed implementation: ``records()``/``encode()`` stream
off a real DBAPI cursor in bounded batches, every connection/engine opened is disposed deterministically
on every exit path, and calling ``records()``/``encode()`` again explicitly re-executes the query (the
replay policy) rather than silently caching or raising. All tests skip cleanly when sqlalchemy is not
installed.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from typing import Any
from unittest import mock

from mixle.data.core import DataSource
from mixle.data.partition import partition_records
from mixle.data.schema import Field, Real, Schema
from mixle.data.sources import sql_source
from mixle.data.sources.sql_source import SqlCursorSource, read_sql
from mixle.data.structure import EXCHANGEABLE, partially_exchangeable

_HAS_SQLALCHEMY = getattr(sql_source, "_sa", None) is not None
if _HAS_SQLALCHEMY:
    import sqlalchemy as sa


def _make_db(rows: list[tuple[Any, ...]], *, columns: tuple[str, ...] = ("id", "val")) -> str:
    """Create a throwaway sqlite db file with a single table ``t`` populated with ``rows``."""
    path = tempfile.mktemp(suffix=".db")
    coldefs = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    raw = sqlite3.connect(path)
    raw.execute(f"CREATE TABLE t ({coldefs})")
    raw.executemany(f"INSERT INTO t ({coldefs}) VALUES ({placeholders})", rows)
    raw.commit()
    raw.close()
    return path


def _install_pull_counter() -> tuple[list[Any], Any]:
    """Install a SQLite UDF (``mark(x)``) on every new connection; returns ``(pulled, remove_fn)``.

    SQLite steps its query engine lazily (``sqlite3_step`` per row), so a UDF embedded in the SELECT
    list fires exactly once per row, exactly when that row is produced -- i.e. when the DBAPI cursor is
    stepped. Counting UDF calls over time is a faithful proxy for "how many rows has the Python side
    pulled off the cursor," which is what a "bounded driver memory" streaming claim should control --
    the same instrumentation strategy the audit finding itself used to reproduce the bug, mirroring
    ``_CountingArray`` in array_data_sources_test.py for the analogous MXR-080-0056 fix. Registered at
    the ``Engine`` *class* level because each call our source makes creates its own ``Engine`` instance
    (the replay/disposal contract this module tests), so a per-instance listener would miss later calls.
    """
    pulled: list[Any] = []

    def _mark(x: Any) -> Any:
        pulled.append(x)
        return x

    def _on_connect(dbapi_conn: Any, _: Any) -> None:
        dbapi_conn.create_function("mark", 1, _mark)

    sa.event.listen(sa.engine.Engine, "connect", _on_connect)

    def _remove() -> None:
        sa.event.remove(sa.engine.Engine, "connect", _on_connect)

    return pulled, _remove


class _RecordingEncoder:
    """Records each batch handed to ``seq_encode``, plus how many rows the driver had produced so far
    (via a ``_install_pull_counter()`` list) -- proves ``encode()`` streams/flushes incrementally rather
    than an eager ``list(self.records())`` first. Mirrors ``_RecordingEncoder`` in
    array_data_sources_test.py, adapted to SQL's row-counting mechanism.
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


@unittest.skipUnless(_HAS_SQLALCHEMY, "sqlalchemy not installed; pip install mixle[sql]")
class SqlSourceBasicsTest(unittest.TestCase):
    """Smoke-tests the connector's ordinary read/columns/schema contract (unchanged by MXR-080-0064)."""

    def setUp(self) -> None:
        self.path = _make_db([(0, 0.5), (1, 1.5), (2, 2.5)])
        self.addCleanup(os.unlink, self.path)

    def test_read_sql_returns_a_sql_cursor_source(self) -> None:
        src = read_sql(f"sqlite:///{self.path}", "SELECT id, val FROM t")
        self.assertIsInstance(src, SqlCursorSource)
        self.assertIsInstance(src, DataSource)  # structural: schema/structure/records()/encode()

    def test_single_column_yields_scalars(self) -> None:
        src = read_sql(f"sqlite:///{self.path}", "SELECT id FROM t ORDER BY id")
        self.assertEqual(list(src.records()), [0, 1, 2])

    def test_multi_column_yields_tuples(self) -> None:
        src = read_sql(f"sqlite:///{self.path}", "SELECT id, val FROM t ORDER BY id")
        self.assertEqual(list(src.records()), [(0, 0.5), (1, 1.5), (2, 2.5)])

    def test_columns_projects_and_reorders(self) -> None:
        src = read_sql(f"sqlite:///{self.path}", "SELECT id, val FROM t ORDER BY id", columns=["val", "id"])
        self.assertEqual(list(src.records()), [(0.5, 0), (1.5, 1), (2.5, 2)])

    def test_schema_applied_by_materialize(self) -> None:
        schema = Schema(fields=(Field(name="id", type=Real()), Field(name="val", type=Real())))
        src = read_sql(f"sqlite:///{self.path}", "SELECT id, val FROM t ORDER BY id", schema=schema)
        out = src.materialize()
        self.assertEqual(out, [(0.0, 0.5), (1.0, 1.5), (2.0, 2.5)])
        self.assertIsInstance(out[0][0], float)  # Real coerces the integer id column to float

    def test_missing_driver_raises_import_error(self) -> None:
        # Mirrors data_layer_test.py's own guard: only meaningful in a driverless environment, a no-op
        # (but still importable/runnable) otherwise.
        if getattr(sql_source, "_sa", None) is None:  # pragma: no cover - not exercised in this venv
            with self.assertRaises(ImportError):
                sql_source.read_sql("sqlite://", "select 1").materialize()


@unittest.skipUnless(_HAS_SQLALCHEMY, "sqlalchemy not installed; pip install mixle[sql]")
class SqlCursorSourceStreamingTest(unittest.TestCase):
    """Regression coverage (audit MXR-080-0064): ``records()``/``encode()`` must stream off the cursor
    in bounded batches, never ``list(...)`` the whole result set before the first record is usable.
    Reproduced first against the pre-fix code (a UDF-instrumented DB showed every row pulled before the
    first record reached the consumer); these tests pin the fixed, incremental behavior instead.
    """

    def test_records_does_not_pull_whole_result_before_first_record(self) -> None:
        n = 2000
        pulled, remove = _install_pull_counter()
        self.addCleanup(remove)
        path = _make_db([(i, float(i)) for i in range(n)])
        self.addCleanup(os.unlink, path)

        src = read_sql(f"sqlite:///{path}", "SELECT mark(id), val FROM t", batch_size=64)
        it = iter(src.records())
        first = next(it)
        # The historical bug: LazySource.materialize()'s `list(self._factory())` (and factory() itself)
        # pulled all n rows before records() could yield even one.
        self.assertLess(len(pulled), n)
        self.assertEqual(first, (0, 0.0))
        it.close()

    def test_encode_flushes_before_entire_result_is_pulled(self) -> None:
        n = 5000
        pulled, remove = _install_pull_counter()
        self.addCleanup(remove)
        path = _make_db([(i, float(i)) for i in range(n)])
        self.addCleanup(os.unlink, path)

        src = read_sql(f"sqlite:///{path}", "SELECT mark(id), val FROM t", batch_size=100)
        encoder = _RecordingEncoder(pulled)
        results = src.encode(encoder, num_chunks=1)

        self.assertTrue(encoder.pulled_before_each_call)
        self.assertLess(encoder.pulled_before_each_call[0], n)
        # Roughly one batch's worth, not pinned exactly: SQLAlchemy's yield_per prefetches slightly
        # ahead internally (observed 101 for batch_size=100), an implementation detail of its fetch
        # strategy that doesn't change the O(batch_size), not O(n), memory bound being asserted here.
        self.assertLessEqual(encoder.pulled_before_each_call[0], 100 + 10)
        self.assertEqual(sum(count for count, _ in results), n)
        self.assertEqual(len(encoder.batches), n // 100)  # 50 flushes of 100 rows each

    def test_encode_matches_list_striding_reference(self) -> None:
        n = 37
        path = _make_db([(i, float(i)) for i in range(n)])
        self.addCleanup(os.unlink, path)
        ref_records = [(i, float(i)) for i in range(n)]

        for nc in (1, 3, 8):
            with self.subTest(num_chunks=nc):
                src = read_sql(f"sqlite:///{path}", "SELECT id, val FROM t ORDER BY id", batch_size=5)
                encoder = _RecordingEncoder([])
                result = src.encode(encoder, num_chunks=nc)
                self.assertEqual(sum(count for count, _ in result), n)
                # Batches may interleave/split differently from partition_records's exact slices (see
                # encode()'s docstring), but the multiset of records recovered must match exactly.
                seen = sorted(row for batch in encoder.batches for row in batch)
                ref_parts = partition_records(ref_records, EXCHANGEABLE, nc)
                expected = sorted(row for part in ref_parts for row in part)
                self.assertEqual(seen, expected)

    def test_num_chunks_one_preserves_original_order(self) -> None:
        n = 41
        path = _make_db([(i, float(i)) for i in range(n)])
        self.addCleanup(os.unlink, path)
        src = read_sql(f"sqlite:///{path}", "SELECT id, val FROM t ORDER BY id", batch_size=7)
        encoder = _RecordingEncoder([])
        src.encode(encoder, num_chunks=1)
        flat = [row for batch in encoder.batches for row in batch]
        self.assertEqual(flat, [(i, float(i)) for i in range(n)])

    def test_empty_result_set_does_not_crash(self) -> None:
        path = _make_db([])
        self.addCleanup(os.unlink, path)
        src = read_sql(f"sqlite:///{path}", "SELECT id, val FROM t")
        self.assertEqual(list(src.records()), [])
        result = src.encode(_RecordingEncoder([]), num_chunks=3)
        self.assertEqual(result, [])

    def test_more_partitions_than_records_does_not_crash(self) -> None:
        path = _make_db([(0, 0.0), (1, 1.0)])
        self.addCleanup(os.unlink, path)
        src = read_sql(f"sqlite:///{path}", "SELECT id, val FROM t")
        result = src.encode(_RecordingEncoder([]), num_chunks=5)
        self.assertEqual(sum(count for count, _ in result), 2)


@unittest.skipUnless(_HAS_SQLALCHEMY, "sqlalchemy not installed; pip install mixle[sql]")
class SqlCursorSourceDisposalTest(unittest.TestCase):
    """Regression coverage (audit MXR-080-0064): the engine/connection opened for a query must be
    disposed deterministically -- on normal completion, on early abandonment, and on an exception raised
    while a row is being consumed downstream -- never left to a connection pool or eventual GC.
    """

    def setUp(self) -> None:
        self.path = _make_db([(i, float(i)) for i in range(50)])
        self.addCleanup(os.unlink, self.path)
        # autospec=True makes the mock a proper bound-method stand-in (correctly receives `self` as
        # each real Engine instance); it does not call through to the real dispose(), which is fine
        # here -- we only need to know it was *called*, not that pooled connections actually closed
        # (the underlying sqlite file is removed via addCleanup regardless of pool state).
        patcher = mock.patch.object(sa.engine.Engine, "dispose", autospec=True)
        self.mock_dispose = patcher.start()
        self.addCleanup(patcher.stop)

    def test_disposed_after_full_iteration(self) -> None:
        src = read_sql(f"sqlite:///{self.path}", "SELECT id, val FROM t")
        list(src.records())
        self.assertEqual(self.mock_dispose.call_count, 1)

    def test_disposed_after_explicit_early_close(self) -> None:
        src = read_sql(f"sqlite:///{self.path}", "SELECT id, val FROM t")
        it = iter(src.records())
        next(it)
        next(it)
        self.assertEqual(self.mock_dispose.call_count, 0)  # not yet -- still mid-stream
        it.close()
        self.assertEqual(self.mock_dispose.call_count, 1)

    def test_disposed_after_exception_during_consumption(self) -> None:
        src = read_sql(f"sqlite:///{self.path}", "SELECT id, val FROM t")
        with self.assertRaises(RuntimeError):
            for i, _rec in enumerate(src.records()):
                if i == 3:
                    raise RuntimeError("boom")
        self.assertEqual(self.mock_dispose.call_count, 1)

    def test_disposed_after_materialize(self) -> None:
        src = read_sql(f"sqlite:///{self.path}", "SELECT id, val FROM t")
        src.materialize()
        self.assertEqual(self.mock_dispose.call_count, 1)

    def test_disposed_after_encode(self) -> None:
        src = read_sql(f"sqlite:///{self.path}", "SELECT id, val FROM t")
        src.encode(_RecordingEncoder([]), num_chunks=2, batch_size=5)
        self.assertEqual(self.mock_dispose.call_count, 1)

    def test_disposed_on_connect_failure(self) -> None:
        bad = read_sql("sqlite:////mixle_nonexistent_dir_xyz/does/not/exist.db", "SELECT 1")
        with self.assertRaises(sa.exc.SQLAlchemyError):
            list(bad.records())
        self.assertEqual(self.mock_dispose.call_count, 1)

    def test_two_independent_records_calls_each_dispose_their_own_engine(self) -> None:
        src = read_sql(f"sqlite:///{self.path}", "SELECT id, val FROM t")
        list(src.records())
        list(src.records())
        self.assertEqual(self.mock_dispose.call_count, 2)


@unittest.skipUnless(_HAS_SQLALCHEMY, "sqlalchemy not installed; pip install mixle[sql]")
class SqlCursorSourceReplayTest(unittest.TestCase):
    """Regression coverage (audit MXR-080-0064): the "explicit replay policy" the finding asks for.

    A live server-side cursor cannot be rewound, so ``SqlCursorSource`` chooses (and documents) replay
    by re-execution: every call to ``records()``/``encode()``/``materialize()`` re-runs the query from
    scratch on a fresh connection, so changes to the underlying table between calls are visible on the
    next call -- exactly as re-running the query by hand would be.
    """

    def setUp(self) -> None:
        self.path = _make_db([(i, float(i)) for i in range(5)])
        self.addCleanup(os.unlink, self.path)

    def test_records_reexecutes_and_sees_intervening_writes(self) -> None:
        src = read_sql(f"sqlite:///{self.path}", "SELECT id, val FROM t")
        first_pass = list(src.records())
        self.assertEqual(len(first_pass), 5)

        raw = sqlite3.connect(self.path)
        raw.execute("INSERT INTO t (id, val) VALUES (99, 99.0)")
        raw.commit()
        raw.close()

        second_pass = list(src.records())
        self.assertEqual(len(second_pass), 6)
        self.assertIn((99, 99.0), second_pass)

    def test_two_records_iterators_are_independent(self) -> None:
        src = read_sql(f"sqlite:///{self.path}", "SELECT id, val FROM t ORDER BY id")
        it1 = iter(src.records())
        it2 = iter(src.records())
        self.assertEqual(next(it1), (0, 0.0))
        self.assertEqual(next(it2), (0, 0.0))  # it2's own fresh execution, unaffected by it1's position
        self.assertEqual(next(it1), (1, 1.0))
        rest1 = list(it1)
        rest2 = list(it2)
        self.assertEqual(rest1, [(2, 2.0), (3, 3.0), (4, 4.0)])
        self.assertEqual(rest2, [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)])


@unittest.skipUnless(_HAS_SQLALCHEMY, "sqlalchemy not installed; pip install mixle[sql]")
class SqlCursorSourceGroupedStreamingTest(unittest.TestCase):
    """Regression coverage (audit MXR-080-0064): partially-exchangeable (grouped) streaming assigns
    whole groups round-robin to partitions, matching ``partition_records``'s reference grouping, but
    (being a single forward pass over a live cursor) requires the query to already yield rows contiguous
    by group key -- enforced with a clear ``ValueError`` rather than silently splitting a group.
    """

    def _make_grouped_db(self, group_sizes: list[int]) -> tuple[str, list[tuple[str, float]]]:
        path = tempfile.mktemp(suffix=".db")
        raw = sqlite3.connect(path)
        raw.execute("CREATE TABLE g (grp TEXT, x REAL)")
        data: list[tuple[str, float]] = []
        for gi, size in enumerate(group_sizes):
            for _ in range(size):
                data.append((f"g{gi}", float(gi)))
        raw.executemany("INSERT INTO g (grp, x) VALUES (?, ?)", data)
        raw.commit()
        raw.close()
        return path, data

    def test_contiguous_groups_match_reference_partitioning(self) -> None:
        path, data = self._make_grouped_db([3, 2, 4, 1])
        self.addCleanup(os.unlink, path)
        # SQL rows are tuples, not dicts, so the group key is extracted positionally (row[0] == grp) --
        # a callable `by` is required for tuple rows regardless of this fix (see structure.group_key).
        structure = partially_exchangeable(lambda row: row[0])
        src = read_sql(f"sqlite:///{path}", "SELECT grp, x FROM g", structure=structure, batch_size=2)
        encoder = _RecordingEncoder([])

        result = src.encode(encoder, num_chunks=2)

        self.assertEqual(sum(count for count, _ in result), len(data))
        seen = sorted(row for batch in encoder.batches for row in batch)
        ref_parts = partition_records(data, structure, 2)
        expected = sorted(row for part in ref_parts for row in part)
        self.assertEqual(seen, expected)

    def test_non_contiguous_groups_raise_value_error(self) -> None:
        path = tempfile.mktemp(suffix=".db")
        raw = sqlite3.connect(path)
        raw.execute("CREATE TABLE g (grp TEXT, x REAL)")
        raw.executemany("INSERT INTO g (grp, x) VALUES (?, ?)", [("a", 1.0), ("b", 2.0), ("a", 3.0)])
        raw.commit()
        raw.close()
        self.addCleanup(os.unlink, path)

        structure = partially_exchangeable(lambda row: row[0])
        src = read_sql(f"sqlite:///{path}", "SELECT grp, x FROM g", structure=structure)
        with self.assertRaises(ValueError):
            src.encode(_RecordingEncoder([]), num_chunks=2)

    def test_single_group_num_chunks_one(self) -> None:
        path, data = self._make_grouped_db([5])
        self.addCleanup(os.unlink, path)
        structure = partially_exchangeable(lambda row: row[0])
        src = read_sql(f"sqlite:///{path}", "SELECT grp, x FROM g", structure=structure, batch_size=2)
        result = src.encode(_RecordingEncoder([]), num_chunks=1)
        self.assertEqual(sum(count for count, _ in result), 5)


@unittest.skipUnless(_HAS_SQLALCHEMY, "sqlalchemy not installed; pip install mixle[sql]")
class SqlCursorSourceChunkSizeTest(unittest.TestCase):
    """``chunk_size`` needs a row count to resolve into a partition count; a streaming cursor does not
    know that up front, so it is resolved via a bounded ``COUNT(*)`` subquery (one integer, never the
    row data) rather than by falling back to full materialization.
    """

    def test_chunk_size_resolves_via_count_star(self) -> None:
        n = 23
        path = _make_db([(i, float(i)) for i in range(n)])
        self.addCleanup(os.unlink, path)
        src = read_sql(f"sqlite:///{path}", "SELECT id, val FROM t")
        result = src.encode(_RecordingEncoder([]), chunk_size=5)  # ceil(23/5) = 5 partitions
        self.assertEqual(sum(count for count, _ in result), n)

    def test_chunk_size_tolerates_trailing_semicolon(self) -> None:
        n = 10
        path = _make_db([(i, float(i)) for i in range(n)])
        self.addCleanup(os.unlink, path)
        src = read_sql(f"sqlite:///{path}", "SELECT id, val FROM t;")
        result = src.encode(_RecordingEncoder([]), chunk_size=4)
        self.assertEqual(sum(count for count, _ in result), n)


@unittest.skipUnless(_HAS_SQLALCHEMY, "sqlalchemy not installed; pip install mixle[sql]")
class SqlCursorSourceValidationTest(unittest.TestCase):
    """``batch_size`` must be validated (positive integer) both at construction and as an ``encode()``
    override, rather than failing confusingly later (e.g. inside SQLAlchemy's ``yield_per``).
    """

    def setUp(self) -> None:
        self.path = _make_db([(0, 0.0)])
        self.addCleanup(os.unlink, self.path)

    def test_zero_batch_size_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            read_sql(f"sqlite:///{self.path}", "SELECT id FROM t", batch_size=0)

    def test_negative_batch_size_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            read_sql(f"sqlite:///{self.path}", "SELECT id FROM t", batch_size=-5)

    def test_bool_batch_size_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            read_sql(f"sqlite:///{self.path}", "SELECT id FROM t", batch_size=True)

    def test_invalid_batch_size_override_rejected_in_encode(self) -> None:
        src = read_sql(f"sqlite:///{self.path}", "SELECT id FROM t")
        with self.assertRaises(ValueError):
            src.encode(_RecordingEncoder([]), batch_size=0)

    def test_valid_batch_size_still_constructs_and_works(self) -> None:
        src = read_sql(f"sqlite:///{self.path}", "SELECT id FROM t", batch_size=10)
        self.assertEqual(list(src.records()), [0])


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(unittest.main())
