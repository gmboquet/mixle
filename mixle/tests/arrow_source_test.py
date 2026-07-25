"""Tests for mixle.data.sources.arrow_source (audit MXR-080-0070).

The module's docstring claimed an Arrow schema is inferred when none is supplied, but both readers
(``read_parquet``, ``read_feather``) unconditionally constructed ``LazySource(..., schema=None)``,
discarding the Arrow table's own field types/nullability -- so ``Schema.conform`` (the schema-aware
validation ``mixle.data.schema`` exists to provide) never ran for either connector's default path.

Fixed by building a real Mixle ``Schema`` from the file's own Arrow schema (a footer/metadata-only
read -- the row data itself stays deferred to first use, like every other ``LazySource`` connector)
whenever the caller omits ``schema``. See ``_arrow_field_type``/``_schema_from_arrow`` in the module
under test for the type-mapping details and its documented, intentional scope: nested/list/struct,
timestamp/date, binary, and decimal columns raise a clear ``TypeError`` during inference rather than
silently mapping to the wrong type or silently discarding schema for the whole table.

Covers: the type/nullability mapping for every Arrow scalar type the mapping supports (both readers),
that materialized records are actually threaded through the inferred schema (not just decorated with
unused metadata), that row data reads stay deferred until first use, that an explicit ``schema=``
still bypasses inference unchanged (legacy behavior), that unsupported column types raise clearly
instead of silently degrading, and that a partitioned Parquet dataset directory (not just a single
file) still infers correctly.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import warnings

import pytest

pa = pytest.importorskip("pyarrow")

import pyarrow.ipc as _pa_ipc  # noqa: E402
import pyarrow.parquet as _pa_parquet  # noqa: E402

from mixle.data import sources as sources_pkg  # noqa: E402
from mixle.data.schema import Boolean, Categorical, Count, Optional, Real, Text  # noqa: E402
from mixle.data.sources.arrow_source import read_feather, read_parquet  # noqa: E402


def _write_parquet(directory: str, name: str, table: pa.Table) -> str:
    path = os.path.join(directory, name)
    _pa_parquet.write_table(table, path)
    return path


def _write_feather(directory: str, name: str, table: pa.Table) -> str:
    # pyarrow.feather.write_feather is deprecated (pyarrow 24+) in favor of writing the IPC file format
    # directly -- Feather V2 *is* the Arrow IPC file format, so a file written this way round-trips
    # through pyarrow.feather.read_table (what read_feather() itself still uses) identically, without
    # pulling a deprecated call into new test code.
    path = os.path.join(directory, name)
    with _pa_ipc.new_file(path, table.schema) as writer:
        writer.write_table(table)
    return path


def _wide_table() -> pa.Table:
    """One table covering every Arrow type the mapping supports, plus a dictionary/categorical column."""
    return pa.table(
        {
            "i8": pa.array([1, -2, 3], type=pa.int8()),
            "i16": pa.array([1, -2, 3], type=pa.int16()),
            "i32": pa.array([1, -2, 3], type=pa.int32()),
            "i64": pa.array([1, -2, 3], type=pa.int64()),
            "u8": pa.array([1, 2, 3], type=pa.uint8()),
            "u16": pa.array([1, 2, 3], type=pa.uint16()),
            "u32": pa.array([1, 2, 3], type=pa.uint32()),
            "u64": pa.array([1, 2, 3], type=pa.uint64()),
            "f32": pa.array([1.5, 2.5, 3.5], type=pa.float32()),
            "f64": pa.array([1.5, 2.5, 3.5], type=pa.float64()),
            "flag": pa.array([True, False, True], type=pa.bool_()),
            "label": pa.array(["a", "b", "c"], type=pa.string()),
            "big_label": pa.array(["a", "b", "c"], type=pa.large_string()),
            "cat": pa.array(["red", "blue", "red"], type=pa.string()).dictionary_encode(),
        }
    )


_EXPECTED_WIDE_TYPES = {
    "i8": Count,
    "i16": Count,
    "i32": Count,
    "i64": Count,
    "u8": Count,
    "u16": Count,
    "u32": Count,
    "u64": Count,
    "f32": Real,
    "f64": Real,
    "flag": Boolean,
    "label": Text,
    "big_label": Text,
    "cat": Categorical,
}


def _unwrap(field_type):
    """Return the inner type of an ``Optional`` wrapper, else the type itself."""
    return field_type.inner if isinstance(field_type, Optional) else field_type


class ArrowSchemaInferenceTest(unittest.TestCase):
    """``schema=None`` (the default) must build a real, correctly-typed Schema for both readers."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="mixle_arrow_infer_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_read_parquet_infers_types_for_every_supported_arrow_type(self):
        path = _write_parquet(self.tmpdir, "wide.parquet", _wide_table())
        src = read_parquet(path)
        self.assertIsNotNone(src.schema)
        by_name = {f.name: f.type for f in src.schema.fields}
        for name, expected_cls in _EXPECTED_WIDE_TYPES.items():
            self.assertIsInstance(_unwrap(by_name[name]), expected_cls, "field %r" % name)

    def test_read_feather_infers_types_for_every_supported_arrow_type(self):
        path = _write_feather(self.tmpdir, "wide.feather", _wide_table())
        src = read_feather(path)
        self.assertIsNotNone(src.schema)
        by_name = {f.name: f.type for f in src.schema.fields}
        for name, expected_cls in _EXPECTED_WIDE_TYPES.items():
            self.assertIsInstance(_unwrap(by_name[name]), expected_cls, "field %r" % name)

    def test_nullable_field_is_optional_non_nullable_field_is_not(self):
        arrow_schema = pa.schema(
            [
                pa.field("required_id", pa.int32(), nullable=False),
                pa.field("optional_score", pa.float64(), nullable=True),
            ]
        )
        table = pa.table(
            {
                "required_id": pa.array([1, 2, 3], type=pa.int32()),
                "optional_score": pa.array([1.0, None, 3.0]),
            },
            schema=arrow_schema,
        )
        path = _write_parquet(self.tmpdir, "nullability.parquet", table)
        src = read_parquet(path)
        by_name = {f.name: f.type for f in src.schema.fields}
        self.assertIsInstance(by_name["required_id"], Count)  # bare, not wrapped
        self.assertIsInstance(by_name["optional_score"], Optional)
        self.assertIsInstance(by_name["optional_score"].inner, Real)

    def test_partitioned_dataset_directory_still_infers_schema(self):
        # read_parquet's docstring claims file/dataset support, and pyarrow.parquet.read_table already
        # transparently reads a directory of files via the dataset API. pyarrow.parquet.read_schema --
        # the more obvious footer-peek API -- raises OSError on a directory, so the module must use
        # pyarrow.dataset instead; this locks that choice in.
        table = pa.table({"id": pa.array([1, 2], type=pa.int32()), "x": pa.array([1.0, 2.0])})
        dataset_dir = os.path.join(self.tmpdir, "dataset")
        os.makedirs(dataset_dir)
        _pa_parquet.write_table(table, os.path.join(dataset_dir, "part-0.parquet"))
        _pa_parquet.write_table(table, os.path.join(dataset_dir, "part-1.parquet"))

        src = read_parquet(dataset_dir)
        self.assertIsNotNone(src.schema)
        self.assertEqual(src.schema.names, ("id", "x"))
        self.assertEqual(len(list(src.records())), 4)

    def test_columns_subset_and_reorder_keeps_schema_and_records_aligned(self):
        table = pa.table(
            {
                "a": pa.array([1, 2], type=pa.int64()),
                "b": pa.array(["x", "y"]),
                "c": pa.array([1.1, 2.2]),
            }
        )
        path = _write_parquet(self.tmpdir, "wide2.parquet", table)
        src = read_parquet(path, columns=["c", "a"])
        self.assertEqual(src.schema.names, ("c", "a"))
        self.assertIsInstance(_unwrap(src.schema.fields[0].type), Real)
        self.assertIsInstance(_unwrap(src.schema.fields[1].type), Count)
        records = list(src.records())
        self.assertEqual(records, [(1.1, 1), (2.2, 2)])

    def test_single_column_selection_still_unwraps_to_scalars(self):
        table = pa.table({"id": pa.array([1, 2, 3], type=pa.int32()), "other": pa.array([1.0, 2.0, 3.0])})
        path = _write_parquet(self.tmpdir, "single.parquet", table)
        src = read_parquet(path, columns=["id"])
        self.assertEqual(len(src.schema.fields), 1)
        records = list(src.records())
        self.assertEqual(records, [1, 2, 3])
        for r in records:
            self.assertIsInstance(r, int)


class ArrowSchemaConformanceTest(unittest.TestCase):
    """Regression coverage: the inferred schema must be threaded through ``Schema.conform`` on every
    materialized record, not merely attached to the source as unused metadata.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="mixle_arrow_conform_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_conformed_records_have_correct_python_types(self):
        table = pa.table({"count": pa.array([1, 2, 3], type=pa.uint16()), "value": pa.array([1.5, None, 3.5])})
        path = _write_parquet(self.tmpdir, "conform.parquet", table)
        records = list(read_parquet(path).records())
        self.assertEqual(records, [(1, 1.5), (2, None), (3, 3.5)])
        for count, value in records:
            self.assertIsInstance(count, int)
            self.assertNotIsInstance(count, bool)
            if value is not None:
                self.assertIsInstance(value, float)

    def test_downstream_conform_rejects_a_negative_count_value(self):
        # Arrow's int32 happily allows negative values; Mixle's Count logical type does not
        # (schema.py's Count.coerce explicitly rejects negative values as "not a genuine count"). This
        # file is perfectly valid Arrow -- if read_parquet's inferred schema is real (Count for an
        # integer column) AND LazySource.materialize() truly threads every record through
        # Schema.conform, reading it back must raise. If schema were still effectively None (or
        # unused), this negative value would pass through silently.
        table = pa.table({"count": pa.array([1, 2, -5], type=pa.int32())})
        path = _write_parquet(self.tmpdir, "negative_count.parquet", table)
        src = read_parquet(path)
        with self.assertRaises(ValueError):
            src.records()

    def test_downstream_conform_rejects_a_negative_count_value_via_feather(self):
        table = pa.table({"count": pa.array([1, 2, -5], type=pa.int32())})
        path = _write_feather(self.tmpdir, "negative_count.feather", table)
        src = read_feather(path)
        with self.assertRaises(ValueError):
            src.records()

    def test_no_explicit_schema_field_count_mismatch_is_impossible_by_construction(self):
        # Sanity check on the inference path itself (not a hand-built schema): the number of inferred
        # fields must equal the number of columns actually emitted per row, for a multi-column table.
        table = pa.table({"a": pa.array([1, 2]), "b": pa.array([3, 4]), "c": pa.array([5, 6])})
        path = _write_parquet(self.tmpdir, "triple.parquet", table)
        src = read_parquet(path)
        self.assertEqual(len(src.schema.fields), 3)
        for record in src.records():
            self.assertEqual(len(record), 3)


class ArrowSchemaInferenceLazinessTest(unittest.TestCase):
    """Building the inferred schema must not force the row data itself to be read early."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="mixle_arrow_lazy_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_row_data_read_stays_deferred_for_parquet(self):
        path = _write_parquet(self.tmpdir, "lazy.parquet", pa.table({"id": pa.array([1, 2, 3])}))
        src = read_parquet(path)
        self.assertIsNotNone(src.schema)  # the footer/metadata peek already happened
        os.remove(path)  # the row data must NOT have been read (or cached) yet
        with self.assertRaises(OSError):
            src.records()  # now forced to actually open the (now-missing) file

    def test_row_data_read_stays_deferred_for_feather(self):
        path = _write_feather(self.tmpdir, "lazy.feather", pa.table({"id": pa.array([1, 2, 3])}))
        src = read_feather(path)
        self.assertIsNotNone(src.schema)
        os.remove(path)
        with self.assertRaises(OSError):
            src.records()

    def test_explicit_schema_skips_inference_and_stays_fully_lazy(self):
        # With an explicit schema, read_parquet must not touch the file at all until materialize --
        # exactly the pre-existing contract, unchanged by this fix.
        from mixle.data.schema import Field, Schema

        path = os.path.join(self.tmpdir, "does_not_exist.parquet")
        explicit = Schema((Field("id", Count()),))
        src = read_parquet(path, schema=explicit)  # must not raise: no file access yet
        self.assertIs(src.schema, explicit)


class ArrowUnsupportedTypeTest(unittest.TestCase):
    """Columns outside the supported mapping must raise clearly, not silently degrade."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="mixle_arrow_unsupported_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_timestamp_column_raises_typeerror_during_inference(self):
        table = pa.table({"id": pa.array([1, 2]), "ts": pa.array([1, 2], type=pa.timestamp("us"))})
        path = _write_parquet(self.tmpdir, "bad.parquet", table)
        with self.assertRaises(TypeError):
            read_parquet(path)

    def test_excluding_the_unsupported_column_avoids_the_raise(self):
        table = pa.table({"id": pa.array([1, 2]), "ts": pa.array([1, 2], type=pa.timestamp("us"))})
        path = _write_parquet(self.tmpdir, "bad2.parquet", table)
        src = read_parquet(path, columns=["id"])  # ts is never inspected
        self.assertEqual(list(src.records()), [1, 2])

    def test_explicit_schema_bypasses_inference_even_with_an_unsupported_column(self):
        from mixle.data.schema import Field, Real, Schema

        table = pa.table({"id": pa.array([1, 2]), "ts": pa.array([1, 2], type=pa.timestamp("us"))})
        path = _write_parquet(self.tmpdir, "bad3.parquet", table)
        explicit = Schema((Field("id", Real()), Field("ts", Real())))
        # ts's raw decoded value is a datetime.datetime; Real.coerce(datetime) raises, but the point
        # here is only that inference itself is skipped -- construction must not raise.
        src = read_parquet(path, schema=explicit)
        self.assertIs(src.schema, explicit)

    def test_struct_column_raises_typeerror_during_inference_for_feather(self):
        table = pa.table({"id": pa.array([1, 2]), "s": pa.array([{"a": 1}, {"a": 2}])})
        path = _write_feather(self.tmpdir, "bad.feather", table)
        with self.assertRaises(TypeError):
            read_feather(path)


class ArrowConnectorRegistryTest(unittest.TestCase):
    """The public ``mixle.data.sources.open(...)`` dispatch must still resolve to the fixed readers."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="mixle_arrow_registry_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_open_parquet_infers_schema(self):
        path = _write_parquet(self.tmpdir, "reg.parquet", pa.table({"id": pa.array([1, 2, 3])}))
        src = sources_pkg.open("parquet", path)
        self.assertIsNotNone(src.schema)

    def test_open_feather_and_arrow_kinds_infer_schema(self):
        path = _write_feather(self.tmpdir, "reg.feather", pa.table({"id": pa.array([1, 2, 3])}))
        for kind in ("feather", "arrow"):
            with self.subTest(kind=kind):
                src = sources_pkg.open(kind, path)
                self.assertIsNotNone(src.schema)


class ArrowFeatherDeprecatedApiTest(unittest.TestCase):
    """``read_feather`` used to call the deprecated ``pyarrow.feather.read_table`` -- pyarrow 24+ emits a
    ``FutureWarning`` on every call ("pyarrow.feather.read_table is deprecated as of 24.0.0. Use
    pyarrow.ipc.open_file() / RecordBatchFileReader instead. Feather V2 is the Arrow IPC file format.").
    Migrated to ``pyarrow.ipc.open_file(...).read_all()`` -- the same IPC reader this module's own
    schema-inference path above (and this test file's ``_write_feather`` helper) already used,
    deprecation-free. Bundled into this audit's test file because it touches the same ``read_feather``
    factory the MXR-080-0070 schema-inference fix lives in; the deprecation itself is unrelated to that
    finding.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="mixle_arrow_feather_deprecation_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_read_feather_emits_no_future_warning(self):
        # A small table of its own (not _wide_table(), whose signed-int columns hold a -2 that Mixle's
        # Count type legitimately rejects by design -- see ArrowSchemaConformanceTest -- unrelated to
        # this fix) since this test reads every column with no `columns` filter.
        table = pa.table({"id": pa.array([1, 2, 3], type=pa.int32()), "label": pa.array(["a", "b", "c"])})
        path = _write_feather(self.tmpdir, "warn.feather", table)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            records = list(read_feather(path).records())
        future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
        self.assertEqual(future_warnings, [], [str(w.message) for w in future_warnings])
        self.assertEqual(records, [(1, "a"), (2, "b"), (3, "c")])

    def test_read_feather_with_columns_emits_no_future_warning(self):
        # Also pins that the manual `columns` projection replacing the old `columns=` read kwarg keeps
        # selecting (and ordering) the right fields, not just that the warning is gone. Uses "u32" (not
        # one of the signed columns, which hold a -2) since Mixle's Count type legitimately rejects
        # negative values by design (see ArrowSchemaConformanceTest) -- unrelated to this fix.
        path = _write_feather(self.tmpdir, "warn_cols.feather", _wide_table())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            records = list(read_feather(path, columns=["cat", "u32"]).records())
        future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
        self.assertEqual(future_warnings, [], [str(w.message) for w in future_warnings])
        self.assertEqual(records, [("red", 1), ("blue", 2), ("red", 3)])
