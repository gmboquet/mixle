"""Regression tests for the remote-connector format/projection gap (MXR-080-0065).

``hadoop_source.read_remote`` opened Feather files in binary mode (implying support) but had no
Feather branch, so every Feather read fell through to "unsupported remote format" despite the module
claiming to delegate to the Arrow readers. JSON and JSONL accepted a ``columns`` argument but never
applied it, always returning whole objects -- unlike the local ``text_source`` connectors, which
project ``columns`` down to a scalar (one requested column) or tuple (several), matching CSV's own
convention in this same file.

The fix: (a) implements Feather via the same Arrow IPC reader
``mixle.data.sources.arrow_source.read_feather`` uses, mirroring the existing Parquet branch (both were
later migrated off the deprecated ``pyarrow.feather.read_table``/``write_feather`` onto
``pyarrow.ipc.open_file(...).read_all()`` / ``pyarrow.ipc.new_file(...)``, per MXR-080-0070's follow-up);
(b) applies ``columns`` projection to JSON/JSONL with semantics identical to
``text_source.read_json``/``read_jsonl`` (missing keys raise ``KeyError``, a single requested column
collapses to a scalar rather than a length-1 tuple); and (c) validates ``fmt`` against the
accepted-format registry (``_BINARY | _TEXT``) up front, before any I/O, instead of only discovering
an unsupported format deep inside a read call via fsspec.

Also covers two smaller, unrelated fixes bundled into this same file because they touch the same
module: the CSV branch's ``columns=[]`` truthiness bug (an explicitly-empty column list was falsy and
fell through to "no filter", disagreeing with the local ``text_source.read_csv`` connector, which
treats ``columns=[]`` as "project to zero columns" via an ``is not None`` check -- same class of "local
vs remote projection semantics must match exactly" bug as MXR-080-0065 above, just a different,
pre-existing instance), and the deprecated ``pyarrow.feather.read_table``/``write_feather`` calls
(pyarrow 24+ emits ``FutureWarning``; migrated to ``pyarrow.ipc.open_file(...).read_all()`` /
``pyarrow.ipc.new_file(...)``, mirroring the fix in ``arrow_source.read_feather``).

fsspec's local filesystem backend (used implicitly for a plain, schemeless path) is the standard way
to exercise fsspec-based code without cloud credentials or a mocking library -- only the storage
backend differs between ``s3://...`` and a local path; the read/projection logic under test here does
not.
"""

import json
import warnings

import pytest

from mixle.data.sources import hadoop_source, text_source

pytest.importorskip("fsspec")  # required unconditionally by read_remote, both binary and text formats


def _write_feather(path, table):
    pytest.importorskip("pyarrow")
    # pyarrow.feather.write_feather is deprecated (pyarrow 24+) in favor of writing the IPC file format
    # directly -- Feather V2 *is* the Arrow IPC file format. Same deprecation-free pattern as
    # arrow_source_test.py's own _write_feather helper.
    import pyarrow.ipc as ipc

    with ipc.new_file(str(path), table.schema) as writer:
        writer.write_table(table)


def _make_table():
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    return pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def _write_parquet(path, table):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    pq.write_table(table, str(path))


# --------------------------------------------------------------------------- Feather (MXR-080-0065a)


def test_feather_round_trip_all_columns(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.feather"
    _write_feather(path, _make_table())

    records = list(hadoop_source.read_remote(str(path), "feather").records())

    assert records == [(1, "x"), (2, "y"), (3, "z")]


def test_feather_round_trip_columns_projection(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.feather"
    _write_feather(path, _make_table())

    records = list(hadoop_source.read_remote(str(path), "feather", columns=["b"]).records())

    assert records == ["x", "y", "z"]


def test_feather_previously_raised_unsupported_format_now_reads(tmp_path):
    """Pins the exact MXR-080-0065 symptom: Feather was opened in binary mode (implying acceptance)
    but had no read branch, so it always raised ``ValueError("unsupported remote format 'feather'")``
    regardless of the file's validity. Confirm that specific failure mode is gone."""
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.feather"
    _write_feather(path, _make_table())

    try:
        hadoop_source.read_remote(str(path), "feather").records()
    except ValueError as e:
        pytest.fail(f"feather must be supported, not raise {e!r}")


# --------------------------------------------------------------------------- JSON (MXR-080-0065b)


def test_json_columns_projection_matches_local_text_source(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"a": 1, "b": "x", "c": 9}, {"a": 2, "b": "y", "c": 8}]))

    remote_one = list(hadoop_source.read_remote(str(path), "json", columns=["b"]).records())
    local_one = list(text_source.read_json(str(path), columns=["b"]).records())
    assert remote_one == local_one == ["x", "y"]  # single column -> scalar, not a length-1 tuple

    remote_two = list(hadoop_source.read_remote(str(path), "json", columns=["b", "a"]).records())
    local_two = list(text_source.read_json(str(path), columns=["b", "a"]).records())
    assert remote_two == local_two == [("x", 1), ("y", 2)]  # order follows the requested columns


def test_json_columns_none_still_returns_whole_objects(tmp_path):
    """Negative control: omitting ``columns`` must keep returning whole objects, unchanged."""
    path = tmp_path / "data.json"
    objs = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path.write_text(json.dumps(objs))

    assert list(hadoop_source.read_remote(str(path), "json").records()) == objs


def test_json_missing_requested_column_raises_keyerror_matching_local(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"a": 1, "b": "x"}]))

    with pytest.raises(KeyError):
        list(hadoop_source.read_remote(str(path), "json", columns=["nope"]).records())
    with pytest.raises(KeyError):
        list(text_source.read_json(str(path), columns=["nope"]).records())


# --------------------------------------------------------------------------- JSONL (MXR-080-0065b)


def test_jsonl_columns_projection_matches_local_text_source(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n')

    remote = list(hadoop_source.read_remote(str(path), "jsonl", columns=["b"]).records())
    local = list(text_source.read_jsonl(str(path), columns=["b"]).records())
    assert remote == local == ["x", "y"]


def test_jsonl_columns_none_still_returns_whole_objects(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n')

    assert list(hadoop_source.read_remote(str(path), "jsonl").records()) == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
    ]


# --------------------------------------------------------------------------- registry / format validation


def test_csv_projection_still_works_unchanged(tmp_path):
    """Negative control: the CSV branch (not part of this finding) must be unaffected by the
    refactor that hoisted format validation and rewrote the JSON/JSONL branch alongside it."""
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,x\n2,y\n")

    assert list(hadoop_source.read_remote(str(path), "csv", columns=["b"]).records()) == ["x", "y"]
    assert list(hadoop_source.read_remote(str(path), "csv").records()) == [("1", "x"), ("2", "y")]


# --------------------------------------------------------------------------- CSV columns=[] truthiness bug


def test_csv_columns_empty_list_matches_local_text_source(tmp_path):
    """``read_remote``'s CSV branch used to build ``idx`` with a truthiness check (``if columns``), so
    an explicitly-empty ``columns=[]`` was falsy and fell through to the "no filter" branch -- returning
    every column instead of projecting to zero columns. ``text_source.read_csv`` already gets this right
    via an ``is not None`` check: ``columns=[]`` means "project to zero columns" there (every row becomes
    ``()``), and ``columns=None`` (omitted) is the only case that means "no filter". Remote and local must
    agree for both inputs.
    """
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,x\n2,y\n")

    with pytest.raises(ValueError):
        hadoop_source.read_remote(str(path), "csv", columns=[])
    with pytest.raises(ValueError):
        text_source.read_csv(str(path), columns=[])


def test_csv_columns_empty_list_differs_from_columns_none(tmp_path):
    """Negative control distinguishing the two: ``columns=[]`` (zero columns) must NOT collapse to the
    same result as ``columns=None`` (no filter, every column) for either connector -- pinning the exact
    symptom of the truthiness bug, where the two were indistinguishable."""
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,x\n2,y\n")

    remote_none = list(hadoop_source.read_remote(str(path), "csv", columns=None).records())
    local_none = list(text_source.read_csv(str(path), columns=None).records())

    with pytest.raises(ValueError):
        hadoop_source.read_remote(str(path), "csv", columns=[])
    with pytest.raises(ValueError):
        text_source.read_csv(str(path), columns=[])
    assert remote_none == local_none == [("1", "x"), ("2", "y")]


# --------------------------------------------------------------------------- Parquet/Feather columns=[] zip bug


def test_parquet_columns_empty_list_matches_local_text_source(tmp_path):
    """``arrow_source._table_records`` -- shared by this module's Parquet and Feather branches via
    ``_table_records(table, None)`` after the table itself is already projected down to zero columns --
    used to build each row via ``zip(*(pydict[c] for c in cols))``. With zero columns, Python's
    ``zip()`` called with zero iterables returns an empty iterator with no way to know how many rows
    the table actually had, so an explicit ``columns=[]`` silently returned zero records regardless of
    the file's real row count, instead of one zero-width record (``()``) per row. Remote Parquet and
    local ``text_source.read_csv`` (which already gets ``columns=[]`` right) must agree.
    """
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.parquet"
    _write_parquet(path, _make_table())
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,x\n2,y\n3,z\n")

    with pytest.raises(ValueError):
        hadoop_source.read_remote(str(path), "parquet", columns=[])
    with pytest.raises(ValueError):
        text_source.read_csv(str(csv_path), columns=[])


def test_feather_columns_empty_list_matches_local_text_source(tmp_path):
    """Same ``_table_records`` zip-with-zero-columns bug as the Parquet case above, reached via this
    module's Feather branch (``table.select([])`` then ``_table_records(table, None)``)."""
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.feather"
    _write_feather(path, _make_table())
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,x\n2,y\n3,z\n")

    with pytest.raises(ValueError):
        hadoop_source.read_remote(str(path), "feather", columns=[])
    with pytest.raises(ValueError):
        text_source.read_csv(str(csv_path), columns=[])


def test_parquet_columns_empty_list_differs_from_columns_none(tmp_path):
    """Negative control pinning the exact symptom: columns=[] (zero columns, row count preserved) must
    NOT collapse to the same result as columns=None (no filter, every column)."""
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.parquet"
    _write_parquet(path, _make_table())

    none = list(hadoop_source.read_remote(str(path), "parquet", columns=None).records())

    with pytest.raises(ValueError):
        hadoop_source.read_remote(str(path), "parquet", columns=[])
    assert none == [(1, "x"), (2, "y"), (3, "z")]


def test_feather_columns_empty_list_differs_from_columns_none(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.feather"
    _write_feather(path, _make_table())

    none = list(hadoop_source.read_remote(str(path), "feather", columns=None).records())

    with pytest.raises(ValueError):
        hadoop_source.read_remote(str(path), "feather", columns=[])
    assert none == [(1, "x"), (2, "y"), (3, "z")]


# --------------------------------------------------------------------------- deprecated pyarrow.feather API


def test_feather_read_emits_no_future_warning(tmp_path):
    """pyarrow 24+ deprecated ``pyarrow.feather.read_table`` in favor of
    ``pyarrow.ipc.open_file(...).read_all()`` (Feather V2 *is* the Arrow IPC file format), emitting a
    ``FutureWarning`` on every call. Confirm the migrated read path no longer triggers it."""
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.feather"
    _write_feather(path, _make_table())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        records = list(hadoop_source.read_remote(str(path), "feather").records())

    future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert not future_warnings, [str(w.message) for w in future_warnings]
    assert records == [(1, "x"), (2, "y"), (3, "z")]  # migration must not change the actual data


def test_feather_write_fixture_emits_no_future_warning(tmp_path):
    """This file's own ``_write_feather`` fixture used to call the equally-deprecated
    ``pyarrow.feather.write_feather``. Confirm the fixture itself (now on ``pyarrow.ipc.new_file``) is
    clean too, so no test in this module silently relies on deprecated pyarrow behavior."""
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.feather"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _write_feather(path, _make_table())

    future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert not future_warnings, [str(w.message) for w in future_warnings]


def test_unsupported_format_raises_before_any_io():
    """The accepted-format registry (``_BINARY | _TEXT``) is now authoritative and checked up front:
    an unsupported ``fmt`` raises immediately, even for a path that does not exist -- not deep inside
    a read call after fsspec has already opened something."""
    with pytest.raises(ValueError, match="unsupported remote format"):
        hadoop_source.read_remote("/nonexistent/path/does/not/matter.xml", "xml")
