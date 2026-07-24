"""Regression tests for the remote-connector format/projection gap (MXR-080-0065).

``hadoop_source.read_remote`` opened Feather files in binary mode (implying support) but had no
Feather branch, so every Feather read fell through to "unsupported remote format" despite the module
claiming to delegate to the Arrow readers. JSON and JSONL accepted a ``columns`` argument but never
applied it, always returning whole objects -- unlike the local ``text_source`` connectors, which
project ``columns`` down to a scalar (one requested column) or tuple (several), matching CSV's own
convention in this same file.

The fix: (a) implements Feather via ``pyarrow.feather.read_table``, the same Arrow reader
``mixle.data.sources.arrow_source.read_feather`` already uses, mirroring the existing Parquet branch;
(b) applies ``columns`` projection to JSON/JSONL with semantics identical to
``text_source.read_json``/``read_jsonl`` (missing keys raise ``KeyError``, a single requested column
collapses to a scalar rather than a length-1 tuple); and (c) validates ``fmt`` against the
accepted-format registry (``_BINARY | _TEXT``) up front, before any I/O, instead of only discovering
an unsupported format deep inside a read call via fsspec.

fsspec's local filesystem backend (used implicitly for a plain, schemeless path) is the standard way
to exercise fsspec-based code without cloud credentials or a mocking library -- only the storage
backend differs between ``s3://...`` and a local path; the read/projection logic under test here does
not.
"""

import json

import pytest

from mixle.data.sources import hadoop_source, text_source

pytest.importorskip("fsspec")  # required unconditionally by read_remote, both binary and text formats


def _write_feather(path, table):
    pytest.importorskip("pyarrow")
    import pyarrow.feather as feather

    feather.write_feather(table, str(path))


def _make_table():
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    return pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})


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


def test_unsupported_format_raises_before_any_io():
    """The accepted-format registry (``_BINARY | _TEXT``) is now authoritative and checked up front:
    an unsupported ``fmt`` raises immediately, even for a path that does not exist -- not deep inside
    a read call after fsspec has already opened something."""
    with pytest.raises(ValueError, match="unsupported remote format"):
        hadoop_source.read_remote("/nonexistent/path/does/not/matter.xml", "xml")
