"""Remote-filesystem connector -- HDFS / S3 / GCS files via ``fsspec``, composing with the text/arrow readers.

Optional: requires ``fsspec`` (and its backend, e.g. ``s3fs``/``gcsfs``/``pyarrow`` for HDFS) --
``pip install mixle[hadoop]``. ``read_remote(path, format=...)`` opens the object store path and
delegates to the matching local reader, so a Parquet/CSV/JSONL file on S3 reads exactly like a local one.
"""

from __future__ import annotations

from typing import Any

from mixle.data.schema import Schema
from mixle.data.structure import EXCHANGEABLE, SampleStructure

try:  # optional dependency
    import fsspec as _fsspec
except ImportError:  # pragma: no cover - exercised only without fsspec
    _fsspec = None

_TEXT = {"csv", "json", "jsonl", "ndjson"}


def read_remote(
    path: str,
    fmt: str,
    columns: list[str] | None = None,
    *,
    structure: SampleStructure = EXCHANGEABLE,
    schema: Schema | None = None,
    **storage_options: Any,
):
    """Read a remote (``s3://`` / ``gcs://`` / ``hdfs://``) file by ``fmt`` via fsspec + the local reader."""
    if _fsspec is None:
        from mixle.utils.optional_deps import require

        require("fsspec", "hadoop")
    from mixle.data.core import LazySource

    def factory():
        with _fsspec.open(path, "rb" if fmt in {"parquet", "feather"} else "rt", **storage_options) as fh:
            if fmt == "parquet":
                from mixle.data.sources.arrow_source import _require_arrow, _table_records  # noqa: PLC0415

                _require_arrow()
                import pyarrow.parquet as pq

                return _table_records(pq.read_table(fh, columns=columns), None)
            if fmt in _TEXT:
                import io

                from mixle.data.sources import text_source

                data = fh.read()
                tmp = io.StringIO(data if isinstance(data, str) else data.decode())
                reader = {"csv": text_source.read_csv, "json": text_source.read_json}.get(fmt, text_source.read_jsonl)
                # text readers take a path; for a stream we re-implement the small read here
                if fmt == "csv":
                    import csv

                    rows = list(csv.reader(tmp))
                    header, body = rows[0], rows[1:]
                    idx = [header.index(c) for c in columns] if columns else list(range(len(header)))
                    return [r[idx[0]] if len(idx) == 1 else tuple(r[i] for i in idx) for r in body]
                import json

                if fmt == "json":
                    return list(json.load(tmp))
                return [json.loads(line) for line in tmp if line.strip()]
            raise ValueError("unsupported remote format %r" % fmt)

    return LazySource(factory, structure, schema)
