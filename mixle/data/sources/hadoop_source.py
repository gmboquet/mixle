"""Remote-filesystem connector -- HDFS / S3 / GCS files via ``fsspec``, composing with the text/arrow readers.

Optional: requires ``fsspec`` (and its backend, e.g. ``s3fs``/``gcsfs``/``pyarrow`` for HDFS) --
``pip install mixle[hadoop]``. ``read_remote(path, format=...)`` opens the object store path and
delegates to the matching local reader, so a Parquet/Feather/CSV/JSON/JSONL file on S3 reads exactly
like a local one -- including ``columns`` projection, applied identically to the local connectors
(:mod:`~mixle.data.sources.arrow_source` / :mod:`~mixle.data.sources.text_source`) rather than ignored.
``fmt`` values outside this set raise ``ValueError`` immediately, before any I/O.
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
_BINARY = {"parquet", "feather"}


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
    if fmt not in _BINARY and fmt not in _TEXT:
        # Validated upfront, against the same registry that decides open-mode below, rather than
        # falling through to this error deep inside the read (MXR-080-0065): every fmt this function
        # accepts past this point has a real branch, and every branch it lacks is rejected before
        # fsspec even opens the path.
        raise ValueError("unsupported remote format %r" % fmt)
    if _fsspec is None:
        from mixle.utils.optional_deps import require

        require("fsspec", "hadoop")
    from mixle.data.core import LazySource

    def factory():
        with _fsspec.open(path, "rb" if fmt in _BINARY else "rt", **storage_options) as fh:
            if fmt in _BINARY:
                from mixle.data.sources.arrow_source import _require_arrow, _table_records  # noqa: PLC0415

                _require_arrow()
                if fmt == "parquet":
                    import pyarrow.parquet as pq

                    table = pq.read_table(fh, columns=columns)
                else:
                    import pyarrow.feather as feather

                    table = feather.read_table(fh, columns=columns)
                return _table_records(table, None)

            import io

            data = fh.read()
            tmp = io.StringIO(data if isinstance(data, str) else data.decode())
            if fmt == "csv":
                import csv

                rows = list(csv.reader(tmp))
                header, body = rows[0], rows[1:]
                idx = [header.index(c) for c in columns] if columns else list(range(len(header)))
                return [r[idx[0]] if len(idx) == 1 else tuple(r[i] for i in idx) for r in body]

            import json

            def _project(obj):
                # Same projection contract as text_source.read_json/read_jsonl's `_select`: missing
                # requested keys raise KeyError (not silently dropped), extra object keys are dropped,
                # a single requested column collapses to a scalar rather than a length-1 tuple.
                if columns is None:
                    return obj
                picked = [obj[c] for c in columns]
                return picked[0] if len(picked) == 1 else tuple(picked)

            if fmt == "json":
                return [_project(obj) for obj in json.load(tmp)]
            return [_project(json.loads(line)) for line in tmp if line.strip()]

    return LazySource(factory, structure, schema)
