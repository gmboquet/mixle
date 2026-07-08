"""Connector registry + the ``open(kind, ...)`` dispatch.

Every connector returns a :class:`~mixle.data.core.DataSource` and is *lazy + optional*: nothing here is
imported at package load, and each connector module guards its driver behind the ``optional_deps`` shim
(``require(name, extra)``). The base install pulls in zero new heavy dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any

# kind -> (connector module, reader function). Resolved lazily on first use.
_KINDS: dict[str, tuple[str, str]] = {
    "csv": ("text_source", "read_csv"),
    "json": ("text_source", "read_json"),
    "jsonl": ("text_source", "read_jsonl"),
    "ndjson": ("text_source", "read_jsonl"),
    "parquet": ("arrow_source", "read_parquet"),
    "feather": ("arrow_source", "read_feather"),
    "arrow": ("arrow_source", "read_feather"),
    "sql": ("sql_source", "read_sql"),
    "dataframe": ("pandas_source", "read_dataframe"),
    "pandas": ("pandas_source", "read_dataframe"),
    "mongo": ("mongo_source", "read_mongo"),
    "remote": ("hadoop_source", "read_remote"),
    "zarr": ("array_source", "read_zarr"),
    "hdf5": ("array_source", "read_hdf5"),
    "memmap": ("array_source", "read_memmap"),
}


def open(kind: str, *args: Any, **kwargs: Any) -> Any:
    """Open a :class:`DataSource` by connector kind.

    Examples: ``open("csv", path)``, ``open("parquet", path, columns=[...])``,
    ``open("sql", url, query="select ...")``. The connector module is imported lazily, and a missing
    driver raises a clear ``pip install mixle[extra]`` message.
    """
    if kind not in _KINDS:
        raise ValueError("unknown source kind %r; known kinds: %s" % (kind, ", ".join(sorted(_KINDS))))
    module_name, reader_name = _KINDS[kind]
    module = importlib.import_module("mixle.data.sources." + module_name)
    return getattr(module, reader_name)(*args, **kwargs)


def kinds() -> list[str]:
    """Return the registered connector kinds."""
    return sorted(_KINDS)
