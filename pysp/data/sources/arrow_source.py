"""Apache Arrow connectors -- Parquet / Arrow / Feather into a :class:`~pysp.data.core.DataSource`.

Optional: requires ``pyarrow`` (``pip install pysparkplug[arrow]``). Schema is inferred from the Arrow
types when none is supplied; reading is columnar and zero-copy where Arrow allows.
"""

from __future__ import annotations

from typing import Any

from pysp.data.core import LazySource
from pysp.data.schema import Schema
from pysp.data.structure import EXCHANGEABLE, SampleStructure

try:  # optional dependency
    import pyarrow.feather as _feather
    import pyarrow.parquet as _pq
except ImportError:  # pragma: no cover - exercised only without pyarrow
    _pq = None
    _feather = None


def _require_arrow() -> None:
    if _pq is None:
        from pysp.utils.optional_deps import require

        require("pyarrow", "arrow")


def _table_records(table: Any, columns: list[str] | None) -> list[Any]:
    cols = columns if columns is not None else table.column_names
    pydict = table.select(cols).to_pydict()
    rows = zip(*(pydict[c] for c in cols))
    return [r[0] if len(cols) == 1 else r for r in rows]


def read_parquet(path: str, columns: list[str] | None = None, *, structure: SampleStructure = EXCHANGEABLE,
                 schema: Schema | None = None) -> LazySource:
    """Read a Parquet file/dataset into a lazy DataSource of scalar/tuple records."""
    def factory():
        _require_arrow()
        return _table_records(_pq.read_table(path, columns=columns), None)
    return LazySource(factory, structure, schema)


def read_feather(path: str, columns: list[str] | None = None, *, structure: SampleStructure = EXCHANGEABLE,
                 schema: Schema | None = None) -> LazySource:
    """Read an Arrow/Feather file into a lazy DataSource of scalar/tuple records."""
    def factory():
        _require_arrow()
        return _table_records(_feather.read_table(path, columns=columns), None)
    return LazySource(factory, structure, schema)
