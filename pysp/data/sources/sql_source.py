"""SQL connector -- any SQLAlchemy URL (Postgres / MySQL / SQLite / ...) into a DataSource.

Optional: requires ``sqlalchemy`` (``pip install pysparkplug[sql]``). One connector covers every RDBMS;
rows stream from a server-side cursor so a large result set is not fully materialized in the driver.
"""

from __future__ import annotations

from pysp.data.core import LazySource
from pysp.data.schema import Schema
from pysp.data.structure import EXCHANGEABLE, SampleStructure

try:  # optional dependency
    import sqlalchemy as _sa
except ImportError:  # pragma: no cover - exercised only without sqlalchemy
    _sa = None


def read_sql(
    url: str,
    query: str,
    columns: list[str] | None = None,
    *,
    structure: SampleStructure = EXCHANGEABLE,
    schema: Schema | None = None,
) -> LazySource:
    """Read the rows of ``query`` against the database at ``url`` into a lazy DataSource."""

    def factory():
        if _sa is None:
            from pysp.utils.optional_deps import require

            require("sqlalchemy", "sql")
        engine = _sa.create_engine(url)
        with engine.connect() as conn:
            result = conn.execute(_sa.text(query))
            keys = list(result.keys())
            idx = [keys.index(c) for c in columns] if columns is not None else list(range(len(keys)))
            rows = []
            for row in result:
                picked = [row[i] for i in idx]
                rows.append(picked[0] if len(picked) == 1 else tuple(picked))
            return rows

    return LazySource(factory, structure, schema)
