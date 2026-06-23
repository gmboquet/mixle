"""MongoDB connector -- a collection query into a :class:`~pysp.data.core.DataSource`.

Optional: requires ``pymongo`` (``pip install pysparkplug[mongo]``). The ``_id`` field is dropped; supply
a :class:`~pysp.data.schema.Schema` to coerce the loosely-typed BSON documents.
"""

from __future__ import annotations

from pysp.data.core import LazySource
from pysp.data.schema import Schema
from pysp.data.structure import EXCHANGEABLE, SampleStructure

try:  # optional dependency
    import pymongo as _pymongo
except ImportError:  # pragma: no cover - exercised only without pymongo
    _pymongo = None


def read_mongo(
    uri: str,
    database: str,
    collection: str,
    query: dict | None = None,
    columns: list[str] | None = None,
    *,
    structure: SampleStructure = EXCHANGEABLE,
    schema: Schema | None = None,
) -> LazySource:
    """Read documents matching ``query`` from ``database.collection`` at ``uri`` into a lazy DataSource."""

    def factory():
        if _pymongo is None:
            from pysp.utils.optional_deps import require

            require("pymongo", "mongo")
        coll = _pymongo.MongoClient(uri)[database][collection]
        projection = {c: 1 for c in columns} if columns else None
        records = []
        for doc in coll.find(query or {}, projection):
            doc.pop("_id", None)
            if columns is None:
                records.append(doc)
            else:
                picked = [doc[c] for c in columns]
                records.append(picked[0] if len(picked) == 1 else tuple(picked))
        return records

    return LazySource(factory, structure, schema)
