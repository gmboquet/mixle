"""Apache Arrow connectors -- Parquet / Arrow / Feather into a :class:`~mixle.data.core.DataSource`.

Optional: requires ``pyarrow`` (``pip install mixle[arrow]``); reading is columnar and zero-copy where
Arrow allows. When ``schema`` is omitted, it is inferred from the file's own Arrow schema -- a
footer/metadata-only read of field types and nullability, not the row data itself, which stays
deferred to first use like every other :class:`~mixle.data.core.LazySource` connector. Passing an
explicit ``schema=`` skips inference entirely (and keeps even that metadata read lazy).

The inferred mapping covers Arrow's common scalar types: signed/unsigned integers -> ``Count``,
``float16``/``float32``/``float64`` -> ``Real``, ``bool`` -> ``Boolean``, ``string``/``large_string`` ->
``Text``, dictionary-encoded columns -> ``Categorical`` (values only -- the fixed category set itself
is not reconstructed), and a nullable Arrow field -> ``Optional(...)``. Arrow types with no Mixle
counterpart (nested/list/struct, timestamp/date, binary, decimal, ...) raise ``TypeError`` during
inference rather than silently guessing a type or dropping schema-wide to unvalidated -- pass an
explicit ``schema=`` for tables with such columns.
"""

from __future__ import annotations

from typing import Any

from mixle.data.core import LazySource
from mixle.data.schema import Boolean, Categorical, Count, Field, Optional, Real, Schema, Text
from mixle.data.structure import EXCHANGEABLE, SampleStructure

try:  # optional dependency
    import pyarrow.dataset as _pa_dataset
    import pyarrow.ipc as _ipc
    import pyarrow.parquet as _pq
    import pyarrow.types as _pat
except ImportError:  # pragma: no cover - exercised only without pyarrow
    _pq = None
    _pa_dataset = None
    _ipc = None
    _pat = None


def _require_arrow() -> None:
    if _pq is None:
        from mixle.utils.optional_deps import require

        require("pyarrow", "arrow")


def _table_records(table: Any, columns: list[str] | None) -> list[Any]:
    cols = columns if columns is not None else table.column_names
    pydict = table.select(cols).to_pydict()
    rows = zip(*(pydict[c] for c in cols))
    return [r[0] if len(cols) == 1 else r for r in rows]


def _arrow_field_type(arrow_type: Any) -> Any:
    """Map one Arrow scalar/dictionary type to its Mixle :class:`~mixle.data.schema.FieldType`.

    Deliberately narrow, matching ``schema.py``'s own closed type vocabulary: signed/unsigned integers
    (``int8``..``int64``, ``uint8``..``uint64``) -> :class:`~mixle.data.schema.Count`;
    ``float16``/``float32``/``float64`` -> :class:`~mixle.data.schema.Real`; ``bool`` ->
    :class:`~mixle.data.schema.Boolean`; ``string``/``large_string`` ->
    :class:`~mixle.data.schema.Text`; dictionary-encoded (categorical) columns ->
    :class:`~mixle.data.schema.Categorical`. ``to_pydict()`` already decodes dictionary values to their
    logical labels, so ``Categorical(categories=None)`` correctly validates "this column holds
    categorical labels" without needing to reconstruct the exact fixed vocabulary from the Arrow
    dictionary (that would need per-chunk dictionary-unification bookkeeping for a marginal precision
    gain -- a reasonable follow-up, not required for this to be a correct mapping).

    Anything else (nested/list/struct, timestamp/date, binary, decimal, ...) has no Mixle counterpart
    today. Raising here (instead of silently falling back to a wrong type, or silently dropping schema
    for the whole table) is deliberate: it is exactly the "connectors silently get wrong" failure mode
    ``schema.py`` itself exists to prevent.
    """
    if _pat.is_boolean(arrow_type):
        return Boolean()
    if _pat.is_integer(arrow_type):
        return Count()
    if _pat.is_floating(arrow_type):
        return Real()
    if _pat.is_string(arrow_type) or _pat.is_large_string(arrow_type):
        return Text()
    if _pat.is_dictionary(arrow_type):
        return Categorical()
    raise TypeError(
        "no Mixle logical-type mapping for Arrow type %r; pass an explicit schema=... to read_parquet/"
        "read_feather for tables with this column (nested/list/struct, timestamp/date, binary, and "
        "decimal types are not auto-inferred)" % (arrow_type,)
    )


def _schema_from_arrow(arrow_schema: Any, columns: list[str]) -> Schema:
    """Build a Mixle :class:`Schema` from an Arrow schema's field types + nullability.

    ``columns`` fixes both which fields are included and their order. Callers below always pass the
    exact same resolved column list to this function and to the later ``_table_records`` call, so the
    field order here is guaranteed to match the row-tuple order records are actually emitted in.
    """
    fields = []
    for name in columns:
        arrow_field = arrow_schema.field(name)
        field_type = _arrow_field_type(arrow_field.type)
        if arrow_field.nullable:
            field_type = Optional(field_type)
        fields.append(Field(name, field_type))
    return Schema(tuple(fields))


def read_parquet(
    path: str,
    columns: list[str] | None = None,
    *,
    structure: SampleStructure = EXCHANGEABLE,
    schema: Schema | None = None,
) -> LazySource:
    """Read a Parquet file/dataset into a lazy DataSource of scalar/tuple records.

    When ``schema`` is omitted, it is inferred from the file/dataset's own Arrow schema via
    ``pyarrow.dataset`` (a footer-only read that resolves a directory of partition files the same way
    ``pyarrow.parquet.read_table`` does; the row data itself is still not read until the source is
    materialized). Pass an explicit ``schema`` to skip inference -- required for files whose columns
    fall outside the supported mapping (see :func:`_arrow_field_type`).
    """
    resolved_columns = columns
    if schema is None:
        _require_arrow()
        arrow_schema = _pa_dataset.dataset(path, format="parquet").schema
        if resolved_columns is None:
            resolved_columns = list(arrow_schema.names)
        schema = _schema_from_arrow(arrow_schema, resolved_columns)

    def factory():
        _require_arrow()
        return _table_records(_pq.read_table(path, columns=resolved_columns), resolved_columns)

    return LazySource(factory, structure, schema)


def read_feather(
    path: str,
    columns: list[str] | None = None,
    *,
    structure: SampleStructure = EXCHANGEABLE,
    schema: Schema | None = None,
) -> LazySource:
    """Read an Arrow/Feather file into a lazy DataSource of scalar/tuple records.

    When ``schema`` is omitted, it is inferred from the file's own Arrow schema via ``pyarrow.ipc`` (a
    footer-only read; the row data itself is still not read until the source is materialized). Pass an
    explicit ``schema`` to skip inference -- required for files whose columns fall outside the
    supported mapping (see :func:`_arrow_field_type`).
    """
    resolved_columns = columns
    if schema is None:
        _require_arrow()
        with _ipc.open_file(path) as reader:
            arrow_schema = reader.schema
        if resolved_columns is None:
            resolved_columns = list(arrow_schema.names)
        schema = _schema_from_arrow(arrow_schema, resolved_columns)

    def factory():
        _require_arrow()
        # pyarrow.feather.read_table is deprecated as of 24.0.0 (Feather V2 *is* the Arrow IPC file
        # format) -- read via the IPC reader directly instead. _table_records already applies its own
        # `columns` selection, so no separate projection step is needed here.
        with _ipc.open_file(path) as reader:
            table = reader.read_all()
        return _table_records(table, resolved_columns)

    return LazySource(factory, structure, schema)
