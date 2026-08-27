"""Pandas DataFrame connector + adapters -- DataFrame columns into the sequence-encoded stats API.

The adapter never imports pandas (it duck-types ``df.columns``/``.loc``/``.itertuples``); the ``pandas``
extra is only needed to *construct* the DataFrame you pass in. ``read_dataframe`` wraps a DataFrame as a
:class:`~mixle.data.core.DataSource` so it funnels into the same encoder contract as every other source.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mixle.data.core import MaterializedSource
from mixle.data.schema import Schema
from mixle.data.structure import EXCHANGEABLE, SampleStructure
from mixle.stats.compute.pdist import DataSequenceEncoder, ParameterEstimator, SequenceEncodableProbabilityDistribution

FieldSpec = str | Sequence[Any] | None

# pandas ships two missing sentinels of its own -- ``pd.NA`` (every nullable extension dtype:
# Float64/Int64/boolean/string, i.e. what ``convert_dtypes()``, ``dtype_backend="numpy_nullable"``
# and a parquet round trip produce) and ``pd.NaT`` (datetime/timedelta). Neither is ``None`` and
# neither is a float, so both used to reach the profiler as ordinary *values*: a nullable numeric
# column was silently modeled as a categorical memorizer with ``<NA>`` as one more level, and every
# unseen number then scored ``-inf``. The same column spelled with ``NaN``/``None`` fits a
# continuous family with a missing-value probability. mixle's missing marker is ``None`` (the
# Optional path keys on it), so pandas' sentinels are mapped onto it at the adapter boundary --
# the one place a DataFrame becomes observation records.
_PANDAS_MISSING_TYPE_NAMES = ("NAType", "NaTType")
_PANDAS_MISSING_TYPES: frozenset[type] = frozenset()
# numpy-backed columns can only carry a pandas sentinel when they hold Python objects ('O') or are
# a datetime64/timedelta64 column, whose missing value IS ``pd.NaT`` ('M'/'m'). A numpy
# float/int/bool/complex column cannot hold one at all, which is the overwhelmingly common case and
# the one worth not walking in Python.
_SENTINEL_CAPABLE_KINDS = frozenset("OMm")


def _pandas_missing_types() -> frozenset[type]:
    """Return pandas' missing-sentinel types, or an empty set when pandas is not loaded.

    The adapter never imports pandas (see the module docstring) -- it reads an *already imported*
    module out of ``sys.modules``. If pandas was never imported, no ``pd.NA``/``pd.NaT`` value can
    exist in the caller's records, so there is nothing to normalize and the scan is skipped. The
    lookup is cached once pandas is present; while it is absent the check is a single dict get, so
    a later ``import pandas`` is still picked up.
    """
    global _PANDAS_MISSING_TYPES
    if _PANDAS_MISSING_TYPES:
        return _PANDAS_MISSING_TYPES
    pandas = sys.modules.get("pandas")
    if pandas is None:
        return frozenset()
    found = set()
    for attribute in ("NA", "NaT"):
        sentinel = getattr(pandas, attribute, None)
        if sentinel is not None and type(sentinel).__name__ in _PANDAS_MISSING_TYPE_NAMES:
            found.add(type(sentinel))
    _PANDAS_MISSING_TYPES = frozenset(found)
    return _PANDAS_MISSING_TYPES


def _column_may_hold_missing(column: Any) -> bool:
    """Report whether a column could contain ``pd.NA``/``pd.NaT``, without walking it in Python.

    Two cheap gates, both of which fail open for a duck-typed frame that does not answer them: the
    column's dtype (a numpy float/int/bool column cannot hold a pandas sentinel at all), then a
    vectorized ``isna()`` (a nullable column with nothing missing needs no rewriting). Without them
    the normalization pass cost ~1s per million rows on ordinary float64 columns -- ~35x the
    ``tolist()`` it follows.
    """
    dtype = getattr(column, "dtype", None)
    if isinstance(dtype, np.dtype):
        if dtype.kind not in _SENTINEL_CAPABLE_KINDS:
            return False
        if dtype.kind == "O":
            # An object cell may itself be a tuple/list carrying a sentinel, which ``isna()`` --
            # a per-cell test -- cannot see. Object columns are rare and already the slow path.
            return True
    isna = getattr(column, "isna", None)
    if callable(isna):
        try:
            return bool(isna().any())
        except Exception:  # noqa: BLE001 - a duck-typed column need not implement isna()
            return True
    return True


def _selection_may_hold_missing(df: Any, source_list: Sequence[Any]) -> bool:
    """Report whether any selected column could contain a pandas missing sentinel."""
    if not _pandas_missing_types():
        return False
    for name in source_list:
        try:
            column = df[name]
        except Exception:  # noqa: BLE001 - a duck-typed frame need not support scalar column access
            return True
        if _column_may_hold_missing(column):
            return True
    return False


def _replace_missing(value: Any, missing_types: frozenset[type]) -> Any:
    if type(value) in missing_types:
        return None
    if isinstance(value, tuple):
        replaced = tuple(_replace_missing(item, missing_types) for item in value)
        return value if all(new is old for new, old in zip(replaced, value)) else replaced
    if isinstance(value, Mapping):
        replaced = {key: _replace_missing(item, missing_types) for key, item in value.items()}
        return value if all(replaced[key] is item for key, item in value.items()) else replaced
    if isinstance(value, list):
        replaced = [_replace_missing(item, missing_types) for item in value]
        return value if all(new is old for new, old in zip(replaced, value)) else replaced
    return value


def normalize_pandas_missing(value: Any) -> Any:
    """Return ``value`` with pandas' own missing sentinels replaced by ``None``.

    ``pd.NA`` and ``pd.NaT`` are pandas' missing markers; mixle's is ``None`` (``NaN`` is also
    understood for float data). Anything else -- including ``NaN`` itself, which the profiler
    already reads as missing -- is returned unchanged, and an unchanged container is returned as the
    same object rather than a copy. Scalars, tuples, lists, and mappings are handled, so record
    rows of any adapter shape can be normalized with one call.
    """
    missing_types = _pandas_missing_types()
    if not missing_types:
        return value
    return _replace_missing(value, missing_types)


def _normalized_records(rows: list[Any], needed: bool) -> list[Any]:
    """Map pandas' missing sentinels onto ``None`` across a freshly built record list."""
    missing_types = _pandas_missing_types()
    if not needed or not missing_types:
        return rows
    return [_replace_missing(row, missing_types) for row in rows]


def _field_source(field: Any) -> Any:
    if isinstance(field, tuple) and len(field) == 2:
        return field[1]
    return field


def _field_name(field: Any) -> Any:
    if isinstance(field, tuple) and len(field) == 2:
        return field[0]
    return field


def dataframe_records(
    df: Any, fields: FieldSpec = None, as_dict: bool = False, *, _dict_keys: str = "logical"
) -> list[Any]:
    """Convert DataFrame columns into observation records for ``seq_encode``.

    A single selected field becomes scalar observations. Multiple selected
    fields become tuple observations in the requested field order, matching the
    data shape expected by composite distributions. When ``as_dict=True``,
    each row is returned as a mapping keyed by the selected source field names.

    pandas' own missing sentinels (``pd.NA`` from any nullable extension dtype, ``pd.NaT`` from a
    datetime/timedelta column) are returned as ``None``, mixle's missing marker, so a nullable
    column fits through the same Optional path as the ``NaN``/``None`` spelling of the same data.
    """
    if fields is None:
        field_list = list(df.columns)
    elif isinstance(fields, str):
        field_list = [fields]
    else:
        field_list = list(fields)

    source_list = [_field_source(name) for name in field_list]
    logical_list = [_field_name(name) for name in field_list]
    if len(set(logical_list)) != len(logical_list):
        raise ValueError(f"logical fields must be unique, got {logical_list!r}")
    missing = [name for name in source_list if name not in df.columns]
    if missing:
        raise KeyError("DataFrame is missing fields: %s" % ", ".join(map(str, missing)))

    if len(field_list) == 0:
        raise ValueError("fields must select at least one DataFrame column.")

    normalize = _selection_may_hold_missing(df, source_list)

    if as_dict:
        dict_keys = logical_list if _dict_keys == "logical" else source_list
        rows = []
        for row in df.loc[:, source_list].itertuples(index=False, name=None):
            rows.append({name: value for name, value in zip(dict_keys, row)})
        return _normalized_records(rows, normalize)

    if len(field_list) == 1:
        return _normalized_records(df[source_list[0]].tolist(), normalize)

    return _normalized_records(list(df.loc[:, source_list].itertuples(index=False, name=None)), normalize)


def seq_encode_dataframe(
    df: Any,
    fields: FieldSpec = None,
    encoder: DataSequenceEncoder | None = None,
    estimator: ParameterEstimator | None = None,
    model: SequenceEncodableProbabilityDistribution | None = None,
    num_chunks: int = 1,
    chunk_size: int | None = None,
):
    """Sequence-encode selected DataFrame columns with the ordinary stats API."""
    from mixle.stats import seq_encode

    def _record_fields_sources(obj: Any) -> tuple[Any, Any] | None:
        """Recover ``(fields, sources)`` from any record-like object via a capability probe.

        Mirrors the duck/``getattr`` probes used in ``mixle.utils.parallel.planner`` -- a record-like model or
        estimator exposes both a ``fields`` and a ``sources`` attribute; anything else is not
        record-shaped and yields ``None``.
        """
        fields_attr = getattr(obj, "fields", None)
        sources_attr = getattr(obj, "sources", None)
        if fields_attr is None or sources_attr is None:
            return None
        return fields_attr, sources_attr

    model_rs = None if model is None else _record_fields_sources(model)
    estimator_rs = None if estimator is None else _record_fields_sources(estimator)

    if fields is None and model_rs is not None:
        fields = tuple(zip(model_rs[0], model_rs[1]))
    elif fields is None and estimator_rs is not None:
        fields = tuple(zip(estimator_rs[0], estimator_rs[1]))
    as_dict = model_rs is not None or estimator_rs is not None
    # Record encoders consume their declared *source* keys. The standalone adapter exposes logical
    # aliases, but this internal bridge retains source-keyed rows for the existing record contract.
    records = dataframe_records(df, fields=fields, as_dict=as_dict, _dict_keys="source")
    return seq_encode(
        records, encoder=encoder, estimator=estimator, model=model, num_chunks=num_chunks, chunk_size=chunk_size
    )


def read_dataframe(
    df: Any,
    fields: FieldSpec = None,
    *,
    as_dict: bool = False,
    structure: SampleStructure = EXCHANGEABLE,
    schema: Schema | None = None,
) -> MaterializedSource:
    """Wrap a pandas DataFrame's selected columns as a DataSource (scalar/tuple/dict records)."""
    return MaterializedSource(dataframe_records(df, fields=fields, as_dict=as_dict), structure, schema)
