"""Pandas DataFrame connector + adapters -- DataFrame columns into the sequence-encoded stats API.

The adapter never imports pandas (it duck-types ``df.columns``/``.loc``/``.itertuples``); the ``pandas``
extra is only needed to *construct* the DataFrame you pass in. ``read_dataframe`` wraps a DataFrame as a
:class:`~mixle.data.core.DataSource` so it funnels into the same encoder contract as every other source.

**Missing cells are canonicalized by column KIND, not by pandas dtype backend.** A gap in a numeric
column becomes ``NaN``; a gap in any other column (string, categorical, boolean, datetime, object)
becomes ``None``. That holds however pandas spelled the gap -- ``NaN``, ``pd.NA`` (nullable
extension dtypes) or ``pd.NaT`` (datetime/timedelta) -- so ``df`` and ``df.convert_dtypes()``
put the SAME missing marker in every cell, the ``OptionalDistribution`` fitted from either frame
carries the same ``missing_value``, and each model scores the other frame. See
:func:`dataframe_records`.

What this does NOT promise is that the two frames' *present* values are spelled alike: a
``float64`` column of whole numbers carries ``1.0`` where its ``convert_dtypes()`` ``Int64`` twin
carries ``1``, and auto-inference reads those as different data (a continuous family versus a
count family). That is a property of the dtype the caller chose, not of the missing marker, it
predates this canonicalization, and both models still accept both frames' records.
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
# continuous family with a missing-value probability. So pandas' sentinels are rewritten to mixle's
# own missing marker at the adapter boundary -- the one place a DataFrame becomes observation
# records. Which marker that is depends on the column's KIND, never on the dtype backend: see
# ``_column_gap_plan``.
_PANDAS_MISSING_TYPE_NAMES = ("NAType", "NaTType")
_PANDAS_MISSING_TYPES: frozenset[type] = frozenset()
# numpy-backed columns can only carry a pandas sentinel when they hold Python objects ('O') or are
# a datetime64/timedelta64 column, whose missing value IS ``pd.NaT`` ('M'/'m'). A numpy
# float/int/bool/complex column cannot hold one at all, which is the overwhelmingly common case and
# the one worth not walking in Python.
_SENTINEL_CAPABLE_KINDS = frozenset("OMm")
# Column kinds whose missing marker is ``NaN`` rather than ``None``: signed int, unsigned int and
# float. Every pandas dtype -- numpy, masked extension ("Float64"/"Int64") and pyarrow-backed --
# answers ``.kind`` with the same one-letter code, so this needs no pandas import and no dtype
# allow-list to maintain.
_NUMERIC_DTYPE_KINDS = frozenset("iuf")
_NAN = float("nan")
# The plan for a column whose contents the adapter cannot characterize: map pandas' own sentinels
# to ``None`` (the pre-0.8.0 behaviour) and rewrite nothing else. Also the no-op plan for a column
# that needs no rewriting at all.
_LEGACY_GAP_PLAN: tuple[Any, bool, bool] = (None, False, False)


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


def _is_nan(value: Any) -> bool:
    """Report whether ``value`` is a float NaN (pandas' gap marker for numpy-backed columns)."""
    return isinstance(value, (float, np.floating)) and value != value


def _object_column_is_numeric(column: Any) -> bool:
    """Report whether an object-dtype column holds numbers (so its gaps are numeric gaps).

    ``object`` is the one dtype whose ``kind`` says nothing about the data: it carries strings,
    tuples, ``Decimal``, and -- from JSON, from ``astype(object)``, from a frame assembled row by
    row -- plain numbers. Deciding the marker on the letter ``'O'`` alone would give the same
    numbers a different missing marker depending only on whether they were stored ``float64`` or
    ``object``, which is the very dtype dependence :func:`_column_gap_plan` exists to remove. Bools
    are excluded deliberately: ``bool`` is an ``int`` subclass, but a boolean column's numpy-backed
    gap is ``object``/``None``, not ``NaN``.

    Present values are scanned until one is not a number, so a string column costs one comparison.
    An empty or all-missing column has nothing to judge and is reported non-numeric, which keeps
    ``None`` -- the more conservative marker -- for a column whose contents nobody has seen.
    """
    missing_types = _pandas_missing_types()
    try:
        values = iter(column)
    except TypeError:
        return False
    seen_present = False
    for value in values:
        if value is None or type(value) in missing_types or _is_nan(value):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            return False
        seen_present = True
    return seen_present


def _column_gap_plan(column: Any) -> tuple[Any, bool, bool]:
    """Return ``(marker, rewrite_nan, rewrite_none)``: how this column's gaps are canonicalized.

    The marker follows what the column HOLDS, never the dtype backend that stored it. numpy-backed
    pandas has no choice about this and defines the convention: a numeric column with a gap is
    ``float64`` holding ``NaN`` (an integer column with a gap is *upcast* to ``float64``/``NaN``,
    and a boolean column with a gap falls back to ``object``/``None``). The nullable extension and
    pyarrow dtypes spell that very same gap ``pd.NA``, a datetime/timedelta column spells it
    ``pd.NaT``, and pandas 3's default ``str`` dtype spells a *string* column's gap ``NaN``. So the
    two rewrite flags are needed in both directions: a numeric column's ``None`` becomes ``NaN``,
    and a non-numeric column's ``NaN`` becomes ``None``.

    Canonicalizing on anything narrower made the *fitted* sentinel a function of the caller's dtype
    backend, and both halves of that were real. Rewriting only pandas' own sentinels, always to
    ``None``, meant a column stored ``float64`` fitted ``OptionalDistribution(...,
    missing_value=nan)`` while the identical column stored ``Float64`` fitted
    ``missing_value=None``; the Optional path compares sentinels by identity, so neither model
    could score the other spelling and a ``convert_dtypes()`` call or a parquet round trip between
    fit and serve rejected the whole batch, in both directions, with a loud ContractError
    (campaign four, T2-02). Rewriting numeric gaps only left the mirror-image defect on the string
    side, where it is SILENT: under pandas 3 a plain string column's gap is ``NaN`` while its
    ``convert_dtypes()`` twin's is ``pd.NA``, so one frame fitted ``missing_value=nan`` and the
    other ``missing_value=None``, and cross-scoring returned ``mean_log_density = -inf`` for the
    whole batch with no error at all. One rule, applied to every column kind at once, is what makes
    ``df`` and ``df.convert_dtypes()`` put the same missing marker in every cell, and therefore
    what makes each frame's fitted model score the other. (Their *present* values can still differ
    -- ``float64`` ``1.0`` versus ``Int64`` ``1`` -- which is the caller's dtype choice showing
    through, not this marker rule; see the module docstring.)

    A column that does not answer ``dtype.kind`` (a duck-typed frame) gets the legacy plan: map
    pandas' own sentinels to ``None`` and touch nothing else. Guessing a marker for a column whose
    contents cannot be characterized would rewrite values the adapter cannot vouch for.
    """
    kind = getattr(getattr(column, "dtype", None), "kind", None)
    if not isinstance(kind, str):
        return _LEGACY_GAP_PLAN
    if kind in _NUMERIC_DTYPE_KINDS or (kind == "O" and _object_column_is_numeric(column)):
        return (_NAN, False, True)
    return (None, True, False)


def _column_may_hold_gap(column: Any, plan: tuple[Any, bool, bool]) -> bool:
    """Report whether a column needs the Python-level rewrite pass, without walking it.

    Three cheap gates, the last of which fails open for a duck-typed frame that does not answer it:
    an object column always walks (an object cell may nest a tuple/list carrying a sentinel, which
    the per-cell ``isna()`` cannot see); a numpy-backed column that cannot hold a gap at all, or
    whose only possible gap is already this column's marker (a ``float64``/``int64`` column, the
    overwhelmingly common case), is skipped outright; anything else asks the vectorized ``isna()``,
    so a nullable column with nothing missing needs no rewriting either. Without these the pass
    cost ~1s per million rows on ordinary float64 columns -- ~35x the ``tolist()`` it follows.
    """
    marker, _rewrite_nan, _rewrite_none = plan
    dtype = getattr(column, "dtype", None)
    if isinstance(dtype, np.dtype):
        if dtype.kind == "O":
            return True
        if dtype.kind in _NUMERIC_DTYPE_KINDS:
            # numpy numeric: the only gap it can store is NaN, and NaN is this column's marker.
            return not _is_nan(marker) and marker is not None
        if dtype.kind not in _SENTINEL_CAPABLE_KINDS:
            return False
    isna = getattr(column, "isna", None)
    if callable(isna):
        try:
            return bool(isna().any())
        except Exception:  # noqa: BLE001 - a duck-typed column need not implement isna()
            return True
    return True


def _selection_gap_plans(df: Any, source_list: Sequence[Any]) -> list[tuple[Any, bool, bool]] | None:
    """Per-column gap plans for the selection, or ``None`` when no column needs rewriting.

    ``None`` means "skip the rewrite pass entirely" -- the common all-numpy-numeric frame, where
    every gap is already the marker its column would canonicalize to. Otherwise the returned list
    is positionally aligned with ``source_list``, and a column that needs no rewriting carries the
    legacy no-op plan so the pass leaves its values alone. A duck-typed frame that refuses scalar
    column access fails open with that same legacy plan, exactly as the previous boolean gate did.
    """
    sentinels = bool(_pandas_missing_types())
    plans: list[tuple[Any, bool, bool]] = []
    needed = False
    for name in source_list:
        try:
            column = df[name]
        except Exception:  # noqa: BLE001 - a duck-typed frame need not support scalar column access
            plans.append(_LEGACY_GAP_PLAN)
            needed = needed or sentinels
            continue
        plan = _column_gap_plan(column)
        if (sentinels or plan[1] or plan[2]) and _column_may_hold_gap(column, plan):
            plans.append(plan)
            needed = True
        else:
            plans.append(_LEGACY_GAP_PLAN)
    return plans if needed else None


def _replace_missing(
    value: Any,
    missing_types: frozenset[type],
    marker: Any = None,
    rewrite_nan: bool = False,
    rewrite_none: bool = False,
) -> Any:
    if type(value) in missing_types:
        return marker
    if rewrite_nan and _is_nan(value):
        return marker
    if rewrite_none and value is None:
        return marker
    # Containers are normalized for pandas' own sentinels only. A cell that is itself a tuple/list
    # of numbers carries its own NaNs, and those are the wrapped field's data (a missing coordinate
    # inside a vector observation), not this column's gap: rewriting them to ``None`` on the
    # strength of the OUTER column's dtype would change values the adapter was never asked about.
    if isinstance(value, tuple):
        replaced = tuple(_replace_missing(item, missing_types, marker) for item in value)
        return value if all(new is old for new, old in zip(replaced, value)) else replaced
    if isinstance(value, Mapping):
        replaced = {key: _replace_missing(item, missing_types, marker) for key, item in value.items()}
        return value if all(replaced[key] is item for key, item in value.items()) else replaced
    if isinstance(value, list):
        replaced = [_replace_missing(item, missing_types, marker) for item in value]
        return value if all(new is old for new, old in zip(replaced, value)) else replaced
    return value


def normalize_pandas_missing(value: Any, marker: Any = None) -> Any:
    """Return ``value`` with pandas' own missing sentinels replaced by ``marker``.

    ``pd.NA`` and ``pd.NaT`` are pandas' missing markers. mixle has two spellings of missing --
    ``None``, and ``NaN`` for numeric data -- and this value-level entry point cannot see which
    column a value came from, so it defaults to ``None``. Callers that DO know the column go
    through :func:`dataframe_records` or :func:`column_records` instead, which canonicalize every
    gap spelling onto the one marker that column's kind determines (``NaN`` for numbers, ``None``
    otherwise) rather than rewriting pandas' sentinels alone. Anything else -- including ``NaN``
    itself, which the profiler already reads as missing -- is returned unchanged, and an unchanged
    container is returned as the same object rather than a copy. Scalars, tuples, lists, and
    mappings are handled, so record rows of any adapter shape can be normalized with one call.
    """
    missing_types = _pandas_missing_types()
    if not missing_types:
        return value
    return _replace_missing(value, missing_types, marker)


def flat_gap_marker(values: Any) -> Any:
    """Return the marker a bare flat sequence of scalars should canonicalize its gaps to.

    :func:`normalize_pandas_missing` cannot see past one value at a time, so it always defaults to
    ``None`` -- correct for a genuinely mixed/non-numeric sequence, but wrong for a numeric one: a
    plain Python list built from ``list(a_nullable_series)`` (``optimize()`` and the encode side
    both fall back to exactly that per-value normalization once the data is no longer a Series or
    DataFrame, so the column's own dtype has already been lost) then fits ``missing_value=None``
    while the identical data still stored as a ``Series`` fits ``missing_value=nan`` via
    :func:`column_records` -- and unlike the Series-vs-Series case (T2-02), a model built either
    way ends up unable to score/re-encode the very ``pd.NA`` values it was fit from, because
    ``pd.NA`` is only sentinel-equivalent to a ``NaN`` missing_value, never to ``None`` (T1-02).

    This makes the same numeric-vs-not judgment :func:`_column_gap_plan` makes from a column's
    ``dtype.kind``, but from the sequence's own present values instead, since there is no dtype
    left to consult. A sequence whose present values are all numbers gets ``NaN``; anything else
    (non-numeric present values, or nothing present to judge) gets ``None``, same as today.
    """
    if not _pandas_missing_types():
        return None
    return _NAN if _object_column_is_numeric(values) else None


def _normalized_scalars(values: list[Any], plan: tuple[Any, bool, bool]) -> list[Any]:
    """Canonicalize the gaps of a single-field record list onto that column's marker."""
    marker, rewrite_nan, rewrite_none = plan
    missing_types = _pandas_missing_types()
    if not missing_types and not rewrite_nan and not rewrite_none:
        return values
    return [_replace_missing(value, missing_types, marker, rewrite_nan, rewrite_none) for value in values]


def _normalized_tuples(rows: list[Any], plans: Sequence[tuple[Any, bool, bool]]) -> list[Any]:
    """Canonicalize gaps across tuple rows, each position taking its own column's plan."""
    missing_types = _pandas_missing_types()
    out = []
    for row in rows:
        replaced = tuple(
            _replace_missing(item, missing_types, plan[0], plan[1], plan[2])
            for item, plan in zip(row, plans, strict=True)
        )
        out.append(row if all(new is old for new, old in zip(replaced, row)) else replaced)
    return out


def _normalized_dicts(rows: list[dict[Any, Any]], plan_by_key: Mapping[Any, tuple[Any, bool, bool]]) -> list[Any]:
    """Canonicalize gaps across mapping rows, each key taking its own column's plan."""
    missing_types = _pandas_missing_types()
    out = []
    for row in rows:
        replaced = {}
        for key, item in row.items():
            marker, rewrite_nan, rewrite_none = plan_by_key.get(key, _LEGACY_GAP_PLAN)
            replaced[key] = _replace_missing(item, missing_types, marker, rewrite_nan, rewrite_none)
        out.append(row if all(replaced[key] is item for key, item in row.items()) else replaced)
    return out


def column_records(column: Any) -> list[Any]:
    """Canonical observation records for ONE pandas column or ``Series``.

    The single-column half of :func:`dataframe_records`: the column's values with every gap
    spelling (``NaN``, ``None``, ``pd.NA``, ``pd.NaT``) collapsed onto the one marker that column's
    kind determines -- ``NaN`` for a column of numbers, ``None`` for everything else. A column
    needing no rewriting is returned as ``tolist()`` gives it, with no extra copy.

    KNOWN GAP, stated rather than implied. A bare ``Series`` reaches the fit verbs on its own
    (``optimize(df["spend"])``) and does NOT yet route through here: that path still rewrites only
    pandas' own sentinels, so the same column fits ``missing_value=nan`` as ``float64`` and
    ``missing_value=None`` as ``Float64`` -- campaign four's T2-02 wearing a different hat, at the
    Series container instead of the frame. Closing it means changing the profiler's and the encoder's
    Series branches together (``mixle.utils.automatic.profiling.normalize_input``,
    ``mixle.inference.estimation._data_records_for_encoding`` and ``mixle.lifecycle._tabular_records``),
    because a model whose sentinel came from one of them and records that came from the other reject
    each other. Until then, ``optimize(column_records(s))`` is the one-call way to fit a Series on
    the same footing as the identical column inside a frame.
    """
    values = column.tolist() if hasattr(column, "tolist") else list(column)
    plan = _column_gap_plan(column)
    sentinels = bool(_pandas_missing_types())
    if not (sentinels or plan[1] or plan[2]) or not _column_may_hold_gap(column, plan):
        return values
    return _normalized_scalars(values, plan)


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

    Missing cells are canonicalized **by what the column holds, not by its dtype backend**: a gap
    in a column of numbers (any int/uint/float dtype -- numpy, nullable or pyarrow -- and an
    ``object`` column whose present values are all numbers) becomes ``NaN``; a gap in any other
    column becomes ``None``. So ``NaN``, ``None``, ``pd.NA`` and ``pd.NaT`` all collapse onto the
    one marker that column's numpy-backed spelling would have produced; ``df`` and
    ``df.convert_dtypes()`` mark every gap the same way, and the ``OptionalDistribution`` fitted
    from either frame carries the same ``missing_value`` and scores both (campaign four, T2-02).
    Present values are NOT respelled -- an ``Int64`` column still yields ``int`` where its
    ``float64`` twin yields ``float`` -- so the two frames can still select different families.

    That covers both directions of the same defect: the numeric one was loud (a ContractError on
    the whole batch), the string one silent (pandas 3 spells a plain ``str`` column's gap ``NaN``
    and its nullable twin's ``pd.NA``, and the mismatched sentinel scored ``-inf`` per row with no
    error). Gaps nested INSIDE a cell -- a tuple/list/mapping observation -- are left as the caller
    wrote them apart from pandas' own sentinels; they are the wrapped field's data, not this
    column's gap.
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

    plans = _selection_gap_plans(df, source_list)

    if as_dict:
        dict_keys = logical_list if _dict_keys == "logical" else source_list
        rows = []
        for row in df.loc[:, source_list].itertuples(index=False, name=None):
            rows.append({name: value for name, value in zip(dict_keys, row)})
        if plans is None:
            return rows
        return _normalized_dicts(rows, dict(zip(dict_keys, plans, strict=True)))

    if len(field_list) == 1:
        values = df[source_list[0]].tolist()
        return values if plans is None else _normalized_scalars(values, plans[0])

    rows = list(df.loc[:, source_list].itertuples(index=False, name=None))
    return rows if plans is None else _normalized_tuples(rows, plans)


def seq_encode_dataframe(
    df: Any,
    fields: FieldSpec = None,
    encoder: DataSequenceEncoder | None = None,
    estimator: ParameterEstimator | None = None,
    model: SequenceEncodableProbabilityDistribution | None = None,
    num_chunks: int = 1,
    chunk_size: int | None = None,
):
    """Sequence-encode selected DataFrame columns with the ordinary stats API.

    The frame-shaped counterpart of :func:`mixle.stats.seq_encode`, which consumes RECORDS and
    refuses a frame outright ("expected a sequence of 2-tuples, got DataFrame"). ``optimize``/
    ``fit`` accept a DataFrame, so the two entry points disagree about what a dataset is and the
    conversion is the caller's job: either encode the frame here, or convert it once with
    :func:`dataframe_records` and hand the records to ``seq_encode`` yourself. Gaps are
    canonicalized exactly as :func:`dataframe_records` canonicalizes them.
    """
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
