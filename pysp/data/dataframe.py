"""Pandas DataFrame adapters for the sequence-encoded stats API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pysp.stats.compute.pdist import DataSequenceEncoder, ParameterEstimator, SequenceEncodableProbabilityDistribution

FieldSpec = str | Sequence[Any] | None


def _field_source(field: Any) -> Any:
    if isinstance(field, tuple) and len(field) == 2:
        return field[1]
    return field


def dataframe_records(df: Any, fields: FieldSpec = None, as_dict: bool = False) -> list[Any]:
    """Convert DataFrame columns into observation records for ``seq_encode``.

    A single selected field becomes scalar observations. Multiple selected
    fields become tuple observations in the requested field order, matching the
    data shape expected by composite distributions. When ``as_dict=True``,
    each row is returned as a mapping keyed by the selected source field names.
    """
    if fields is None:
        field_list = list(df.columns)
    elif isinstance(fields, str):
        field_list = [fields]
    else:
        field_list = list(fields)

    source_list = [_field_source(name) for name in field_list]
    missing = [name for name in source_list if name not in df.columns]
    if missing:
        raise KeyError("DataFrame is missing fields: %s" % ", ".join(map(str, missing)))

    if len(field_list) == 0:
        raise ValueError("fields must select at least one DataFrame column.")

    if as_dict:
        rows = []
        for row in df.loc[:, source_list].itertuples(index=False, name=None):
            rows.append({name: value for name, value in zip(source_list, row)})
        return rows

    if len(field_list) == 1:
        return df[source_list[0]].tolist()

    return list(df.loc[:, source_list].itertuples(index=False, name=None))


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
    from pysp.stats import seq_encode

    def _record_fields_sources(obj: Any) -> tuple[Any, Any] | None:
        """Recover ``(fields, sources)`` from any record-like object via a capability probe.

        Mirrors the duck/``getattr`` probes used in ``pysp.planner`` -- a record-like model or
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
    records = dataframe_records(df, fields=fields, as_dict=as_dict)
    return seq_encode(
        records, encoder=encoder, estimator=estimator, model=model, num_chunks=num_chunks, chunk_size=chunk_size
    )
