"""Pandas DataFrame adapters for the sequence-encoded stats API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pysp.stats.pdist import DataSequenceEncoder, ParameterEstimator, SequenceEncodableProbabilityDistribution

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
    from pysp.stats.record import RecordDistribution, RecordEstimator

    if fields is None and model is not None and isinstance(model, RecordDistribution):
        fields = tuple(zip(model.fields, model.sources))
    elif fields is None and estimator is not None and isinstance(estimator, RecordEstimator):
        fields = tuple(zip(estimator.fields, estimator.sources))
    as_dict = isinstance(model, RecordDistribution) or isinstance(estimator, RecordEstimator)
    records = dataframe_records(df, fields=fields, as_dict=as_dict)
    return seq_encode(
        records, encoder=encoder, estimator=estimator, model=model, num_chunks=num_chunks, chunk_size=chunk_size
    )
