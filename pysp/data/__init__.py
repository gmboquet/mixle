"""Data layer for pysparkplug -- a typed, structure-aware way to get records into the encoder contract.

The core abstraction is :class:`~pysp.data.core.DataSource`: a lazy, typed (:class:`~pysp.data.schema.Schema`),
structured (:class:`~pysp.data.structure.SampleStructure`) reference to data. It is purely additive --
``seq_encode(list)`` and ``seq_encode(rdd)`` are unchanged; a ``DataSource`` is simply a third accepted
input that funnels into the same encoder contract. The DataFrame / graph / RDD adapters below remain
input/representation helpers (not probability distributions), so they live outside ``pysp.stats``.
"""

from pysp.data.core import DataSource, LazySource, MaterializedSource, as_source
from pysp.data.dataframe import dataframe_records, seq_encode_dataframe
from pysp.data.graph_data import GraphDataEncoder, GraphObservation
from pysp.data.rdd_sampler import sample_rdd, sample_seq_as_rdd, take_sample
from pysp.data.schema import (
    Boolean,
    Categorical,
    Count,
    Field,
    FieldType,
    Nested,
    Optional,
    Real,
    Schema,
    Text,
    Timestamp,
    Vector,
)
from pysp.data.sources import kinds as source_kinds
from pysp.data.sources import open as open_source
from pysp.data.structure import EXCHANGEABLE, IID, SEQUENTIAL, SampleStructure, partially_exchangeable

__all__ = [
    # core abstraction
    "DataSource",
    "MaterializedSource",
    "LazySource",
    "as_source",
    "open_source",
    "source_kinds",
    # schema / logical types
    "Schema",
    "Field",
    "FieldType",
    "Real",
    "Count",
    "Categorical",
    "Boolean",
    "Vector",
    "Timestamp",
    "Text",
    "Optional",
    "Nested",
    # sample structure
    "SampleStructure",
    "IID",
    "EXCHANGEABLE",
    "SEQUENTIAL",
    "partially_exchangeable",
    # input/representation adapters
    "GraphDataEncoder",
    "GraphObservation",
    "dataframe_records",
    "seq_encode_dataframe",
    "sample_rdd",
    "sample_seq_as_rdd",
    "take_sample",
]
