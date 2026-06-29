"""Data layer for mixle -- a typed, structure-aware way to get records into the encoder contract.

The core abstraction is :class:`~mixle.data.core.DataSource`: a lazy, typed (:class:`~mixle.data.schema.Schema`),
structured (:class:`~mixle.data.structure.SampleStructure`) reference to data. It is purely additive --
``seq_encode(list)`` and ``seq_encode(rdd)`` are unchanged; a ``DataSource`` is simply a third accepted
input that funnels into the same encoder contract. The DataFrame / graph / RDD adapters below remain
input/representation helpers (not probability distributions), so they live outside ``mixle.stats``.
"""

from mixle.data.core import DataSource, LazySource, MaterializedSource, as_source
from mixle.data.encoded_io import load_encoded, save_encoded
from mixle.data.hashing import dataset_hash, model_hash
from mixle.data.schema import (
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
from mixle.data.sources import kinds as source_kinds
from mixle.data.sources import open as open_source
from mixle.data.structure import EXCHANGEABLE, IID, SEQUENTIAL, SampleStructure, partially_exchangeable
from mixle.data.validate import DataReport, check_dataset

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
    # reproducibility: hashing, validation, encoded-data serialization
    "dataset_hash",
    "model_hash",
    "check_dataset",
    "DataReport",
    "save_encoded",
    "load_encoded",
    # input/representation adapters
    "GraphDataEncoder",
    "GraphObservation",
    "dataframe_records",
    "seq_encode_dataframe",
    "sample_rdd",
    "sample_seq_as_rdd",
    "take_sample",
]

# The input/representation adapters are imported lazily. ``graph_source`` defines GraphDataEncoder,
# whose base class (DataSequenceEncoder) lives under ``mixle.stats``, while mixle.stats's graph
# distributions import GraphDataEncoder back from here. Importing graph_source eagerly at package
# load therefore deadlocks whenever ``mixle.data`` is imported before ``mixle.stats``. Deferring the
# import -- and fully loading mixle.stats first for the graph adapter, which pulls graph_source in
# cleanly as a side effect -- breaks the cycle without forcing an import order on callers.
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mixle.data.sources.graph_source import GraphDataEncoder, GraphObservation
    from mixle.data.sources.pandas_source import dataframe_records, seq_encode_dataframe
    from mixle.data.sources.spark_source import sample_rdd, sample_seq_as_rdd, take_sample

# name -> (defining module, needs mixle.stats loaded first)
_LAZY_ADAPTERS = {
    "GraphDataEncoder": ("mixle.data.sources.graph_source", True),
    "GraphObservation": ("mixle.data.sources.graph_source", True),
    "dataframe_records": ("mixle.data.sources.pandas_source", False),
    "seq_encode_dataframe": ("mixle.data.sources.pandas_source", False),
    "sample_rdd": ("mixle.data.sources.spark_source", False),
    "sample_seq_as_rdd": ("mixle.data.sources.spark_source", False),
    "take_sample": ("mixle.data.sources.spark_source", False),
}


def __getattr__(name):
    entry = _LAZY_ADAPTERS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, needs_stats = entry
    if needs_stats:
        importlib.import_module("mixle.stats")  # loads graph_source cleanly; see note above
    return getattr(importlib.import_module(module_path), name)


def __dir__():
    return sorted(__all__)
