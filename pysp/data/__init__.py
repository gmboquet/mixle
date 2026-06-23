"""Data layer for pysparkplug -- a typed, structure-aware way to get records into the encoder contract.

The core abstraction is :class:`~pysp.data.core.DataSource`: a lazy, typed (:class:`~pysp.data.schema.Schema`),
structured (:class:`~pysp.data.structure.SampleStructure`) reference to data. It is purely additive --
``seq_encode(list)`` and ``seq_encode(rdd)`` are unchanged; a ``DataSource`` is simply a third accepted
input that funnels into the same encoder contract. The DataFrame / graph / RDD adapters below remain
input/representation helpers (not probability distributions), so they live outside ``pysp.stats``.
"""

from pysp.data.core import DataSource, LazySource, MaterializedSource, as_source
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

# The input/representation adapters are imported lazily. ``graph_source`` defines GraphDataEncoder,
# whose base class (DataSequenceEncoder) lives under ``pysp.stats``, while pysp.stats's graph
# distributions import GraphDataEncoder back from here. Importing graph_source eagerly at package
# load therefore deadlocks whenever ``pysp.data`` is imported before ``pysp.stats``. Deferring the
# import -- and fully loading pysp.stats first for the graph adapter, which pulls graph_source in
# cleanly as a side effect -- breaks the cycle without forcing an import order on callers.
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pysp.data.sources.graph_source import GraphDataEncoder, GraphObservation
    from pysp.data.sources.pandas_source import dataframe_records, seq_encode_dataframe
    from pysp.data.sources.spark_source import sample_rdd, sample_seq_as_rdd, take_sample

# name -> (defining module, needs pysp.stats loaded first)
_LAZY_ADAPTERS = {
    "GraphDataEncoder": ("pysp.data.sources.graph_source", True),
    "GraphObservation": ("pysp.data.sources.graph_source", True),
    "dataframe_records": ("pysp.data.sources.pandas_source", False),
    "seq_encode_dataframe": ("pysp.data.sources.pandas_source", False),
    "sample_rdd": ("pysp.data.sources.spark_source", False),
    "sample_seq_as_rdd": ("pysp.data.sources.spark_source", False),
    "take_sample": ("pysp.data.sources.spark_source", False),
}


def __getattr__(name):
    entry = _LAZY_ADAPTERS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, needs_stats = entry
    if needs_stats:
        importlib.import_module("pysp.stats")  # loads graph_source cleanly; see note above
    return getattr(importlib.import_module(module_path), name)


def __dir__():
    return sorted(__all__)
