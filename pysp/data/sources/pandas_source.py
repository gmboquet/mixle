"""Pandas DataFrame -> :class:`~pysp.data.core.DataSource` (reuses the duck-typed DataFrame adapter).

The adapter itself never imports pandas; you only need the ``pandas`` extra to *construct* the DataFrame
you pass in. This funnels a DataFrame into the same encoder contract as every other source.
"""

from __future__ import annotations

from typing import Any

from pysp.data.core import MaterializedSource
from pysp.data.dataframe import dataframe_records
from pysp.data.schema import Schema
from pysp.data.structure import EXCHANGEABLE, SampleStructure


def read_dataframe(df: Any, fields: Any = None, *, as_dict: bool = False,
                   structure: SampleStructure = EXCHANGEABLE, schema: Schema | None = None) -> MaterializedSource:
    """Wrap a pandas DataFrame's selected columns as a DataSource (scalar/tuple/dict records)."""
    return MaterializedSource(dataframe_records(df, fields=fields, as_dict=as_dict), structure, schema)
