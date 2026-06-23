"""Text connectors -- CSV / JSON / JSONL (NDJSON) into a :class:`~pysp.data.core.DataSource`.

Pure standard library: usable in the base install with no extra. A :class:`~pysp.data.schema.Schema`
turns the raw strings into the Python types encoders expect (CSV columns are all strings until coerced).
"""

from __future__ import annotations

import csv
import json
from typing import Any

from pysp.data.core import LazySource
from pysp.data.schema import Schema
from pysp.data.structure import EXCHANGEABLE, SampleStructure


def _select(values: list[Any], columns: list[int] | None) -> Any:
    picked = values if columns is None else [values[i] for i in columns]
    return picked[0] if len(picked) == 1 else tuple(picked)


def read_csv(
    path: str,
    columns: list[str] | None = None,
    *,
    structure: SampleStructure = EXCHANGEABLE,
    schema: Schema | None = None,
) -> LazySource:
    """Read a CSV file (header row required) into a lazy DataSource of scalar/tuple records."""

    def factory():
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            idx = [header.index(c) for c in columns] if columns is not None else None
            for row in reader:
                yield _select(row, idx)

    return LazySource(factory, structure, schema)


def read_jsonl(
    path: str,
    columns: list[str] | None = None,
    *,
    structure: SampleStructure = EXCHANGEABLE,
    schema: Schema | None = None,
) -> LazySource:
    """Read NDJSON / JSONL (one JSON object per line) into a lazy DataSource of dict/tuple records."""

    def factory():
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield obj if columns is None else _select([obj[c] for c in columns], None)

    return LazySource(factory, structure, schema)


def read_json(
    path: str,
    columns: list[str] | None = None,
    *,
    structure: SampleStructure = EXCHANGEABLE,
    schema: Schema | None = None,
) -> LazySource:
    """Read a JSON file holding an array of record objects into a lazy DataSource."""

    def factory():
        with open(path) as fh:
            for obj in json.load(fh):
                yield obj if columns is None else _select([obj[c] for c in columns], None)

    return LazySource(factory, structure, schema)
