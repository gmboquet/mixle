"""Structure-aware partitioning -- one place that splits a dataset into chunks while honoring its
:class:`~pysp.data.structure.SampleStructure`.

This replaces the inline striding in ``seq_encode`` (``data[i::C]``). For strideable structures
(IID / exchangeable / sequential-records) it reproduces that striding exactly, so the existing fast path
is bit-identical. For ``PARTIALLY_EXCHANGEABLE`` data it strides at the *group* level -- every record of
a group lands in the same partition -- so a hierarchical model never sees a group split across chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pysp.data.structure import EXCHANGEABLE, SampleStructure


def partition_records(records: Sequence[Any], structure: SampleStructure, n: int) -> list[list[Any]]:
    """Split ``records`` into ``n`` partitions respecting ``structure``.

    Strideable structures give ``records[k::n]`` (identical to the historical chunking); a
    partially-exchangeable structure groups by its key and round-robins whole groups across partitions.
    """
    n = max(1, int(n))
    if structure.strides_records:
        return [list(records[k::n]) for k in range(n)]
    groups: dict[Any, list[Any]] = {}
    order: list[Any] = []
    for r in records:
        key = structure.group_key(r)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    parts: list[list[Any]] = [[] for _ in range(n)]
    for i, key in enumerate(order):
        parts[i % n].extend(groups[key])
    return parts


def num_chunks_for(size: int, num_chunks: int = 1, chunk_size: int | None = None) -> int:
    """Resolve the chunk count from an explicit ``num_chunks`` or a target ``chunk_size`` (as seq_encode does)."""
    if chunk_size is not None:
        import math

        return max(1, int(math.ceil(float(size) / float(chunk_size))))
    return max(1, int(num_chunks))


def encode_partitions(records: Sequence[Any], encoder: Any, structure: SampleStructure = EXCHANGEABLE,
                      num_chunks: int = 1, chunk_size: int | None = None) -> list[tuple[int, Any]]:
    """Partition ``records`` by ``structure`` and ``encoder.seq_encode`` each part -> ``[(count, payload)]``."""
    n = num_chunks_for(len(records), num_chunks, chunk_size)
    return [(len(part), encoder.seq_encode(part)) for part in partition_records(records, structure, n)]
