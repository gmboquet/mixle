"""Structure-aware partitioning -- one place that splits a dataset into chunks while honoring its
:class:`~mixle.data.structure.SampleStructure`.

This replaces the inline striding in ``seq_encode`` (``data[i::C]``). For strideable structures
(IID / exchangeable / sequential-records) it reproduces that striding exactly, so the existing fast path
is bit-identical. For ``PARTIALLY_EXCHANGEABLE`` data it strides at the *group* level -- every record of
a group lands in the same partition -- so a hierarchical model never sees a group split across chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mixle.data.structure import EXCHANGEABLE, SampleStructure


def _positive_int(name: str, value: Any) -> int:
    """Validate that ``value`` is an exact positive ``int`` (rejects ``bool``, other types, and
    nonpositive/fractional values) and return it. Used to fail partition controls clearly at entry
    rather than letting them silently collapse to one chunk or blow up deep inside arithmetic."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def partition_records(records: Sequence[Any], structure: SampleStructure, n: int) -> list[list[Any]]:
    """Split ``records`` into ``n`` partitions respecting ``structure``.

    Strideable structures give ``records[k::n]`` (identical to the historical chunking); a
    partially-exchangeable structure groups by its key and round-robins whole groups across partitions.
    ``n`` must be a positive integer -- a nonpositive or fractional ``n`` used to silently collapse to
    a single partition instead of failing.
    """
    n = _positive_int("n", n)
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
    # Deterministic longest-processing-time placement preserves whole groups while bounding the
    # imbalance to the largest group. Encounter order breaks equal-size/equal-load ties.
    parts: list[list[Any]] = [[] for _ in range(n)]
    loads = [0] * n
    ordered_groups = sorted(enumerate(order), key=lambda item: (-len(groups[item[1]]), item[0]))
    for _, key in ordered_groups:
        target = min(range(n), key=lambda i: (loads[i], i))
        parts[target].extend(groups[key])
        loads[target] += len(groups[key])
    return parts


def num_chunks_for(size: int, num_chunks: int = 1, chunk_size: int | None = None) -> int:
    """Resolve the chunk count from an explicit ``num_chunks`` or a target ``chunk_size`` (as seq_encode does).

    Both controls are validated as exact positive integers before any arithmetic: ``chunk_size=0``
    used to raise a bare ``ZeroDivisionError``, and a negative/fractional ``chunk_size`` or a
    nonpositive/fractional ``num_chunks`` used to silently collapse to one chunk instead of failing --
    these bound memory, so silently weakening them can materially change an execution plan. A
    non-default ``num_chunks`` supplied together with ``chunk_size`` is rejected as conflicting rather
    than silently letting ``chunk_size`` win.
    """
    _positive_int("num_chunks", num_chunks)
    if chunk_size is not None:
        _positive_int("chunk_size", chunk_size)
        if num_chunks != 1:
            raise ValueError(
                f"num_chunks={num_chunks!r} and chunk_size={chunk_size!r} were both given explicitly; "
                "pass only one (chunk_size derives the chunk count from size)."
            )
        import math

        return max(1, int(math.ceil(float(size) / float(chunk_size))))
    return num_chunks


def encode_partitions(
    records: Sequence[Any],
    encoder: Any,
    structure: SampleStructure = EXCHANGEABLE,
    num_chunks: int = 1,
    chunk_size: int | None = None,
) -> list[tuple[int, Any]]:
    """Partition ``records`` by ``structure`` and ``encoder.seq_encode`` each part -> ``[(count, payload)]``."""
    n = num_chunks_for(len(records), num_chunks, chunk_size)
    return [(len(part), encoder.seq_encode(part)) for part in partition_records(records, structure, n) if part]
