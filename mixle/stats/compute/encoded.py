"""Metadata wrappers and transfer helpers for sequence-encoded data batches.

The module attaches count, byte-size, encoder, and engine metadata to encoded
payloads and moves numeric fields to resident engines without disturbing object
or string metadata.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.engines import ComputeEngine, engine_of
from mixle.stats.compute.pdist import DataSequenceEncoder, encoded_nbytes


@dataclass(frozen=True)
class EncodedData:
    """A one-chunk encoded payload with planner-visible metadata."""

    count: int
    payload: Any
    engine: ComputeEngine
    nbytes: int
    encoder: DataSequenceEncoder | None = None

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        count: int,
        encoder: DataSequenceEncoder | None = None,
        engine: ComputeEngine | None = None,
    ) -> EncodedData:
        """Wrap an already encoded payload with count, engine, and byte metadata."""
        engine = engine_of(payload) if engine is None else engine
        size = encoder.nbytes(payload) if encoder is not None else encoded_nbytes(payload)
        return cls(count=int(count), payload=payload, engine=engine, nbytes=int(size), encoder=encoder)

    @classmethod
    def from_data(cls, data: Any, encoder: DataSequenceEncoder, engine: ComputeEngine | None = None) -> EncodedData:
        """Encode raw data once and attach planner-visible metadata."""
        payload = encoder.seq_encode(data)
        size = encoder.nbytes(payload)
        if engine is not None:
            payload = move_encoded_payload(payload, engine)
        engine = engine_of(payload) if engine is None else engine
        return cls(count=int(len(data)), payload=payload, engine=engine, nbytes=int(size), encoder=encoder)

    def as_seq_chunk(self) -> tuple[int, Any]:
        """Return the legacy ``(count, encoded_payload)`` chunk tuple."""
        return self.count, self.payload

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        yield self.as_seq_chunk()

    def __len__(self) -> int:
        return 1


@dataclass(frozen=True)
class ResidentEncodedPayload:
    """Pair a host encoding with a resident engine encoding for one chunk."""

    host_payload: Any
    engine_payload: Any


def as_encoded_data(
    payload: Any, count: int, encoder: DataSequenceEncoder | None = None, engine: ComputeEngine | None = None
) -> EncodedData:
    """Wrap an existing encoded payload with count, engine, and byte metadata."""
    return EncodedData.from_payload(payload, count=count, encoder=encoder, engine=engine)


def move_encoded_payload(payload: Any, engine: ComputeEngine) -> Any:
    """Move numeric encoded arrays into ``engine`` while preserving object fields.

    Encoders remain backend-agnostic and produce their historical Python/NumPy
    payloads.  Orchestrators can call this exactly once after encoding a shard
    so scoring kernels see resident engine arrays.  Object/string arrays and
    non-array Python metadata stay on the host because many distribution
    encodings intentionally carry labels, maps, or structural metadata.
    """
    if isinstance(payload, np.ndarray):
        if payload.dtype.kind in ("O", "U", "S"):
            return payload
        return engine.asarray(payload)
    if isinstance(payload, tuple):
        return tuple(move_encoded_payload(value, engine) for value in payload)
    if isinstance(payload, list):
        return [move_encoded_payload(value, engine) for value in payload]
    if isinstance(payload, dict):
        return {key: move_encoded_payload(value, engine) for key, value in payload.items()}
    return payload
