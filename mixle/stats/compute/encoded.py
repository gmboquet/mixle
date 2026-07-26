"""Metadata wrappers and transfer helpers for sequence-encoded data batches.

The module attaches count, byte-size, encoder, and engine metadata to encoded
payloads and moves numeric fields to resident engines without disturbing object
or string metadata.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from operator import index
from typing import Any

import numpy as np

from mixle.engines import ComputeEngine, engine_of
from mixle.stats.compute.pdist import DataSequenceEncoder, encoded_nbytes


def _engine_owner_matches(actual: ComputeEngine, expected: ComputeEngine) -> bool:
    """Return whether two engines own the same backend/device/mesh placement."""
    return (
        type(actual) is type(expected)
        and str(getattr(actual, "device", None)) == str(getattr(expected, "device", None))
        and getattr(actual, "mesh", None) == getattr(expected, "mesh", None)
    )


def _payload_engine_leaves(payload: Any) -> list[ComputeEngine]:
    """Find numeric resident-array owners while ignoring host-only metadata."""
    resident = getattr(payload, "engine_payload", None)
    if resident is not None and hasattr(payload, "host_payload"):
        return _payload_engine_leaves(resident)
    if isinstance(payload, np.ndarray) and payload.dtype.kind in ("O", "U", "S"):
        return []
    if isinstance(payload, dict):
        return [engine for value in payload.values() for engine in _payload_engine_leaves(value)]
    if isinstance(payload, (list, tuple)):
        return [engine for value in payload for engine in _payload_engine_leaves(value)]
    owner = engine_of(payload, default=None)
    return [] if owner is None else [owner]


def _infer_payload_engine(payload: Any) -> ComputeEngine:
    owners = _payload_engine_leaves(payload)
    if not owners:
        from mixle.engines import NUMPY_ENGINE

        return NUMPY_ENGINE
    owner = owners[0]
    if any(not _engine_owner_matches(candidate, owner) for candidate in owners[1:]):
        raise TypeError("encoded payload contains arrays owned by incompatible compute engines")
    return owner


def _validate_nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        value = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class EncodedData:
    """A one-chunk encoded payload with planner-visible metadata."""

    count: int
    payload: Any
    engine: ComputeEngine
    nbytes: int
    encoder: DataSequenceEncoder | None = None

    def __post_init__(self) -> None:
        count = _validate_nonnegative_integer(self.count, name="count")
        declared_nbytes = _validate_nonnegative_integer(self.nbytes, name="nbytes")
        if not isinstance(self.engine, ComputeEngine):
            raise TypeError("engine must be a ComputeEngine")
        if self.encoder is not None and not isinstance(self.encoder, DataSequenceEncoder):
            raise TypeError("encoder must be a DataSequenceEncoder or None")

        owners = _payload_engine_leaves(self.payload)
        if owners:
            if any(not _engine_owner_matches(owner, self.engine) for owner in owners):
                raise ValueError("declared engine does not own every numeric array in encoded payload")
        else:
            from mixle.engines import NUMPY_ENGINE

            if not _engine_owner_matches(self.engine, NUMPY_ENGINE):
                raise ValueError("a metadata-only encoded payload must remain on the NumPy host engine")

        measured_nbytes = encoded_nbytes(self.payload)
        if declared_nbytes != measured_nbytes:
            raise ValueError(
                f"nbytes={declared_nbytes} does not match final encoded payload size {measured_nbytes}"
            )
        if self.encoder is None:
            from mixle.stats.compute.pdist import _infer_encoded_row_count

            measured_count = _infer_encoded_row_count(self.payload)
            if measured_count is None:
                raise ValueError("an encoder is required when payload row count cannot be inferred")
        else:
            measured_count = self.encoder.row_count(self.payload)
        measured_count = _validate_nonnegative_integer(measured_count, name="encoded row count")
        if measured_count != count:
            raise ValueError(f"count={count} does not match encoder-measured row count {measured_count}")
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "nbytes", measured_nbytes)

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        count: int,
        encoder: DataSequenceEncoder | None = None,
        engine: ComputeEngine | None = None,
    ) -> EncodedData:
        """Wrap an already encoded payload with count, engine, and byte metadata."""
        engine = _infer_payload_engine(payload) if engine is None else engine
        size = encoded_nbytes(payload)
        return cls(count=count, payload=payload, engine=engine, nbytes=size, encoder=encoder)

    @classmethod
    def from_data(cls, data: Any, encoder: DataSequenceEncoder, engine: ComputeEngine | None = None) -> EncodedData:
        """Encode raw data once and attach planner-visible metadata."""
        payload = encoder.seq_encode(data)
        if engine is not None:
            payload = move_encoded_payload(payload, engine)
        engine = _infer_payload_engine(payload) if engine is None else engine
        size = encoded_nbytes(payload)
        return cls(count=len(data), payload=payload, engine=engine, nbytes=size, encoder=encoder)

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
