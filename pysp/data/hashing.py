"""Stable content hashing of training datasets, for reproducible model provenance.

``dataset_hash(data)`` returns a hex SHA-256 over a canonical byte encoding of the records, so the exact
dataset that trained a model can be fingerprinted and recorded in its header (see
``pysp.inference.provenance``). The hash is *order-sensitive* (the same records in a different order hash
differently) -- it identifies an exact training sequence; pass ``sort=True`` for an order-insensitive
fingerprint (records are hashed independently and combined commutatively).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _canonical(obj: Any) -> bytes:
    """Deterministic bytes for a record component (numbers, strings, arrays, tuples, dicts, None)."""
    if obj is None:
        return b"N"
    if isinstance(obj, (bool, np.bool_)):
        return b"b1" if obj else b"b0"
    if isinstance(obj, (int, np.integer)):
        return b"i" + repr(int(obj)).encode()
    if isinstance(obj, (float, np.floating)):
        # struct-stable float bytes; NaN normalized so missing entries hash consistently
        f = float(obj)
        return b"fNaN" if f != f else b"f" + np.float64(f).tobytes()
    if isinstance(obj, (bytes, bytearray)):
        return b"y" + bytes(obj)
    if isinstance(obj, str):
        return b"s" + obj.encode("utf-8")
    if isinstance(obj, np.ndarray):
        return (
            b"a" + str(obj.dtype).encode() + b":" + str(obj.shape).encode() + b":" + np.ascontiguousarray(obj).tobytes()
        )
    if isinstance(obj, Mapping):
        return b"d{" + b",".join(_canonical(k) + b":" + _canonical(v) for k, v in sorted(obj.items(), key=repr)) + b"}"
    if isinstance(obj, (tuple, list)):
        return b"t[" + b",".join(_canonical(v) for v in obj) + b"]"
    return b"r" + repr(obj).encode()  # last resort: stable repr


def model_hash(model: Any) -> str:
    """Hex SHA-256 fingerprint of a fitted model's parameters (its serialized state).

    Stable across processes: hashes the canonical form of ``to_serializable(model)``, so the same model
    always yields the same hash and two models hash equal iff their serialized parameters match. Used to
    fingerprint a checkpoint and chain EM iteration lineage (see ``pysp.inference.provenance``)."""
    from pysp.utils.serialization import ensure_pysp_serialization_registry, to_serializable

    ensure_pysp_serialization_registry()
    # a fitted model may carry a non-serializable provenance header (attached post-fit); the fingerprint is
    # of the parameters, so detach it for the canonical serialization (mirrors ModelRegistry.register).
    attached = getattr(model, "header", None)
    had_attr = hasattr(model, "__dict__") and "header" in vars(model)
    if had_attr:
        del model.header
    try:
        payload = to_serializable(model)
    finally:
        if had_attr:
            model.header = attached
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _records(data: Any) -> Iterable[Any]:
    if hasattr(data, "records") and callable(data.records):  # a pysp.data DataSource
        return data.records()
    return data


def dataset_hash(data: Any, *, sort: bool = False, max_records: int | None = None) -> str:
    """Hex SHA-256 fingerprint of ``data`` (a sequence of records or a ``DataSource``).

    ``sort=False`` (default) is order-sensitive (exact training sequence). ``sort=True`` combines per-record
    hashes commutatively for an order-insensitive fingerprint. ``max_records`` truncates (the count is mixed
    in, so a truncated hash never collides with a full one)."""
    recs = _records(data)
    if sort:
        acc = 0
        n = 0
        for i, r in enumerate(recs):
            if max_records is not None and i >= max_records:
                break
            d = int.from_bytes(hashlib.sha256(_canonical(r)).digest(), "big")
            acc = (acc + d) % (1 << 256)  # commutative -> order-insensitive
            n += 1
        h = hashlib.sha256()
        h.update(b"sorted")
        h.update(acc.to_bytes(32, "big"))
        h.update(str(n).encode())
        return h.hexdigest()
    h = hashlib.sha256()
    n = 0
    for i, r in enumerate(recs):
        if max_records is not None and i >= max_records:
            break
        h.update(_canonical(r))
        h.update(b"|")
        n += 1
    h.update(b"#")
    h.update(str(n).encode())
    return h.hexdigest()
