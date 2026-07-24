"""Stable content hashing of training datasets, for reproducible model provenance.

``dataset_hash(data)`` returns a hex SHA-256 over a canonical byte encoding of the records, so the exact
dataset that trained a model can be fingerprinted and recorded in its header (see
``mixle.inference.production.provenance``). The hash is *order-sensitive* (the same records in a different order hash
differently) -- it identifies an exact training sequence; pass ``sort=True`` for an order-insensitive
fingerprint (records are hashed independently and combined commutatively).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _len_prefixed(payload: bytes) -> bytes:
    """8-byte big-endian length prefix + ``payload``.

    Every variable-length byte string ``_canonical`` emits goes through this, so its extent is explicit
    rather than inferred by scanning for a separator byte -- one the payload's own content could also
    contain.
    """
    return len(payload).to_bytes(8, "big") + payload


def _canonical(obj: Any) -> bytes:
    """Deterministic, self-delimiting bytes for a record component (numbers, strings, arrays, tuples, dicts, None).

    Every case is a tag byte plus content whose length is either fixed by the tag alone (bool, None, a
    non-NaN float's 8 raw bytes) or given by an explicit prefix (:func:`_len_prefixed` for strings/bytes/
    array fields, an 8-byte big-endian count for dict/list/tuple) -- never inferred from a bare separator
    character. That makes every encoding self-delimiting: concatenating several ``_canonical`` outputs
    (done both by the dict/list/tuple cases below and by ``dataset_hash``'s record loop) can always be
    split back into exactly the pieces that produced it, so two structurally different inputs can never
    land on identical bytes (short of an actual SHA-256 collision).

    A prior version joined dict/list elements with a bare ``,``/``:`` and encoded strings/bytes as a tag
    plus raw, un-length-prefixed content. A string, key, or record containing those separator bytes could
    then make one structure's join collide byte-for-byte with a different structure's join -- e.g.
    ``["X", "Y"]`` and ``["X,sY"]`` both encoded as ``b"t[sX,sY]"`` -- so distinct data could hash
    identically. See ``mixle/tests/data/test_hashing.py`` for the regression coverage.
    """
    if obj is None:
        return b"N"
    if isinstance(obj, (bool, np.bool_)):
        return b"b1" if obj else b"b0"
    if isinstance(obj, (int, np.integer)):
        return b"i" + _len_prefixed(repr(int(obj)).encode())
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if f != f:
            # NaN has many bit patterns; normalize to one canonical marker so missing entries hash
            # consistently. Its own tag ("F", not "f") keeps it a fixed, unambiguous length rather than a
            # same-tag, shorter-content special case of the line below -- exactly the kind of tag-sharing,
            # content-dependent-length ambiguity this rewrite eliminates everywhere else.
            return b"F"
        return b"f" + np.float64(f).tobytes()
    if isinstance(obj, (bytes, bytearray)):
        return b"y" + _len_prefixed(bytes(obj))
    if isinstance(obj, str):
        return b"s" + _len_prefixed(obj.encode("utf-8"))
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return (
            b"a"
            + _len_prefixed(str(arr.dtype).encode())
            + _len_prefixed(str(arr.shape).encode())
            + _len_prefixed(arr.tobytes())
        )
    if isinstance(obj, Mapping):
        # sort by each key's own canonical bytes (not repr of the (k, v) pair): well-defined since dict
        # keys are unique, and -- unlike repr -- doesn't also depend on the value or risk an insertion-
        # order-dependent tie between two structurally-equal dicts built in different key orders.
        pairs = sorted(((_canonical(k), v) for k, v in obj.items()), key=lambda kv: kv[0])
        body = b"".join(k_bytes + _canonical(v) for k_bytes, v in pairs)
        return b"d" + len(pairs).to_bytes(8, "big") + body
    if isinstance(obj, (tuple, list)):
        return b"t" + len(obj).to_bytes(8, "big") + b"".join(_canonical(v) for v in obj)
    return b"r" + _len_prefixed(repr(obj).encode())  # last resort: stable repr


def model_hash(model: Any) -> str:
    """Hex SHA-256 fingerprint of a fitted model's parameters (its serialized state).

    Stable across processes: hashes the canonical form of ``to_serializable(model)``, so the same model
    always yields the same hash and two models hash equal iff their serialized parameters match. Used to
    fingerprint a checkpoint and chain EM iteration lineage (see ``mixle.inference.production.provenance``)."""
    from mixle.utils.serialization import ensure_pysp_serialization_registry, to_serializable

    ensure_pysp_serialization_registry()
    # a fitted model may carry a non-serializable provenance header (attached post-fit); the fingerprint is
    # of the parameters, so detach it for the canonical serialization (mirrors Registry.register).
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
    if hasattr(data, "records") and callable(data.records):  # a mixle.data DataSource
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
        h.update(_canonical(r))  # self-delimiting (see _canonical) -- no inter-record separator needed
        n += 1
    h.update(b"#")
    h.update(str(n).encode())
    return h.hexdigest()
