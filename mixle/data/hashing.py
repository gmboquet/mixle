"""Stable content hashing of training datasets, for reproducible model provenance.

``dataset_hash(data)`` returns a hex SHA-256 over a canonical byte encoding of the records, so the exact
dataset that trained a model can be fingerprinted and recorded in its header (see
``mixle.inference.production.provenance``). The hash is *order-sensitive* (the same records in a different order hash
differently) -- it identifies an exact training sequence; pass ``sort=True`` for an order-insensitive
fingerprint (records are hashed independently and combined commutatively).
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
from collections.abc import Iterable, Mapping, Sized
from collections.abc import Set as AbstractSet
from copy import copy
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
    """Deterministic, self-delimiting bytes for a record component, over a CLOSED set of types.

    Every case is a tag byte plus content whose length is either fixed by the tag alone (bool, None, a
    non-NaN float's 8 raw bytes) or given by an explicit prefix (:func:`_len_prefixed` for strings/bytes/
    array fields, an 8-byte big-endian count for dict/list/tuple/set) -- never inferred from a bare
    separator character. That makes every encoding self-delimiting: concatenating several ``_canonical``
    outputs (done both by the container cases below and by ``dataset_hash``'s record loop) can always be
    split back into exactly the pieces that produced it, so two structurally different inputs can never
    land on identical bytes (short of an actual SHA-256 collision).

    A prior version joined dict/list elements with a bare ``,``/``:`` and encoded strings/bytes as a tag
    plus raw, un-length-prefixed content. A string, key, or record containing those separator bytes could
    then make one structure's join collide byte-for-byte with a different structure's join -- e.g.
    ``["X", "Y"]`` and ``["X,sY"]`` both encoded as ``b"t[sX,sY]"`` -- so distinct data could hash
    identically.

    Three further defects fixed in MXR-080-1601, all of which broke the "structurally different inputs
    cannot collide, equal content cannot differ" contract these digests are used as provenance for:

    * lists and tuples shared the ``b"t"`` tag, so ``dataset_hash([[1, 2]]) == dataset_hash([(1, 2)])``.
      They now carry distinct tags.
    * sets had no case of their own and fell through to ``repr``, whose element order follows the
      per-process string hash seed -- the same set record produced different digests under
      ``PYTHONHASHSEED=1`` and ``PYTHONHASHSEED=2``. Sets now sort by their elements' own canonical
      bytes, which is content-determined and seed-independent.
    * every other unsupported object also fell through to ``repr``, which for an ordinary instance
      embeds its memory address -- so two distinct instances holding the same field value hashed
      differently, run to run. That fallback is gone: dataclass instances are encoded structurally by
      their declared fields, and anything else raises instead of being fingerprinted by
      process-dependent text.

    Note that closing the schema changes the digest of any record containing a list (its tag moved off
    ``b"t"``). That is unavoidable: the old encoding's whole problem is that two different structures
    produced the same bytes, and no fix for a collision can leave both colliding values unchanged.
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
        if arr.dtype.hasobject:
            # object-dtype (or object-field-containing structured) arrays store PyObject* pointers in
            # their buffer, not the elements' own bytes -- arr.tobytes() would serialize those pointers
            # (process-specific memory addresses), so the identical logical array hashes differently
            # from one process/run to the next. Recurse into each element's actual value instead, the
            # same way list/tuple elements are handled below; each _canonical(v) call is already
            # self-delimiting, so wrapping their concatenation in the same _len_prefixed used for the
            # tobytes() fast path keeps the outer "a" framing identical either way.
            payload = b"".join(_canonical(v) for v in arr.flat)
        else:
            payload = arr.tobytes()
        return (
            b"a"
            + _len_prefixed(str(arr.dtype).encode())
            + _len_prefixed(str(arr.shape).encode())
            + _len_prefixed(payload)
        )
    if isinstance(obj, Mapping):
        # sort by each key's own canonical bytes (not repr of the (k, v) pair): well-defined since dict
        # keys are unique, and -- unlike repr -- doesn't also depend on the value or risk an insertion-
        # order-dependent tie between two structurally-equal dicts built in different key orders.
        pairs = sorted(((_canonical(k), v) for k, v in obj.items()), key=lambda kv: kv[0])
        body = b"".join(k_bytes + _canonical(v) for k_bytes, v in pairs)
        return b"d" + len(pairs).to_bytes(8, "big") + body
    if isinstance(obj, tuple):
        return b"t" + len(obj).to_bytes(8, "big") + b"".join(_canonical(v) for v in obj)
    if isinstance(obj, list):
        # distinct tag from tuple: sharing b"t" made [1, 2] and (1, 2) hash identically (MXR-080-1601)
        return b"l" + len(obj).to_bytes(8, "big") + b"".join(_canonical(v) for v in obj)
    if isinstance(obj, (set, frozenset, AbstractSet)):
        # sorted by each element's OWN canonical bytes, not by iteration order (which follows the
        # per-process string hash seed) and not by repr -- so an equal set hashes equally in every
        # process. Sets are unordered, so this is the only well-defined encoding of one.
        members = sorted(_canonical(v) for v in obj)
        return b"e" + len(members).to_bytes(8, "big") + b"".join(members)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # structured records get a structural encoding keyed on their declared fields, in declaration
        # order, qualified by their type -- never repr(), which embeds the instance's memory address.
        field_values = [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]
        body = b"".join(_canonical(name) + _canonical(value) for name, value in field_values)
        return (
            b"D"
            + _len_prefixed(f"{type(obj).__module__}.{type(obj).__qualname__}".encode())
            + len(field_values).to_bytes(8, "big")
            + body
        )
    raise TypeError(
        f"dataset_hash/model_hash cannot canonically encode {type(obj).__module__}.{type(obj).__qualname__}. "
        "These digests are training-data provenance, so an unsupported value is rejected rather than "
        "fingerprinted by repr(), whose text is process-dependent for an ordinary object (it embeds the "
        "instance's memory address) and would report a different identity for equal content on every run. "
        "Supported: None, bool, int, float, bytes, str, numpy arrays, mappings, lists, tuples, sets, and "
        "dataclass instances. Convert anything else to one of those first."
    )


def model_hash(model: Any) -> str:
    """Hex SHA-256 fingerprint of a fitted model's parameters (its serialized state).

    Stable across processes: hashes the canonical form of ``to_serializable(model)``, so the same model
    always yields the same hash and two models hash equal iff their serialized parameters match. Used to
    fingerprint a checkpoint and chain EM iteration lineage (see ``mixle.inference.production.provenance``)."""
    from mixle.utils.serialization import ensure_pysp_serialization_registry, to_serializable

    ensure_pysp_serialization_registry()
    # a fitted model may carry a non-serializable provenance header (attached post-fit); the fingerprint is
    # of the parameters, so detach it for the canonical serialization (mirrors Registry.register).
    had_attr = hasattr(model, "__dict__") and "header" in vars(model)
    subject = model
    if had_attr:
        subject = copy(model)
        vars(subject).pop("header", None)
    payload = _without_fit_provenance(to_serializable(subject))
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _without_fit_provenance(payload: Any) -> Any:
    """Strip fit-provenance envelopes so the fingerprint stays a fingerprint of the PARAMETERS.

    Serialized models carry their fit receipt beside their state (MXR-080-1190/1202). That receipt
    describes how the parameters were produced -- iteration count, convergence, repairs -- and two
    models with identical parameters are the same model whichever run emitted them. Hashing it would
    make content identity depend on fitting history, which is what ``model_hash`` exists not to do,
    and would break replay verification against a receipt recorded from a separate run.
    """
    if isinstance(payload, dict):
        return {key: _without_fit_provenance(value) for key, value in payload.items() if key != "fit_provenance"}
    if isinstance(payload, list):
        return [_without_fit_provenance(item) for item in payload]
    return payload


def _records(data: Any) -> Iterable[Any]:
    if hasattr(data, "records") and callable(data.records):  # a mixle.data DataSource
        return data.records()
    return data


def dataset_hash(data: Any, *, sort: bool = False, max_records: int | None = None) -> str:
    """Hex SHA-256 fingerprint of ``data`` (a sequence of records or a ``DataSource``).

    ``sort=False`` (default) is order-sensitive (exact training sequence). ``sort=True`` combines per-record
    hashes commutatively for an order-insensitive fingerprint. ``max_records`` truncates (the count *and*
    whether truncation actually happened are both mixed in, so a hash of a truncated prefix never collides
    with a hash of a genuinely complete dataset of that same visible length)."""
    if isinstance(max_records, bool) or (
        max_records is not None and (not isinstance(max_records, int) or max_records < 0)
    ):
        raise ValueError(f"max_records must be a non-negative integer or None, got {max_records!r}")
    recs = _records(data)
    original = recs
    if max_records is None:
        bounded = recs
        truncation_marker = b"#"
    else:
        bounded = itertools.islice(recs, max_records)
        if isinstance(original, Sized):
            truncation_marker = b"T" if len(original) > max_records else b"#"
        else:
            # An unsized iterable cannot prove whether an exactly-full prefix was complete without
            # consuming a record that is not represented in the digest. ``P`` honestly identifies
            # that bounded-prefix contract; a short prefix is corrected to complete after exhaustion.
            truncation_marker = b"P"
    if sort:
        acc = 0
        n = 0
        for r in bounded:
            d = int.from_bytes(hashlib.sha256(_canonical(r)).digest(), "big")
            acc = (acc + d) % (1 << 256)  # commutative -> order-insensitive
            n += 1
        if max_records is not None and truncation_marker == b"P" and n < max_records:
            truncation_marker = b"#"
        h = hashlib.sha256()
        h.update(b"sorted")
        h.update(acc.to_bytes(32, "big"))
        # "T"/"#" marks whether max_records actually cut off further records, not just how many were
        # hashed -- otherwise dataset_hash([1, 2, 3], max_records=2) and dataset_hash([1, 2]) hash the
        # same records with the same count and collide despite one being a truncated prefix of a larger
        # dataset and the other genuinely complete.
        h.update(truncation_marker)
        h.update(str(n).encode())
        return h.hexdigest()
    h = hashlib.sha256()
    n = 0
    for r in bounded:
        h.update(_canonical(r))  # self-delimiting (see _canonical) -- no inter-record separator needed
        n += 1
    if max_records is not None and truncation_marker == b"P" and n < max_records:
        truncation_marker = b"#"
    h.update(truncation_marker)
    h.update(str(n).encode())
    return h.hexdigest()
