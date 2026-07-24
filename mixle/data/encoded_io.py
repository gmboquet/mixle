"""Serialize encoded (``seq_encode``) data to disk, so an expensive encode is done once and reused.

``seq_encode`` turns raw records into an encoder-specific payload (nested tuples of NumPy arrays). Fitting
re-encodes every call; for large datasets that is the dominant cost. ``save_encoded``/``load_encoded``
persist the encoded payload (with a content digest and the encoder's identity) so subsequent fits load it
directly. The body is pickle (the payloads are internal numeric structures with no safe-JSON registry
representation); the header carrying the digest and the encoder's structural signature (``str(encoder)``)
is plain JSON, deliberately never pickle, so reading it cannot itself execute code -- only the
digest-verified body is ever unpickled, and only after its digest is checked.

This digest is corruption-detection, not authentication: it is computed from and stored inside the same
file, so it catches truncation/bit-rot but cannot prove the file was not tampered with by whoever could
already write to ``path`` -- callers should still treat ``load_encoded`` like any other local pickle load
and only point it at a path they trust, exactly as :func:`mixle.lifecycle.Model.load` documents for its
own pickle-format artifacts.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from typing import Any

_MAGIC = b"PSPENC1\n"


def save_encoded(encoded: Any, path: str, *, encoder: Any = None) -> str:
    """Write ``encoded`` (the output of ``encoder.seq_encode(...)``) to ``path``; return its hex digest.

    ``encoder`` (optional) records the encoder's structural signature -- ``str(encoder)``, not just its
    class name -- so a load against a structurally different encoder is flagged. Comparing class names
    alone would let e.g. a one-field ``CompositeDataEncoder`` be accepted for a two-field
    ``CompositeDataEncoder`` request, since both share the same class name."""
    body = pickle.dumps(encoded, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(body).hexdigest()
    meta = {"digest": digest, "encoder": str(encoder) if encoder is not None else None}
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(json.dumps(meta).encode("utf-8"))
        f.write(b"\n")
        f.write(body)
    return digest


def load_encoded(path: str, *, encoder: Any = None) -> Any:
    """Load encoded data written by :func:`save_encoded`, verifying its integrity digest.

    The header is parsed as JSON (never pickle) and the body's digest is checked BEFORE it is
    unpickled, so a truncated or corrupted file is rejected before any deserialization runs. If
    ``encoder`` is given, its structural signature (``str(encoder)``) must match the one recorded at
    save time (else ``ValueError``). This deliberately checks more than the encoder's class: a
    ``DataSequenceEncoder`` subclass is expected to make ``__str__`` structural (mirroring what its
    ``__eq__`` checks) -- e.g. ``CompositeDataEncoder.__str__`` recurses into its component encoders
    and so reflects field count, and ``DiagonalGaussianDataEncoder.__str__`` includes its ``dim`` --
    so a shape mismatch under the same class name (e.g. a one-field vs. two-field
    ``CompositeDataEncoder``) is caught, not just a mismatch in class."""
    with open(path, "rb") as f:
        if f.read(len(_MAGIC)) != _MAGIC:
            raise ValueError(f"{path!r} is not a mixle encoded-data file")
        meta_line = b""
        while True:
            c = f.read(1)
            if c in (b"\n", b""):
                break
            meta_line += c
        try:
            meta = json.loads(meta_line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"{path!r} has a corrupt header") from exc
        body = f.read()
    if hashlib.sha256(body).hexdigest() != meta.get("digest"):
        raise ValueError(f"{path!r} failed its integrity check (corrupt or truncated)")
    if encoder is not None and meta.get("encoder") is not None and str(encoder) != meta["encoder"]:
        raise ValueError(f"encoder mismatch: file was encoded with {meta['encoder']}, got {encoder}")
    return pickle.loads(body)  # noqa: S301 - digest-verified above; still a local-trust artifact, see module docstring
