"""Serialize encoded (``seq_encode``) data to disk, so an expensive encode is done once and reused.

``seq_encode`` turns raw records into an encoder-specific payload (nested tuples of NumPy arrays). Fitting
re-encodes every call; for large datasets that is the dominant cost. ``save_encoded``/``load_encoded``
persist the encoded payload (with an integrity digest and the encoder's identity) so subsequent fits load
it directly. Format is pickle (the payloads are internal numeric structures); a content digest is stored
and verified on load, and the encoder's class name is recorded so a payload is not silently loaded against
a mismatched encoder.
"""

from __future__ import annotations

import hashlib
import pickle
from typing import Any

_MAGIC = b"PSPENC1\n"


def save_encoded(encoded: Any, path: str, *, encoder: Any = None) -> str:
    """Write ``encoded`` (the output of ``encoder.seq_encode(...)``) to ``path``; return its hex digest.

    ``encoder`` (optional) records the encoder class so a load against a different encoder is flagged."""
    body = pickle.dumps(encoded, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(body).hexdigest()
    meta = {"digest": digest, "encoder": type(encoder).__name__ if encoder is not None else None}
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
        f.write(b"\n")
        f.write(body)
    return digest


def load_encoded(path: str, *, encoder: Any = None) -> Any:
    """Load encoded data written by :func:`save_encoded`, verifying its integrity digest.

    If ``encoder`` is given, its class must match the one recorded at save time (else ``ValueError``)."""
    with open(path, "rb") as f:
        if f.read(len(_MAGIC)) != _MAGIC:
            raise ValueError(f"{path!r} is not a pysp encoded-data file")
        meta_line = b""
        while True:
            c = f.read(1)
            if c in (b"\n", b""):
                break
            meta_line += c
        meta = pickle.loads(meta_line)
        body = f.read()
    if hashlib.sha256(body).hexdigest() != meta["digest"]:
        raise ValueError(f"{path!r} failed its integrity check (corrupt or truncated)")
    if encoder is not None and meta["encoder"] is not None and type(encoder).__name__ != meta["encoder"]:
        raise ValueError(f"encoder mismatch: file was encoded with {meta['encoder']}, got {type(encoder).__name__}")
    return pickle.loads(body)
