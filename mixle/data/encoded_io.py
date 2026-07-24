"""Serialize encoded (``seq_encode``) data to disk, so an expensive encode is done once and reused.

``seq_encode`` turns raw records into an encoder-specific payload (nested tuples of NumPy arrays). Fitting
re-encodes every call; for large datasets that is the dominant cost. ``save_encoded``/``load_encoded``
persist the encoded payload (with a content digest and the encoder's identity) so subsequent fits load it
directly. The body is pickle (the payloads are internal numeric structures with no safe-JSON registry
representation); the header carrying the digest and the encoder's structural signature (``str(encoder)``)
is plain JSON, deliberately never pickle, so reading it cannot itself execute code -- only the
digest-verified body is ever unpickled, and only after its digest is checked.

The digest covers the *whole* envelope -- every header field (currently just the encoder signature) plus
the body -- not the body alone. An earlier version hashed only the body, so the header's ``encoder`` field
could be edited on disk without touching the body or the stored digest, and ``load_encoded`` would still
accept the payload under the forged identity (MXR-080-0052). Header fields are folded into the digest via
:func:`mixle.data.hashing._canonical`'s self-delimiting canonical encoding -- the same helper
``dataset_hash``/``model_hash`` use -- so on-disk JSON key order or whitespace can never produce a false
"corrupt" mismatch, and the header/body split can never be ambiguous with a different split producing the
same bytes. The on-disk magic changed (``PSPENC1`` -> ``PSPENC2``) alongside the digest scope, so a file
written by the older, narrower-digest code is rejected outright as an unrecognized format rather than
failing the new digest check in a way that reads like corruption.

This digest is corruption-detection, not authentication: it is computed from and stored inside the same
file, so it catches truncation/bit-rot/header-tampering but cannot prove the file was not replaced wholesale
by whoever could already write to ``path`` -- callers should still treat ``load_encoded`` like any other
local pickle load and only point it at a path they trust, exactly as :func:`mixle.lifecycle.Model.load`
documents for its own pickle-format artifacts.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from typing import Any

from mixle.data.hashing import _canonical

_MAGIC = b"PSPENC2\n"


def _envelope_digest(header_fields: dict[str, Any], body: bytes) -> str:
    """SHA-256 over the whole envelope: ``header_fields`` (every header field except ``digest``
    itself, which cannot cover its own value) plus ``body``.

    ``header_fields`` is hashed via its :func:`mixle.data.hashing._canonical` encoding rather than
    raw JSON bytes, for two reasons: (1) it makes the digest independent of on-disk key order and
    whitespace, so re-serializing the header parsed at load time can never produce a false
    "corrupt" mismatch against a value computed from differently-formatted-but-identical-content
    bytes at save time; (2) ``_canonical``'s encoding is self-delimiting (length-prefixed, no bare
    separators), so concatenating it with ``body`` before hashing can never be ambiguous with a
    different (header, body) split landing on the same bytes. See MXR-080-0052: a prior version
    hashed the body alone, so any header field (e.g. the recorded encoder identity) could be edited
    on disk without touching the body or the stored digest, and the corruption check missed it."""
    return hashlib.sha256(_canonical(header_fields) + body).hexdigest()


def save_encoded(encoded: Any, path: str, *, encoder: Any = None) -> str:
    """Write ``encoded`` (the output of ``encoder.seq_encode(...)``) to ``path``; return its hex digest.

    ``encoder`` (optional) records the encoder's structural signature -- ``str(encoder)``, not just its
    class name -- so a load against a structurally different encoder is flagged. Comparing class names
    alone would let e.g. a one-field ``CompositeDataEncoder`` be accepted for a two-field
    ``CompositeDataEncoder`` request, since both share the same class name. The returned digest (also
    stored in the header) covers this signature as well as the body -- see :func:`_envelope_digest`."""
    body = pickle.dumps(encoded, protocol=pickle.HIGHEST_PROTOCOL)
    header_fields: dict[str, Any] = {"encoder": str(encoder) if encoder is not None else None}
    digest = _envelope_digest(header_fields, body)
    meta = {"digest": digest, **header_fields}
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(json.dumps(meta).encode("utf-8"))
        f.write(b"\n")
        f.write(body)
    return digest


def load_encoded(path: str, *, encoder: Any = None) -> Any:
    """Load encoded data written by :func:`save_encoded`, verifying its integrity digest.

    The header is parsed as JSON (never pickle) and the envelope digest -- header fields and body
    together, see :func:`_envelope_digest` -- is checked BEFORE the body is unpickled, so a
    truncated, corrupted, or header-tampered file is rejected before any deserialization runs. If
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
    header_fields = {k: v for k, v in meta.items() if k != "digest"}
    if _envelope_digest(header_fields, body) != meta.get("digest"):
        raise ValueError(f"{path!r} failed its integrity check (corrupt, truncated, or a tampered header)")
    if encoder is not None and meta.get("encoder") is not None and str(encoder) != meta["encoder"]:
        raise ValueError(f"encoder mismatch: file was encoded with {meta['encoder']}, got {encoder}")
    return pickle.loads(body)  # noqa: S301 - digest-verified above; still a local-trust artifact, see module docstring
