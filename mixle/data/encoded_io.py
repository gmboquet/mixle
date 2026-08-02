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
same bytes. The current on-disk magic is ``PSPENC3``; it adds mandatory, versioned encoder binding and
rejects older unbound artifacts rather than treating an opaque payload as checked reusable data. A file
written by the older, narrower-digest code is rejected outright as an unrecognized format rather than
failing the new digest check in a way that reads like corruption.

This digest is corruption-detection, not authentication: it is computed from and stored inside the same
file, so it catches truncation/bit-rot/header-tampering but cannot prove the file was not replaced wholesale
by whoever could already write to ``path``. A forged artifact carries a valid digest for its own contents,
a matching encoder signature, and a body that executes on load. ``load_encoded`` therefore requires an
explicit ``trusted=True`` (MXR-080-1873): the decision belongs at the call site, where the provenance of
``path`` is known, not in a docstring the caller may never read. This mirrors what
:func:`mixle.lifecycle.Model.load` documents for its own pickle-format artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from typing import Any

from mixle.data.hashing import _canonical

_MAGIC = b"PSPENC3\n"


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

    ``encoder`` is required and records the encoder's versioned structural signature -- not just its
    class name -- so a load against a structurally different encoder is flagged. Comparing class names
    alone would let e.g. a one-field ``CompositeDataEncoder`` be accepted for a two-field
    ``CompositeDataEncoder`` request, since both share the same class name. The returned digest (also
    stored in the header) covers this signature as well as the body -- see :func:`_envelope_digest`."""
    if encoder is None:
        raise ValueError("save_encoded requires encoder= so the opaque payload has a structural type binding")
    body = pickle.dumps(encoded, protocol=pickle.HIGHEST_PROTOCOL)
    header_fields: dict[str, Any] = {
        "format_version": 3,
        "encoder": {
            "module": type(encoder).__module__,
            "qualname": type(encoder).__qualname__,
            "signature": str(encoder),
        },
    }
    digest = _envelope_digest(header_fields, body)
    meta = {"digest": digest, **header_fields}
    destination = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(destination)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(destination)}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(_MAGIC)
            f.write(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            f.write(b"\n")
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, destination)
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return digest


def load_encoded(path: str, *, encoder: Any = None, trusted: bool | None = None) -> Any:
    """Load encoded data written by :func:`save_encoded`, verifying its integrity digest.

    The header is parsed as JSON (never pickle) and the envelope digest -- header fields and body
    together, see :func:`_envelope_digest` -- is checked BEFORE the body is unpickled, so a
    truncated, corrupted, or header-tampered file is rejected before any deserialization runs. The
    required ``encoder`` argument's structural signature must match the one recorded at
    save time (else ``ValueError``). This deliberately checks more than the encoder's class: a
    ``DataSequenceEncoder`` subclass is expected to make ``__str__`` structural (mirroring what its
    ``__eq__`` checks) -- e.g. ``CompositeDataEncoder.__str__`` recurses into its component encoders
    and so reflects field count, and ``DiagonalGaussianDataEncoder.__str__`` includes its ``dim`` --
    so a shape mismatch under the same class name (e.g. a one-field vs. two-field
    ``CompositeDataEncoder``) is caught, not just a mismatch in class.

    ``trusted=True`` is required, and is the caller's statement that ``path`` is under their control.
    The body is pickle, so loading it is equivalent to executing whatever is in the file, and NONE of
    the checks above change that: the digest is computed from and stored inside the same file, so it
    detects truncation and header tampering but is worthless against a file replaced wholesale -- a
    forged artifact carries a valid digest for its own contents and a matching encoder signature, and
    executes on load (MXR-080-1873). The module docstring said as much and the parameter did not
    exist, so the warning was addressed to whoever read the source rather than to whoever called the
    function. Making it an argument puts the decision at the call site, where the provenance of
    ``path`` is actually known.
    """
    if encoder is None:
        raise ValueError("load_encoded requires encoder= to verify the payload's structural type binding")
    if trusted is not True:
        raise ValueError(
            f"load_encoded({path!r}) requires trusted=True. The body is pickle, so loading it executes "
            "whatever the file contains; the integrity digest is stored in the same file and cannot "
            "distinguish a forged artifact from a genuine one. Pass trusted=True only for a path you "
            "control, exactly as you would for any other local pickle load."
            if trusted is None
            else f"load_encoded({path!r}) was called with trusted={trusted!r}; it will not unpickle a "
            "path the caller has not vouched for."
        )
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
    expected_encoder = {
        "module": type(encoder).__module__,
        "qualname": type(encoder).__qualname__,
        "signature": str(encoder),
    }
    if meta.get("format_version") != 3 or meta.get("encoder") != expected_encoder:
        raise ValueError(f"encoder mismatch: file was encoded with {meta.get('encoder')!r}, got {expected_encoder!r}")
    return pickle.loads(body)  # noqa: S301 - digest-verified above; still a local-trust artifact, see module docstring
