"""Serialization + input-validation helpers shared by the neural-leaf families (``mixle.models``).

The neural leaves wrap a live ``torch.nn.Module``, which the generic JSON encoder in
:mod:`mixle.utils.serialization` cannot walk (a module is not a registered mixle class). These helpers give
every neural leaf a working ``to_dict``/``from_dict`` and the recursive ``__pysp_getstate__``/``__pysp_setstate__``
hooks (so a leaf inside a ``MixtureDistribution`` serializes too) by persisting an environment-bound module
artifact.

The module round-trips through ``torch.save``/``torch.load`` of a ``pickle`` byte buffer -- which requires the
wrapped nn.Module class to be reachable at module level (that is why the ``build_*`` helpers were hoisted). The
bytes are base64-encoded so the whole payload is plain JSON. They are not a portable or safe interchange
format: loading still requires the producing Python/Torch environment and access to the module's class.

That byte buffer is still a full-object pickle, so decoding it executes arbitrary code for a malicious
input -- exactly like unpickling an untrusted file -- even though the surrounding artifact is nominally
"JSON format". :func:`module_from_bytes` refuses to run unless the caller has opened
``mixle.utils.serialization.trusted_deserialization()``; see that function for why.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import io
import pickle
import platform
from collections.abc import Mapping
from typing import Any

import numpy as np

MAX_NEURAL_MODULE_BYTES = 256 * 1024 * 1024
_NEURAL_MODULE_FORMAT = "torch-full-object-pickle/v1"
_PAYLOAD_FIELDS = frozenset(
    {
        "__neural_module__",
        "format",
        "environment_bound",
        "decoded_bytes",
        "sha256",
        "environment",
    }
)


def _serialization_error(message: str) -> Exception:
    from mixle.utils.serialization import SerializationError

    return SerializationError(message)


def _require_trusted_deserialization() -> None:
    from mixle.utils.serialization import deserialization_is_trusted

    if not deserialization_is_trusted():
        raise _serialization_error(
            "refusing to deserialize an embedded torch module: this executes arbitrary code from the "
            "artifact, the same as pickle.load on an untrusted file. Only load a model from a source "
            "you trust, and do so inside 'with mixle.utils.serialization.trusted_deserialization():'."
        )


def _environment() -> dict[str, str]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }


def module_to_bytes(module: Any) -> bytes:
    """Serialize a torch module to a size-bounded, environment-bound pickle."""
    import torch

    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    buf = io.BytesIO()
    torch.save(module, buf, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    data = buf.getvalue()
    if not data:
        raise ValueError("torch.save produced an empty neural-module artifact")
    if len(data) > MAX_NEURAL_MODULE_BYTES:
        raise ValueError(f"neural-module artifact is {len(data)} bytes; maximum is {MAX_NEURAL_MODULE_BYTES} bytes")
    return data


def module_from_bytes(data: bytes) -> Any:
    """Reconstruct a torch nn.Module previously encoded by :func:`module_to_bytes`.

    This unpickles a full object graph (architecture + weights), which executes arbitrary code for a
    malicious byte string -- exactly like ``pickle.load`` on an untrusted file, regardless of whether
    the caller arrived here through a nominally "JSON" artifact (a NeuralLeaf's state embeds this blob
    base64-encoded inside otherwise-safe JSON). Refuses by default; the caller must open
    ``mixle.utils.serialization.trusted_deserialization()`` around a load it knows the source of.
    """
    _require_trusted_deserialization()
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise _serialization_error("embedded torch module must be a bytes-like object")
    raw = bytes(data)
    if not raw:
        raise _serialization_error("embedded torch module is empty")
    if len(raw) > MAX_NEURAL_MODULE_BYTES:
        raise _serialization_error(
            f"embedded torch module is {len(raw)} bytes; maximum is {MAX_NEURAL_MODULE_BYTES} bytes"
        )
    import torch

    buf = io.BytesIO(raw)
    try:
        module = torch.load(buf, weights_only=False)  # full module (arch + weights); trust gate above
    except TypeError:  # torch < 2.0 has no weights_only kwarg
        buf.seek(0)
        module = torch.load(buf)
    if not isinstance(module, torch.nn.Module):
        raise _serialization_error("embedded torch artifact did not contain a torch.nn.Module")
    return module


def encode_module(module: Any) -> dict[str, Any]:
    """Return a self-describing JSON envelope for an environment-bound torch pickle."""
    data = module_to_bytes(module)
    return {
        "__neural_module__": base64.b64encode(data).decode("ascii"),
        "format": _NEURAL_MODULE_FORMAT,
        "environment_bound": True,
        "decoded_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "environment": _environment(),
    }


def decode_module(payload: Any) -> Any:
    """Decode a trusted, bounded neural-module envelope.

    Trust is checked before inspecting or base64-decoding the embedded bytes. This keeps an untrusted caller
    from using this path as an unbounded allocation primitive even when it knows loading would later fail.
    """
    _require_trusted_deserialization()
    if not isinstance(payload, Mapping):
        raise _serialization_error("neural-module payload must be a mapping")
    if set(payload) != _PAYLOAD_FIELDS:
        missing = sorted(_PAYLOAD_FIELDS - set(payload))
        extra = sorted(set(payload) - _PAYLOAD_FIELDS)
        raise _serialization_error(f"invalid neural-module payload fields: missing={missing}, extra={extra}")
    if payload["format"] != _NEURAL_MODULE_FORMAT:
        raise _serialization_error(f"unsupported neural-module format: {payload['format']!r}")
    if payload["environment_bound"] is not True:
        raise _serialization_error("neural-module payload must declare environment_bound=true")

    decoded_bytes = payload["decoded_bytes"]
    if type(decoded_bytes) is not int or not 0 < decoded_bytes <= MAX_NEURAL_MODULE_BYTES:
        raise _serialization_error(f"decoded_bytes must be an integer from 1 through {MAX_NEURAL_MODULE_BYTES}")
    encoded = payload["__neural_module__"]
    if not isinstance(encoded, str):
        raise _serialization_error("__neural_module__ must be an ASCII base64 string")
    max_encoded_bytes = 4 * ((MAX_NEURAL_MODULE_BYTES + 2) // 3)
    if not encoded or len(encoded) > max_encoded_bytes:
        raise _serialization_error(f"encoded neural module must contain at most {max_encoded_bytes} base64 characters")
    try:
        encoded_ascii = encoded.encode("ascii")
    except UnicodeEncodeError as exc:
        raise _serialization_error("__neural_module__ must contain only ASCII base64 characters") from exc
    try:
        data = base64.b64decode(encoded_ascii, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise _serialization_error("__neural_module__ is not strict base64") from exc
    if len(data) != decoded_bytes:
        raise _serialization_error(f"decoded neural-module length {len(data)} does not match declared {decoded_bytes}")

    digest = payload["sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise _serialization_error("neural-module sha256 must be a 64-character hexadecimal digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise _serialization_error("neural-module sha256 must be hexadecimal") from exc
    actual_digest = hashlib.sha256(data).hexdigest()
    if not hmac.compare_digest(actual_digest, digest.lower()):
        raise _serialization_error("neural-module sha256 does not match its decoded bytes")

    environment = payload["environment"]
    if not isinstance(environment, Mapping) or set(environment) != {"python", "torch"}:
        raise _serialization_error("neural-module environment must contain exactly python and torch versions")
    current_environment = _environment()
    if dict(environment) != current_environment:
        raise _serialization_error(
            f"neural-module environment {dict(environment)!r} does not match {current_environment!r}"
        )
    return module_from_bytes(data)


def check_finite(x: np.ndarray, where: str) -> np.ndarray:
    """Raise a clear error if ``x`` has any non-finite entry, so a NaN cannot silently poison a mixture E-step.

    A neural leaf that returned NaN log-density would corrupt every responsibility in the E-step without a
    diagnosable failure; validating at the density boundary turns that into an immediate, named error instead.
    """
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "%s received non-finite input (NaN or inf); a neural leaf cannot score it and it would poison a "
            "mixture E-step. Clean the data before fitting/scoring." % where
        )
    return arr
