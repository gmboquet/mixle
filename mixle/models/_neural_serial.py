"""Serialization + input-validation helpers shared by the neural-leaf families (``mixle.models``).

The neural leaves wrap a live ``torch.nn.Module``, which the generic JSON encoder in
:mod:`mixle.utils.serialization` cannot walk (a module is not a registered mixle class). These helpers give
every neural leaf a working ``to_dict``/``from_dict`` and the recursive ``__pysp_getstate__``/``__pysp_setstate__``
hooks (so a leaf inside a ``MixtureDistribution`` serializes too) by persisting the module as portable bytes.

The module round-trips through ``torch.save``/``torch.load`` of a ``pickle`` byte buffer -- which requires the
wrapped nn.Module class to be reachable at module level (that is why the ``build_*`` helpers were hoisted). The
bytes are base64-encoded so the whole payload is plain JSON.

That byte buffer is still a full-object pickle, so decoding it executes arbitrary code for a malicious
input -- exactly like unpickling an untrusted file -- even though the surrounding artifact is nominally
"JSON format". :func:`module_from_bytes` refuses to run unless the caller has opened
``mixle.utils.serialization.trusted_deserialization()``; see that function for why.
"""

from __future__ import annotations

import base64
import io
import pickle
from typing import Any

import numpy as np


def module_to_bytes(module: Any) -> bytes:
    """Serialize a torch nn.Module (architecture + weights) to portable bytes via ``torch.save``."""
    import torch

    buf = io.BytesIO()
    torch.save(module, buf, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    return buf.getvalue()


def module_from_bytes(data: bytes) -> Any:
    """Reconstruct a torch nn.Module previously encoded by :func:`module_to_bytes`.

    This unpickles a full object graph (architecture + weights), which executes arbitrary code for a
    malicious byte string -- exactly like ``pickle.load`` on an untrusted file, regardless of whether
    the caller arrived here through a nominally "JSON" artifact (a NeuralLeaf's state embeds this blob
    base64-encoded inside otherwise-safe JSON). Refuses by default; the caller must open
    ``mixle.utils.serialization.trusted_deserialization()`` around a load it knows the source of.
    """
    from mixle.utils.serialization import SerializationError, deserialization_is_trusted

    if not deserialization_is_trusted():
        raise SerializationError(
            "refusing to deserialize an embedded torch module: this executes arbitrary code from the "
            "artifact, the same as pickle.load on an untrusted file. Only load a model from a source "
            "you trust, and do so inside 'with mixle.utils.serialization.trusted_deserialization():'."
        )
    import torch

    buf = io.BytesIO(bytes(data))
    try:
        return torch.load(buf, weights_only=False)  # full module (arch + weights); trust gate above
    except TypeError:  # torch < 2.0 has no weights_only kwarg
        buf.seek(0)
        return torch.load(buf)


def encode_module(module: Any) -> dict[str, str]:
    """A JSON-safe tagged dict for a torch module (base64 of :func:`module_to_bytes`)."""
    return {"__neural_module__": base64.b64encode(module_to_bytes(module)).decode("ascii")}


def decode_module(payload: Any) -> Any:
    """Inverse of :func:`encode_module`."""
    return module_from_bytes(base64.b64decode(payload["__neural_module__"].encode("ascii")))


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
