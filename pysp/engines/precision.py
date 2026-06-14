"""Floating-point precision helpers for compute engines."""

from __future__ import annotations

from typing import Any

import numpy as np

_ALIASES = {
    "16": "float16",
    "half": "float16",
    "fp16": "float16",
    "float16": "float16",
    "32": "float32",
    "single": "float32",
    "float": "float32",
    "fp32": "float32",
    "float32": "float32",
    "64": "float64",
    "double": "float64",
    "fp64": "float64",
    "float64": "float64",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


def precision_name(precision: Any) -> str:
    """Return a readable canonical precision name."""
    if precision is None:
        return "default"
    text = str(precision).replace("torch.", "").replace("numpy.", "").replace("np.", "")
    text = text.replace("<class '", "").replace("'>", "")
    text = text.split(".")[-1].lower()
    return _ALIASES.get(text, text)


def normalize_numpy_dtype(precision: Any) -> np.dtype | None:
    """Normalize a precision specifier to a NumPy floating dtype."""
    if precision is None:
        return None
    name = precision_name(precision)
    if name == "bfloat16":
        raise ValueError("NumPyEngine does not support bfloat16 precision.")
    try:
        dtype = np.dtype(name)
    except TypeError:
        dtype = np.dtype(precision)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError("precision must be a floating-point dtype, got %r." % (precision,))
    return dtype


def normalize_torch_dtype(precision: Any, torch_module: Any) -> Any:
    """Normalize a precision specifier to a Torch floating dtype."""
    if precision is None:
        return None
    if torch_module is not None and isinstance(precision, torch_module.dtype):
        dtype = precision
    else:
        name = precision_name(precision)
        lookup = {
            "float16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
            "float32": torch_module.float32,
            "float64": torch_module.float64,
        }
        if name not in lookup:
            raise ValueError("Unknown Torch floating precision %r." % (precision,))
        dtype = lookup[name]
    if not dtype.is_floating_point:
        raise ValueError("precision must be a floating-point dtype, got %r." % (precision,))
    return dtype


def engine_with_precision(engine: Any, precision: Any) -> Any:
    """Return ``engine`` adjusted to the requested floating precision."""
    if precision is None:
        return engine
    if engine is None:
        from pysp.engines.numpy_engine import NumpyEngine

        return NumpyEngine(dtype=precision)
    fn = getattr(engine, "with_precision", None)
    if not callable(fn):
        raise TypeError("%s does not support precision adjustment." % type(engine).__name__)
    return fn(precision)
