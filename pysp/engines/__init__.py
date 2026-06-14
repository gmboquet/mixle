"""Compute engines for backend-neutral pysparkplug kernels."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from pysp.engines.base import ComputeEngine
from pysp.engines.numpy_engine import NUMPY_ENGINE, NumpyEngine
from pysp.engines.precision import engine_with_precision, normalize_numpy_dtype, normalize_torch_dtype, precision_name
from pysp.engines.symbolic_engine import SYMBOLIC_ENGINE, SymbolicEngine, SymbolicExpression, is_symbolic_payload
from pysp.engines.symbolic_export import to_latex, to_sage, to_sympy
from pysp.engines.torch_engine import TorchEngine, torch

__all__ = [
    "ComputeEngine",
    "NumpyEngine",
    "SymbolicEngine",
    "SymbolicExpression",
    "SYMBOLIC_ENGINE",
    "TorchEngine",
    "NUMPY_ENGINE",
    "engine_of",
    "engine_with_precision",
    "normalize_numpy_dtype",
    "normalize_torch_dtype",
    "precision_name",
    "register_array_type",
    "to_latex",
    "to_numpy",
    "to_sage",
    "to_sympy",
]


_ARRAY_ENGINE_REGISTRY: dict[type[Any], ComputeEngine] = {
    np.ndarray: NUMPY_ENGINE,
    np.generic: NUMPY_ENGINE,
}

if torch is not None:
    _ARRAY_ENGINE_REGISTRY[torch.Tensor] = TorchEngine()
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover - depends on torch build
        DTensor = None
    if DTensor is not None:
        _ARRAY_ENGINE_REGISTRY[DTensor] = TorchEngine()
else:
    DTensor = None


def register_array_type(array_type: type[Any], engine: ComputeEngine) -> None:
    """Register an array/tensor type with its owning engine."""
    _ARRAY_ENGINE_REGISTRY[array_type] = engine


def _direct_engine(x: Any) -> ComputeEngine | None:
    explicit = getattr(x, "__pysp_engine__", None)
    if explicit is not None:
        return explicit
    # object arrays of symbolic nodes are ndarrays, so they must be routed to
    # the symbolic engine before the np.ndarray -> NumpyEngine registry rule
    if is_symbolic_payload(x):
        return SYMBOLIC_ENGINE
    for cls, engine in _ARRAY_ENGINE_REGISTRY.items():
        if isinstance(x, cls):
            if torch is not None and cls is torch.Tensor and isinstance(engine, TorchEngine):
                return TorchEngine(device=str(x.device), dtype=x.dtype)
            if DTensor is not None and cls is DTensor and isinstance(engine, TorchEngine):
                local = x.to_local()
                return TorchEngine(device=str(local.device), dtype=x.dtype, mesh=x.device_mesh)
            return engine
    return None


def _child_values(x: Any) -> Iterable[Any]:
    if isinstance(x, dict):
        return x.values()
    if isinstance(x, (list, tuple)):
        return x
    return ()


def engine_of(x: Any, default: ComputeEngine = NUMPY_ENGINE) -> ComputeEngine:
    """Return the ComputeEngine associated with an array or encoded payload.

    Nested encodings are scanned recursively.  Mixing arrays owned by different
    engine classes is an error because silent host/device mixing is almost
    always a performance or correctness bug.
    """
    direct = _direct_engine(x)
    if direct is not None:
        return direct

    found: ComputeEngine | None = None
    for child in _child_values(x):
        child_engine = engine_of(child, default=None)
        if child_engine is None:
            continue
        if found is None:
            found = child_engine
        elif type(found) is not type(child_engine):
            raise TypeError("mixed compute engines in encoded payload: %s and %s" % (found.name, child_engine.name))
    return default if found is None else found


def to_numpy(x: Any) -> Any:
    """Convert an engine array/tensor payload to NumPy at an explicit boundary."""
    return engine_of(x).to_numpy(x)
