"""Public compute-engine registry and precision front door for Mixle kernels.

This namespace exposes the NumPy, Torch, JAX, symbolic, and precision-aware
engine utilities used by backend-neutral scoring, estimation, and data-transfer
paths.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

# Precision spectrum: extended precision, format codecs, error tracing, and the auto-routing front door.
# All pure-numpy; the arbitrary-precision (MPFR) tail in mixle.engines.highprec stays lazily imported so
# the engines package never eagerly requires gmpy2.
from mixle.engines.affine import AffineForm, allocate_precision
from mixle.engines.base import ComputeEngine
from mixle.engines.error_tracing import Interval, float64_sum_is_accurate, sum_error_bound
from mixle.engines.extended import DoubleDouble, dd_dot, dd_sum
from mixle.engines.formats import CodebookFormat, FixedPointFormat, FloatFormat
from mixle.engines.jax_engine import JaxEngine
from mixle.engines.jax_engine import jax as _jax
from mixle.engines.numpy_engine import FUSED_NUMPY_ENGINE, NUMPY_ENGINE, NumpyEngine
from mixle.engines.precision import (
    auto_precision,
    engine_with_precision,
    normalize_numpy_dtype,
    normalize_torch_dtype,
    precision_name,
)
from mixle.engines.spectrum import accurate_sum, cast, sum_certificate
from mixle.engines.symbolic_engine import SYMBOLIC_ENGINE, SymbolicEngine, SymbolicExpression, is_symbolic_payload
from mixle.engines.symbolic_export import to_latex, to_sage, to_sympy
from mixle.engines.torch_engine import TorchEngine, torch

__all__ = [
    "ComputeEngine",
    "NumpyEngine",
    "SymbolicEngine",
    "SymbolicExpression",
    "SYMBOLIC_ENGINE",
    "TorchEngine",
    "JaxEngine",
    "NUMPY_ENGINE",
    "FUSED_NUMPY_ENGINE",
    "auto_precision",
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
    # precision spectrum
    "DoubleDouble",
    "dd_sum",
    "dd_dot",
    "FloatFormat",
    "FixedPointFormat",
    "CodebookFormat",
    "Interval",
    "sum_error_bound",
    "float64_sum_is_accurate",
    "AffineForm",
    "allocate_precision",
    "accurate_sum",
    "cast",
    "sum_certificate",
]


_ARRAY_ENGINE_REGISTRY: dict[type[Any], ComputeEngine] = {
    np.ndarray: NUMPY_ENGINE,
    np.generic: NUMPY_ENGINE,
}

if torch is not None:
    _ARRAY_ENGINE_REGISTRY[torch.Tensor] = TorchEngine()
    try:  # public path (torch >= 2.5), then the private module torch 2.0-2.4 ship it under
        from torch.distributed.tensor import DTensor
    except ImportError:
        try:
            from torch.distributed._tensor import DTensor
        except ImportError:  # pragma: no cover - depends on torch build
            DTensor = None
    if DTensor is not None:
        _ARRAY_ENGINE_REGISTRY[DTensor] = TorchEngine()
else:
    DTensor = None

if _jax is not None:
    _ARRAY_ENGINE_REGISTRY[_jax.Array] = JaxEngine()


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
            if _jax is not None and cls is _jax.Array and isinstance(engine, JaxEngine):
                return JaxEngine(dtype=x.dtype)
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
