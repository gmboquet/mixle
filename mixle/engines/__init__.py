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
from mixle.engines.jax_engine import JaxEngine, jax_array_placement
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
    "is_symbolic_payload",
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


def register_array_type(array_type: type[Any], engine: ComputeEngine, *, override: bool = False) -> None:
    """Register an array/tensor type with its owning engine.

    Existing ownership cannot be replaced unless ``override=True`` is explicit. Registering ``object``
    is forbidden because it would capture every otherwise-unregistered Python value process-wide.
    """
    if not isinstance(array_type, type):
        raise TypeError(f"array_type must be an actual type, got {type(array_type).__name__}.")
    if array_type is object:
        raise ValueError("registering object as an array type is forbidden because it shadows every Python value.")
    if not isinstance(engine, ComputeEngine):
        raise TypeError(f"engine must be a ComputeEngine instance, got {type(engine).__name__}.")
    if type(override) is not bool:
        raise TypeError(f"override must be a bool, got {type(override).__name__}.")
    existing = _ARRAY_ENGINE_REGISTRY.get(array_type)
    if existing is not None and existing is not engine and not override:
        raise ValueError(
            f"{array_type.__module__}.{array_type.__qualname__} is already registered to {existing.name}; "
            "pass override=True to replace it explicitly."
        )
    _ARRAY_ENGINE_REGISTRY[array_type] = engine


def _resolve_registered_type(x: Any) -> tuple[type[Any], ComputeEngine] | None:
    """Return the ``(type, engine)`` registry entry for the most specific registered
    ancestor of ``type(x)``.

    Walks ``type(x).__mro__`` (most-derived class first) instead of scanning
    ``_ARRAY_ENGINE_REGISTRY`` in insertion order, so a more specific subclass
    registration always wins over a more generic ancestor's -- e.g. an
    ``np.ndarray`` subclass registered to a different engine than plain
    ``np.ndarray`` resolves to its own registration even when the generic
    ``np.ndarray`` entry was registered first.
    """
    for cls in type(x).__mro__:
        engine = _ARRAY_ENGINE_REGISTRY.get(cls)
        if engine is not None:
            return cls, engine
    return None


def _direct_engine(x: Any) -> ComputeEngine | None:
    explicit = getattr(x, "__pysp_engine__", None)
    if explicit is not None:
        if not isinstance(explicit, ComputeEngine):
            raise TypeError(
                "__pysp_engine__ must be a ComputeEngine instance, got %r of type %s."
                % (explicit, type(explicit).__name__)
            )
        return explicit
    # object arrays of symbolic nodes are ndarrays, so they must be routed to
    # the symbolic engine before the np.ndarray -> NumpyEngine registry rule.
    # is_symbolic_payload validates EVERY element's ownership (MXR-080-0151), so a
    # mixed-ownership object array raises TypeError here rather than resolving to
    # whichever engine the registry check below would have guessed.
    if is_symbolic_payload(x):
        return SYMBOLIC_ENGINE
    resolved = _resolve_registered_type(x)
    if resolved is None:
        return None
    cls, engine = resolved
    if torch is not None and cls is torch.Tensor and isinstance(engine, TorchEngine):
        # A tensor's own storage dtype is only a meaningful *floating-point policy* when
        # it actually is floating -- forwarding an integer/Boolean dtype as the engine's
        # float policy makes TorchEngine's constructor reject it outright, so fall back to
        # the engine's default float policy for those instead of failing discovery.
        dt = x.dtype if x.dtype.is_floating_point else None
        return TorchEngine(device=str(x.device), dtype=dt)
    if DTensor is not None and cls is DTensor and isinstance(engine, TorchEngine):
        local = x.to_local()
        dt = x.dtype if x.dtype.is_floating_point else None
        return TorchEngine(device=str(local.device), dtype=dt, mesh=x.device_mesh)
    if _jax is not None and cls is _jax.Array and isinstance(engine, JaxEngine):
        # A JAX array's own storage dtype is only a meaningful *floating-point policy* when it
        # actually is floating -- forwarding an integer/Boolean dtype as the engine's float policy
        # makes JaxEngine's constructor reject it outright (via normalize_numpy_dtype), so fall back
        # to the engine's default float policy for those instead of failing discovery. Mirrors the
        # torch.Tensor/DTensor branches above (MXR-080-0122).
        dt = x.dtype if np.issubdtype(x.dtype, np.floating) else None
        # ...and, like those branches, carry the array's ACTUAL placement rather than letting the
        # constructor resolve its default CPU device: without it every JAX array claimed CPU
        # ownership however it was really placed, so _engines_compatible's device comparison could
        # not distinguish a GPU/TPU array from a host one (MXR-080-1561).
        return JaxEngine(device=jax_array_placement(x), dtype=dt)
    return engine


def _child_values(x: Any) -> Iterable[Any]:
    if isinstance(x, dict):
        return x.values()
    if isinstance(x, (list, tuple)):
        return x
    return ()


def _engines_compatible(a: ComputeEngine, b: ComputeEngine) -> bool:
    """Whether two engines are interchangeable for dispatch purposes.

    Two engines must agree on backend (Python class), device placement, and
    mesh/distributed placement to be considered the same execution context --
    comparing the Python class alone lets, e.g., a ``cuda:0`` Torch tensor and
    a ``cuda:1`` Torch tensor pass as "homogeneous". Dtype/precision policy is
    compared too, but only when BOTH engines carry an explicit, caller-chosen
    policy (``dtype_explicit``, set by :class:`TorchEngine` and
    :class:`JaxEngine`): an engine discovered from a non-floating leaf (an
    integer index or Boolean mask tensor/array, say) has no real precision
    opinion of its own and must not conflict with a genuinely floating
    sibling leaf purely because of its filled-in default dtype. Engines that
    don't declare ``dtype_explicit`` (NumPy, symbolic) default to "always
    opinionated", preserving a strict dtype comparison for them.
    """
    if type(a) is not type(b):
        return False
    if str(getattr(a, "device", None)) != str(getattr(b, "device", None)):
        return False
    if getattr(a, "mesh", None) != getattr(b, "mesh", None):
        return False
    if not getattr(a, "dtype_explicit", True) or not getattr(b, "dtype_explicit", True):
        return True
    return getattr(a, "dtype", None) == getattr(b, "dtype", None)


def engine_of(x: Any, default: ComputeEngine = NUMPY_ENGINE) -> ComputeEngine:
    """Return the ComputeEngine associated with an array or encoded payload.

    Nested encodings are scanned recursively. Mixing arrays whose engines
    disagree on backend, device, mesh/placement, or an explicitly-chosen dtype
    policy is an error, because silent host/device/precision mixing is almost
    always a performance or correctness bug (see :func:`_engines_compatible`).
    """
    return _engine_of(x, default, set())


def _engine_of(x: Any, default: ComputeEngine | None, active_containers: set[int]) -> ComputeEngine | None:
    direct = _direct_engine(x)
    if direct is not None:
        return direct

    is_container = isinstance(x, (dict, list, tuple))
    identity = id(x)
    if is_container:
        if identity in active_containers:
            raise ValueError("cyclic container encountered during compute-engine discovery.")
        active_containers.add(identity)
    found: ComputeEngine | None = None
    try:
        for child in _child_values(x):
            child_engine = _engine_of(child, None, active_containers)
            if child_engine is None:
                continue
            if found is None:
                found = child_engine
            elif not _engines_compatible(found, child_engine):
                raise TypeError("mixed compute engines in encoded payload: %s and %s" % (found.name, child_engine.name))
            elif not getattr(found, "dtype_explicit", True) and getattr(child_engine, "dtype_explicit", True):
                found = child_engine  # prefer the more-opinionated (explicit-precision) engine
    finally:
        if is_container:
            active_containers.remove(identity)
    return default if found is None else found


def _contains_engine_value(x: Any, _active_containers: set[int] | None = None) -> bool:
    """Return whether ``x`` is, or recursively contains, a directly engine-owned value.

    Used by :func:`to_numpy` to decide whether a dict/list/tuple must be
    walked leaf by leaf -- each leaf handed to its OWN owning engine -- rather
    than handed whole to a single engine's conversion routine, which is only
    correct for plain nested Python data with no engine-owned array inside.
    """
    if _direct_engine(x) is not None:
        return True
    if not isinstance(x, (dict, list, tuple)):
        return False
    active_containers = set() if _active_containers is None else _active_containers
    identity = id(x)
    if identity in active_containers:
        raise ValueError("cyclic container encountered while inspecting compute-engine ownership.")
    active_containers.add(identity)
    try:
        return any(_contains_engine_value(value, active_containers) for value in _child_values(x))
    finally:
        active_containers.remove(identity)


def to_numpy(x: Any) -> Any:
    """Convert an engine array/tensor payload to NumPy at an explicit boundary.

    A dict, list, or tuple that contains an engine-owned value is converted
    leaf by leaf, preserving container structure and resolving each leaf's own
    owning engine independently. Resolving a single engine for the whole
    container up front and handing it the container as-is -- the previous
    behavior -- fails to stack a ragged list of Torch tensors, leaves
    dictionary values unconverted, and can hand a device tensor to
    ``np.asarray`` without transferring it to host memory first. Plain nested
    data with no engine-owned value is still handed to the resolved engine as
    one unit, unchanged.
    """
    if isinstance(x, dict) and _contains_engine_value(x):
        return {k: to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)) and _contains_engine_value(x):
        converted = [to_numpy(v) for v in x]
        return tuple(converted) if isinstance(x, tuple) else converted
    return engine_of(x).to_numpy(x)
