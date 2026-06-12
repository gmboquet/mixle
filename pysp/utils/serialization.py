"""Safe JSON serialization helpers for pysp distribution and estimator objects.

The legacy ``load_models(eval_string)`` path reconstructed models by executing
their repr strings.  This module instead serializes distribution state as a
small tagged JSON value graph and reconstructs only classes registered from
pysp's own distribution modules.
"""
from __future__ import annotations

import base64
import importlib
import inspect
import json
import math
import pkgutil
from collections.abc import Callable
from typing import Any, Dict, Iterable, Optional, Set, Type

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - scipy is a package dependency in normal use.
    sp = None


TAG = "__pysp_type__"

_CLASS_REGISTRY: Dict[str, Type[Any]] = {}
_CLASS_IDS: Dict[Type[Any], str] = {}
_CALLABLE_REGISTRY: Dict[str, Callable[..., Any]] = {}
_CALLABLE_IDS: Dict[Callable[..., Any], str] = {}
_REGISTRY_READY = False
_OPTIONAL_IMPORT_NAMES = {"torch", "umap", "pyspark"}


class SerializationError(ValueError):
    """Raised when an object cannot be serialized or decoded safely."""


def _type_id(cls: Type[Any]) -> str:
    return "%s.%s" % (cls.__module__, cls.__name__)


def register_serializable_class(cls: Type[Any], type_id: Optional[str] = None) -> Type[Any]:
    """Register a class that may be reconstructed from serialized state.

    Deserialization never imports a class named in the payload.  The class must
    already be present in this registry, which is populated from pysp package
    modules by ``ensure_pysp_serialization_registry``.
    """
    tid = type_id or _type_id(cls)
    previous = _CLASS_REGISTRY.get(tid)
    if previous is not None and previous is not cls:
        raise SerializationError("type id %r is already registered for %r" % (tid, previous))
    _CLASS_REGISTRY[tid] = cls
    _CLASS_IDS[cls] = tid
    return cls


def register_serializable_callable(fn: Callable[..., Any],
                                   callable_id: Optional[str] = None) -> Callable[..., Any]:
    """Register a callable that may appear inside a serialized distribution.

    This is intentionally explicit.  Arbitrary lambdas or local functions
    cannot be made safe by JSON alone; callers that need SelectDistribution-like
    routing should register a stable process-local callable id.
    """
    if callable_id is None:
        module = getattr(fn, "__module__", None)
        qualname = getattr(fn, "__qualname__", None)
        if not module or not qualname or "<lambda>" in qualname or "<locals>" in qualname:
            raise SerializationError("callable_id is required for lambdas and local callables")
        callable_id = "%s:%s" % (module, qualname)

    previous = _CALLABLE_REGISTRY.get(callable_id)
    if previous is not None and previous is not fn:
        raise SerializationError("callable id %r is already registered" % callable_id)
    _CALLABLE_REGISTRY[callable_id] = fn
    _CALLABLE_IDS[fn] = callable_id
    return fn


def serializable_class_ids() -> Set[str]:
    """Return the registered class ids, primarily for diagnostics/tests."""
    ensure_pysp_serialization_registry()
    return set(_CLASS_REGISTRY.keys())


def _iter_distribution_modules(package_name: str) -> Iterable[Any]:
    package = importlib.import_module(package_name)
    yield package
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    prefix = package.__name__ + "."
    for info in pkgutil.iter_modules(package_path, prefix):
        try:
            yield importlib.import_module(info.name)
        except ModuleNotFoundError as err:
            if err.name in _OPTIONAL_IMPORT_NAMES:
                continue
            raise


def ensure_pysp_serialization_registry() -> None:
    """Populate the closed registry of pysp classes that can be decoded."""
    global _REGISTRY_READY
    if _REGISTRY_READY:
        return

    from pysp.stats.pdist import ProbabilityDistribution as StatsDistribution
    from pysp.stats.pdist import ParameterEstimator as StatsEstimator
    from pysp.bstats.pdist import ProbabilityDistribution as BStatsDistribution
    from pysp.bstats.pdist import ParameterEstimator as BStatsEstimator

    for package_name in ("pysp.stats", "pysp.bstats"):
        for module in _iter_distribution_modules(package_name):
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ != module.__name__:
                    continue
                if issubclass(cls, (StatsDistribution, BStatsDistribution, StatsEstimator, BStatsEstimator)):
                    register_serializable_class(cls)
                elif cls.__module__ == "pysp.stats.transform" and cls.__name__.endswith("Transform"):
                    register_serializable_class(cls)

    try:
        automatic = importlib.import_module("pysp.utils.automatic")
        for _, cls in inspect.getmembers(automatic, inspect.isclass):
            if cls.__module__ == automatic.__name__ and issubclass(cls, (StatsDistribution, StatsEstimator)):
                register_serializable_class(cls)
    except Exception:
        # Automatic estimator support is optional for the serializer.  The core
        # stats/bstats registries above should still be available.
        pass

    _REGISTRY_READY = True


def _cycle_enter(value: Any, active: Set[int]) -> int:
    obj_id = id(value)
    if obj_id in active:
        raise SerializationError("cyclic object graph cannot be serialized")
    active.add(obj_id)
    return obj_id


def _cycle_leave(obj_id: int, active: Set[int]) -> None:
    active.remove(obj_id)


def _encode_float(value: float) -> Any:
    value = float(value)
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return {TAG: "float", "value": "nan"}
    return {TAG: "float", "value": "inf" if value > 0.0 else "-inf"}


def _decode_float(value: str) -> float:
    if value == "nan":
        return float("nan")
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    raise SerializationError("unknown special float value %r" % value)


def _sort_key(value: Any) -> str:
    return "%s.%s:%r" % (type(value).__module__, type(value).__qualname__, value)


def _encode_ndarray(value: np.ndarray, active: Set[int]) -> Dict[str, Any]:
    obj_id = _cycle_enter(value, active)
    try:
        return {
            TAG: "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": _encode(value.tolist(), active),
        }
    finally:
        _cycle_leave(obj_id, active)


def _decode_ndarray(payload: Dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(payload["dtype"])
    shape = tuple(int(u) for u in payload["shape"])
    data = _decode(payload["data"])
    return np.asarray(data, dtype=dtype).reshape(shape)


def _encode_sparse(value: Any, active: Set[int]) -> Dict[str, Any]:
    obj_id = _cycle_enter(value, active)
    try:
        coo = value.tocoo()
        return {
            TAG: "sparse",
            "format": value.getformat(),
            "shape": list(value.shape),
            "dtype": str(coo.data.dtype),
            "row": _encode(coo.row, active),
            "col": _encode(coo.col, active),
            "data": _encode(coo.data, active),
        }
    finally:
        _cycle_leave(obj_id, active)


def _decode_sparse(payload: Dict[str, Any]) -> Any:
    if sp is None:
        raise SerializationError("scipy.sparse is required to decode sparse matrices")
    row = np.asarray(_decode(payload["row"]), dtype=np.int64)
    col = np.asarray(_decode(payload["col"]), dtype=np.int64)
    data = np.asarray(_decode(payload["data"]), dtype=np.dtype(payload["dtype"]))
    shape = tuple(int(u) for u in payload["shape"])
    return sp.coo_matrix((data, (row, col)), shape=shape).asformat(payload["format"])


def _encode_dict(value: Dict[Any, Any], active: Set[int]) -> Dict[str, Any]:
    obj_id = _cycle_enter(value, active)
    try:
        items = sorted(value.items(), key=lambda u: _sort_key(u[0]))
        return {
            TAG: "dict",
            "items": [[_encode(k, active), _encode(v, active)] for k, v in items],
        }
    finally:
        _cycle_leave(obj_id, active)


def _decode_dict(payload: Dict[str, Any]) -> Dict[Any, Any]:
    return {_decode(k): _decode(v) for k, v in payload["items"]}


def _encode_sequence(tag: str, value: Iterable[Any], active: Set[int]) -> Dict[str, Any]:
    value_list = list(value)
    return {TAG: tag, "items": [_encode(v, active) for v in value_list]}


def _encode_object(value: Any, active: Set[int]) -> Dict[str, Any]:
    ensure_pysp_serialization_registry()
    cls = value.__class__
    tid = _CLASS_IDS.get(cls)
    if tid is None:
        raise SerializationError("class %s is not registered for pysp JSON serialization" % _type_id(cls))
    if not hasattr(value, "__dict__"):
        raise SerializationError("registered class %s has no __dict__ state" % tid)

    obj_id = _cycle_enter(value, active)
    try:
        return {
            TAG: "object",
            "type": tid,
            "state": _encode(dict(value.__dict__), active),
        }
    finally:
        _cycle_leave(obj_id, active)


def _decode_object(payload: Dict[str, Any]) -> Any:
    ensure_pysp_serialization_registry()
    tid = payload["type"]
    cls = _CLASS_REGISTRY.get(tid)
    if cls is None:
        raise SerializationError("type id %r is not registered for pysp JSON deserialization" % tid)
    state = _decode(payload["state"])
    if not isinstance(state, dict):
        raise SerializationError("serialized object state for %r is not a dict" % tid)
    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    return obj


def _encode_callable(value: Callable[..., Any]) -> Dict[str, Any]:
    callable_id = _CALLABLE_IDS.get(value)
    if callable_id is None:
        raise SerializationError(
            "callable %r is not registered; use register_serializable_callable()" % (value,))
    return {TAG: "callable", "id": callable_id}


def _decode_callable(payload: Dict[str, Any]) -> Callable[..., Any]:
    callable_id = payload["id"]
    fn = _CALLABLE_REGISTRY.get(callable_id)
    if fn is None:
        raise SerializationError("callable id %r is not registered" % callable_id)
    return fn


def _encode(value: Any, active: Set[int]) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return _encode_float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _encode_float(float(value))
    if isinstance(value, bytes):
        return {TAG: "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, np.ndarray):
        return _encode_ndarray(value, active)
    if sp is not None and sp.issparse(value):
        return _encode_sparse(value, active)
    if isinstance(value, tuple):
        obj_id = _cycle_enter(value, active)
        try:
            return _encode_sequence("tuple", value, active)
        finally:
            _cycle_leave(obj_id, active)
    if isinstance(value, range):
        return {TAG: "range", "start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, list):
        obj_id = _cycle_enter(value, active)
        try:
            return [_encode(v, active) for v in value]
        finally:
            _cycle_leave(obj_id, active)
    if isinstance(value, frozenset):
        return _encode_sequence("frozenset", sorted(value, key=_sort_key), active)
    if isinstance(value, set):
        return _encode_sequence("set", sorted(value, key=_sort_key), active)
    if isinstance(value, dict):
        return _encode_dict(value, active)
    if callable(value):
        return _encode_callable(value)
    if hasattr(value, "__dict__"):
        return _encode_object(value, active)
    raise SerializationError("objects of type %s are not JSON serializable by pysp" % _type_id(value.__class__))


def _decode(payload: Any) -> Any:
    if payload is None or isinstance(payload, (bool, str, int, float)):
        return payload
    if isinstance(payload, list):
        return [_decode(v) for v in payload]
    if not isinstance(payload, dict):
        raise SerializationError("unexpected serialized value of type %s" % type(payload).__name__)

    tag = payload.get(TAG)
    if tag is None:
        raise SerializationError("serialized dict is missing %r" % TAG)
    if tag == "float":
        return _decode_float(payload["value"])
    if tag == "bytes":
        return base64.b64decode(payload["data"].encode("ascii"))
    if tag == "ndarray":
        return _decode_ndarray(payload)
    if tag == "sparse":
        return _decode_sparse(payload)
    if tag == "dict":
        return _decode_dict(payload)
    if tag == "tuple":
        return tuple(_decode(v) for v in payload["items"])
    if tag == "set":
        return set(_decode(v) for v in payload["items"])
    if tag == "frozenset":
        return frozenset(_decode(v) for v in payload["items"])
    if tag == "range":
        return range(int(payload["start"]), int(payload["stop"]), int(payload["step"]))
    if tag == "callable":
        return _decode_callable(payload)
    if tag == "object":
        return _decode_object(payload)
    raise SerializationError("unknown pysp JSON tag %r" % tag)


def to_serializable(value: Any) -> Any:
    """Convert a pysp model/value to a JSON-compatible tagged value."""
    return _encode(value, set())


def from_serializable(payload: Any) -> Any:
    """Decode a value produced by ``to_serializable``."""
    return _decode(payload)


def to_json(value: Any, **kwargs: Any) -> str:
    """Serialize a pysp model/value to strict JSON."""
    dump_kwargs = dict(kwargs)
    dump_kwargs["allow_nan"] = False
    dump_kwargs.setdefault("sort_keys", True)
    if "indent" not in dump_kwargs:
        dump_kwargs.setdefault("separators", (",", ":"))
    return json.dumps(to_serializable(value), **dump_kwargs)


def from_json(text: str) -> Any:
    """Deserialize a value produced by ``to_json``."""
    return from_serializable(json.loads(text))
