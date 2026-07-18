"""Safe JSON serialization helpers for mixle distribution and estimator objects.

The legacy ``load_models(eval_string)`` path reconstructed models by executing
their repr strings.  This module instead serializes distribution state as a
small tagged JSON value graph and reconstructs only classes registered from
mixle's own distribution modules.
"""

from __future__ import annotations

import base64
import contextlib
import contextvars
import importlib
import inspect
import json
import math
import pkgutil
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - scipy is a package dependency in normal use.  # noqa: BLE001
    sp = None


TAG = "__pysp_type__"

_CLASS_REGISTRY: dict[str, type[Any]] = {}
_CLASS_IDS: dict[type[Any], str] = {}
_CALLABLE_REGISTRY: dict[str, Callable[..., Any]] = {}
_CALLABLE_IDS: dict[Callable[..., Any], str] = {}
_REGISTRY_READY = False
_OPTIONAL_IMPORT_NAMES = {"torch", "umap", "pyspark"}

# Trust gate for code-executing deserialization (currently: an embedded torch module, persisted as a
# full-object pickle -- see mixle.models._neural_serial). The type-tagged registry walk above this gate
# is closed (only registered mixle classes are reconstructed, never an arbitrary imported class), but a
# NeuralLeaf-family object's state embeds a pickle blob that DOES execute arbitrary code on load, which
# defeats that guarantee wherever such a leaf is nested. Default-closed: any decode that reaches an
# embedded module blob without this gate open raises SerializationError instead of silently unpickling
# it, so "this artifact is JSON-format" is no longer a false safety claim. Callers that DO trust the
# artifact's source open the gate explicitly with :func:`trusted_deserialization`.
_TRUST_CODE_EXECUTION: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "mixle_trust_code_execution", default=False
)


class SerializationError(ValueError):
    """Raised when an object cannot be serialized or decoded safely."""


def deserialization_is_trusted() -> bool:
    """Whether the current context has opted into code-executing deserialization.

    Checked by :mod:`mixle.models._neural_serial` before unpickling an embedded torch module. Not
    needed for the ordinary registry-based path (:func:`from_serializable` on a payload with no
    embedded module blob), which never executes code regardless of this flag.
    """
    return _TRUST_CODE_EXECUTION.get()


@contextlib.contextmanager
def trusted_deserialization() -> Iterator[None]:
    """Permit code-executing deserialization (an embedded torch module) for this ``with`` block.

    Only enter this around an artifact whose SOURCE you trust: within the block, decoding a
    NeuralLeaf-family object (or anything else that persists live code/objects via pickle) executes
    that pickle's ``__reduce__``/``__setstate__`` arbitrarily, exactly like ``pickle.load`` on an
    untrusted file. Nested/re-entrant use is safe (the gate stays open until the outermost block
    exits); safe to use across threads/async tasks via ``contextvars`` propagation.
    """
    token = _TRUST_CODE_EXECUTION.set(True)
    try:
        yield
    finally:
        _TRUST_CODE_EXECUTION.reset(token)


def _type_id(cls: type[Any]) -> str:
    return "%s.%s" % (cls.__module__, cls.__name__)


def register_serializable_class(cls: type[Any], type_id: str | None = None) -> type[Any]:
    """Register a class that may be reconstructed from serialized state.

    Deserialization never imports a class named in the payload.  The class must
    already be present in this registry, which is populated from mixle package
    modules by ``ensure_pysp_serialization_registry``.
    """
    tid = type_id or _type_id(cls)
    previous = _CLASS_REGISTRY.get(tid)
    if previous is not None and previous is not cls:
        raise SerializationError("type id %r is already registered for %r" % (tid, previous))
    _CLASS_REGISTRY[tid] = cls
    _CLASS_IDS[cls] = tid
    return cls


def register_serializable_callable(fn: Callable[..., Any], callable_id: str | None = None) -> Callable[..., Any]:
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


def serializable_class_ids() -> set[str]:
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
    # walk_packages (not iter_modules) so distributions grouped into subpackages
    # (mixle.stats.univariate.continuous, mixle.stats.latent, ...) are still discovered for the registry.
    for info in pkgutil.walk_packages(package_path, prefix):
        try:
            yield importlib.import_module(info.name)
        except ModuleNotFoundError as err:
            if err.name in _OPTIONAL_IMPORT_NAMES:
                continue
            raise


def ensure_pysp_serialization_registry() -> None:
    """Populate the closed registry of mixle classes that can be decoded."""
    global _REGISTRY_READY
    if _REGISTRY_READY:
        return

    from mixle.stats.compute.pdist import ParameterEstimator as StatsEstimator
    from mixle.stats.compute.pdist import ProbabilityDistribution as StatsDistribution

    for package_name in ("mixle.stats", "mixle.analysis"):
        for module in _iter_distribution_modules(package_name):
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ != module.__name__:
                    continue
                if issubclass(cls, (StatsDistribution, StatsEstimator)):
                    register_serializable_class(cls)
                elif cls.__module__ == "mixle.stats.combinator.transform" and cls.__name__.endswith("Transform"):
                    register_serializable_class(cls)
                elif getattr(cls, "__pysp_serializable__", False):
                    register_serializable_class(cls)

    try:
        automatic = importlib.import_module("mixle.utils.automatic")
        for _, cls in inspect.getmembers(automatic, inspect.isclass):
            if cls.__module__ == automatic.__name__ and issubclass(cls, (StatsDistribution, StatsEstimator)):
                register_serializable_class(cls)
    except Exception:  # noqa: BLE001
        # Automatic estimator support is optional for the serializer.  The core
        # stats/bstats registries above should still be available.
        pass

    # Structure-learning distributions (DependencyTreeDistribution and its regression/GLM edges) live in
    # mixle.inference, outside the stats walk above, but opt in explicitly via __pysp_serializable__ so a
    # learned structured model -- e.g. a distilled structured classifier -- persists through the json artifact path.
    try:
        structure = importlib.import_module("mixle.inference.structure")
        for _, cls in inspect.getmembers(structure, inspect.isclass):
            if cls.__module__ == structure.__name__ and getattr(cls, "__pysp_serializable__", False):
                register_serializable_class(cls)
    except Exception:  # noqa: BLE001
        # Optional: the core stats registry above is enough for pure-stats models.
        pass

    # Heterogeneous Bayesian networks (HeterogeneousBayesianNetwork + its per-child factor classes) live
    # in mixle.inference.bayesian_network -- same opt-in mechanism, same reason: optimize(data)'s automatic
    # structure-discovery path (F10.1) returns one of these, and it must survive a save/reload round trip
    # through the same safe json artifact path as everything else, not fall back to raw pickle.
    try:
        bn = importlib.import_module("mixle.inference.bayesian_network")
        for _, cls in inspect.getmembers(bn, inspect.isclass):
            if cls.__module__ == bn.__name__ and getattr(cls, "__pysp_serializable__", False):
                register_serializable_class(cls)
    except Exception:  # noqa: BLE001
        # Optional: the core stats registry above is enough for pure-stats models.
        pass

    _REGISTRY_READY = True


def _cycle_enter(value: Any, active: set[int]) -> int:
    obj_id = id(value)
    if obj_id in active:
        raise SerializationError("cyclic object graph cannot be serialized")
    active.add(obj_id)
    return obj_id


def _cycle_leave(obj_id: int, active: set[int]) -> None:
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


def _encode_ndarray(value: np.ndarray, active: set[int]) -> dict[str, Any]:
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


def _decode_ndarray(payload: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(payload["dtype"])
    shape = tuple(int(u) for u in payload["shape"])
    data = _decode(payload["data"])
    return np.asarray(data, dtype=dtype).reshape(shape)


def _encode_sparse(value: Any, active: set[int]) -> dict[str, Any]:
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


def _decode_sparse(payload: dict[str, Any]) -> Any:
    if sp is None:
        raise SerializationError("scipy.sparse is required to decode sparse matrices")
    row = np.asarray(_decode(payload["row"]), dtype=np.int64)
    col = np.asarray(_decode(payload["col"]), dtype=np.int64)
    data = np.asarray(_decode(payload["data"]), dtype=np.dtype(payload["dtype"]))
    shape = tuple(int(u) for u in payload["shape"])
    return sp.coo_matrix((data, (row, col)), shape=shape).asformat(payload["format"])


def _encode_dict(value: dict[Any, Any], active: set[int]) -> dict[str, Any]:
    obj_id = _cycle_enter(value, active)
    try:
        items = sorted(value.items(), key=lambda u: _sort_key(u[0]))
        return {
            TAG: "dict",
            "items": [[_encode(k, active), _encode(v, active)] for k, v in items],
        }
    finally:
        _cycle_leave(obj_id, active)


def _decode_dict(payload: dict[str, Any]) -> dict[Any, Any]:
    return {_decode(k): _decode(v) for k, v in payload["items"]}


def _encode_sequence(tag: str, value: Iterable[Any], active: set[int]) -> dict[str, Any]:
    value_list = list(value)
    return {TAG: tag, "items": [_encode(v, active) for v in value_list]}


def _encode_object(value: Any, active: set[int]) -> dict[str, Any]:
    ensure_pysp_serialization_registry()
    cls = value.__class__
    tid = _CLASS_IDS.get(cls)
    if tid is None:
        raise SerializationError("class %s is not registered for mixle JSON serialization" % _type_id(cls))
    if not hasattr(value, "__dict__"):
        raise SerializationError("registered class %s has no __dict__ state" % tid)

    state_getter = getattr(value, "__pysp_getstate__", None)
    obj_id = _cycle_enter(value, active)
    try:
        state = state_getter() if callable(state_getter) else dict(value.__dict__)
        return {
            TAG: "object",
            "type": tid,
            "state": _encode(state, active),
        }
    finally:
        _cycle_leave(obj_id, active)


def _decode_object(payload: dict[str, Any]) -> Any:
    ensure_pysp_serialization_registry()
    tid = payload["type"]
    cls = _CLASS_REGISTRY.get(tid)
    if cls is None:
        raise SerializationError("type id %r is not registered for mixle JSON deserialization" % tid)
    state = _decode(payload["state"])
    if not isinstance(state, dict):
        raise SerializationError("serialized object state for %r is not a dict" % tid)
    obj = cls.__new__(cls)
    state_setter = getattr(obj, "__pysp_setstate__", None)
    if callable(state_setter):
        state_setter(state)
    else:
        obj.__dict__.update(state)
    return obj


def _encode_callable(value: Callable[..., Any]) -> dict[str, Any]:
    callable_id = _CALLABLE_IDS.get(value)
    if callable_id is None:
        raise SerializationError("callable %r is not registered; use register_serializable_callable()" % (value,))
    return {TAG: "callable", "id": callable_id}


def _decode_callable(payload: dict[str, Any]) -> Callable[..., Any]:
    callable_id = payload["id"]
    fn = _CALLABLE_REGISTRY.get(callable_id)
    if fn is None:
        raise SerializationError("callable id %r is not registered" % callable_id)
    return fn


def _encode(value: Any, active: set[int]) -> Any:
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
    if callable(value) or hasattr(value, "__dict__"):
        # Instances of a registered serializable class encode via their object state even when they
        # are callable (e.g. a data-carrying routing object), so they round-trip from their __dict__
        # without needing a process-local callable id. Plain functions/lambdas still need one.
        ensure_pysp_serialization_registry()
        if callable(value) and value.__class__ not in _CLASS_IDS:
            return _encode_callable(value)
        return _encode_object(value, active)
    raise SerializationError("objects of type %s are not JSON serializable by mixle" % _type_id(value.__class__))


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
    raise SerializationError("unknown mixle JSON tag %r" % tag)


def to_serializable(value: Any) -> Any:
    """Convert a mixle model/value to a JSON-compatible tagged value."""
    return _encode(value, set())


def from_serializable(payload: Any) -> Any:
    """Decode a value produced by ``to_serializable``."""
    return _decode(payload)


def to_json(value: Any, **kwargs: Any) -> str:
    """Serialize a mixle model/value to strict JSON."""
    dump_kwargs = dict(kwargs)
    dump_kwargs["allow_nan"] = False
    dump_kwargs.setdefault("sort_keys", True)
    if "indent" not in dump_kwargs:
        dump_kwargs.setdefault("separators", (",", ":"))
    return json.dumps(to_serializable(value), **dump_kwargs)


def from_json(text: str) -> Any:
    """Deserialize a value produced by ``to_json``."""
    return from_serializable(json.loads(text))
