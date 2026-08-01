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
import threading
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - scipy is a package dependency in normal use.  # noqa: BLE001
    sp = None


TAG = "__pysp_type__"
OBJECT_SCHEMA_VERSION = 1
MAX_DECODE_DEPTH = 128
MAX_DECODE_NODES = 1_000_000
MAX_DECODE_CONTAINER_ITEMS = 1_000_000
MAX_DECODE_BINARY_BYTES = 256 * 1024 * 1024
MAX_JSON_CHARS = 384 * 1024 * 1024

_CLASS_REGISTRY: dict[str, type[Any]] = {}
_CLASS_IDS: dict[type[Any], str] = {}
_CALLABLE_REGISTRY: dict[str, Callable[..., Any]] = {}
_CALLABLE_IDS: dict[Callable[..., Any], str] = {}
_REGISTRY_READY = False
_REGISTRY_LOCK = threading.RLock()
_OPTIONAL_IMPORT_NAMES = {"torch", "umap", "pyspark"}
_OPTIONAL_SERIALIZATION_MODULES = (
    "mixle.models.dpo_leaf",
    "mixle.models.energy",
    "mixle.models.feature_map",
    "mixle.models.grad_leaf",
    "mixle.models.mixture_density",
    "mixle.models.neural_density",
    "mixle.models.neural_families",
    "mixle.models.neural_leaf",
    "mixle.models.pinn",
    "mixle.models.softmax_leaf",
    "mixle.models.streaming_transformer_leaf",
)

# Persistence compatibility is narrower than runtime serializability. These schemas have historical
# fixtures and an explicit v0 -> v1 migration. Every other registered type remains usable for
# same-version round trips but is honestly recorded as provisional in the schema manifest.
# Attributes that describe how an object was PRODUCED, not what it is. They are deliberately dropped
# from the payload and excluded from the declared-field check below: a distribution's serialized state
# is its parameters, and provenance is not one of them.
#
# Not round-tripping them is the point, not a limitation. A deserialized model reports
# ``fit_provenance() is None`` -- "no fit produced this object in this process" -- rather than a
# reconstructed receipt, for the same reason a deserialized authorization is not a grant
# (MXR-080-1725): a record claiming a fit converged is only a record claiming it.
_NON_STATE_ATTRIBUTES = frozenset({"_fit_provenance", "_numerical_repairs"})

_STABLE_STATE_FIELDS: dict[str, frozenset[str]] = {
    "mixle.stats.univariate.continuous.exponential.ExponentialDistribution": frozenset(
        {
            "beta",
            "conj_prior_params",
            "expected_nparams",
            "has_conj_prior",
            "keys",
            "log_beta",
            "name",
            "prior",
        }
    ),
    "mixle.stats.univariate.continuous.gaussian.GaussianDistribution": frozenset(
        {
            "const",
            "expected_nparams",
            "has_conj_prior",
            "keys",
            "log_const",
            "mu",
            "name",
            "prior",
            "sigma2",
        }
    ),
    "mixle.stats.univariate.discrete.categorical.CategoricalDistribution": frozenset(
        {
            "conj_prior_params",
            "default_value",
            "expected_nparams",
            "has_conj_prior",
            "is_normalized_probability",
            "keys",
            "log1p_default_value",
            "log_default_value",
            "name",
            "no_default",
            "pmap",
            "prior",
            "scoring_only",
        }
    ),
    "mixle.stats.univariate.discrete.poisson.PoissonDistribution": frozenset(
        {
            "conj_prior_params",
            "has_conj_prior",
            "keys",
            "lam",
            "log_lambda",
            "name",
            "prior",
        }
    ),
    "mixle.stats.latent.mixture.MixtureDistribution": frozenset(
        {
            "components",
            "conj_prior_params",
            "expected_nparams",
            "has_conj_prior",
            "log_w",
            "name",
            "num_components",
            "prior",
            "w",
            "zw",
        }
    ),
}


def _migrate_stable_v0_state(tid: str, state: dict[str, Any]) -> dict[str, Any]:
    """Apply the five reviewed legacy-schema migrations without mutating the decoded artifact."""
    migrated = dict(state)
    if tid in {
        "mixle.stats.univariate.continuous.exponential.ExponentialDistribution",
        "mixle.stats.univariate.continuous.gaussian.GaussianDistribution",
        "mixle.stats.univariate.discrete.poisson.PoissonDistribution",
    }:
        migrated["keys"] = None
    elif tid == "mixle.stats.univariate.discrete.categorical.CategoricalDistribution":
        # v0 artifacts predate both flags, so neither can be read off the payload. `scoring_only`
        # is an author declaration with no v0 equivalent -- every v0 object was an ordinary
        # distribution -- so it migrates to False. `is_normalized_probability` must NOT simply be
        # asserted True: v0 deliberately permitted unnormalized and empty pmaps, so hard-coding
        # the claim mislabels those artifacts and hands a non-summing-to-1 vector to the sampler.
        # Derive it from the persisted parameters instead, matching what the constructor computes.
        pmap = migrated.get("pmap")
        weights = list(pmap.values()) if isinstance(pmap, dict) else [pair[1] for pair in (pmap or ())]
        normalized = (
            bool(weights)
            and migrated.get("default_value", 0.0) == 0.0
            and math.isclose(sum(weights), 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12)
        )
        migrated.update(keys=None, is_normalized_probability=normalized, scoring_only=False)
    elif tid != "mixle.stats.latent.mixture.MixtureDistribution":
        raise SerializationError("no v0 -> v1 migration is registered for %r" % tid)
    return migrated


# Trust gate for code-executing deserialization (currently: an embedded torch module, persisted as a
# full-object pickle -- see mixle.models._neural_serial). The type-tagged registry walk above this gate
# is closed (only registered mixle classes are reconstructed, never an arbitrary imported class), but a
# NeuralLeaf-family object's state embeds a pickle blob that DOES execute arbitrary code on load, which
# defeats that guarantee wherever such a leaf is nested. Default-closed: any decode that reaches an
# embedded module blob without this gate open raises SerializationError instead of silently unpickling
# it, so "this artifact is JSON-format" is no longer a false safety claim. Callers that DO trust the
# artifact's source open the gate explicitly with :func:`trusted_deserialization`.
class _TrustScope:
    """One synchronous, task-local code-execution authority."""

    def __init__(self) -> None:
        self.thread_id = threading.get_ident()
        self.task = _current_task()
        self.active = True


def _current_task() -> Any:
    try:
        import asyncio

        return asyncio.current_task()
    except RuntimeError:
        return None


_TRUST_CODE_EXECUTION: contextvars.ContextVar[_TrustScope | None] = contextvars.ContextVar(
    "mixle_trust_code_execution", default=None
)


class SerializationError(ValueError):
    """Raised when an object cannot be serialized or decoded safely."""


def deserialization_is_trusted() -> bool:
    """Whether the current context has opted into code-executing deserialization.

    Checked by :mod:`mixle.models._neural_serial` before unpickling an embedded torch module. Not
    needed for the ordinary registry-based path (:func:`from_serializable` on a payload with no
    embedded module blob), which never executes code regardless of this flag.
    """
    scope = _TRUST_CODE_EXECUTION.get()
    return bool(
        scope is not None
        and scope.active
        and scope.thread_id == threading.get_ident()
        and scope.task is _current_task()
    )


@contextlib.contextmanager
def trusted_deserialization() -> Iterator[None]:
    """Permit code-executing deserialization (an embedded torch module) for this ``with`` block.

    Only enter this around an artifact whose SOURCE you trust: within the block, decoding a
    NeuralLeaf-family object (or anything else that persists live code/objects via pickle) executes
    that pickle's ``__reduce__``/``__setstate__`` arbitrarily, exactly like ``pickle.load`` on an
    untrusted file. Nested/re-entrant use is safe (the gate stays open until the outermost block
    exits).

    The authority is deliberately NOT inherited by child work. ``contextvars`` are copied into child
    asyncio tasks and into an explicitly copied context on another thread, so ambient propagation
    would let a task spawned inside this block keep decoding trusted artifacts for its whole
    lifetime -- including after the parent block exited. :class:`_TrustScope` therefore pins the
    owning thread and task, and :func:`deserialization_is_trusted` reports ``False`` anywhere but
    that one synchronous decode context. Open the block again in the child if it genuinely needs
    the authority.
    """
    scope = _TrustScope()
    token = _TRUST_CODE_EXECUTION.set(scope)
    try:
        yield
    finally:
        # ContextVars are copied into child tasks. Mutating this shared scope invalidates those copies
        # when the lexical block exits; the task-identity check above also prevents use while the
        # parent block is still active.
        scope.active = False
        _TRUST_CODE_EXECUTION.reset(token)


def _type_id(cls: type[Any]) -> str:
    return "%s.%s" % (cls.__module__, cls.__name__)


def _register_class(
    registry: dict[str, type[Any]],
    class_ids: dict[type[Any], str],
    cls: type[Any],
    type_id: str | None = None,
) -> None:
    tid = type_id or _type_id(cls)
    previous = registry.get(tid)
    if previous is not None and previous is not cls:
        raise SerializationError("type id %r is already registered for %r" % (tid, previous))
    previous_id = class_ids.get(cls)
    if previous_id is not None and previous_id != tid:
        raise SerializationError("class %r is already registered under type id %r" % (cls, previous_id))
    registry[tid] = cls
    class_ids[cls] = tid


def register_serializable_class(cls: type[Any], type_id: str | None = None) -> type[Any]:
    """Register a class that may be reconstructed from serialized state.

    Deserialization never imports a class named in the payload.  The class must
    already be present in this registry, which is populated from mixle package
    modules by ``ensure_pysp_serialization_registry``.
    """
    with _REGISTRY_LOCK:
        _register_class(_CLASS_REGISTRY, _CLASS_IDS, cls, type_id)
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


def serializable_schema_records(profile: str = "full") -> dict[str, dict[str, Any]]:
    """Return the declared per-type schema records for one generation profile.

    ``base`` excludes torch-backed ``mixle.models`` types. ``full`` includes every type available
    after the optional serialization-module discovery. A generator must separately attest that the
    profile's dependency inventory is installed before treating this record as complete.
    """
    if profile not in {"base", "full"}:
        raise ValueError("serialization profile must be 'base' or 'full'")
    ensure_pysp_serialization_registry()
    ids = sorted(tid for tid in _CLASS_REGISTRY if profile == "full" or not tid.startswith("mixle.models."))
    records: dict[str, dict[str, Any]] = {}
    for tid in ids:
        cls = _CLASS_REGISTRY[tid]
        stable_fields = _STABLE_STATE_FIELDS.get(tid)
        records[tid] = {
            "state_version": OBJECT_SCHEMA_VERSION,
            "stability": "stable" if stable_fields is not None else "provisional",
            "codec": (
                "class-owned"
                if callable(getattr(cls, "__pysp_getstate__", None))
                and callable(getattr(cls, "__pysp_setstate__", None))
                else "constructor-validated"
            ),
            "state_fields": sorted(stable_fields) if stable_fields is not None else None,
            "migrations": ["0->1"] if stable_fields is not None else [],
        }
    return records


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
    global _CLASS_IDS, _CLASS_REGISTRY, _REGISTRY_READY
    if _REGISTRY_READY:
        return

    with _REGISTRY_LOCK:
        if _REGISTRY_READY:
            return

        registry = dict(_CLASS_REGISTRY)
        class_ids = dict(_CLASS_IDS)

        from mixle.stats.compute.pdist import ParameterEstimator as StatsEstimator
        from mixle.stats.compute.pdist import ProbabilityDistribution as StatsDistribution

        for package_name in ("mixle.stats", "mixle.analysis"):
            for module in _iter_distribution_modules(package_name):
                for _, cls in inspect.getmembers(module, inspect.isclass):
                    if cls.__module__ != module.__name__:
                        continue
                    if issubclass(cls, (StatsDistribution, StatsEstimator)):
                        _register_class(registry, class_ids, cls)
                    elif cls.__module__ == "mixle.stats.combinator.transform" and cls.__name__.endswith("Transform"):
                        _register_class(registry, class_ids, cls)
                    elif getattr(cls, "__pysp_serializable__", False):
                        _register_class(registry, class_ids, cls)

        automatic = importlib.import_module("mixle.utils.automatic")
        for _, cls in inspect.getmembers(automatic, inspect.isclass):
            if cls.__module__ == automatic.__name__ and issubclass(cls, (StatsDistribution, StatsEstimator)):
                _register_class(registry, class_ids, cls)

        # Structure-learning distributions (DependencyTreeDistribution and its regression/GLM edges) live in
        # mixle.inference, outside the stats walk above, but opt in explicitly via __pysp_serializable__ so a
        # learned structured model -- e.g. a distilled structured classifier -- persists through the json artifact path.
        structure = importlib.import_module("mixle.inference.structure")
        for _, cls in inspect.getmembers(structure, inspect.isclass):
            if cls.__module__ == structure.__name__ and getattr(cls, "__pysp_serializable__", False):
                _register_class(registry, class_ids, cls)

        # Heterogeneous Bayesian networks (HeterogeneousBayesianNetwork + its per-child factor classes) live
        # in mixle.inference.bayesian_network -- same opt-in mechanism, same reason: optimize(data)'s automatic
        # structure-discovery path (F10.1) returns one of these, and it must survive a save/reload round trip
        # through the same safe json artifact path as everything else, not fall back to raw pickle.
        bn = importlib.import_module("mixle.inference.bayesian_network")
        for _, cls in inspect.getmembers(bn, inspect.isclass):
            if cls.__module__ == bn.__name__ and getattr(cls, "__pysp_serializable__", False):
                _register_class(registry, class_ids, cls)

        # Neural serialization types are optional and are deliberately discovered from an explicit
        # module inventory rather than from import side effects. Missing optional dependencies leave
        # the base profile complete; full-profile generation refuses to run unless its inventory is
        # present.
        for module_name in _OPTIONAL_SERIALIZATION_MODULES:
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as err:
                if err.name in _OPTIONAL_IMPORT_NAMES:
                    continue
                raise
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if (
                    cls.__module__ == module.__name__
                    and not cls.__name__.startswith("_")
                    and getattr(cls, "__pysp_serializable__", False)
                ):
                    _register_class(registry, class_ids, cls)

        # Decorator registrations that occurred while discovery imported modules belong to this same
        # atomic snapshot. Merge them explicitly instead of losing them when the local snapshot is
        # published (TypeDispatch is the current non-subclass example).
        for tid, cls in _CLASS_REGISTRY.items():
            _register_class(registry, class_ids, cls, tid)

        # Publish one complete snapshot. A failed discovery leaves the previous
        # registry intact and _REGISTRY_READY false, so a later call can retry.
        _CLASS_REGISTRY = registry
        _CLASS_IDS = class_ids
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


def _require_fields(payload: dict[str, Any], required: set[str], optional: set[str] | None = None) -> None:
    """Validate one tagged payload as a closed schema."""
    allowed = required | (optional or set())
    actual = set(payload)
    missing = required - actual
    extra = actual - allowed
    if missing or extra:
        raise SerializationError(
            "invalid %r payload fields (missing=%r, extra=%r)" % (payload.get(TAG), sorted(missing), sorted(extra))
        )


def _exact_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise SerializationError("%s must be an exact non-Boolean integer" % name)
    result = int(value)
    if minimum is not None and result < minimum:
        raise SerializationError("%s must be >= %d" % (name, minimum))
    return result


def _decode_base64(value: Any, *, name: str) -> bytes:
    if not isinstance(value, str):
        raise SerializationError("%s must be an ASCII base64 string" % name)
    if len(value) > ((MAX_DECODE_BINARY_BYTES + 2) // 3) * 4:
        raise SerializationError("%s exceeds the %d-byte decode limit" % (name, MAX_DECODE_BINARY_BYTES))
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise SerializationError("%s is not valid base64" % name) from exc
    if len(decoded) > MAX_DECODE_BINARY_BYTES:
        raise SerializationError("%s exceeds the %d-byte decode limit" % (name, MAX_DECODE_BINARY_BYTES))
    return decoded


def _canonical_sort_key(value: Any) -> bytes:
    """Canonical encoded bytes used for deterministic dict/set ordering."""
    encoded = to_serializable(value)
    try:
        return json.dumps(encoded, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SerializationError("value has no canonical JSON ordering key") from exc


def _encode_ndarray(value: np.ndarray, active: set[int], memo: dict[int, str]) -> dict[str, Any]:
    obj_id = _cycle_enter(value, active)
    try:
        if value.dtype.hasobject:
            raise SerializationError("object-dtype ndarrays are not supported by the canonical binary codec")
        contiguous = np.ascontiguousarray(value)
        dtype_spec: Any
        if value.dtype.fields is None:
            dtype_spec = {"kind": "str", "value": value.dtype.str}
        else:
            dtype_spec = {"kind": "descr", "value": _encode(value.dtype.descr, active, memo)}
        return {
            TAG: "ndarray",
            "dtype": dtype_spec,
            "shape": list(value.shape),
            "data": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
        }
    finally:
        _cycle_leave(obj_id, active)


def _decode_ndarray(payload: dict[str, Any], references: dict[str, Any], depth: int, budget: list[int]) -> np.ndarray:
    _require_fields(payload, {TAG, "dtype", "shape", "data"}, {"id"})
    dtype_payload = payload["dtype"]
    if isinstance(dtype_payload, str):  # legacy v0.7/v0.8 payload
        dtype = np.dtype(dtype_payload)
        shape = tuple(_exact_int(u, name="ndarray shape", minimum=0) for u in payload["shape"])
        data = _decode(payload["data"], references, depth + 1, budget)
        try:
            return np.asarray(data, dtype=dtype).reshape(shape)
        except (TypeError, ValueError) as exc:
            raise SerializationError("invalid legacy ndarray payload") from exc
    if not isinstance(dtype_payload, dict):
        raise SerializationError("ndarray dtype must be a tagged dtype descriptor")
    _require_fields(dtype_payload, {"kind", "value"})
    if dtype_payload["kind"] == "str" and isinstance(dtype_payload["value"], str):
        dtype = np.dtype(dtype_payload["value"])
    elif dtype_payload["kind"] == "descr":
        dtype = np.dtype(_decode(dtype_payload["value"], references, depth + 1, budget))
    else:
        raise SerializationError("invalid ndarray dtype descriptor")
    if not isinstance(payload["shape"], list):
        raise SerializationError("ndarray shape must be a list")
    if len(payload["shape"]) > MAX_DECODE_DEPTH:
        raise SerializationError("ndarray has too many dimensions")
    shape = tuple(_exact_int(u, name="ndarray shape", minimum=0) for u in payload["shape"])
    raw = _decode_base64(payload["data"], name="ndarray data")
    expected = math.prod(shape) * dtype.itemsize
    if expected > MAX_DECODE_BINARY_BYTES:
        raise SerializationError("ndarray exceeds the %d-byte decode limit" % MAX_DECODE_BINARY_BYTES)
    if len(raw) != expected:
        raise SerializationError("ndarray byte length %d does not match expected %d" % (len(raw), expected))
    return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)


def _encode_sparse(value: Any, active: set[int], memo: dict[int, str]) -> dict[str, Any]:
    obj_id = _cycle_enter(value, active)
    try:
        coo = value.tocoo()
        return {
            TAG: "sparse",
            "format": value.getformat(),
            "shape": list(value.shape),
            "dtype": str(coo.data.dtype),
            "row": _encode(coo.row, active, memo),
            "col": _encode(coo.col, active, memo),
            "data": _encode(coo.data, active, memo),
        }
    finally:
        _cycle_leave(obj_id, active)


def _decode_sparse(payload: dict[str, Any], references: dict[str, Any], depth: int, budget: list[int]) -> Any:
    _require_fields(payload, {TAG, "format", "shape", "dtype", "row", "col", "data"}, {"id"})
    if sp is None:
        raise SerializationError("scipy.sparse is required to decode sparse matrices")
    row = np.asarray(_decode(payload["row"], references, depth + 1, budget), dtype=np.int64)
    col = np.asarray(_decode(payload["col"], references, depth + 1, budget), dtype=np.int64)
    data = np.asarray(_decode(payload["data"], references, depth + 1, budget), dtype=np.dtype(payload["dtype"]))
    if not isinstance(payload["shape"], list) or len(payload["shape"]) != 2:
        raise SerializationError("sparse shape must contain exactly two dimensions")
    shape = tuple(_exact_int(u, name="sparse shape", minimum=0) for u in payload["shape"])
    if row.shape != col.shape or row.shape != data.shape:
        raise SerializationError("sparse row, column, and data arrays must have identical shape")
    if np.any(row < 0) or np.any(col < 0) or np.any(row >= shape[0]) or np.any(col >= shape[1]):
        raise SerializationError("sparse coordinates fall outside the declared shape")
    if payload["format"] not in {"coo", "csr", "csc", "bsr", "dia", "dok", "lil"}:
        raise SerializationError("unsupported sparse format %r" % payload["format"])
    return sp.coo_matrix((data, (row, col)), shape=shape).asformat(payload["format"])


def _encode_dict(value: dict[Any, Any], active: set[int], memo: dict[int, str]) -> dict[str, Any]:
    obj_id = _cycle_enter(value, active)
    try:
        encoded_keys = [(_canonical_sort_key(key), key, item) for key, item in value.items()]
        encoded_keys.sort(key=lambda row: row[0])
        if any(encoded_keys[i - 1][0] == encoded_keys[i][0] for i in range(1, len(encoded_keys))):
            raise SerializationError("dictionary keys have colliding canonical encodings")
        return {
            TAG: "dict",
            "items": [[_encode(k, active, memo), _encode(v, active, memo)] for _, k, v in encoded_keys],
        }
    finally:
        _cycle_leave(obj_id, active)


def _decode_dict(payload: dict[str, Any], references: dict[str, Any], depth: int, budget: list[int]) -> dict[Any, Any]:
    _require_fields(payload, {TAG, "items"}, {"id"})
    if not isinstance(payload["items"], list):
        raise SerializationError("dict items must be a list")
    if len(payload["items"]) > MAX_DECODE_CONTAINER_ITEMS:
        raise SerializationError("dict exceeds the %d-item decode limit" % MAX_DECODE_CONTAINER_ITEMS)
    result: dict[Any, Any] = {}
    for pair in payload["items"]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise SerializationError("each serialized dict item must be a [key, value] pair")
        key = _decode(pair[0], references, depth + 1, budget)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise SerializationError("decoded dictionary key is unhashable") from exc
        if duplicate:
            raise SerializationError("serialized dictionary contains duplicate decoded keys")
        result[key] = _decode(pair[1], references, depth + 1, budget)
    return result


def _encode_sequence(tag: str, value: Iterable[Any], active: set[int], memo: dict[int, str]) -> dict[str, Any]:
    value_list = list(value)
    return {TAG: tag, "items": [_encode(v, active, memo) for v in value_list]}


def _encode_object(value: Any, active: set[int], memo: dict[int, str]) -> dict[str, Any]:
    ensure_pysp_serialization_registry()
    cls = value.__class__
    tid = _CLASS_IDS.get(cls)
    if tid is None:
        raise SerializationError("class %s is not registered for mixle JSON serialization" % _type_id(cls))
    if not hasattr(value, "__dict__"):
        raise SerializationError("registered class %s has no __dict__ state" % tid)

    object_id = id(value)
    if object_id in active:
        raise SerializationError("cyclic object graph cannot be serialized")
    reference_id = memo.get(object_id)
    if reference_id is not None:
        return {TAG: "ref", "id": reference_id}
    reference_id = "o%d" % len(memo)
    memo[object_id] = reference_id
    state_getter = getattr(value, "__pysp_getstate__", None)
    obj_id = _cycle_enter(value, active)
    try:
        state = state_getter() if callable(state_getter) else dict(value.__dict__)
        for attribute in _NON_STATE_ATTRIBUTES:
            state.pop(attribute, None)
        stable_fields = _STABLE_STATE_FIELDS.get(tid)
        if stable_fields is not None and set(state) != stable_fields:
            raise SerializationError(
                "state for stable constructor-owned schema %r does not match its declared fields "
                "(missing=%r, extra=%r)" % (tid, sorted(stable_fields - set(state)), sorted(set(state) - stable_fields))
            )
        return {
            TAG: "object",
            "id": reference_id,
            "type": tid,
            "schema_version": OBJECT_SCHEMA_VERSION,
            "state": _encode(state, active, memo),
        }
    finally:
        _cycle_leave(obj_id, active)


def _constructor_decode(cls: type[Any], state: dict[str, Any], tid: str) -> Any:
    """Construct and validate a registered object without bypassing its invariants."""
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError) as exc:
        raise SerializationError(
            "registered class %r has no safe decode hook or inspectable constructor" % tid
        ) from exc

    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    accepts_kwargs = False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_kwargs = True
        elif parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            if parameter.name in state:
                kwargs[parameter.name] = state[parameter.name]
            elif parameter.default is inspect.Parameter.empty:
                missing.append(parameter.name)
        elif parameter.kind == inspect.Parameter.POSITIONAL_ONLY and parameter.default is inspect.Parameter.empty:
            missing.append(parameter.name)
    if missing:
        raise SerializationError(
            "registered class %r requires a class-owned __pysp_setstate__ hook; "
            "constructor fields are absent: %s" % (tid, ", ".join(missing))
        )
    if accepts_kwargs:
        kwargs.update({key: value for key, value in state.items() if key not in kwargs})

    try:
        obj = cls(**kwargs)
    except Exception as exc:
        raise SerializationError("constructor rejected serialized state for %r" % tid) from exc
    if not hasattr(obj, "__dict__"):
        raise SerializationError("constructor for registered class %r produced no state dictionary" % tid)

    # The constructor owns both validation and normalization. Its resulting
    # state must exactly match the artifact, including the field set; otherwise
    # accepting the artifact would reintroduce an invariant-bypassing update.
    try:
        constructed = dict(obj.__dict__)
        for attribute in _NON_STATE_ATTRIBUTES:
            # Symmetric with the write side: the constructor may set provenance (a rebuilt covariance
            # records its own jitter heal), and the artifact never carries it. Comparing it would fail
            # a round trip over a difference that is not parameter state.
            constructed.pop(attribute, None)
        encoded_artifact = json.dumps(to_serializable(state), sort_keys=True, separators=(",", ":"), allow_nan=False)
        encoded_constructed = json.dumps(
            to_serializable(constructed), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise SerializationError("cannot validate serialized state schema for %r" % tid) from exc
    if encoded_artifact != encoded_constructed:
        raise SerializationError(
            "serialized state for %r does not match its constructor-owned schema; "
            "define __pysp_setstate__ for validated custom state" % tid
        )
    return obj


def _decode_object(payload: dict[str, Any], references: dict[str, Any], depth: int, budget: list[int]) -> Any:
    _require_fields(payload, {TAG, "type", "state"}, {"id", "schema_version"})
    ensure_pysp_serialization_registry()
    tid = payload["type"]
    if not isinstance(tid, str):
        raise SerializationError("serialized object type id must be a string")
    cls = _CLASS_REGISTRY.get(tid)
    if cls is None:
        raise SerializationError("type id %r is not registered for mixle JSON deserialization" % tid)
    schema_version = payload.get("schema_version", 0)
    schema_version = _exact_int(schema_version, name="object schema_version", minimum=0)
    legacy_schema = schema_version == 0
    if legacy_schema:
        if tid not in _STABLE_STATE_FIELDS:
            raise SerializationError(
                "legacy unversioned state for provisional type %r is unsupported; "
                "load it with the originating Mixle version and re-save it" % tid
            )
        schema_version = OBJECT_SCHEMA_VERSION
    if schema_version != OBJECT_SCHEMA_VERSION:
        raise SerializationError(
            "unsupported schema_version %d for %r; supported version is %d"
            % (schema_version, tid, OBJECT_SCHEMA_VERSION)
        )
    reference_id = payload.get("id")
    if reference_id is not None:
        if not isinstance(reference_id, str) or not reference_id:
            raise SerializationError("serialized object reference id must be a non-empty string")
        if reference_id in references:
            raise SerializationError("duplicate serialized object reference id %r" % reference_id)
    state = _decode(payload["state"], references, depth + 1, budget)
    if not isinstance(state, dict):
        raise SerializationError("serialized object state for %r is not a dict" % tid)
    if legacy_schema:
        state = _migrate_stable_v0_state(tid, state)
    stable_fields = _STABLE_STATE_FIELDS.get(tid)
    if stable_fields is not None and set(state) != stable_fields:
        raise SerializationError(
            "serialized state for stable constructor-owned schema %r does not match version %d fields "
            "(missing=%r, extra=%r)"
            % (
                tid,
                schema_version,
                sorted(stable_fields - set(state)),
                sorted(set(state) - stable_fields),
            )
        )
    if callable(getattr(cls, "__pysp_setstate__", None)):
        obj = cls.__new__(cls)
        obj.__pysp_setstate__(state)
    else:
        obj = _constructor_decode(cls, state, tid)
    if reference_id is not None:
        references[reference_id] = obj
    return obj


def _encode_callable(value: Callable[..., Any]) -> dict[str, Any]:
    callable_id = _CALLABLE_IDS.get(value)
    if callable_id is None:
        raise SerializationError("callable %r is not registered; use register_serializable_callable()" % (value,))
    return {TAG: "callable", "id": callable_id}


def _decode_callable(payload: dict[str, Any]) -> Callable[..., Any]:
    _require_fields(payload, {TAG, "id"})
    callable_id = payload["id"]
    if not isinstance(callable_id, str):
        raise SerializationError("serialized callable id must be a string")
    fn = _CALLABLE_REGISTRY.get(callable_id)
    if fn is None:
        raise SerializationError("callable id %r is not registered" % callable_id)
    return fn


def _encode(value: Any, active: set[int], memo: dict[int, str]) -> Any:
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
        return _encode_ndarray(value, active, memo)
    if sp is not None and sp.issparse(value):
        return _encode_sparse(value, active, memo)
    if isinstance(value, tuple):
        obj_id = _cycle_enter(value, active)
        try:
            return _encode_sequence("tuple", value, active, memo)
        finally:
            _cycle_leave(obj_id, active)
    if isinstance(value, range):
        return {TAG: "range", "start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, list):
        obj_id = _cycle_enter(value, active)
        try:
            return [_encode(v, active, memo) for v in value]
        finally:
            _cycle_leave(obj_id, active)
    if isinstance(value, frozenset):
        return _encode_sequence("frozenset", sorted(value, key=_canonical_sort_key), active, memo)
    if isinstance(value, set):
        return _encode_sequence("set", sorted(value, key=_canonical_sort_key), active, memo)
    if isinstance(value, dict):
        return _encode_dict(value, active, memo)
    if callable(value) or hasattr(value, "__dict__"):
        # Instances of a registered serializable class encode via their object state even when they
        # are callable (e.g. a data-carrying routing object), so they round-trip from their __dict__
        # without needing a process-local callable id. Plain functions/lambdas still need one.
        ensure_pysp_serialization_registry()
        if callable(value) and value.__class__ not in _CLASS_IDS:
            return _encode_callable(value)
        return _encode_object(value, active, memo)
    raise SerializationError("objects of type %s are not JSON serializable by mixle" % _type_id(value.__class__))


def _decode(payload: Any, references: dict[str, Any], depth: int, budget: list[int]) -> Any:
    if depth > MAX_DECODE_DEPTH:
        raise SerializationError("serialized value exceeds the maximum decode depth of %d" % MAX_DECODE_DEPTH)
    budget[0] += 1
    if budget[0] > MAX_DECODE_NODES:
        raise SerializationError("serialized value exceeds the maximum decode node count of %d" % MAX_DECODE_NODES)
    if payload is None or isinstance(payload, (bool, str, int)):
        return payload
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise SerializationError("raw non-finite JSON numbers are forbidden; use the tagged float form")
        return payload
    if isinstance(payload, list):
        if len(payload) > MAX_DECODE_CONTAINER_ITEMS:
            raise SerializationError("list exceeds the %d-item decode limit" % MAX_DECODE_CONTAINER_ITEMS)
        return [_decode(v, references, depth + 1, budget) for v in payload]
    if not isinstance(payload, dict):
        raise SerializationError("unexpected serialized value of type %s" % type(payload).__name__)

    tag = payload.get(TAG)
    if tag is None:
        raise SerializationError("serialized dict is missing %r" % TAG)
    if tag == "float":
        _require_fields(payload, {TAG, "value"})
        if not isinstance(payload["value"], str):
            raise SerializationError("special float value must be a string")
        return _decode_float(payload["value"])
    if tag == "bytes":
        _require_fields(payload, {TAG, "data"})
        return _decode_base64(payload["data"], name="bytes data")
    if tag == "ndarray":
        return _decode_ndarray(payload, references, depth, budget)
    if tag == "sparse":
        return _decode_sparse(payload, references, depth, budget)
    if tag == "dict":
        return _decode_dict(payload, references, depth, budget)
    if tag == "tuple":
        _require_fields(payload, {TAG, "items"}, {"id"})
        if not isinstance(payload["items"], list):
            raise SerializationError("tuple items must be a list")
        if len(payload["items"]) > MAX_DECODE_CONTAINER_ITEMS:
            raise SerializationError("tuple exceeds the %d-item decode limit" % MAX_DECODE_CONTAINER_ITEMS)
        return tuple(_decode(v, references, depth + 1, budget) for v in payload["items"])
    if tag == "set":
        _require_fields(payload, {TAG, "items"}, {"id"})
        if not isinstance(payload["items"], list):
            raise SerializationError("set items must be a list")
        if len(payload["items"]) > MAX_DECODE_CONTAINER_ITEMS:
            raise SerializationError("set exceeds the %d-item decode limit" % MAX_DECODE_CONTAINER_ITEMS)
        values = [_decode(v, references, depth + 1, budget) for v in payload["items"]]
        try:
            result = set(values)
        except TypeError as exc:
            raise SerializationError("decoded set member is unhashable") from exc
        if len(result) != len(values):
            raise SerializationError("serialized set contains duplicate decoded members")
        return result
    if tag == "frozenset":
        _require_fields(payload, {TAG, "items"}, {"id"})
        if not isinstance(payload["items"], list):
            raise SerializationError("frozenset items must be a list")
        if len(payload["items"]) > MAX_DECODE_CONTAINER_ITEMS:
            raise SerializationError("frozenset exceeds the %d-item decode limit" % MAX_DECODE_CONTAINER_ITEMS)
        values = [_decode(v, references, depth + 1, budget) for v in payload["items"]]
        try:
            result = frozenset(values)
        except TypeError as exc:
            raise SerializationError("decoded frozenset member is unhashable") from exc
        if len(result) != len(values):
            raise SerializationError("serialized frozenset contains duplicate decoded members")
        return result
    if tag == "range":
        _require_fields(payload, {TAG, "start", "stop", "step"})
        start = _exact_int(payload["start"], name="range start")
        stop = _exact_int(payload["stop"], name="range stop")
        step = _exact_int(payload["step"], name="range step")
        if step == 0:
            raise SerializationError("range step cannot be zero")
        return range(start, stop, step)
    if tag == "callable":
        return _decode_callable(payload)
    if tag == "object":
        return _decode_object(payload, references, depth, budget)
    if tag == "ref":
        _require_fields(payload, {TAG, "id"})
        reference_id = payload["id"]
        if not isinstance(reference_id, str) or reference_id not in references:
            raise SerializationError("unresolved serialized object reference %r" % reference_id)
        return references[reference_id]
    raise SerializationError("unknown mixle JSON tag %r" % tag)


def to_serializable(value: Any) -> Any:
    """Convert a mixle model/value to a JSON-compatible tagged value."""
    return _encode(value, set(), {})


def from_serializable(payload: Any) -> Any:
    """Decode a value produced by ``to_serializable``."""
    return _decode(payload, {}, 0, [0])


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
    if not isinstance(text, str):
        raise TypeError("serialized JSON input must be a string")
    if len(text) > MAX_JSON_CHARS:
        raise SerializationError("serialized JSON exceeds the %d-character decode limit" % MAX_JSON_CHARS)

    def reject_constant(value: str) -> Any:
        raise SerializationError("non-standard JSON constant %r is forbidden" % value)

    try:
        payload = json.loads(text, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise SerializationError("invalid JSON") from exc
    return from_serializable(payload)
