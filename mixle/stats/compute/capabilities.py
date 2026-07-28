"""Distribution capability metadata for engine and planner decisions.

The registry records which distribution families can execute on each compute
engine, which kernels are generic or specialized, and why some families remain
intentionally NumPy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import getattr_static, signature
from threading import RLock
from typing import Any

KNOWN_COMPUTE_ENGINES = frozenset({"jax", "numpy", "symbolic", "torch"})
KERNEL_STATUSES = frozenset(
    {
        "explicit_stacked",
        "generic",
        "generic_composite",
        "generic_latent",
        "generic_object",
        "generic_table",
        "legacy_numpy",
        "numba_adapter",
        "numpy_only",
    }
)

# Engines that compose safely through combinators and wrappers: every combinator/wrapper kernel is
# verified on these. A leaf may additionally declare a scoring-only engine (e.g. 'jax') for direct
# fitting, but composition does NOT propagate it -- combinators (via intersect_engine_ready) and
# delegating wrappers (via delegated_engine_ready) cap to this set so a model never *claims* an engine
# its kernel does not actually support. Widen this only after verifying the new engine on every
# combinator/wrapper kernel.
COMPOSITION_ENGINES: tuple[str, ...] = ("numpy", "torch")


def delegated_engine_ready(child_engine_ready: tuple[str, ...]) -> tuple[str, ...]:
    """Engines a delegating wrapper (Weighted/Ignored/Transform) may report: the child's engines capped
    to the composition-safe set. A no-op for numpy/torch children; it only drops leaf-only engines the
    wrapper kernel has not been verified to support."""
    have = set(child_engine_ready)
    return tuple(name for name in COMPOSITION_ENGINES if name in have)


@dataclass(frozen=True)
class DistributionCapabilities:
    """Runtime capability metadata for a distribution family."""

    engine_ready: tuple[str, ...] = ("numpy",)
    kernel_status: str = "generic"
    numpy_only_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.engine_ready, tuple) or not self.engine_ready:
            raise TypeError("engine_ready must be a non-empty tuple of engine names.")
        if any(not isinstance(name, str) or not name.strip() for name in self.engine_ready):
            raise ValueError("engine_ready must contain non-empty engine names.")
        if len(self.engine_ready) != len(set(self.engine_ready)):
            raise ValueError("engine_ready must not contain duplicate engine names.")
        unknown = set(self.engine_ready).difference(KNOWN_COMPUTE_ENGINES)
        if unknown:
            raise ValueError(f"engine_ready contains unknown engines: {sorted(unknown)!r}.")
        if self.engine_ready[0] != "numpy":
            raise ValueError("engine_ready must include 'numpy' first as the reference execution path.")
        if not isinstance(self.kernel_status, str) or self.kernel_status not in KERNEL_STATUSES:
            raise ValueError(f"kernel_status must be one of {sorted(KERNEL_STATUSES)!r}.")
        reason = self.numpy_only_reason
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError("numpy_only_reason must be a non-empty string or None.")
        if self.kernel_status == "numpy_only":
            if self.engine_ready != ("numpy",) or reason is None:
                raise ValueError("a numpy_only kernel requires exactly the NumPy engine and a reason.")
        elif reason is not None:
            raise ValueError("numpy_only_reason is only valid when kernel_status is 'numpy_only'.")
        if self.kernel_status == "legacy_numpy" and self.engine_ready != ("numpy",):
            raise ValueError("a legacy_numpy kernel may only claim the NumPy engine.")

    def supports_engine(self, engine: Any) -> bool:
        """Return whether this metadata allows execution on ``engine``."""
        name = "numpy" if engine is None else getattr(engine, "name", str(engine))
        return name in self.engine_ready

    @property
    def is_permanently_numpy_only(self) -> bool:
        """Return true for families intentionally excluded from tensor engines."""
        return self.engine_ready == ("numpy",) and self.numpy_only_reason is not None


@dataclass(frozen=True, slots=True)
class CapabilitiesNotApplicable:
    """Explicit hook result requesting registry/default capability lookup."""

    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("a capabilities not-applicable result requires a non-empty reason.")


_CAPABILITIES: dict[type[Any], DistributionCapabilities] = {}
_CAPABILITIES_LOCK = RLock()


def register_capabilities(dist_type: type[Any], capabilities: DistributionCapabilities) -> None:
    """Register capability metadata for a class, rejecting accidental replacement."""
    if not isinstance(dist_type, type):
        raise TypeError("capability registration key must be a class.")
    if not isinstance(capabilities, DistributionCapabilities):
        raise TypeError("registered capabilities must be DistributionCapabilities.")
    with _CAPABILITIES_LOCK:
        if dist_type in _CAPABILITIES:
            raise KeyError(f"capabilities are already registered for {dist_type.__qualname__}.")
        _CAPABILITIES[dist_type] = capabilities


def replace_capabilities(dist_type: type[Any], capabilities: DistributionCapabilities) -> None:
    """Explicitly replace capability metadata already registered for ``dist_type``."""
    if not isinstance(dist_type, type):
        raise TypeError("capability registration key must be a class.")
    if not isinstance(capabilities, DistributionCapabilities):
        raise TypeError("registered capabilities must be DistributionCapabilities.")
    with _CAPABILITIES_LOCK:
        if dist_type not in _CAPABILITIES:
            raise KeyError(f"no capabilities are registered for {dist_type.__qualname__}.")
        _CAPABILITIES[dist_type] = capabilities


def unregister_capabilities(dist_type: type[Any]) -> DistributionCapabilities:
    """Remove and return capability metadata registered directly for ``dist_type``."""
    if not isinstance(dist_type, type):
        raise TypeError("capability registration key must be a class.")
    with _CAPABILITIES_LOCK:
        try:
            return _CAPABILITIES.pop(dist_type)
        except KeyError:
            raise KeyError(f"no capabilities are registered for {dist_type.__qualname__}.") from None


def registered_capability_types() -> tuple[type[Any], ...]:
    """Return distribution classes with explicitly registered capabilities."""
    with _CAPABILITIES_LOCK:
        registered = tuple(_CAPABILITIES)
    return tuple(sorted(registered, key=lambda cls: (cls.__module__, cls.__name__)))


def numpy_only_distribution_types() -> tuple[type[Any], ...]:
    """Return families intentionally kept on the NumPy execution path.

    This excludes transitional ``legacy_numpy`` families: those may gain
    backend declarations later.  The returned families have permanent
    distribution-owned reasons explaining why generic tensor engines are not a
    good fit.
    """
    with _CAPABILITIES_LOCK:
        registry = dict(_CAPABILITIES)
    return tuple(
        dist_type for dist_type in registered_capability_types() if registry[dist_type].is_permanently_numpy_only
    )


def _hook_result(hook: Any, owner: type[Any]) -> DistributionCapabilities | None:
    try:
        signature(hook).bind()
    except TypeError as exc:
        raise TypeError(f"{owner.__qualname__}.compute_capabilities must be callable without arguments.") from exc
    result = hook()
    if isinstance(result, CapabilitiesNotApplicable):
        return None
    if not isinstance(result, DistributionCapabilities):
        raise TypeError(
            f"{owner.__qualname__}.compute_capabilities() must return DistributionCapabilities "
            "or CapabilitiesNotApplicable."
        )
    return result


def capabilities_for(x: Any) -> DistributionCapabilities:
    """Return registered capabilities for a distribution instance or class."""
    cls = x if isinstance(x, type) else type(x)

    # A class that declares its own `engine_ready` outranks a compute_capabilities() hook it merely
    # INHERITED. Without this, a subclass narrowing its engine support -- the documented way to say
    # "this variant is numpy-only" -- was silently overruled by the parent's hook and kept claiming
    # torch and jax. A declaration that does nothing is worse than no declaration at all: callers
    # dispatch on supported_engines() and would place the model on an engine it just disclaimed.
    # A hook defined ON the class still wins, since that is the more specific statement of intent.
    own_engine_ready = "engine_ready" in getattr(cls, "__dict__", {})
    own_hook = "compute_capabilities" in getattr(cls, "__dict__", {})
    if own_engine_ready and not own_hook:
        return DistributionCapabilities(
            engine_ready=tuple(cls.engine_ready),
            kernel_status=getattr(cls, "kernel_status", "generic"),
            numpy_only_reason=getattr(cls, "numpy_only_reason", None),
        )

    if not isinstance(x, type):
        hook = getattr(x, "compute_capabilities", None)
        if callable(hook):
            result = _hook_result(hook, cls)
            if result is not None:
                return result

    # Only classmethod/staticmethod descriptors are class-level hooks. A plain instance method visible
    # on ``cls`` is deliberately left for instance lookup instead of being called without ``self``.
    descriptor = getattr_static(cls, "compute_capabilities", None)
    if isinstance(descriptor, (classmethod, staticmethod)):
        result = _hook_result(cls.compute_capabilities, cls)
        if result is not None:
            return result

    if "engine_ready" in getattr(cls, "__dict__", {}):
        return DistributionCapabilities(
            engine_ready=tuple(cls.engine_ready),
            kernel_status=getattr(cls, "kernel_status", "generic"),
            numpy_only_reason=getattr(cls, "numpy_only_reason", None),
        )

    with _CAPABILITIES_LOCK:
        registry = dict(_CAPABILITIES)
    direct = registry.get(cls)
    if direct is not None:
        return direct

    for base in cls.mro()[1:]:
        caps = registry.get(base)
        if caps is not None:
            return caps
    engine_ready = getattr(cls, "engine_ready", ("numpy",))
    return DistributionCapabilities(engine_ready=tuple(engine_ready))


def intersect_engine_ready(
    children: tuple[Any, ...], preferred_order: tuple[str, ...] = COMPOSITION_ENGINES
) -> tuple[str, ...]:
    """Return the engine names supported by every child distribution."""
    if not children:
        return ("numpy",)
    ready = set(capabilities_for(children[0]).engine_ready)
    for child in children[1:]:
        ready &= set(capabilities_for(child).engine_ready)
    return tuple(name for name in preferred_order if name in ready)


def compute_capabilities_from_hook(x: Any) -> DistributionCapabilities:
    """Compatibility helper for callers that need a direct hook result."""
    return capabilities_for(x)


def supported_engines(x: Any) -> tuple[str, ...]:
    """Return engine names supported by a distribution instance or class."""
    return capabilities_for(x).engine_ready
