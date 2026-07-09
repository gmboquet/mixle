"""Distribution capability metadata for engine and planner decisions.

The registry records which distribution families can execute on each compute
engine, which kernels are generic or specialized, and why some families remain
intentionally NumPy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

    def supports_engine(self, engine: Any) -> bool:
        """Return whether this metadata allows execution on ``engine``."""
        name = "numpy" if engine is None else getattr(engine, "name", str(engine))
        return name in self.engine_ready

    @property
    def is_permanently_numpy_only(self) -> bool:
        """Return true for families intentionally excluded from tensor engines."""
        return self.engine_ready == ("numpy",) and self.numpy_only_reason is not None


_CAPABILITIES: dict[type[Any], DistributionCapabilities] = {}


def register_capabilities(dist_type: type[Any], capabilities: DistributionCapabilities) -> None:
    """Register capability metadata for a distribution class."""
    _CAPABILITIES[dist_type] = capabilities


def registered_capability_types() -> tuple[type[Any], ...]:
    """Return distribution classes with explicitly registered capabilities."""
    return tuple(sorted(_CAPABILITIES.keys(), key=lambda cls: (cls.__module__, cls.__name__)))


def numpy_only_distribution_types() -> tuple[type[Any], ...]:
    """Return families intentionally kept on the NumPy execution path.

    This excludes transitional ``legacy_numpy`` families: those may gain
    backend declarations later.  The returned families have permanent
    distribution-owned reasons explaining why generic tensor engines are not a
    good fit.
    """
    return tuple(
        dist_type for dist_type in registered_capability_types() if _CAPABILITIES[dist_type].is_permanently_numpy_only
    )


def capabilities_for(x: Any) -> DistributionCapabilities:
    """Return registered capabilities for a distribution instance or class."""
    cls = x if isinstance(x, type) else type(x)
    if "engine_ready" in getattr(cls, "__dict__", {}):
        return DistributionCapabilities(
            engine_ready=tuple(cls.engine_ready),
            kernel_status=getattr(cls, "kernel_status", "generic"),
            numpy_only_reason=getattr(cls, "numpy_only_reason", None),
        )

    if not isinstance(x, type):
        hook = getattr(x, "compute_capabilities", None)
        if callable(hook):
            return hook()

    hook = getattr(cls, "compute_capabilities", None)
    if callable(hook):
        try:
            return hook()
        except TypeError:
            pass

    direct = _CAPABILITIES.get(cls)
    if direct is not None:
        return direct

    for base in cls.mro()[1:]:
        caps = _CAPABILITIES.get(base)
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
    hook = getattr(x, "compute_capabilities", None)
    if callable(hook):
        return hook()
    return capabilities_for(x)


def supported_engines(x: Any) -> tuple[str, ...]:
    """Return engine names supported by a distribution instance or class."""
    return capabilities_for(x).engine_ready
