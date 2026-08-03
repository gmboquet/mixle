"""Detector registry for automatic model selection.

A :class:`Detector` describes one candidate distribution family for the profiler / ``get_estimator``:
*when* it applies (a support gate over the data), its *BIC code length* on the data (for selection), how
to *build* its estimator, and (optionally) its *CDF* for goodness-of-fit. Families register here so the
candidate set can be extended without editing the core profiling/selection code -- every module in this
package self-registers on import, and the registry is discovered lazily the first time it is queried.

This is purely *additive*: the profiler's built-in candidates (gaussian / student_t / mixture / lognormal
/ gamma for continuous; poisson / categorical for integer) are unchanged; registered detectors are scored
and built *alongside* them, so a richer family only ever wins when its BIC actually beats the builtins.

A detector receives the data two ways, matching the leaf profiler:

* ``applies(arr)`` and ``score(arr, nobs)`` and ``cdf(arr)`` take the expanded value array (a NumPy array
  of the observed scalars, repeated by their counts);
* ``factory(vdict, pseudo_count, emp_suff_stat, use_bstats)`` takes the value->count map and returns a
  ``ParameterEstimator`` -- the same signature as the built-in ``get_*_estimator`` factories.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal


@dataclass(frozen=True)
class Detector:
    """One candidate distribution family for automatic selection."""

    name: str
    kind: Literal["continuous", "discrete"]
    applies: Callable[[Any], bool]  # (arr) -> bool : support gate
    score: Callable[[Any, int], float | None]  # (arr, nobs) -> BIC bits/obs (None if the fit fails)
    factory: Callable[..., Any]  # (vdict, pseudo_count, emp_suff_stat, use_bstats) -> ParameterEstimator
    cdf: Callable[[Any], Any] | None = None  # (arr) -> ndarray of F(x) for the PIT, or None
    n_params: int = 2


_REGISTRY = MappingProxyType(
    {
        "continuous": MappingProxyType({}),
        "discrete": MappingProxyType({}),
    }
)
_DISCOVERED = False
_DISCOVERY_STAGING: dict[str, dict[str, Detector]] | None = None
_REGISTRY_LOCK = threading.RLock()


def _validate(detector: Detector) -> None:
    if not isinstance(detector, Detector):
        raise TypeError("detector must be a Detector instance")
    if not isinstance(detector.name, str) or not detector.name.strip():
        raise ValueError("detector name must be a non-empty string")
    if detector.kind not in {"continuous", "discrete"}:
        raise ValueError("detector kind must be 'continuous' or 'discrete'")
    for name in ("applies", "score", "factory"):
        if not callable(getattr(detector, name)):
            raise TypeError("detector %s must be callable" % name)
    if detector.cdf is not None and not callable(detector.cdf):
        raise TypeError("detector cdf must be callable or None")
    if isinstance(detector.n_params, bool) or not isinstance(detector.n_params, int) or detector.n_params < 0:
        raise ValueError("detector n_params must be a non-negative integer")


def _mutable_snapshot() -> dict[str, dict[str, Detector]]:
    return {kind: dict(entries) for kind, entries in _REGISTRY.items()}


def _publish(registry: dict[str, dict[str, Detector]]) -> None:
    global _REGISTRY
    _REGISTRY = MappingProxyType(
        {
            "continuous": MappingProxyType(dict(registry["continuous"])),
            "discrete": MappingProxyType(dict(registry["discrete"])),
        }
    )


def _add(registry: dict[str, dict[str, Detector]], detector: Detector) -> None:
    for kind, entries in registry.items():
        previous = entries.get(detector.name)
        if previous is not None and previous is not detector:
            raise ValueError("detector %r is already registered for kind %r" % (detector.name, kind))
    registry[detector.kind][detector.name] = detector


def register(detector: Detector) -> Detector:
    """Register one validated detector without replacing an existing name."""
    _validate(detector)
    with _REGISTRY_LOCK:
        if _DISCOVERY_STAGING is not None:
            _add(_DISCOVERY_STAGING, detector)
        else:
            registry = _mutable_snapshot()
            _add(registry, detector)
            _publish(registry)
    return detector


def _discover() -> None:
    global _DISCOVERED, _DISCOVERY_STAGING
    if _DISCOVERED:
        return
    with _REGISTRY_LOCK:
        if _DISCOVERED:
            return
        staging = _mutable_snapshot()
        before_modules = set(sys.modules)
        _DISCOVERY_STAGING = staging
        try:
            modules = sorted(
                __name__ + "." + info.name for info in pkgutil.iter_modules(__path__) if not info.name.startswith("_")
            )
            for module_name in modules:
                importlib.import_module(module_name)
        except Exception:
            # A retry must execute self-registration again for modules imported
            # during the failed transaction rather than reusing their cache.
            for module_name in set(sys.modules) - before_modules:
                if module_name.startswith(__name__ + "."):
                    sys.modules.pop(module_name, None)
            raise
        else:
            _publish(staging)
            _DISCOVERED = True
        finally:
            _DISCOVERY_STAGING = None


def continuous_detectors() -> list[Detector]:
    """Registered continuous-support candidate families."""
    _discover()
    return list(_REGISTRY["continuous"].values())


def discrete_detectors() -> list[Detector]:
    """Registered integer/discrete-support candidate families."""
    _discover()
    return list(_REGISTRY["discrete"].values())


def get_detector(name: str) -> Detector | None:
    """Return the registered detector with this name (any kind), or None."""
    _discover()
    for kind in _REGISTRY.values():
        if name in kind:
            return kind[name]
    return None


__all__ = ["Detector", "register", "continuous_detectors", "discrete_detectors", "get_detector"]
