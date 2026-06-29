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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Detector:
    """One candidate distribution family for automatic selection."""

    name: str
    kind: str  # "continuous" | "discrete"
    applies: Callable[[Any], bool]  # (arr) -> bool : support gate
    score: Callable[[Any, int], float | None]  # (arr, nobs) -> BIC bits/obs (None if the fit fails)
    factory: Callable[..., Any]  # (vdict, pseudo_count, emp_suff_stat, use_bstats) -> ParameterEstimator
    cdf: Callable[[Any], Any] | None = None  # (arr) -> ndarray of F(x) for the PIT, or None
    n_params: int = 2


_REGISTRY: dict[str, dict[str, Detector]] = {"continuous": {}, "discrete": {}}
_DISCOVERED = False


def register(detector: Detector) -> Detector:
    """Register a detector (idempotent on name within its kind). Returns it, so it can decorate a build."""
    _REGISTRY.setdefault(detector.kind, {})[detector.name] = detector
    return detector


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    for info in pkgutil.iter_modules(__path__):
        if not info.name.startswith("_"):
            importlib.import_module(__name__ + "." + info.name)


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
