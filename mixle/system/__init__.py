"""Local application facade: answer/ingest/improve a Mixle system, with its cost ledger, degradation
policy, held-out scorecard, meta-improvement allocator, and model registry.

Formerly six cross-importing top-level modules (``mixle.system``, ``mixle.spend``, ``mixle.fault``,
``mixle.scorecard``, ``mixle.meta``, ``mixle.registry``) -- one cohesive package now:

* :mod:`mixle.system.core` -- :class:`System`, :class:`SystemConfig`, :class:`Query`: the
  answer/ingest/improve facade.
* :mod:`mixle.system.spend` -- :class:`Spend`: the budget ledger ``System`` is metered against.
* :mod:`mixle.system.fault` -- :func:`with_fallback`, :func:`abstain_on_timeout`, :func:`route_past`,
  :class:`DegradedResult`: named degradation modes instead of silently-successful fallbacks.
* :mod:`mixle.system.scorecard` -- :class:`SystemScorecard`, :func:`evaluate`, :func:`detect_regression`:
  measuring a ``System`` against a held-out question set.
* :mod:`mixle.system.meta` -- :class:`ImprovementOption`, :func:`improve_by_regret`: spend an
  improvement budget on the highest realized gain-per-dollar option, stopping on regression.
* :mod:`mixle.system.registry` -- :class:`Registry`, :class:`RegistryEntry`: the local directory-backed
  catalog of fitted task models these workflows read from and write into.

Every name below is importable directly from ``mixle.system`` (``from mixle.system import System,
Spend, Registry``); the individual submodules remain the place to look for the source and docstrings.
The old flat top-level names (``mixle.spend``, ``mixle.fault``, ``mixle.scorecard``, ``mixle.meta``,
``mixle.registry``) still work via a deprecation shim; new code should import from here.

Resolution is lazy (PEP 562, mirroring :mod:`mixle` itself): importing one submodule -- e.g.
``mixle.system.fault``, which callers such as :mod:`mixle.doe.oracle` use for its zero-dependency
degradation helpers -- never pulls in the other five. ``core.py`` alone reaches :mod:`mixle.task`; an
eager ``__init__`` that imported every submodule up front would force that whole chain onto any caller
of the leaf modules, including a cycle back through :mod:`mixle.task.propose`.
"""

from __future__ import annotations

import importlib
from typing import Any

_SOURCE = {
    "Query": "core",
    "System": "core",
    "SystemConfig": "core",
    "Spend": "spend",
    "DegradedResult": "fault",
    "with_fallback": "fault",
    "abstain_on_timeout": "fault",
    "route_past": "fault",
    "SystemScorecard": "scorecard",
    "RegressionReport": "scorecard",
    "evaluate": "scorecard",
    "detect_regression": "scorecard",
    "ImprovementOption": "meta",
    "MetaImprovementReport": "meta",
    "improve_by_regret": "meta",
    "Registry": "registry",
    "RegistryEntry": "registry",
}

__all__ = sorted(_SOURCE)


def __getattr__(name: str) -> Any:
    submodule = _SOURCE.get(name)
    if submodule is None:
        raise AttributeError(f"module 'mixle.system' has no attribute {name!r}")
    return getattr(importlib.import_module(f"mixle.system.{submodule}"), name)


def __dir__() -> list[str]:
    return __all__
