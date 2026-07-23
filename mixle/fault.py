"""Compatibility shim: ``mixle.fault`` moved to :mod:`mixle.system.fault`.

The degradation policy (``with_fallback``, ``abstain_on_timeout``, ``route_past``, ``DegradedResult``)
is now part of one cohesive package, :mod:`mixle.system`, instead of six cross-importing top-level
modules. Existing ``from mixle.fault import X`` imports keep working (with a
:class:`DeprecationWarning`); new code should import from ``mixle.system`` (or ``mixle.system.fault``)
directly.
"""

from __future__ import annotations

import warnings
from typing import Any

_MOVED = "mixle.system.fault"


def __getattr__(name: str) -> Any:
    import importlib

    try:
        value = getattr(importlib.import_module(_MOVED), name)
    except AttributeError:
        raise AttributeError(f"module 'mixle.fault' has no attribute {name!r}") from None
    warnings.warn(
        f"mixle.fault is deprecated and moved to {_MOVED}; import {name!r} from there instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return value


def __dir__() -> list[str]:
    import importlib

    return sorted(n for n in dir(importlib.import_module(_MOVED)) if not n.startswith("_"))
