"""Compatibility shim: ``mixle.meta`` moved to :mod:`mixle.system.meta`.

The heuristic improvement-budget allocator (``ImprovementOption``, ``improve_by_regret``) is now part
of one cohesive package, :mod:`mixle.system`, instead of six cross-importing top-level modules.
Existing ``from mixle.meta import X`` imports keep working (with a :class:`DeprecationWarning`); new
code should import from ``mixle.system`` (or ``mixle.system.meta``) directly.
"""

from __future__ import annotations

import warnings
from typing import Any

_MOVED = "mixle.system.meta"
__all__ = ["ImprovementOption", "MetaImprovementReport", "improve_by_regret"]  # noqa: F822


def __getattr__(name: str) -> Any:
    import importlib

    try:
        value = getattr(importlib.import_module(_MOVED), name)
    except AttributeError:
        raise AttributeError(f"module 'mixle.meta' has no attribute {name!r}") from None
    warnings.warn(
        f"mixle.meta is deprecated and moved to {_MOVED}; import {name!r} from there instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
