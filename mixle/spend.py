"""Compatibility shim: ``mixle.spend`` moved to :mod:`mixle.system.spend`.

``Spend`` and its siblings (``System``, degradation modes, the scorecard, the meta-improvement
allocator, and the registry) are now one cohesive package, :mod:`mixle.system`, instead of six
cross-importing top-level modules. Existing ``from mixle.spend import X`` imports keep working (with a
:class:`DeprecationWarning`); new code should import from ``mixle.system`` (or ``mixle.system.spend``)
directly.
"""

from __future__ import annotations

import warnings
from typing import Any

_MOVED = "mixle.system.spend"


def __getattr__(name: str) -> Any:
    import importlib

    try:
        value = getattr(importlib.import_module(_MOVED), name)
    except AttributeError:
        raise AttributeError(f"module 'mixle.spend' has no attribute {name!r}") from None
    warnings.warn(
        f"mixle.spend is deprecated and moved to {_MOVED}; import {name!r} from there instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return value


def __dir__() -> list[str]:
    import importlib

    return sorted(n for n in dir(importlib.import_module(_MOVED)) if not n.startswith("_"))
