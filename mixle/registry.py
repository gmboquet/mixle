"""Compatibility shim: ``mixle.registry`` moved to :mod:`mixle.system.registry`.

The local directory-backed model catalog (``Registry``, ``RegistryEntry``) is now part of one cohesive
package, :mod:`mixle.system`, instead of six cross-importing top-level modules. Existing
``from mixle.registry import X`` imports keep working (with a :class:`DeprecationWarning`); new code
should import from ``mixle.system`` (or ``mixle.system.registry``) directly.
"""

from __future__ import annotations

import warnings
from typing import Any

_MOVED = "mixle.system.registry"
__all__ = ["Registry", "RegistryEntry"]  # noqa: F822


def __getattr__(name: str) -> Any:
    import importlib

    try:
        value = getattr(importlib.import_module(_MOVED), name)
    except AttributeError:
        raise AttributeError(f"module 'mixle.registry' has no attribute {name!r}") from None
    warnings.warn(
        f"mixle.registry is deprecated and moved to {_MOVED}; import {name!r} from there instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
