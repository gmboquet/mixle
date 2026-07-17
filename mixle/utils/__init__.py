"""General utility namespace.

Most utility subpackages stay lazy.  A few user-facing review helpers are
also exposed here so examples do not have to teach implementation paths.
"""

from __future__ import annotations

import importlib

_SUBMODULES = [
    "automatic",
    "metrics",
    "optional_deps",
    "optsutil",
    "parallel",
    "pvalues",
    "special",
    "vector",
]

_LAZY_EXPORTS = {
    "analyze_structure": ("mixle.utils.automatic", "analyze_structure"),
    "htsne": ("mixle.utils.hvis", "htsne"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr = _LAZY_EXPORTS[name]
        return getattr(importlib.import_module(module_name), attr)
    if name in _SUBMODULES:
        return importlib.import_module("mixle.utils." + name)
    raise AttributeError("module 'mixle.utils' has no attribute %r" % name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_SUBMODULES) | set(_LAZY_EXPORTS))


__all__ = [
    *_SUBMODULES,
    *_LAZY_EXPORTS,
]
