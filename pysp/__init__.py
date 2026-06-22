"""pysparkplug — a capability-oriented probability/statistics library.

The structure (see ``docs/ARCHITECTURE.md`` and ``docs/CAPABILITIES.md``):

* **Objects** — the families: :mod:`pysp.dist`, :mod:`pysp.process`, :mod:`pysp.graph`,
  :mod:`pysp.relations`.
* **Concerns** — what you can do, each its own module: :mod:`pysp.enumeration`,
  :mod:`pysp.sampling`, :mod:`pysp.inference`, :mod:`pysp.ops`.
* **Kernel** — :mod:`pysp.contracts` (every ABC/Protocol in one import) and the capability meta-layer
  re-exported here (:func:`supports`, :func:`capabilities`, :func:`describe`, :func:`catalog`,
  :func:`what_supports`, :func:`require`).

Start with ``pysp.describe(x)`` to see what any object can do.
"""

from pysp.capability import capabilities, catalog, describe, require, supports, what_supports

# Top-level namespaces resolved lazily so ``import pysp`` stays cheap and ``pysp.dist`` / ``pysp.ops``
# / ``pysp.enumeration`` work without importing the whole tree up front.
_NAMESPACES = (
    "dist",
    "process",
    "graph",
    "enumeration",
    "sampling",
    "inference",
    "ops",
    "contracts",
)


def __getattr__(name: str):  # PEP 562 — lazy submodule namespaces
    if name in _NAMESPACES:
        import importlib

        return importlib.import_module("pysp." + name)
    raise AttributeError("module 'pysp' has no attribute %r" % name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_NAMESPACES))


__all__ = [
    "supports",
    "capabilities",
    "describe",
    "catalog",
    "what_supports",
    "require",
    *_NAMESPACES,
    "stats",
    "utils",
    "models",
    "parallel",
    "src",
]
