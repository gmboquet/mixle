"""pysparkplug — a capability-oriented probability/statistics library.

The structure (see ``docs/ARCHITECTURE.md`` and ``docs/CAPABILITIES.md``):

* **Objects** — the families: :mod:`pysp.dist` (the umbrella over every distribution, including the
  graph / ranking / set / Markov families), :mod:`pysp.process` (stochastic processes),
  :mod:`pysp.models` (generic / applied models — GPs, neural nets, random forests, knowledge graphs,
  POMDPs — which aren't full Distribution-contract families but still participate in some concerns),
  and :mod:`pysp.relations`.
* **Concerns** — what you can do, each its own module: :mod:`pysp.enumeration`,
  :mod:`pysp.inference`, :mod:`pysp.ops`. (Drawing from a model is intrinsic behavior, not a separate
  concern: :func:`pysp.stats.sample` is the verb, and the ``Posterior`` hierarchy that inference
  produces lives in :mod:`pysp.stats.compute.posterior` / :mod:`pysp.inference.posterior`.)
* **Kernel** — :mod:`pysp.contracts` (every ABC/Protocol in one import) and the capability meta-layer
  re-exported here (:func:`supports`, :func:`capabilities`, :func:`describe`, :func:`catalog`,
  :func:`what_supports`, :func:`require`).

Start with ``pysp.describe(x)`` to see what any object can do.
"""

from pysp.capability import capabilities, catalog, describe, require, summarize, supports, what_supports

# Top-level namespaces resolved lazily so ``import pysp`` stays cheap and ``pysp.dist`` / ``pysp.ops``
# / ``pysp.enumeration`` work without importing the whole tree up front.
_NAMESPACES = (
    "dist",
    "process",
    "models",
    "enumeration",
    "inference",
    "ops",
    "contracts",
)


def __getattr__(name: str):  # PEP 562 — resolve any pysp submodule (incl. the namespaces) lazily
    if not name.startswith("_"):
        import importlib

        try:
            return importlib.import_module("pysp." + name)
        except ModuleNotFoundError:
            pass
    raise AttributeError("module 'pysp' has no attribute %r" % name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_NAMESPACES))


__all__ = [
    "supports",
    "capabilities",
    "describe",
    "summarize",
    "catalog",
    "what_supports",
    "require",
    *_NAMESPACES,
    "stats",
    "utils",
    "parallel",
    "src",
]
