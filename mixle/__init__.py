"""mixle — a capability-oriented probability/statistics library.

The structure (see ``docs/architecture.md`` and ``docs/contracts.md``):

* **Objects** — the families: :mod:`mixle.dist` (the umbrella over every distribution, including the
  graph / ranking / set / Markov families), :mod:`mixle.process` (stochastic processes),
  :mod:`mixle.models` (generic / applied models — GPs, neural nets, random forests, knowledge graphs,
  POMDPs — which aren't full Distribution-contract families but still participate in some concerns),
  and :mod:`mixle.relations`.
* **Concerns** — what you can do, each its own module: :mod:`mixle.enumeration`,
  :mod:`mixle.inference`, :mod:`mixle.ops`. (Drawing from a model is intrinsic behavior, not a separate
  concern: :func:`mixle.stats.sample` is the verb, and the ``Posterior`` hierarchy that inference
  produces lives in :mod:`mixle.stats.compute.posterior` / :mod:`mixle.inference.posterior`.)
* **Kernel** — :mod:`mixle.contracts` (every ABC/Protocol in one import) and the capability meta-layer
  re-exported here (:func:`supports`, :func:`capabilities`, :func:`describe`, :func:`catalog`,
  :func:`what_supports`, :func:`require`).

Start with ``mixle.describe(x)`` to see what any object can do.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from mixle.capability import capabilities, catalog, describe, require, summarize, supports, what_supports

try:
    __version__ = _pkg_version("mixle")
except PackageNotFoundError:  # running from a source tree with no installed distribution metadata
    __version__ = "0+unknown"

# Top-level namespaces resolved lazily so ``import mixle`` stays lightweight and ``mixle.dist`` / ``mixle.ops``
# / ``mixle.enumeration`` work without importing the whole tree up front.
_NAMESPACES = (
    "dist",
    "process",
    "models",
    "enumeration",
    "inference",
    "ops",
    "contracts",
    "semantics",
)


def __getattr__(name: str):  # PEP 562 — resolve any mixle submodule (incl. the namespaces) lazily
    if name in ("Model", "propose"):  # the lifecycle facade, kept lazy so `import mixle` stays lightweight
        import mixle.lifecycle as _lc

        return getattr(_lc, name)
    if not name.startswith("_"):
        import importlib

        module_name = "mixle." + name
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
    raise AttributeError("module 'mixle' has no attribute %r" % name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_NAMESPACES) | {"Model", "propose"})


__all__ = [
    "__version__",
    "Model",
    "propose",
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
]
