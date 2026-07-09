"""mixle.contracts — every contract (ABC / Protocol) in one import.

The capability vocabulary was correct but spread across several modules (`capability`, `pdist`,
`relations`, `engines.base`, `planner`, `utils.em`, `doe._contracts`). This module gathers them so
the whole contract surface has one home:

    from mixle.contracts import Distribution, Enumerable, Conditionable, Relation, ComputeEngine, \
        Surrogate, EMStrategy

The light contracts (the object cast + the capability protocols) are imported eagerly; the heavier
subsystem roles are resolved lazily (PEP 562 ``__getattr__``) so this stays a lightweight leaf import with no
cycles. See ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# --- the capability protocols (the facet vocabulary) ---
from mixle.capability import (
    Conditionable,
    ConjugateUpdatable,
    EngineResidentEStep,
    Enumerable,
    ExponentialFamily,
    FiniteSupport,
    LatentStructured,
    Marginalizable,
    Neutral,
    PosteriorPredictive,
    PredicateCapability,
    RankableByIndex,
    SetValued,
    SupportsBackendComponentScoring,
    SupportsBackendScoring,
    SupportsStackedBackend,
    TemporalPointProcess,
    Transform,
)

# --- the universal object cast (core contracts) ---
from mixle.stats.compute.pdist import (
    ConditionalSampler,
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulator,
    StatisticAccumulatorFactory,
)
from mixle.stats.compute.pdist import (
    SequenceEncodableProbabilityDistribution as Distribution,
)

if TYPE_CHECKING:  # let static tools / __all__ see the lazily-resolved subsystem contracts
    from mixle.doe._contracts import Acquisition, Criterion, Surrogate
    from mixle.engines.base import ComputeEngine
    from mixle.enumeration.quantization.semiring import DecomposableSemiring
    from mixle.inference.em import EMStrategy
    from mixle.relations import Relation
    from mixle.stats.compute.kernel import Kernel, KernelFactory
    from mixle.utils.parallel.planner import EncodedFold

# Heavier subsystem-role contracts — imported on first access to keep this module a light leaf.
_LAZY: dict[str, tuple[str, str]] = {
    "Relation": ("mixle.relations", "Relation"),
    "ComputeEngine": ("mixle.engines.base", "ComputeEngine"),
    "Kernel": ("mixle.stats.compute.kernel", "Kernel"),
    "KernelFactory": ("mixle.stats.compute.kernel", "KernelFactory"),
    "DecomposableSemiring": ("mixle.enumeration.quantization.semiring", "DecomposableSemiring"),
    "EMStrategy": ("mixle.inference.em", "EMStrategy"),
    "EncodedFold": ("mixle.utils.parallel.planner", "EncodedFold"),
    "Surrogate": ("mixle.doe._contracts", "Surrogate"),
    "Acquisition": ("mixle.doe._contracts", "Acquisition"),
    "Criterion": ("mixle.doe._contracts", "Criterion"),
}


def __getattr__(name: str):  # PEP 562 — resolve subsystem contracts lazily
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError("module 'mixle.contracts' has no attribute %r" % name)
    import importlib

    module, attr = target
    return getattr(importlib.import_module(module), attr)


__all__ = [
    # object cast
    "Distribution",
    "DistributionSampler",
    "ConditionalSampler",
    "DistributionEnumerator",
    "ParameterEstimator",
    "StatisticAccumulator",
    "SequenceEncodableStatisticAccumulator",
    "StatisticAccumulatorFactory",
    "DataSequenceEncoder",
    # capability protocols
    "Enumerable",
    "FiniteSupport",
    "RankableByIndex",
    "ExponentialFamily",
    "ConjugateUpdatable",
    "Conditionable",
    "Marginalizable",
    "LatentStructured",
    "PosteriorPredictive",
    "TemporalPointProcess",
    "SetValued",
    "Neutral",
    "Transform",
    "EngineResidentEStep",
    "SupportsBackendScoring",
    "SupportsBackendComponentScoring",
    "SupportsStackedBackend",
    "PredicateCapability",
    # subsystem roles (lazy)
    "Relation",
    "ComputeEngine",
    "Kernel",
    "KernelFactory",
    "DecomposableSemiring",
    "EMStrategy",
    "EncodedFold",
    "Surrogate",
    "Acquisition",
    "Criterion",
]
