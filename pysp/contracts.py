"""pysp.contracts — every contract (ABC / Protocol) in one import.

The capability vocabulary was correct but spread across ~8 modules (`capability`, `pdist`, `relations`,
`engines.base`, `planner`, `utils.em`, `ppl._operator`, `doe._contracts`). This module gathers them so
the whole contract surface has one home:

    from pysp.contracts import Distribution, Enumerable, Conditionable, Relation, ComputeEngine, \
        ForwardOperator, Surrogate, EMStrategy

The light contracts (the object cast + the capability protocols) are imported eagerly; the heavier
subsystem roles are resolved lazily (PEP 562 ``__getattr__``) so this stays a cheap leaf import with no
cycles. See ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# --- the capability protocols (the facet vocabulary) ---
from pysp.capability import (
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
from pysp.stats.compute.pdist import (
    ConditionalSampler,
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    ParameterEstimator,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulator,
    StatisticAccumulatorFactory,
)
from pysp.stats.compute.pdist import (
    SequenceEncodableProbabilityDistribution as Distribution,
)

if TYPE_CHECKING:  # let static tools / __all__ see the lazily-resolved subsystem contracts
    from pysp.doe._contracts import Acquisition, Criterion, Surrogate
    from pysp.engines.base import ComputeEngine
    from pysp.enumeration.quantization.semiring import DecomposableSemiring
    from pysp.inference.em import EMStrategy
    from pysp.ppl.physics._operator import ForwardOperator
    from pysp.ppl.physics.dynamics import DynamicsOperator
    from pysp.relations import Relation
    from pysp.stats.compute.kernel import Kernel, KernelFactory
    from pysp.utils.parallel.planner import EncodedFold

# Heavier subsystem-role contracts — imported on first access to keep this module a light leaf.
_LAZY: dict[str, tuple[str, str]] = {
    "Relation": ("pysp.relations", "Relation"),
    "ComputeEngine": ("pysp.engines.base", "ComputeEngine"),
    "Kernel": ("pysp.stats.compute.kernel", "Kernel"),
    "KernelFactory": ("pysp.stats.compute.kernel", "KernelFactory"),
    "DecomposableSemiring": ("pysp.enumeration.quantization.semiring", "DecomposableSemiring"),
    "EMStrategy": ("pysp.inference.em", "EMStrategy"),
    "EncodedFold": ("pysp.utils.parallel.planner", "EncodedFold"),
    "DynamicsOperator": ("pysp.ppl.physics.dynamics", "DynamicsOperator"),
    "ForwardOperator": ("pysp.ppl.physics._operator", "ForwardOperator"),
    "Surrogate": ("pysp.doe._contracts", "Surrogate"),
    "Acquisition": ("pysp.doe._contracts", "Acquisition"),
    "Criterion": ("pysp.doe._contracts", "Criterion"),
}


def __getattr__(name: str):  # PEP 562 — resolve subsystem contracts lazily
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError("module 'pysp.contracts' has no attribute %r" % name)
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
    "DynamicsOperator",
    "ForwardOperator",
    "Surrogate",
    "Acquisition",
    "Criterion",
]
