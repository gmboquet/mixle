"""The model-parallel decomposition contract.

A distribution optionally declares *how its parameters and sufficient statistics may be split across
devices*, via :meth:`SequenceEncodableProbabilityDistribution.decomposition`. The descriptor names the
axis the node splits along (its mixture components, its composite factors, its sequence/document units,
...), the reduction that recombines per-shard sufficient statistics, whether the split is exact, and
which children are *shared* (held whole and reduced, e.g. an HMM transition matrix or LDA topics).

This is the unifying primitive the structural planner and the model-parallel executor both consume. It
is strictly opt-in: any node that does not override ``decomposition()`` reports
:meth:`Decomposition.atomic` (replicated, not split), so the contract never disturbs the existing
data-parallel-with-replicated-model path. The cross-shard reduction is always the same additive
``combine()`` monoid the accumulators already implement -- this introduces no new reduction algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DecompAxis(Enum):
    """The dimension a node splits along for model parallelism."""

    NONE = "none"  # atomic / not split (leaves, unannotated nodes)
    COMPONENT = "component"  # mixture / latent components (the stacked-kernel axis)
    FACTOR = "factor"  # independent coordinates of a composite / record
    SEQUENCE = "sequence"  # iid units: sequences, documents (the data axis)
    TOPIC = "topic"  # shared latent topics (hierarchical / LDA)
    STATE = "state"  # HMM emission states (transition matrix stays shared)


class ReductionOp(Enum):
    """How per-shard sufficient statistics recombine at the cross-shard boundary."""

    SUM = "sum"  # additive combine() monoid -- exact, embarrassingly parallel
    LOGSUMEXP_RESPONSIBILITY = "logsumexp"  # responsibilities live INSIDE a shard; boundary is SUM + scalar all-reduce
    REPLICATE = "replicate"  # not split on this axis; held whole / agreed across shards


@dataclass(frozen=True)
class Decomposition:
    """How a distribution may be split across devices for model parallelism.

    Attributes:
        axis: the dimension the node splits along.
        num_units: the number of splittable units along ``axis`` (components, factors, units).
        reduction: how per-shard sufficient statistics recombine across shards.
        exact: whether sharding this axis is exact EM (vs a restricted/approximate family).
        child_roles: names of the split children (e.g. ``("component",)``), informational.
        shared_children: children held whole on every shard and reduced, not split -- e.g. an HMM
            ``transitions`` matrix or LDA ``topics`` (the structural statement of what cannot be split).
        engine_axis: the tensor axis for ``ComputeEngine.place_component_axis`` (DTensor sharding), or
            ``None`` when there is no homogeneous stacked-parameter tensor to shard.
        key_pooling: whether per-shard estimates must route through keyed-tying
            (``tie_component_shard_values``) before the M-step.
    """

    axis: DecompAxis = DecompAxis.NONE
    num_units: int = 1
    reduction: ReductionOp = ReductionOp.REPLICATE
    exact: bool = True
    child_roles: tuple[str, ...] = ()
    shared_children: tuple[str, ...] = ()
    engine_axis: int | None = None
    key_pooling: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def atomic(cls) -> Decomposition:
        """The default: this node is not split -- it is replicated across shards."""
        return cls()

    @property
    def is_shardable(self) -> bool:
        """True when this node declares a real (non-atomic) split axis with more than one unit."""
        return self.axis is not DecompAxis.NONE and self.num_units > 1


# Registry for classes that cannot carry a ``decomposition()`` method (parity with capabilities_for).
_DECOMPOSITIONS: dict[type[Any], Decomposition] = {}


def register_decomposition(dist_type: type[Any], decomposition: Decomposition) -> None:
    """Register a decomposition descriptor for a distribution class."""
    _DECOMPOSITIONS[dist_type] = decomposition


def decomposition_for(x: Any) -> Decomposition:
    """Return the decomposition descriptor for a distribution instance or class.

    Lookup mirrors :func:`mixle.stats.compute.capabilities.capabilities_for`: an instance
    ``decomposition()`` hook wins, then the class registry, then the MRO, then the atomic default.
    """
    if not isinstance(x, type):
        hook = getattr(x, "decomposition", None)
        if callable(hook):
            try:
                return hook()
            except TypeError:
                pass
    cls = x if isinstance(x, type) else type(x)
    direct = _DECOMPOSITIONS.get(cls)
    if direct is not None:
        return direct
    for base in cls.mro()[1:]:
        found = _DECOMPOSITIONS.get(base)
        if found is not None:
            return found
    return Decomposition.atomic()


__all__ = [
    "DecompAxis",
    "ReductionOp",
    "Decomposition",
    "register_decomposition",
    "decomposition_for",
]
