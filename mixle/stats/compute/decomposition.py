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
from inspect import signature
from operator import index
from threading import RLock
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

    def __post_init__(self) -> None:
        if not isinstance(self.axis, DecompAxis):
            raise TypeError("decomposition axis must be a DecompAxis.")
        if not isinstance(self.reduction, ReductionOp):
            raise TypeError("decomposition reduction must be a ReductionOp.")
        if isinstance(self.num_units, bool):
            raise TypeError("decomposition num_units must be an integer, not a boolean.")
        try:
            num_units = index(self.num_units)
        except TypeError as exc:
            raise TypeError("decomposition num_units must be an integer.") from exc
        if num_units <= 0:
            raise ValueError("decomposition num_units must be positive.")
        object.__setattr__(self, "num_units", num_units)
        if not isinstance(self.exact, bool):
            raise TypeError("decomposition exact must be a boolean.")
        if not isinstance(self.key_pooling, bool):
            raise TypeError("decomposition key_pooling must be a boolean.")
        self._validate_roles("child_roles", self.child_roles)
        self._validate_roles("shared_children", self.shared_children)
        overlap = set(self.child_roles).intersection(self.shared_children)
        if overlap:
            raise ValueError(f"split and shared child roles overlap: {sorted(overlap)!r}.")
        if self.child_roles and len(self.child_roles) != num_units:
            raise ValueError("decomposition child_roles must be empty or contain exactly num_units names.")
        if not isinstance(self.extra, dict):
            raise TypeError("decomposition extra metadata must be a dict.")

        if self.axis is DecompAxis.NONE:
            if num_units != 1 or self.reduction is not ReductionOp.REPLICATE:
                raise ValueError("an atomic decomposition must have one unit and use REPLICATE.")
            if self.child_roles or self.shared_children or self.engine_axis is not None or self.key_pooling:
                raise ValueError("an atomic decomposition cannot declare children, an engine axis, or key pooling.")
            if not self.exact:
                raise ValueError("an atomic decomposition must be exact.")
        elif self.reduction is ReductionOp.REPLICATE:
            raise ValueError("a non-atomic decomposition cannot use REPLICATE reduction.")

        if self.reduction is ReductionOp.LOGSUMEXP_RESPONSIBILITY and self.axis is not DecompAxis.COMPONENT:
            raise ValueError("LOGSUMEXP_RESPONSIBILITY reduction requires the COMPONENT axis.")
        if self.key_pooling and self.axis is not DecompAxis.COMPONENT:
            raise ValueError("key_pooling requires the COMPONENT axis.")
        if self.engine_axis is not None:
            if isinstance(self.engine_axis, bool):
                raise TypeError("decomposition engine_axis must be an integer or None, not a boolean.")
            try:
                engine_axis = index(self.engine_axis)
            except TypeError as exc:
                raise TypeError("decomposition engine_axis must be an integer or None.") from exc
            if engine_axis < 0:
                raise ValueError("decomposition engine_axis must be non-negative.")
            if self.axis is not DecompAxis.COMPONENT:
                raise ValueError("decomposition engine_axis is only defined for the COMPONENT axis.")
            object.__setattr__(self, "engine_axis", engine_axis)

    @staticmethod
    def _validate_roles(label: str, roles: Any) -> None:
        if not isinstance(roles, tuple):
            raise TypeError(f"decomposition {label} must be a tuple.")
        if any(not isinstance(role, str) or not role.strip() for role in roles):
            raise ValueError(f"decomposition {label} must contain non-empty strings.")
        if len(roles) != len(set(roles)):
            raise ValueError(f"decomposition {label} must not contain duplicate names.")

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
_DECOMPOSITIONS_LOCK = RLock()


def register_decomposition(dist_type: type[Any], decomposition: Decomposition) -> None:
    """Register a descriptor for a class, rejecting accidental replacement."""
    if not isinstance(dist_type, type):
        raise TypeError("decomposition registration key must be a class.")
    if not isinstance(decomposition, Decomposition):
        raise TypeError("registered decomposition must be a Decomposition.")
    with _DECOMPOSITIONS_LOCK:
        if dist_type in _DECOMPOSITIONS:
            raise KeyError(f"a decomposition is already registered for {dist_type.__qualname__}.")
        _DECOMPOSITIONS[dist_type] = decomposition


def replace_decomposition(dist_type: type[Any], decomposition: Decomposition) -> None:
    """Explicitly replace a descriptor already registered for ``dist_type``."""
    if not isinstance(dist_type, type):
        raise TypeError("decomposition registration key must be a class.")
    if not isinstance(decomposition, Decomposition):
        raise TypeError("registered decomposition must be a Decomposition.")
    with _DECOMPOSITIONS_LOCK:
        if dist_type not in _DECOMPOSITIONS:
            raise KeyError(f"no decomposition is registered for {dist_type.__qualname__}.")
        _DECOMPOSITIONS[dist_type] = decomposition


def unregister_decomposition(dist_type: type[Any]) -> Decomposition:
    """Remove and return the descriptor registered directly for ``dist_type``."""
    if not isinstance(dist_type, type):
        raise TypeError("decomposition registration key must be a class.")
    with _DECOMPOSITIONS_LOCK:
        try:
            return _DECOMPOSITIONS.pop(dist_type)
        except KeyError:
            raise KeyError(f"no decomposition is registered for {dist_type.__qualname__}.") from None


def decomposition_for(x: Any) -> Decomposition:
    """Return the decomposition descriptor for a distribution instance or class.

    Lookup mirrors :func:`mixle.stats.compute.capabilities.capabilities_for`: an instance
    ``decomposition()`` hook wins, then the class registry, then the MRO, then the atomic default.
    """
    if not isinstance(x, type):
        hook = getattr(x, "decomposition", None)
        if callable(hook):
            try:
                signature(hook).bind()
            except TypeError as exc:
                raise TypeError(f"{type(x).__qualname__}.decomposition must be callable without arguments.") from exc
            descriptor = hook()
            if not isinstance(descriptor, Decomposition):
                raise TypeError(f"{type(x).__qualname__}.decomposition() must return a Decomposition.")
            return descriptor
    cls = x if isinstance(x, type) else type(x)
    with _DECOMPOSITIONS_LOCK:
        registry = dict(_DECOMPOSITIONS)
    direct = registry.get(cls)
    if direct is not None:
        return direct
    for base in cls.mro()[1:]:
        found = registry.get(base)
        if found is not None:
            return found
    return Decomposition.atomic()


__all__ = [
    "DecompAxis",
    "ReductionOp",
    "Decomposition",
    "decomposition_for",
    "register_decomposition",
    "replace_decomposition",
    "unregister_decomposition",
]
