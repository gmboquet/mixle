"""Contracts shared by distributed gradient-training backends.

The encoded-data backends reduce sufficient statistics.  Frontier neural
training has a different lifecycle: materialize a process mesh, transform a
module, bind optimizer state to logical parameters, execute steps, and save a
reshardable training state.  Keeping that lifecycle separate prevents a Ray or
Megatron integration from pretending to be an ``EncodedDataHandle``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral, Real
from typing import Any, Protocol, runtime_checkable

from mixle.utils.immutable import detach_receipt_container


class ParallelAxis(StrEnum):
    """Named dimensions understood by Mixle and external training engines."""

    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    TP = "tp"
    PP = "pp"
    CP = "cp"
    EP = "ep"
    ETP = "etp"


class PayloadKind(StrEnum):
    """Value communicated when an update is distributed."""

    GRADIENT = "gradient"
    SUFFICIENT_STATISTIC = "sufficient_statistic"
    PARAMETER = "parameter"
    ACTIVATION = "activation"
    KV_BLOCK = "kv_block"
    TOKEN = "token"
    MESSAGE = "message"


class CollectiveKind(StrEnum):
    """Collective or point-to-point operation required by an update."""

    NONE = "none"
    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_GATHER = "all_gather"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"
    P2P = "p2p"
    CUSTOM = "custom"


class StateLayout(StrEnum):
    """Physical ownership of mutable training state."""

    REPLICATED = "replicated"
    SHARDED = "sharded"
    PIPELINE_LOCAL = "pipeline_local"
    EXPERT_LOCAL = "expert_local"
    OFFLOADED = "offloaded"


@dataclass(frozen=True)
class ParallelPlan:
    """An explicit N-D process mesh and batch schedule.

    Every process belongs to exactly one coordinate in the product below.
    Keeping replicate and shard data-parallel dimensions separate permits
    HSDP without conflating it with tensor or context parallelism.
    """

    dp_replicate: int = 1
    dp_shard: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1
    etp: int = 1
    microbatches: int = 1
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        names = (
            "dp_replicate",
            "dp_shard",
            "tp",
            "pp",
            "cp",
            "ep",
            "etp",
            "microbatches",
            "gradient_accumulation_steps",
        )
        for name in names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be an exact positive integer.")
            object.__setattr__(self, name, int(value))
        if self.data_parallel_size % self.ep:
            raise ValueError("expert parallelism must divide the data-parallel domain.")

    @property
    def axis_sizes(self) -> dict[ParallelAxis, int]:
        return {
            ParallelAxis.DP_REPLICATE: self.dp_replicate,
            ParallelAxis.DP_SHARD: self.dp_shard,
            ParallelAxis.TP: self.tp,
            ParallelAxis.PP: self.pp,
            ParallelAxis.CP: self.cp,
            ParallelAxis.EP: self.ep,
            ParallelAxis.ETP: self.etp,
        }

    @property
    def world_size(self) -> int:
        """Number of physical ranks.

        EP and ETP are overlapping process groups cut from the data/model
        domains, not extra orthogonal DeviceMesh dimensions.
        """

        result = 1
        for axis in (
            ParallelAxis.DP_REPLICATE,
            ParallelAxis.DP_SHARD,
            ParallelAxis.TP,
            ParallelAxis.PP,
            ParallelAxis.CP,
        ):
            result *= self.axis_sizes[axis]
        return result

    @property
    def data_parallel_size(self) -> int:
        return self.dp_replicate * self.dp_shard

    @property
    def active_axes(self) -> tuple[ParallelAxis, ...]:
        return tuple(axis for axis, size in self.axis_sizes.items() if size > 1)

    @property
    def mesh(self) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Return stable DeviceMesh names and shape, including a 1-rank DP dimension."""

        physical = {
            ParallelAxis.DP_REPLICATE,
            ParallelAxis.DP_SHARD,
            ParallelAxis.TP,
            ParallelAxis.PP,
            ParallelAxis.CP,
        }
        axes = tuple(axis for axis in self.active_axes if axis in physical) or (ParallelAxis.DP_REPLICATE,)
        return tuple(axis.value for axis in axes), tuple(self.axis_sizes[axis] for axis in axes)

    def size(self, axis: ParallelAxis | str) -> int:
        key = axis if isinstance(axis, ParallelAxis) else ParallelAxis(str(axis))
        return self.axis_sizes[key]

    def validate_world_size(self, actual: int) -> None:
        if isinstance(actual, bool) or not isinstance(actual, Integral) or actual < 1:
            raise ValueError("actual world size must be an exact positive integer.")
        actual = int(actual)
        if actual != self.world_size:
            raise ValueError(
                "parallel plan requires world_size=%d, but the process group has world_size=%d."
                % (self.world_size, actual)
            )

    def as_dict(self) -> dict[str, int]:
        payload = {axis.value: size for axis, size in self.axis_sizes.items()}
        payload.update(
            microbatches=self.microbatches,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            world_size=self.world_size,
            data_parallel_size=self.data_parallel_size,
        )
        return payload


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend actually executes, rather than merely accepting as flags."""

    name: str
    axes: frozenset[ParallelAxis]
    precisions: frozenset[str] = frozenset({"fp32", "bf16"})
    distributed_optimizer: bool = False
    reshardable_checkpoint: bool = False
    elastic_restart: bool = False
    requirements: tuple[str, ...] = ()
    incompatible_axis_sets: tuple[frozenset[ParallelAxis], ...] = ()

    def validate(self, plan: ParallelPlan, *, precision: str = "fp32") -> None:
        missing = tuple(axis for axis in plan.active_axes if axis not in self.axes)
        if missing:
            raise NotImplementedError(
                "%s does not execute requested parallel axes: %s"
                % (self.name, ", ".join(axis.value for axis in missing))
            )
        if precision not in self.precisions:
            raise ValueError(
                "%s does not support precision=%r; supported: %s"
                % (self.name, precision, ", ".join(sorted(self.precisions)))
            )
        active = frozenset(plan.active_axes)
        for incompatible in self.incompatible_axis_sets:
            if incompatible.issubset(active):
                raise NotImplementedError(
                    "%s does not compose axes %s in one session."
                    % (self.name, "+".join(sorted(axis.value for axis in incompatible)))
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "axes": sorted(axis.value for axis in self.axes),
            "precisions": sorted(self.precisions),
            "distributed_optimizer": self.distributed_optimizer,
            "reshardable_checkpoint": self.reshardable_checkpoint,
            "elastic_restart": self.elastic_restart,
            "requirements": list(self.requirements),
            "incompatible_axis_sets": [sorted(axis.value for axis in axes) for axes in self.incompatible_axis_sets],
        }


@dataclass(frozen=True)
class ParameterLayout:
    """Stable logical identity and global layout for one parameter."""

    logical_id: str
    global_shape: tuple[int, ...]
    placements: tuple[tuple[str, str], ...] = ()
    shared_group: str | None = None
    optimizer_state: StateLayout = StateLayout.REPLICATED

    def __post_init__(self) -> None:
        if not isinstance(self.logical_id, str) or not self.logical_id.strip():
            raise ValueError("logical_id must be a non-empty string.")
        if not isinstance(self.global_shape, tuple):
            raise TypeError("global_shape must be a tuple of exact non-negative integers.")
        canonical_shape = []
        for size in self.global_shape:
            if isinstance(size, bool) or not isinstance(size, Integral) or size < 0:
                raise ValueError("global_shape must contain exact non-negative integers.")
            canonical_shape.append(int(size))
        object.__setattr__(self, "global_shape", tuple(canonical_shape))
        if not isinstance(self.placements, tuple):
            raise TypeError("placements must be a tuple of (axis, placement) pairs.")
        canonical_placements = []
        seen_axes = set()
        for placement in self.placements:
            if not isinstance(placement, tuple) or len(placement) != 2:
                raise ValueError("each placement must be an (axis, placement) pair.")
            axis, value = placement
            if not isinstance(axis, str) or not axis.strip() or not isinstance(value, str) or not value.strip():
                raise ValueError("placement axes and values must be non-empty strings.")
            if axis in seen_axes:
                raise ValueError(f"parameter placement axis {axis!r} is duplicated.")
            seen_axes.add(axis)
            canonical_placements.append((axis, value))
        object.__setattr__(self, "placements", tuple(canonical_placements))
        if self.shared_group is not None and (not isinstance(self.shared_group, str) or not self.shared_group.strip()):
            raise ValueError("shared_group must be a non-empty string or None.")
        if not isinstance(self.optimizer_state, StateLayout):
            raise TypeError("optimizer_state must be a StateLayout.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_id": self.logical_id,
            "global_shape": list(self.global_shape),
            "placements": dict(self.placements),
            "shared_group": self.shared_group,
            "optimizer_state": self.optimizer_state.value,
        }


@dataclass(frozen=True)
class DistributedUpdate:
    """Executable communication declaration with explicit numerical evidence."""

    node_id: str
    payload: PayloadKind
    collective: CollectiveKind
    mesh_axes: tuple[ParallelAxis, ...]
    state_layout: StateLayout
    exact: bool
    notes: tuple[str, ...] = ()
    contract_exact: bool = False
    determinism_observed: bool | None = None
    maximum_absolute_error: float | None = None
    maximum_relative_error: float | None = None
    numerics_evidence_id: str | None = None
    numerics_sample_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("distributed update node_id must be non-empty.")
        if not isinstance(self.payload, PayloadKind):
            raise TypeError("payload must be a PayloadKind.")
        if not isinstance(self.collective, CollectiveKind):
            raise TypeError("collective must be a CollectiveKind.")
        if not isinstance(self.state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if not isinstance(self.mesh_axes, tuple) or any(not isinstance(axis, ParallelAxis) for axis in self.mesh_axes):
            raise TypeError("mesh_axes must be a tuple of ParallelAxis values.")
        if len(set(self.mesh_axes)) != len(self.mesh_axes):
            raise ValueError("mesh_axes must not contain duplicates.")
        if not isinstance(self.exact, bool) or not isinstance(self.contract_exact, bool):
            raise TypeError("exact and contract_exact must be booleans.")
        if self.determinism_observed is not None and not isinstance(self.determinism_observed, bool):
            raise TypeError("determinism_observed must be boolean or None.")
        if not isinstance(self.notes, tuple) or any(not isinstance(note, str) for note in self.notes):
            raise TypeError("notes must be a tuple of strings.")
        if self.exact and self.collective is not CollectiveKind.NONE:
            raise ValueError("a distributed collective cannot claim guaranteed numerical exactness.")
        errors = (self.maximum_absolute_error, self.maximum_relative_error)
        if any(value is not None and (not math.isfinite(value) or value < 0.0) for value in errors):
            raise ValueError("observed collective errors must be finite and non-negative.")
        if (
            isinstance(self.numerics_sample_count, bool)
            or not isinstance(self.numerics_sample_count, Integral)
            or self.numerics_sample_count < 0
        ):
            raise ValueError("numerics_sample_count must be an exact non-negative integer.")
        object.__setattr__(self, "numerics_sample_count", int(self.numerics_sample_count))
        if self.numerics_evidence_id is not None and (
            not isinstance(self.numerics_evidence_id, str) or not self.numerics_evidence_id.strip()
        ):
            raise ValueError("numerics_evidence_id must be a non-empty string or None.")
        has_evidence = self.numerics_evidence_id is not None
        if has_evidence != (self.numerics_sample_count > 0):
            raise ValueError("collective numerical evidence requires both an id and positive sample count.")
        if has_evidence and (
            self.determinism_observed is None
            or self.maximum_absolute_error is None
            or self.maximum_relative_error is None
        ):
            raise ValueError("collective numerical evidence must report determinism and both error bounds.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "payload": self.payload.value,
            "collective": self.collective.value,
            "mesh_axes": [axis.value for axis in self.mesh_axes],
            "state_layout": self.state_layout.value,
            "exact": self.exact,
            "contract_exact": self.contract_exact,
            "determinism_observed": self.determinism_observed,
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "numerics_evidence_id": self.numerics_evidence_id,
            "numerics_sample_count": self.numerics_sample_count,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class StepReceipt:
    """Unambiguous accounting for one committed optimizer update."""

    step: int
    loss: float
    local_examples: int
    local_tokens: int
    microbatches: int
    accumulation_steps: int
    data_parallel_size: int
    optimizer: str
    precision: str
    collective_bytes: int = 0
    skipped: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    global_examples_observed: int | None = None
    global_tokens_observed: int | None = None

    def __post_init__(self) -> None:
        # detach, not freeze: a consumer type-tests this with isinstance(extra, dict), which a
        # mappingproxy fails (MXR-080-1876).
        object.__setattr__(self, "extra", detach_receipt_container(self.extra))
        integer_domains = {
            "step": 0,
            "local_examples": 0,
            "local_tokens": 0,
            "microbatches": 1,
            "accumulation_steps": 1,
            "data_parallel_size": 1,
            "collective_bytes": 0,
        }
        for name, minimum in integer_domains.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                qualifier = "positive" if minimum == 1 else "non-negative"
                raise ValueError(f"{name} must be an exact {qualifier} integer.")
            object.__setattr__(self, name, int(value))
        if not isinstance(self.loss, Real) or isinstance(self.loss, bool):
            raise TypeError("loss must be a finite non-negative real number.")
        loss = float(self.loss)
        if not math.isfinite(loss) or loss < 0.0:
            raise ValueError("loss must be a finite non-negative real number.")
        object.__setattr__(self, "loss", loss)
        if not isinstance(self.optimizer, str) or not self.optimizer.strip():
            raise ValueError("optimizer must be a non-empty string.")
        if not isinstance(self.precision, str) or not self.precision.strip():
            raise ValueError("precision must be a non-empty string.")
        if not isinstance(self.skipped, bool):
            raise TypeError("skipped must be boolean.")
        if not isinstance(self.extra, dict):
            raise TypeError("extra must be a dictionary.")
        for name in ("global_examples_observed", "global_tokens_observed"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                    raise ValueError(f"{name} must be None or an exact non-negative integer.")
                object.__setattr__(self, name, int(value))

    @property
    def global_examples(self) -> int:
        if self.global_examples_observed is not None:
            return self.global_examples_observed
        return self.local_examples * self.data_parallel_size

    @property
    def global_tokens(self) -> int:
        if self.global_tokens_observed is not None:
            return self.global_tokens_observed
        return self.local_tokens * self.data_parallel_size

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "local_examples": self.local_examples,
            "local_tokens": self.local_tokens,
            "microbatches": self.microbatches,
            "accumulation_steps": self.accumulation_steps,
            "data_parallel_size": self.data_parallel_size,
            "global_examples": self.global_examples,
            "global_tokens": self.global_tokens,
            "optimizer": self.optimizer,
            "precision": self.precision,
            "collective_bytes": self.collective_bytes,
            "skipped": self.skipped,
            "extra": dict(self.extra),
        }


@runtime_checkable
class DistributedTrainingSession(Protocol):
    """A materialized model, optimizer, mesh, and resumable step clock."""

    plan: ParallelPlan
    capabilities: BackendCapabilities
    step: int

    def train_batch(self, inputs: Any, targets: Any) -> StepReceipt: ...

    def finish_accumulation(self) -> StepReceipt | None: ...

    def discard_accumulation(self) -> StepReceipt | None: ...

    def close(self) -> None: ...


@runtime_checkable
class DistributedTrainingBackend(Protocol):
    """Structural extension point implemented by Torch, Megatron, or other engines."""

    capabilities: BackendCapabilities

    def prepare(self, module: Any, *, plan: ParallelPlan, **kwargs: Any) -> DistributedTrainingSession: ...


_TRAINING_BACKENDS: dict[str, Any] = {}


def register_training_backend(name: str, backend: Any, *, override: bool = False) -> None:
    """Register a backend instance or lazy factory without importing optional dependencies."""

    key = str(name).strip().lower().replace("-", "_")
    if not key:
        raise ValueError("training backend name must be non-empty.")
    if key in _TRAINING_BACKENDS and _TRAINING_BACKENDS[key] is not backend and not override:
        raise ValueError("distributed training backend %r is already registered." % key)
    _TRAINING_BACKENDS[key] = backend


def available_training_backends() -> tuple[str, ...]:
    return tuple(sorted(_TRAINING_BACKENDS))


def get_training_backend(name: str) -> Any:
    """Resolve a registered backend, calling a zero-argument lazy factory once."""

    key = str(name).strip().lower().replace("-", "_")
    if key not in _TRAINING_BACKENDS:
        raise ValueError(
            "unknown distributed training backend %r; registered backends: %s"
            % (name, ", ".join(available_training_backends()))
        )
    backend = _TRAINING_BACKENDS[key]
    if not isinstance(backend, DistributedTrainingBackend) and callable(backend):
        backend = backend()
        _TRAINING_BACKENDS[key] = backend
    if not isinstance(backend, DistributedTrainingBackend):
        raise TypeError("registered backend %r does not implement DistributedTrainingBackend." % key)
    return backend


__all__ = [
    "BackendCapabilities",
    "CollectiveKind",
    "DistributedTrainingBackend",
    "DistributedTrainingSession",
    "DistributedUpdate",
    "ParallelAxis",
    "ParallelPlan",
    "ParameterLayout",
    "PayloadKind",
    "StateLayout",
    "StepReceipt",
    "available_training_backends",
    "get_training_backend",
    "register_training_backend",
]
