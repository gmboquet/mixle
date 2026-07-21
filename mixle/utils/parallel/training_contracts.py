"""Contracts shared by distributed gradient-training backends.

The encoded-data backends reduce sufficient statistics.  Frontier neural
training has a different lifecycle: materialize a process mesh, transform a
module, bind optimizer state to logical parameters, execute steps, and save a
reshardable training state.  Keeping that lifecycle separate prevents a Ray or
Megatron integration from pretending to be an ``EncodedDataHandle``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


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
        dimensions = self.axis_sizes
        if any(size < 1 for size in dimensions.values()):
            raise ValueError("parallel dimensions must all be positive.")
        if self.microbatches < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("microbatches and gradient accumulation must be positive.")
        if self.data_parallel_size % self.ep:
            raise ValueError("expert parallelism must divide the data-parallel domain.")

    @property
    def axis_sizes(self) -> dict[ParallelAxis, int]:
        return {
            ParallelAxis.DP_REPLICATE: int(self.dp_replicate),
            ParallelAxis.DP_SHARD: int(self.dp_shard),
            ParallelAxis.TP: int(self.tp),
            ParallelAxis.PP: int(self.pp),
            ParallelAxis.CP: int(self.cp),
            ParallelAxis.EP: int(self.ep),
            ParallelAxis.ETP: int(self.etp),
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
        if int(actual) != self.world_size:
            raise ValueError(
                "parallel plan requires world_size=%d, but the process group has world_size=%d."
                % (self.world_size, int(actual))
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
    """Executable communication declaration for one typed update node."""

    node_id: str
    payload: PayloadKind
    collective: CollectiveKind
    mesh_axes: tuple[ParallelAxis, ...]
    state_layout: StateLayout
    exact: bool
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "payload": self.payload.value,
            "collective": self.collective.value,
            "mesh_axes": [axis.value for axis in self.mesh_axes],
            "state_layout": self.state_layout.value,
            "exact": self.exact,
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

    @property
    def global_examples(self) -> int:
        return self.local_examples * self.data_parallel_size

    @property
    def global_tokens(self) -> int:
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
