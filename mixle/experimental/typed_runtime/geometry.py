"""Parameter-geometry optimizer routing, curvature transforms, and batch receipts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.contracts import CurvatureKind, UpdateContract, UpdateKind


class ParameterRole(StrEnum):
    """Optimization-relevant role inferred from shape and module path."""

    EXACT_STATISTICAL = "exact_statistical"
    MATRIX = "matrix"
    EMBEDDING = "embedding"
    NORMALIZATION = "normalization"
    BIAS = "bias"
    SCALAR = "scalar"
    VECTOR = "vector"
    ROUTER = "router"
    LOW_RANK_ADAPTER = "low_rank_adapter"
    SPARSE_EXPERT = "sparse_expert"
    OTHER = "other"


class OptimizerFamily(StrEnum):
    """Optimizer/update family chosen for a parameter block."""

    EXACT = "exact"
    FROZEN = "frozen"
    ADAMW = "adamw"
    SGD_MOMENTUM = "sgd_momentum"
    DIAGONAL_ADAPTIVE = "diagonal_adaptive"
    MUON = "muon"
    KRONECKER = "kronecker"
    NATURAL_GRADIENT = "natural_gradient"
    PROXIMAL = "proximal"
    LOW_RANK_ADAPTIVE = "low_rank_adaptive"
    DISCRETE_SEARCH = "discrete_search"


@dataclass(frozen=True)
class ParameterDescriptor:
    """Shape, role, and storage facts for one named parameter."""

    name: str
    shape: tuple[int, ...]
    numel: int
    itemsize: int
    role: ParameterRole
    requires_grad: bool = True
    shared_group: str | None = None

    @property
    def parameter_bytes(self) -> int:
        """Dense parameter footprint."""

        return self.numel * self.itemsize

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible descriptor."""

        return {
            "name": self.name,
            "shape": list(self.shape),
            "numel": self.numel,
            "itemsize": self.itemsize,
            "parameter_bytes": self.parameter_bytes,
            "role": self.role.value,
            "requires_grad": self.requires_grad,
            "shared_group": self.shared_group,
        }


def _parameter_role(name: str, shape: tuple[int, ...]) -> ParameterRole:
    lower = name.lower()
    if "lora" in lower or "adapter" in lower:
        return ParameterRole.LOW_RANK_ADAPTER
    if "expert" in lower:
        return ParameterRole.SPARSE_EXPERT
    if "router" in lower or "gate" in lower:
        return ParameterRole.ROUTER
    if "embed" in lower or "embedding" in lower:
        return ParameterRole.EMBEDDING
    if lower.endswith("bias") or ".bias" in lower:
        return ParameterRole.BIAS
    if "norm" in lower or "layernorm" in lower or "rmsnorm" in lower:
        return ParameterRole.NORMALIZATION
    if not shape:
        return ParameterRole.SCALAR
    if len(shape) == 2:
        return ParameterRole.MATRIX
    if len(shape) == 1:
        return ParameterRole.VECTOR
    return ParameterRole.OTHER


def describe_parameters(module: Any) -> tuple[ParameterDescriptor, ...]:
    """Describe a torch-like module without importing torch."""

    named = getattr(module, "named_parameters", None)
    if callable(named):
        rows = tuple(named())
    else:
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            raise TypeError("module must expose named_parameters() or parameters().")
        rows = tuple(("parameter_%d" % index, parameter) for index, parameter in enumerate(parameters()))
    descriptors = []
    seen: dict[int, str] = {}
    for name, parameter in rows:
        shape = tuple(int(value) for value in getattr(parameter, "shape", ()))
        numel_fn = getattr(parameter, "numel", None)
        numel = int(numel_fn()) if callable(numel_fn) else int(np.prod(shape) if shape else 1)
        element_size = getattr(parameter, "element_size", None)
        itemsize = int(element_size()) if callable(element_size) else int(getattr(parameter, "itemsize", 4))
        identity = id(parameter)
        shared_group = seen.get(identity)
        if shared_group is None:
            seen[identity] = str(name)
        descriptors.append(
            ParameterDescriptor(
                str(name),
                shape,
                numel,
                itemsize,
                _parameter_role(str(name), shape),
                bool(getattr(parameter, "requires_grad", True)),
                shared_group,
            )
        )
    return tuple(descriptors)


@dataclass(frozen=True)
class GeometryRouterConfig:
    """Conservative matrix thresholds and optimizer-state memory limit."""

    matrix_min_elements: int = 4_096
    matrix_min_dimension: int = 16
    max_state_to_parameter_ratio: float = 4.0
    use_muon: bool = True
    use_kronecker: bool = True

    def __post_init__(self) -> None:
        if self.matrix_min_elements < 1 or self.matrix_min_dimension < 1:
            raise ValueError("matrix routing thresholds must be positive.")
        if self.max_state_to_parameter_ratio <= 0.0:
            raise ValueError("max_state_to_parameter_ratio must be positive.")


@dataclass(frozen=True)
class ParameterRoute:
    """Chosen family, fallback, state cost, and scheduling behavior."""

    parameter: ParameterDescriptor
    family: OptimizerFamily
    fallback_family: OptimizerFamily
    reason: str
    optimizer_state_bytes: int
    curvature_kind: CurvatureKind
    separate_clock: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible route."""

        return {
            "parameter": self.parameter.as_dict(),
            "family": self.family.value,
            "fallback_family": self.fallback_family.value,
            "reason": self.reason,
            "optimizer_state_bytes": self.optimizer_state_bytes,
            "curvature_kind": self.curvature_kind.value,
            "separate_clock": self.separate_clock,
        }


@dataclass(frozen=True)
class OptimizerPlan:
    """Per-parameter geometry routes for one typed node."""

    routes: tuple[ParameterRoute, ...]

    @property
    def optimizer_state_bytes(self) -> int:
        """Total planned optimizer and curvature state."""

        return sum(route.optimizer_state_bytes for route in self.routes)

    def route(self, name: str) -> ParameterRoute:
        """Return a route by parameter name."""

        return next(route for route in self.routes if route.parameter.name == name)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible plan."""

        return {
            "optimizer_state_bytes": self.optimizer_state_bytes,
            "routes": [route.as_dict() for route in self.routes],
        }


def _state_bytes(family: OptimizerFamily, parameter: ParameterDescriptor) -> int:
    scalar_bytes = max(4, parameter.itemsize)
    if family in (OptimizerFamily.EXACT, OptimizerFamily.FROZEN, OptimizerFamily.DISCRETE_SEARCH):
        return 0
    if family in (OptimizerFamily.ADAMW, OptimizerFamily.DIAGONAL_ADAPTIVE, OptimizerFamily.LOW_RANK_ADAPTIVE):
        return 2 * parameter.numel * scalar_bytes
    if family in (OptimizerFamily.MUON, OptimizerFamily.SGD_MOMENTUM):
        return parameter.numel * scalar_bytes
    if family is OptimizerFamily.KRONECKER and len(parameter.shape) == 2:
        rows, columns = parameter.shape
        return (parameter.numel + rows * rows + columns * columns) * scalar_bytes
    if family is OptimizerFamily.NATURAL_GRADIENT:
        return min(parameter.numel * parameter.numel, 4 * parameter.numel) * scalar_bytes
    return parameter.numel * scalar_bytes


def _route_one(
    parameter: ParameterDescriptor,
    contract: UpdateContract,
    config: GeometryRouterConfig,
) -> ParameterRoute:
    fallback = OptimizerFamily.ADAMW
    separate = parameter.role is ParameterRole.SPARSE_EXPERT
    if not parameter.requires_grad or contract.update_kind is UpdateKind.FROZEN:
        family, reason = OptimizerFamily.FROZEN, "parameter or typed node is frozen"
        fallback = OptimizerFamily.FROZEN
    elif contract.update_kind is UpdateKind.EXACT_CLOSED_FORM:
        family, reason = OptimizerFamily.EXACT, "closed-form statistical parameters bypass neural optimizers"
        fallback = OptimizerFamily.EXACT
    elif contract.update_kind in (UpdateKind.PROXIMAL,):
        family, reason = OptimizerFamily.PROXIMAL, "typed update requires projection/proximal geometry"
    elif contract.update_kind in (UpdateKind.DISCRETE_SEARCH,):
        family, reason = OptimizerFamily.DISCRETE_SEARCH, "non-differentiable typed search block"
        fallback = OptimizerFamily.DISCRETE_SEARCH
    elif parameter.role is ParameterRole.ROUTER and contract.curvature_kind is CurvatureKind.FISHER:
        family, reason = OptimizerFamily.NATURAL_GRADIENT, "probabilistic router has Fisher geometry"
    elif parameter.role is ParameterRole.LOW_RANK_ADAPTER:
        family, reason = OptimizerFamily.LOW_RANK_ADAPTIVE, "low-rank adapter keeps factor-specific state"
    elif parameter.role in (
        ParameterRole.EMBEDDING,
        ParameterRole.NORMALIZATION,
        ParameterRole.BIAS,
        ParameterRole.SCALAR,
        ParameterRole.VECTOR,
    ):
        family, reason = OptimizerFamily.ADAMW, "embedding/vector/scalar defaults conservatively to AdamW"
    elif (
        len(parameter.shape) == 2
        and parameter.numel >= config.matrix_min_elements
        and min(parameter.shape) >= config.matrix_min_dimension
    ):
        if contract.curvature_kind is CurvatureKind.KRONECKER and config.use_kronecker:
            family, reason = OptimizerFamily.KRONECKER, "large matrix with declared Kronecker curvature"
        elif config.use_muon:
            family, reason = OptimizerFamily.MUON, "large hidden matrix eligible for orthogonalized updates"
        else:
            family, reason = OptimizerFamily.ADAMW, "matrix optimizer disabled by router configuration"
    else:
        family, reason = OptimizerFamily.ADAMW, "small or unclassified trainable parameter"

    state_bytes = _state_bytes(family, parameter)
    if family in (
        OptimizerFamily.KRONECKER,
        OptimizerFamily.MUON,
        OptimizerFamily.NATURAL_GRADIENT,
    ) and state_bytes > config.max_state_to_parameter_ratio * max(parameter.parameter_bytes, 1):
        family = OptimizerFamily.ADAMW
        state_bytes = _state_bytes(family, parameter)
        reason = "geometry state exceeds configured memory ratio; fell back to AdamW"
    return ParameterRoute(parameter, family, fallback, reason, state_bytes, contract.curvature_kind, separate)


def route_optimizer_geometry(
    parameters: tuple[ParameterDescriptor, ...],
    contract: UpdateContract,
    config: GeometryRouterConfig | None = None,
) -> OptimizerPlan:
    """Build a conservative per-parameter optimizer plan."""

    config = config or GeometryRouterConfig()
    return OptimizerPlan(tuple(_route_one(parameter, contract, config) for parameter in parameters))


@dataclass(frozen=True)
class OptimizerEvidence:
    """Measured candidate performance at a fixed target definition."""

    parameter_name: str
    family: OptimizerFamily
    target_name: str
    target_achieved: bool
    time_to_target_seconds: float | None
    consumed_tokens: int
    optimizer_updates: int
    state_bytes: int
    collective_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.parameter_name or not self.target_name:
            raise ValueError("optimizer evidence parameter and target names must be non-empty.")
        if self.time_to_target_seconds is not None and self.time_to_target_seconds < 0.0:
            raise ValueError("time_to_target_seconds must be non-negative.")
        counts = (self.consumed_tokens, self.optimizer_updates, self.state_bytes, self.collective_bytes)
        if any(value < 0 for value in counts):
            raise ValueError("optimizer evidence work counters must be non-negative.")


def apply_optimizer_evidence(
    plan: OptimizerPlan,
    evidence: tuple[OptimizerEvidence, ...],
    *,
    minimum_time_improvement: float = 0.02,
) -> OptimizerPlan:
    """Fall back when a routed family does not beat measured AdamW time-to-target."""

    if not 0.0 <= minimum_time_improvement < 1.0:
        raise ValueError("minimum_time_improvement must be in [0, 1).")
    lookup = {(row.parameter_name, row.family): row for row in evidence}
    routes = []
    for route in plan.routes:
        if route.family in (OptimizerFamily.EXACT, OptimizerFamily.FROZEN, OptimizerFamily.ADAMW):
            routes.append(route)
            continue
        candidate = lookup.get((route.parameter.name, route.family))
        baseline = lookup.get((route.parameter.name, OptimizerFamily.ADAMW))
        fallback_reason = None
        if candidate is not None and baseline is not None:
            if not candidate.target_achieved:
                fallback_reason = "%s failed target; measured fallback to AdamW" % route.family.value
            elif baseline.target_achieved and baseline.time_to_target_seconds is not None:
                if (
                    candidate.time_to_target_seconds is None
                    or candidate.time_to_target_seconds
                    >= (1.0 - minimum_time_improvement) * baseline.time_to_target_seconds
                ):
                    fallback_reason = "%s did not beat AdamW time-to-target after overhead" % route.family.value
        if fallback_reason is None:
            routes.append(route)
        else:
            routes.append(
                replace(
                    route,
                    family=OptimizerFamily.ADAMW,
                    reason=fallback_reason,
                    optimizer_state_bytes=_state_bytes(OptimizerFamily.ADAMW, route.parameter),
                )
            )
    return OptimizerPlan(tuple(routes))


@dataclass(frozen=True)
class BatchSemanticsReceipt:
    """Unambiguous microbatch, accumulation, world-size, and update accounting."""

    examples_per_microbatch: int
    tokens_per_microbatch: int
    responsibility_mass_per_microbatch: float
    accumulation_steps: int
    data_parallel_world_size: int
    optimizer_updates: int
    loss_reduction: str
    loss_scale: float
    schedule_position: int

    def __post_init__(self) -> None:
        positive = (
            self.examples_per_microbatch,
            self.tokens_per_microbatch,
            self.accumulation_steps,
            self.data_parallel_world_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("microbatch size, accumulation, and world size must be positive.")
        if self.responsibility_mass_per_microbatch < 0.0 or self.optimizer_updates < 0:
            raise ValueError("responsibility mass and optimizer updates must be non-negative.")
        if not self.loss_reduction or self.loss_scale <= 0.0 or self.schedule_position < 0:
            raise ValueError("loss semantics and schedule position must be valid.")

    @property
    def effective_global_examples(self) -> int:
        return self.examples_per_microbatch * self.accumulation_steps * self.data_parallel_world_size

    @property
    def effective_global_tokens(self) -> int:
        return self.tokens_per_microbatch * self.accumulation_steps * self.data_parallel_world_size

    @property
    def effective_responsibility_mass(self) -> float:
        return self.responsibility_mass_per_microbatch * self.accumulation_steps * self.data_parallel_world_size

    def as_dict(self) -> dict[str, Any]:
        return {
            "examples_per_microbatch": self.examples_per_microbatch,
            "tokens_per_microbatch": self.tokens_per_microbatch,
            "responsibility_mass_per_microbatch": self.responsibility_mass_per_microbatch,
            "accumulation_steps": self.accumulation_steps,
            "data_parallel_world_size": self.data_parallel_world_size,
            "optimizer_updates": self.optimizer_updates,
            "loss_reduction": self.loss_reduction,
            "loss_scale": self.loss_scale,
            "schedule_position": self.schedule_position,
            "effective_global_examples": self.effective_global_examples,
            "effective_global_tokens": self.effective_global_tokens,
            "effective_responsibility_mass": self.effective_responsibility_mass,
        }


def orthogonalized_matrix_direction(gradient: Any) -> np.ndarray:
    """Exact polar factor used as a reference for Muon-style orthogonalization."""

    value = np.asarray(gradient, dtype=np.float64)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("orthogonalized direction requires a finite matrix.")
    left, _, right = np.linalg.svd(value, full_matrices=False)
    direction = left @ right
    return direction * math.sqrt(max(1.0, value.shape[0] / value.shape[1]))


def kronecker_precondition(
    gradient: Any,
    row_factor: Any,
    column_factor: Any,
    *,
    damping: float = 1.0e-6,
) -> np.ndarray:
    """Apply inverse fourth-root Kronecker factors to a matrix gradient."""

    gradient = np.asarray(gradient, dtype=np.float64)
    row_factor = np.asarray(row_factor, dtype=np.float64)
    column_factor = np.asarray(column_factor, dtype=np.float64)
    if gradient.ndim != 2:
        raise ValueError("Kronecker preconditioning requires a matrix gradient.")
    if row_factor.shape != (gradient.shape[0], gradient.shape[0]) or column_factor.shape != (
        gradient.shape[1],
        gradient.shape[1],
    ):
        raise ValueError("Kronecker factor shapes do not match gradient axes.")
    if damping <= 0.0:
        raise ValueError("Kronecker damping must be positive.")

    def inverse_quarter(matrix: np.ndarray) -> np.ndarray:
        values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
        powers = np.maximum(values, 0.0) + damping
        return (vectors * powers[None, :] ** -0.25) @ vectors.T

    return inverse_quarter(row_factor) @ gradient @ inverse_quarter(column_factor)


def natural_gradient_direction(gradient: Any, fisher: Any, *, damping: float = 1.0e-6) -> np.ndarray:
    """Solve a damped Fisher system for a natural-gradient direction."""

    gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
    fisher = np.asarray(fisher, dtype=np.float64)
    if fisher.shape != (gradient.size, gradient.size):
        raise ValueError("Fisher shape must match flattened gradient size.")
    if damping <= 0.0 or not np.all(np.isfinite(fisher)):
        raise ValueError("natural-gradient Fisher and damping must be finite/positive.")
    symmetric = 0.5 * (fisher + fisher.T)
    return np.linalg.solve(symmetric + damping * np.eye(gradient.size), gradient)


@dataclass(frozen=True)
class CurvatureSketch:
    """Versioned curvature factors that may be reused until their staleness bound."""

    key: str
    kind: CurvatureKind
    factors: tuple[np.ndarray, ...]
    model_version: int
    observations: float


@dataclass
class CurvatureCache:
    """Share versioned curvature sketches across related parameters/experts."""

    max_version_lag: int = 1
    _sketches: dict[str, CurvatureSketch] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_version_lag < 0:
            raise ValueError("max_version_lag must be non-negative.")

    def put(self, sketch: CurvatureSketch) -> None:
        if sketch.model_version < 0 or sketch.observations < 0.0 or not sketch.key:
            raise ValueError("curvature sketch metadata must be non-negative and named.")
        self._sketches[sketch.key] = sketch

    def get(self, key: str, *, model_version: int) -> CurvatureSketch | None:
        sketch = self._sketches.get(key)
        if sketch is None or model_version - sketch.model_version > self.max_version_lag:
            return None
        if model_version < sketch.model_version:
            raise ValueError("cannot read curvature from a future model version.")
        return sketch


__all__ = [
    "BatchSemanticsReceipt",
    "CurvatureCache",
    "CurvatureSketch",
    "GeometryRouterConfig",
    "OptimizerEvidence",
    "OptimizerFamily",
    "OptimizerPlan",
    "ParameterDescriptor",
    "ParameterRole",
    "ParameterRoute",
    "apply_optimizer_evidence",
    "describe_parameters",
    "kronecker_precondition",
    "natural_gradient_direction",
    "orthogonalized_matrix_direction",
    "route_optimizer_geometry",
]
