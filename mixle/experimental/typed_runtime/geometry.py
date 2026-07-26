"""Parameter-geometry optimizer routing, curvature transforms, and batch receipts."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.contracts import CurvatureKind, UpdateContract, UpdateKind
from mixle.experimental.typed_runtime.proposal import payload_fingerprint


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
    aliases: tuple[str, ...] = ()
    alias_roles: tuple[ParameterRole, ...] = ()

    def __post_init__(self) -> None:
        aliases = self.aliases or (self.name,)
        roles = self.alias_roles or (self.role,)
        if not self.name or len(set(aliases)) != len(aliases) or self.name not in aliases:
            raise ValueError("parameter descriptor requires unique aliases including its primary name.")
        if len(roles) != len(aliases):
            raise ValueError("parameter alias roles must align with aliases.")
        if self.numel < 0 or self.itemsize < 1 or any(dimension < 0 for dimension in self.shape):
            raise ValueError("parameter shape, size, and itemsize must be non-negative/positive.")
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "alias_roles", roles)

    @property
    def identity_fingerprint(self) -> str:
        """Stable logical identity used to bind measurements and curvature."""

        return payload_fingerprint(
            (
                self.name,
                self.aliases,
                tuple(role.value for role in self.alias_roles),
                self.shape,
                self.numel,
                self.itemsize,
                self.shared_group,
            )
        )

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
            "aliases": list(self.aliases),
            "alias_roles": [role.value for role in self.alias_roles],
            "identity_fingerprint": self.identity_fingerprint,
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
        try:
            signature = inspect.signature(named)
        except (TypeError, ValueError):
            signature = None
        supports_duplicates = signature is not None and (
            "remove_duplicate" in signature.parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        )
        rows = tuple(named(remove_duplicate=False)) if supports_duplicates else tuple(named())
    else:
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            raise TypeError("module must expose named_parameters() or parameters().")
        rows = tuple(("parameter_%d" % index, parameter) for index, parameter in enumerate(parameters()))
    by_name: dict[str, int] = {}
    groups: dict[int, tuple[Any, list[str]]] = {}
    for raw_name, parameter in rows:
        name = str(raw_name)
        previous = by_name.get(name)
        if previous is not None and previous != id(parameter):
            raise ValueError("module parameter name resolves to multiple parameter objects: %s" % name)
        by_name[name] = id(parameter)
        if id(parameter) not in groups:
            groups[id(parameter)] = (parameter, [])
        groups[id(parameter)][1].append(name)

    descriptors = []
    for parameter, names in groups.values():
        name = names[0]
        shape = tuple(int(value) for value in getattr(parameter, "shape", ()))
        numel_fn = getattr(parameter, "numel", None)
        numel = int(numel_fn()) if callable(numel_fn) else int(np.prod(shape) if shape else 1)
        element_size = getattr(parameter, "element_size", None)
        itemsize = int(element_size()) if callable(element_size) else int(getattr(parameter, "itemsize", 4))
        alias_roles = tuple(_parameter_role(alias, shape) for alias in names)
        role = alias_roles[0] if len(set(alias_roles)) == 1 else ParameterRole.OTHER
        shared_group = "shared:%s" % name if len(names) > 1 else None
        descriptors.append(
            ParameterDescriptor(
                name,
                shape,
                numel,
                itemsize,
                role,
                bool(getattr(parameter, "requires_grad", True)),
                shared_group,
                tuple(names),
                alias_roles,
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
    elif parameter.shared_group is not None and parameter.role is ParameterRole.OTHER:
        family, reason = (
            OptimizerFamily.ADAMW,
            "shared aliases span optimization roles; conservative shared-group route",
        )
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
    parameter_fingerprint: str
    model_version: int
    target_achieved: bool
    time_to_target_seconds: float | None
    consumed_tokens: int
    optimizer_updates: int
    state_bytes: int
    collective_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.parameter_name or not self.target_name or not self.parameter_fingerprint:
            raise ValueError("optimizer evidence parameter, fingerprint, and target names must be non-empty.")
        if self.model_version < 0:
            raise ValueError("optimizer evidence model_version must be non-negative.")
        if self.time_to_target_seconds is not None and (
            not math.isfinite(self.time_to_target_seconds) or self.time_to_target_seconds < 0.0
        ):
            raise ValueError("time_to_target_seconds must be finite and non-negative.")
        if self.target_achieved and self.time_to_target_seconds is None:
            raise ValueError("achieved optimizer targets require a measured time_to_target_seconds.")
        counts = (self.consumed_tokens, self.optimizer_updates, self.state_bytes, self.collective_bytes)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("optimizer evidence work counters must be non-negative integers.")


def apply_optimizer_evidence(
    plan: OptimizerPlan,
    evidence: tuple[OptimizerEvidence, ...],
    *,
    target_name: str,
    model_version: int,
    minimum_time_improvement: float = 0.02,
) -> OptimizerPlan:
    """Fall back when a routed family does not beat measured AdamW time-to-target."""

    if not target_name or model_version < 0:
        raise ValueError("optimizer evidence target_name and model_version must be valid.")
    if not math.isfinite(minimum_time_improvement) or not 0.0 <= minimum_time_improvement < 1.0:
        raise ValueError("minimum_time_improvement must be in [0, 1).")
    parameters = {route.parameter.name: route.parameter for route in plan.routes}
    lookup = {}
    for row in evidence:
        parameter = parameters.get(row.parameter_name)
        if parameter is None:
            raise ValueError("optimizer evidence refers to an unknown parameter: %s" % row.parameter_name)
        if row.target_name != target_name:
            raise ValueError("optimizer evidence target does not match the requested target.")
        if row.model_version != model_version:
            raise ValueError("optimizer evidence model version does not match the routed model.")
        if row.parameter_fingerprint != parameter.identity_fingerprint:
            raise ValueError("optimizer evidence parameter fingerprint does not match the route.")
        key = (row.parameter_name, row.family)
        if key in lookup:
            raise ValueError("duplicate optimizer evidence for one parameter/family/target.")
        lookup[key] = row
    routes = []
    for route in plan.routes:
        if route.family in (OptimizerFamily.EXACT, OptimizerFamily.FROZEN, OptimizerFamily.ADAMW):
            routes.append(route)
            continue
        candidate = lookup.get((route.parameter.name, route.family))
        baseline = lookup.get((route.parameter.name, OptimizerFamily.ADAMW))
        fallback_reason = None
        if candidate is None or baseline is None:
            fallback_reason = "%s lacks complete target-bound evidence; fell back to AdamW" % route.family.value
        elif not candidate.target_achieved:
            fallback_reason = "%s failed target; measured fallback to AdamW" % route.family.value
        elif baseline.target_achieved and (
            candidate.time_to_target_seconds
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
    if not np.all(np.isfinite(gradient)):
        raise ValueError("Kronecker gradient must be finite.")
    if row_factor.shape != (gradient.shape[0], gradient.shape[0]) or column_factor.shape != (
        gradient.shape[1],
        gradient.shape[1],
    ):
        raise ValueError("Kronecker factor shapes do not match gradient axes.")
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("Kronecker damping must be finite and positive.")

    def inverse_quarter(matrix: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(matrix)) or not np.allclose(
            matrix,
            matrix.T,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError("Kronecker factors must be finite and symmetric.")
        values, vectors = np.linalg.eigh(matrix)
        if float(np.min(values)) < 0.0:
            raise ValueError("Kronecker factors must be positive semidefinite.")
        powers = values + damping
        return (vectors * powers[None, :] ** -0.25) @ vectors.T

    result = inverse_quarter(row_factor) @ gradient @ inverse_quarter(column_factor)
    if not np.all(np.isfinite(result)):
        raise ValueError("Kronecker preconditioning produced a non-finite direction.")
    return result


def natural_gradient_direction(gradient: Any, fisher: Any, *, damping: float = 1.0e-6) -> np.ndarray:
    """Solve a damped Fisher system for a natural-gradient direction."""

    gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
    fisher = np.asarray(fisher, dtype=np.float64)
    if not np.all(np.isfinite(gradient)):
        raise ValueError("natural-gradient input gradient must be finite.")
    if fisher.shape != (gradient.size, gradient.size):
        raise ValueError("Fisher shape must match flattened gradient size.")
    if (
        not math.isfinite(damping)
        or damping <= 0.0
        or not np.all(np.isfinite(fisher))
        or not np.allclose(fisher, fisher.T, rtol=1.0e-10, atol=1.0e-12)
    ):
        raise ValueError("natural-gradient Fisher and damping must be finite/positive.")
    eigenvalues = np.linalg.eigvalsh(fisher)
    if float(np.min(eigenvalues)) < 0.0:
        raise ValueError("natural-gradient Fisher must be positive semidefinite.")
    result = np.linalg.solve(fisher + damping * np.eye(gradient.size), gradient)
    if not np.all(np.isfinite(result)):
        raise ValueError("natural-gradient solve produced a non-finite direction.")
    return result


@dataclass(frozen=True)
class CurvatureSketch:
    """Versioned curvature factors that may be reused until their staleness bound."""

    key: str
    kind: CurvatureKind
    factors: tuple[np.ndarray, ...]
    model_id: str
    parameter_fingerprint: str
    model_version: int
    observations: float

    def __post_init__(self) -> None:
        if not self.key or not self.model_id or not self.parameter_fingerprint:
            raise ValueError("curvature sketch key, model, and parameter identity must be non-empty.")
        if self.model_version < 0 or not math.isfinite(self.observations) or self.observations <= 0.0:
            raise ValueError("curvature sketch version and observations must be valid.")
        if not self.factors:
            raise ValueError("curvature sketch requires at least one factor.")
        if self.kind is CurvatureKind.UNAVAILABLE:
            raise ValueError("unavailable curvature cannot be cached as a factor sketch.")
        if self.kind is CurvatureKind.KRONECKER and len(self.factors) != 2:
            raise ValueError("Kronecker curvature sketches require row and column factors.")
        if self.kind in (CurvatureKind.FISHER, CurvatureKind.DIAGONAL) and len(self.factors) != 1:
            raise ValueError("Fisher/diagonal curvature sketches require exactly one factor.")
        frozen = []
        for factor in self.factors:
            value = np.array(factor, dtype=np.float64, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError("curvature sketch factors must be finite.")
            if self.kind in (CurvatureKind.KRONECKER, CurvatureKind.FISHER):
                if value.ndim != 2 or value.shape[0] != value.shape[1]:
                    raise ValueError("Kronecker/Fisher curvature factors must be square matrices.")
                if not np.allclose(value, value.T, rtol=1.0e-10, atol=1.0e-12):
                    raise ValueError("Kronecker/Fisher curvature factors must be symmetric.")
                eigenvalues = np.linalg.eigvalsh(value)
                if float(np.min(eigenvalues)) < 0.0:
                    raise ValueError("Kronecker/Fisher curvature factors must be positive semidefinite.")
            elif self.kind is CurvatureKind.DIAGONAL and (
                value.ndim != 1 or np.any(value < 0.0)
            ):
                raise ValueError("diagonal curvature factors must be non-negative vectors.")
            immutable = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
            frozen.append(immutable)
        object.__setattr__(self, "factors", tuple(frozen))

    @property
    def factor_fingerprint(self) -> str:
        """Content identity for immutable cached factors."""

        return payload_fingerprint(self.factors)


@dataclass
class CurvatureCache:
    """Share versioned curvature sketches across related parameters/experts."""

    max_version_lag: int = 0
    _sketches: dict[str, CurvatureSketch] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_version_lag < 0:
            raise ValueError("max_version_lag must be non-negative.")

    def put(self, sketch: CurvatureSketch) -> None:
        existing = self._sketches.get(sketch.key)
        if existing is not None:
            if (
                existing.model_id != sketch.model_id
                or existing.parameter_fingerprint != sketch.parameter_fingerprint
            ):
                raise ValueError("curvature cache key cannot be reused across model/parameter identities.")
            if sketch.model_version < existing.model_version:
                raise ValueError("curvature cache cannot replace a newer sketch with a stale one.")
        self._sketches[sketch.key] = sketch

    def get(
        self,
        key: str,
        *,
        model_id: str,
        parameter_fingerprint: str,
        model_version: int,
    ) -> CurvatureSketch | None:
        sketch = self._sketches.get(key)
        if sketch is None:
            return None
        if sketch.model_id != model_id or sketch.parameter_fingerprint != parameter_fingerprint:
            raise ValueError("curvature cache identity does not match the requested model/parameter.")
        if model_version < sketch.model_version:
            raise ValueError("cannot read curvature from a future model version.")
        if model_version - sketch.model_version > self.max_version_lag:
            return None
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
