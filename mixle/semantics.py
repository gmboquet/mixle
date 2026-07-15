"""Minimal domain-neutral semantics for uncertain values and inference artifacts.

The contracts in this module describe *what a quantity and result mean*.  They
deliberately exclude inquiry graphs, domain laws, solvers, jobs, and storage.
Operational metadata is serialized for audit but excluded from semantic
identity, so moving an unchanged problem between backends cannot change it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from importlib.resources import files
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "SEMANTICS_SCHEMA_VERSION",
    "CalibrationArtifact",
    "CapabilityExtension",
    "ConstraintSpec",
    "DecisionArtifact",
    "LikelihoodSpec",
    "ObservationSpec",
    "PosteriorArtifact",
    "PredictiveArtifact",
    "PriorSpec",
    "TraceEvent",
    "TraceSink",
    "TransformKind",
    "TransformSpec",
    "UncertaintyComponent",
    "UncertaintyKind",
    "ValueRole",
    "ValueSpec",
    "canonical_json",
    "load_reference_fixture",
    "semantic_digest",
    "to_record",
]

SEMANTICS_SCHEMA_VERSION = "1.0.0"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _normalize(value: Any, *, semantic: bool) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            item.name: _normalize(getattr(value, item.name), semantic=semantic)
            for item in dataclasses.fields(value)
            if not semantic or item.metadata.get("semantic", True)
        }
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical mappings require string keys")
        return {key: _normalize(value[key], semantic=semantic) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_normalize(item, semantic=semantic) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"value is not finite canonical JSON: {type(value).__name__}")


def canonical_json(value: Any, *, semantic: bool = True) -> bytes:
    return json.dumps(
        _normalize(value, semantic=semantic), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def semantic_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value, semantic=True)).hexdigest()


def to_record(value: Any) -> dict[str, Any]:
    record = _normalize(value, semantic=False)
    if not isinstance(record, dict):
        raise TypeError("record root must be an object")
    return record


def _require_id(value: str, label: str) -> None:
    if not _ID.fullmatch(value):
        raise ValueError(f"{label} must be a portable non-empty identifier")


class ValueRole(StrEnum):
    FIXED = "fixed"
    FREE = "free"
    OBSERVED = "observed"
    LATENT = "latent"
    DERIVED = "derived"
    CONTROLLED = "controlled"


class TransformKind(StrEnum):
    IDENTITY = "identity"
    LOG = "log"
    LOGIT = "logit"
    AFFINE = "affine"


class UncertaintyKind(StrEnum):
    ALEATORIC = "aleatoric"
    EPISTEMIC = "epistemic"
    MEASUREMENT = "measurement"
    MODEL_DISCREPANCY = "model_discrepancy"
    NUMERICAL = "numerical"


@dataclass(frozen=True)
class ConstraintSpec:
    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    allowed_values: tuple[str | int | float | bool, ...] = ()
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.lower is not None and not math.isfinite(self.lower):
            raise ValueError("constraint lower bound must be finite")
        if self.upper is not None and not math.isfinite(self.upper):
            raise ValueError("constraint upper bound must be finite")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("constraint lower bound exceeds upper bound")
        if len(self.allowed_values) != len(set(self.allowed_values)):
            raise ValueError("constraint allowed_values must be unique")

    def accepts(self, value: Any) -> bool:
        values = value if isinstance(value, (tuple, list)) else (value,)
        for item in values:
            if self.allowed_values and item not in self.allowed_values:
                return False
            if self.lower is not None and (item < self.lower if self.lower_inclusive else item <= self.lower):
                return False
            if self.upper is not None and (item > self.upper if self.upper_inclusive else item >= self.upper):
                return False
        return True

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> ConstraintSpec:
        return cls(**{**value, "allowed_values": tuple(value.get("allowed_values", ()))})


@dataclass(frozen=True)
class TransformSpec:
    kind: TransformKind = TransformKind.IDENTITY
    scale: float = 1.0
    offset: float = 0.0
    lower: float = 0.0
    upper: float = 1.0
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not all(math.isfinite(item) for item in (self.scale, self.offset, self.lower, self.upper)):
            raise ValueError("transform parameters must be finite")
        if self.kind is TransformKind.AFFINE and self.scale == 0:
            raise ValueError("affine transform scale cannot be zero")
        if self.kind is TransformKind.LOGIT and self.lower >= self.upper:
            raise ValueError("logit transform requires lower < upper")

    def forward(self, value: float) -> float:
        if self.kind is TransformKind.IDENTITY:
            return float(value)
        if self.kind is TransformKind.LOG:
            if value <= 0:
                raise ValueError("log transform requires a positive value")
            return math.log(value)
        if self.kind is TransformKind.LOGIT:
            if not self.lower < value < self.upper:
                raise ValueError("logit value lies outside the open transform interval")
            return math.log((value - self.lower) / (self.upper - value))
        return self.scale * value + self.offset

    def inverse(self, value: float) -> float:
        if self.kind is TransformKind.IDENTITY:
            return float(value)
        if self.kind is TransformKind.LOG:
            return math.exp(value)
        if self.kind is TransformKind.LOGIT:
            probability = 1.0 / (1.0 + math.exp(-value))
            return self.lower + (self.upper - self.lower) * probability
        return (value - self.offset) / self.scale

    def log_abs_det_jacobian(self, value: float) -> float:
        """Log absolute derivative of ``forward`` at a natural-space value."""
        if self.kind is TransformKind.IDENTITY:
            return 0.0
        if self.kind is TransformKind.LOG:
            if value <= 0:
                raise ValueError("log transform requires a positive value")
            return -math.log(value)
        if self.kind is TransformKind.LOGIT:
            if not self.lower < value < self.upper:
                raise ValueError("logit value lies outside the open transform interval")
            return math.log(self.upper - self.lower) - math.log(value - self.lower) - math.log(self.upper - value)
        return math.log(abs(self.scale))

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> TransformSpec:
        return cls(**{**value, "kind": TransformKind(value.get("kind", "identity"))})


@dataclass(frozen=True)
class PriorSpec:
    id: str
    family: str
    parameters: Mapping[str, Any]
    unit: str = "1"
    source_ref: str | None = None
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.id, "prior id")
        _require_id(self.family, "prior family")
        if not self.unit:
            raise ValueError("prior unit is required; use '1' for dimensionless")
        canonical_json(self.parameters)

    @property
    def identity(self) -> str:
        return semantic_digest(self)

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> PriorSpec:
        return cls(**value)


@dataclass(frozen=True)
class ValueSpec:
    id: str
    role: ValueRole
    unit: str
    dtype: str = "float64"
    shape: tuple[int, ...] = ()
    transform: TransformSpec = field(default_factory=TransformSpec)
    constraint: ConstraintSpec | None = None
    prior: PriorSpec | None = None
    value: Any = None
    expression: str | None = None
    dependencies: tuple[str, ...] = ()
    description: str = ""
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.id, "value id")
        if not self.unit:
            raise ValueError("value unit is required; use '1' for dimensionless")
        if any(not isinstance(size, int) or size <= 0 for size in self.shape):
            raise ValueError("value shape dimensions must be positive integers")
        if self.role in {ValueRole.FREE, ValueRole.LATENT} and self.prior is None:
            raise ValueError(f"{self.role.value} values require a prior")
        if self.role in {ValueRole.FIXED, ValueRole.CONTROLLED} and self.value is None:
            raise ValueError(f"{self.role.value} values require a value")
        if self.role is ValueRole.DERIVED and (not self.expression or not self.dependencies):
            raise ValueError("derived values require an expression and dependencies")
        if self.role is not ValueRole.DERIVED and (self.expression or self.dependencies):
            raise ValueError("only derived values may declare expression dependencies")
        if self.role in {ValueRole.FIXED, ValueRole.CONTROLLED, ValueRole.OBSERVED, ValueRole.DERIVED} and self.prior:
            raise ValueError(f"{self.role.value} values cannot declare a prior")
        if self.constraint and self.value is not None and not self.constraint.accepts(self.value):
            raise ValueError("value violates its declared constraint")
        canonical_json(self.value)

    @property
    def identity(self) -> str:
        return semantic_digest(self)

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> ValueSpec:
        return cls(
            **{
                **value,
                "role": ValueRole(value["role"]),
                "shape": tuple(value.get("shape", ())),
                "dependencies": tuple(value.get("dependencies", ())),
                "transform": TransformSpec.from_record(value.get("transform", {})),
                "constraint": ConstraintSpec.from_record(value["constraint"]) if value.get("constraint") else None,
                "prior": PriorSpec.from_record(value["prior"]) if value.get("prior") else None,
            }
        )


@dataclass(frozen=True)
class LikelihoodSpec:
    id: str
    family: str
    observation_ids: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    discrepancy_ref: str | None = None
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.id, "likelihood id")
        _require_id(self.family, "likelihood family")
        if not self.observation_ids or len(self.observation_ids) != len(set(self.observation_ids)):
            raise ValueError("likelihood requires unique observation ids")
        canonical_json(self.parameters)

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> LikelihoodSpec:
        return cls(**{**value, "observation_ids": tuple(value["observation_ids"])})


@dataclass(frozen=True)
class ObservationSpec:
    id: str
    value_spec_id: str
    content_digest: str
    data_ref: str = field(metadata={"semantic": False})
    likelihood: LikelihoodSpec
    measurement_uncertainty: float | None = None
    unit: str | None = None
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.id, "observation id")
        _require_id(self.value_spec_id, "observation value id")
        if not _SHA256.fullmatch(self.content_digest) or not self.data_ref:
            raise ValueError("observation content digest and data reference are required")
        if self.id not in self.likelihood.observation_ids:
            raise ValueError("observation likelihood must reference the observation id")
        if self.measurement_uncertainty is not None and self.measurement_uncertainty < 0:
            raise ValueError("measurement uncertainty cannot be negative")

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> ObservationSpec:
        return cls(**{**value, "likelihood": LikelihoodSpec.from_record(value["likelihood"])})


@dataclass(frozen=True)
class UncertaintyComponent:
    id: str
    kind: UncertaintyKind
    measure: str
    value: float | None = None
    artifact_digest: str | None = None
    unit: str = "1"
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.id, "uncertainty id")
        if (self.value is None) == (self.artifact_digest is None):
            raise ValueError("uncertainty requires exactly one scalar value or artifact digest")
        if self.value is not None and (not math.isfinite(self.value) or self.value < 0):
            raise ValueError("uncertainty scalar must be finite and nonnegative")
        if self.artifact_digest is not None and not _SHA256.fullmatch(self.artifact_digest):
            raise ValueError("uncertainty artifact digest must be SHA-256")

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> UncertaintyComponent:
        return cls(**{**value, "kind": UncertaintyKind(value["kind"])})


@dataclass(frozen=True)
class PosteriorArtifact:
    id: str
    values: tuple[ValueSpec, ...]
    observations: tuple[ObservationSpec, ...]
    likelihood: LikelihoodSpec
    method: str
    random_seed: int
    summary: Mapping[str, Any]
    uncertainty: tuple[UncertaintyComponent, ...]
    sample_digest: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    sample_ref: str | None = field(default=None, metadata={"semantic": False})
    backend_id: str | None = field(default=None, metadata={"semantic": False})
    job_id: str | None = field(default=None, metadata={"semantic": False})
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.id, "posterior id")
        if not self.values or not self.observations or not self.method:
            raise ValueError("posterior values, observations, and method are required")
        value_ids = {item.id for item in self.values}
        observation_ids = {item.id for item in self.observations}
        if len(value_ids) != len(self.values) or len(observation_ids) != len(self.observations):
            raise ValueError("posterior value and observation ids must be unique")
        if any(item.value_spec_id not in value_ids for item in self.observations):
            raise ValueError("posterior observation references an unknown value")
        if set(self.likelihood.observation_ids) != observation_ids:
            raise ValueError("posterior likelihood must close over exactly its observations")
        if self.sample_digest is not None and not _SHA256.fullmatch(self.sample_digest):
            raise ValueError("posterior sample digest must be SHA-256")
        canonical_json(self.summary)
        canonical_json(self.diagnostics)

    @property
    def identity(self) -> str:
        return semantic_digest(self)

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> PosteriorArtifact:
        return cls(
            **{
                **value,
                "values": tuple(ValueSpec.from_record(item) for item in value["values"]),
                "observations": tuple(ObservationSpec.from_record(item) for item in value["observations"]),
                "likelihood": LikelihoodSpec.from_record(value["likelihood"]),
                "uncertainty": tuple(UncertaintyComponent.from_record(item) for item in value["uncertainty"]),
            }
        )


@dataclass(frozen=True)
class PredictiveArtifact:
    id: str
    posterior_identity: str
    target_value_ids: tuple[str, ...]
    content_digest: str
    uncertainty: tuple[UncertaintyComponent, ...]
    method: str
    backend_id: str | None = field(default=None, metadata={"semantic": False})
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.id, "predictive id")
        if not _SHA256.fullmatch(self.posterior_identity) or not _SHA256.fullmatch(self.content_digest):
            raise ValueError("predictive posterior and content identities must be SHA-256")
        if not self.target_value_ids or not self.method:
            raise ValueError("predictive targets and method are required")


@dataclass(frozen=True)
class CalibrationArtifact:
    id: str
    target_identity: str
    method: str
    metrics: Mapping[str, float]
    slice: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.id, "calibration id")
        if not _SHA256.fullmatch(self.target_identity) or not self.method or not self.metrics:
            raise ValueError("calibration target, method, and metrics are required")
        canonical_json(self.metrics)


@dataclass(frozen=True)
class DecisionArtifact:
    id: str
    alternatives: tuple[str, ...]
    selected: str
    utility: Mapping[str, float]
    posterior_identity: str
    risk_measure: str
    assumptions: tuple[str, ...] = ()
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_id(self.id, "decision id")
        if len(self.alternatives) < 2 or len(set(self.alternatives)) != len(self.alternatives):
            raise ValueError("decision alternatives must contain at least two unique values")
        if self.selected not in self.alternatives or set(self.utility) != set(self.alternatives):
            raise ValueError("decision selection and utility must close over the alternatives")
        if not _SHA256.fullmatch(self.posterior_identity) or not self.risk_measure:
            raise ValueError("decision posterior identity and risk measure are required")


@dataclass(frozen=True)
class CapabilityExtension:
    id: str
    owner_project: str
    input_schema_uri: str
    output_schema_uri: str
    maturity: str
    schema_version: str = SEMANTICS_SCHEMA_VERSION


@dataclass(frozen=True)
class TraceEvent:
    trace_id: str
    sequence: int
    event_type: str
    semantic_identity: str
    payload: Mapping[str, Any]
    occurred_at: str
    schema_version: str = SEMANTICS_SCHEMA_VERSION


@runtime_checkable
class TraceSink(Protocol):
    def emit(self, event: TraceEvent) -> None: ...


def load_reference_fixture() -> dict[str, Any]:
    """Load the packaged cross-project value/prior/observation fixture."""
    return json.loads(files("mixle").joinpath("fixtures/quantitative-semantics-v1.json").read_text())
