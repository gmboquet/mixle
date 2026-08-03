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
from datetime import datetime
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

#: The closed set of value dtypes this schema version defines. ``dtype`` used to be an arbitrary
#: unchecked string, so a value could be certified as ``"not-a-dtype"``.
DTYPES: frozenset[str] = frozenset({"float64", "float32", "int64", "int32", "bool", "str"})

_DTYPE_TYPES: dict[str, tuple[type, ...]] = {
    "float64": (int, float),
    "float32": (int, float),
    "int64": (int,),
    "int32": (int,),
    "bool": (bool,),
    "str": (str,),
}


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


def _require_schema_version(value: Any, label: str) -> None:
    """Reject a record that does not declare exactly the schema version this module implements.

    Every contract here carries ``schema_version``, but nothing checked it, so a record declaring
    ``"future/999"`` constructed normally and was then interpreted under the 1.0 field rules and given
    an authoritative semantic identity. A version this build does not implement must stay unreadable
    (a caller can migrate it explicitly) rather than be silently reinterpreted as a current contract.
    """
    if value != SEMANTICS_SCHEMA_VERSION:
        raise ValueError(
            f"{label} declares schema_version {value!r}, which this build does not implement; "
            f"only {SEMANTICS_SCHEMA_VERSION!r} is supported -- migrate the record explicitly"
        )


def _canonical_mapping(value: Any, label: str) -> dict[str, Any]:
    """Deep-normalize a caller-owned mapping into an owned canonical value.

    Frozen records used to retain the caller's mapping BY REFERENCE, so a durable semantic identity
    changed underneath itself: constructing a ``PriorSpec`` from ``{"mu": 0}`` and then setting the
    original mapping's ``"mu"`` to ``9`` changed that same object's ``identity`` digest. Normalizing
    (which also validates canonical-JSON-ness, as the old ``canonical_json(...)`` call did) produces
    fresh nested containers the caller has no handle on.
    """
    normalized = _normalize(value, semantic=False)
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must be a mapping with string keys")
    return normalized


def _own_sequence(value: Any, label: str) -> tuple[Any, ...]:
    """Copy a caller-owned sequence field into the tuple the contract declares (MXR-080-1903).

    One boundary, two defects that only show up together:

    * *retained mutable evidence.* These fields are annotated ``tuple[...]`` but nothing enforced it,
      so a record built from a ``list`` kept the CALLER's list by reference.  Appending to that list
      afterwards changed what the record contains -- while ``identity``, the digest cached at
      construction, went on reporting the content the record had when it was built.  A durable record
      then advertises an identity that no longer matches it, and validated invariants (a decision's
      ``utility`` closing over its ``alternatives``, a posterior's likelihood closing over its
      observations) silently stop holding.
    * *a string read as a sequence of characters.* ``alternatives="ab"`` and
      ``observation_ids="o1"`` satisfied every length/uniqueness/membership check by iterating
      characters, and ``to_record`` then emitted a JSON string where the schema declares an array.

    Only ``list``/``tuple`` are admitted.  Unordered containers are refused rather than sorted:
    element order is part of the canonical JSON the semantic digest is taken over, so accepting a
    ``set`` would make a record's durable identity depend on iteration order.  Generators are refused
    for the same reason a second pass over one yields nothing -- these fields are re-read by the
    validations below.

    Not checked here, deliberately: the *elements*.  Each record validates its own element types
    immediately after this call, and a shared helper cannot know whether a field holds ids, specs, or
    scalars.
    """
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(
            f"{label} must be a list or tuple of items, not {type(value).__name__}; a string here "
            "would be read as its individual characters"
        )
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} must be a list or tuple, got {type(value).__name__}")
    return tuple(value)


def _measured_shape(value: Any, label: str) -> tuple[int, ...]:
    """The shape ``value`` actually has: ``()`` for a scalar, ``(n, ...)`` for rectangular nested lists."""
    if not isinstance(value, (list, tuple)):
        return ()
    inner = {_measured_shape(item, label) for item in value}
    if len(inner) > 1:
        raise ValueError(f"{label} is a ragged array; a declared shape requires rectangular data")
    return (len(value), *(inner.pop() if inner else ()))


def _check_dtype(value: Any, dtype: str, label: str) -> None:
    """Every leaf of ``value`` must belong to ``dtype``'s Python domain."""
    if isinstance(value, (list, tuple)):
        for item in value:
            _check_dtype(item, dtype, label)
        return
    allowed = _DTYPE_TYPES[dtype]
    # bool subclasses int, so it satisfies an int/float isinstance check it has no business satisfying
    if isinstance(value, bool) and dtype != "bool":
        raise ValueError(f"{label} holds a Boolean but declares dtype {dtype!r}")
    if not isinstance(value, allowed):
        raise ValueError(f"{label} holds {type(value).__name__} but declares dtype {dtype!r}")
    if dtype in ("float64", "float32") and not math.isfinite(value):
        raise ValueError(f"{label} holds a non-finite number")


def _coerce_enum(value: Any, enum_cls: type[StrEnum], label: str) -> StrEnum:
    """Coerce ``value`` into a member of ``enum_cls``, as every ``from_record`` already does.

    Constructors below branch on these fields with identity checks against specific members
    (e.g. ``self.kind is TransformKind.LOG``), which a plain string never matches. Without this,
    direct construction (bypassing ``from_record``) silently falls through to a default branch
    instead of the requested behavior or a clear error.
    """
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a valid {enum_cls.__name__} value, got {value!r}") from exc


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
        _require_schema_version(self.schema_version, "constraint")
        object.__setattr__(self, "allowed_values", _own_sequence(self.allowed_values, "constraint allowed_values"))
        if self.lower is not None and not math.isfinite(self.lower):
            raise ValueError("constraint lower bound must be finite")
        if self.upper is not None and not math.isfinite(self.upper):
            raise ValueError("constraint upper bound must be finite")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("constraint lower bound exceeds upper bound")
        if len(self.allowed_values) != len(set(self.allowed_values)):
            raise ValueError("constraint allowed_values must be unique")

    def accepts(self, value: Any) -> bool:
        """Whether ``value`` (a scalar, or a non-empty vector of scalars) satisfies this constraint.

        Admissibility is decided by explicit checks, never by IEEE comparison behavior: NaN used to be
        accepted by a bounded constraint because BOTH ordered comparisons against it are false, and an
        empty list was accepted because the validation loop ran zero times -- a vacuous container
        certified as satisfying every bound. A bounded constraint also has no ordering to apply to a
        Boolean or a non-numeric value (``bool`` only compares because it subclasses ``int``), so those
        are rejected rather than silently ranked.
        """
        if isinstance(value, (tuple, list)):
            if not value:
                return False  # a vacuous container satisfies nothing; it carries no value to check
            values: tuple[Any, ...] = tuple(value)
        else:
            values = (value,)
        bounded = self.lower is not None or self.upper is not None
        for item in values:
            if self.allowed_values and item not in self.allowed_values:
                return False
            if not bounded:
                continue
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return False
            if not math.isfinite(item):
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
        _require_schema_version(self.schema_version, "transform")
        object.__setattr__(self, "kind", _coerce_enum(self.kind, TransformKind, "transform kind"))
        if not all(math.isfinite(item) for item in (self.scale, self.offset, self.lower, self.upper)):
            raise ValueError("transform parameters must be finite")
        if self.kind is TransformKind.AFFINE and self.scale == 0:
            raise ValueError("affine transform scale cannot be zero")
        if self.kind is TransformKind.LOGIT and self.lower >= self.upper:
            raise ValueError("logit transform requires lower < upper")

    def forward(self, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("transform input must be finite")
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
        if not math.isfinite(value):
            raise ValueError("transform input must be finite")
        if self.kind is TransformKind.IDENTITY:
            return float(value)
        if self.kind is TransformKind.LOG:
            return math.exp(value)
        if self.kind is TransformKind.LOGIT:
            if value >= 0.0:
                exp_neg = math.exp(-value)
                probability = 1.0 / (1.0 + exp_neg)
            else:
                exp_pos = math.exp(value)
                probability = exp_pos / (1.0 + exp_pos)
            return self.lower + (self.upper - self.lower) * probability
        return (value - self.offset) / self.scale

    def log_abs_det_jacobian(self, value: float) -> float:
        """Log absolute derivative of ``forward`` at a natural-space value."""
        if not math.isfinite(value):
            raise ValueError("transform input must be finite")
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
        return cls(**{**value, "kind": _coerce_enum(value.get("kind", "identity"), TransformKind, "transform kind")})


@dataclass(frozen=True)
class PriorSpec:
    id: str
    family: str
    parameters: Mapping[str, Any]
    unit: str = "1"
    source_ref: str | None = None
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version, "prior")
        _require_id(self.id, "prior id")
        _require_id(self.family, "prior family")
        if not self.unit:
            raise ValueError("prior unit is required; use '1' for dimensionless")
        object.__setattr__(self, "parameters", _canonical_mapping(self.parameters, "prior parameters"))
        object.__setattr__(self, "_identity", semantic_digest(self))

    @property
    def identity(self) -> str:
        """The content digest fixed at construction -- a durable record cannot drift into a new identity."""
        return self._identity

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
        _require_schema_version(self.schema_version, "value")
        object.__setattr__(self, "role", _coerce_enum(self.role, ValueRole, "value role"))
        object.__setattr__(self, "shape", _own_sequence(self.shape, "value shape"))
        object.__setattr__(self, "dependencies", _own_sequence(self.dependencies, "value dependencies"))
        _require_id(self.id, "value id")
        if not self.unit:
            raise ValueError("value unit is required; use '1' for dimensionless")
        if self.dtype not in DTYPES:
            raise ValueError(f"value dtype {self.dtype!r} is not one of {sorted(DTYPES)}")
        # `bool` subclasses `int`, so a Boolean dimension passed the old integer test as size 1
        if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in self.shape):
            raise ValueError("value shape dimensions must be positive non-Boolean integers")
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
        # a prior is a distribution OVER THIS QUANTITY, so it is stated in this quantity's unit: a
        # value declared in metres used to accept a prior declared in kilograms, because units were
        # only checked for truthiness and never reconciled along the value/prior chain.
        if self.prior is not None and self.prior.unit != self.unit:
            raise ValueError(
                f"value {self.id!r} is measured in {self.unit!r} but its prior declares {self.prior.unit!r}; "
                "prior and value must share one unit (canonicalize the strings; this is an identity "
                "comparison, not a dimensional algebra)"
            )
        # dependencies were only checked for non-emptiness, so a derived value could declare itself as
        # its own dependency (expression "d+1", dependencies ("d",)) and still receive a stable
        # identity despite having no well-founded evaluation order at all.
        for dependency in self.dependencies:
            _require_id(dependency, "value dependency")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("value dependencies must be unique")
        if self.id in self.dependencies:
            raise ValueError(f"derived value {self.id!r} depends on itself")
        # the declared shape/dtype is a contract about the value, not a decoration beside it: a scalar
        # used to be certified as a two-vector, and `dtype` accepted any string whatsoever.
        if self.value is not None:
            measured = _measured_shape(self.value, f"value {self.id!r}")
            if measured != tuple(self.shape):
                raise ValueError(f"value {self.id!r} has shape {measured} but declares shape {tuple(self.shape)}")
            _check_dtype(self.value, self.dtype, f"value {self.id!r}")
        if self.constraint and self.value is not None and not self.constraint.accepts(self.value):
            raise ValueError("value violates its declared constraint")
        object.__setattr__(self, "value", _normalize(self.value, semantic=False))
        object.__setattr__(self, "_identity", semantic_digest(self))

    @property
    def identity(self) -> str:
        """The content digest fixed at construction -- a durable record cannot drift into a new identity."""
        return self._identity

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> ValueSpec:
        return cls(
            **{
                **value,
                "role": _coerce_enum(value["role"], ValueRole, "value role"),
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
        _require_schema_version(self.schema_version, "likelihood")
        object.__setattr__(self, "observation_ids", _own_sequence(self.observation_ids, "likelihood observation_ids"))
        _require_id(self.id, "likelihood id")
        _require_id(self.family, "likelihood family")
        if not self.observation_ids or len(self.observation_ids) != len(set(self.observation_ids)):
            raise ValueError("likelihood requires unique observation ids")
        object.__setattr__(self, "parameters", _canonical_mapping(self.parameters, "likelihood parameters"))

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
        _require_schema_version(self.schema_version, "observation")
        _require_id(self.id, "observation id")
        _require_id(self.value_spec_id, "observation value id")
        if not _SHA256.fullmatch(self.content_digest) or not self.data_ref:
            raise ValueError("observation content digest and data reference are required")
        if self.id not in self.likelihood.observation_ids:
            raise ValueError("observation likelihood must reference the observation id")
        if self.measurement_uncertainty is not None and (
            not math.isfinite(self.measurement_uncertainty) or self.measurement_uncertainty < 0
        ):
            raise ValueError("measurement uncertainty must be finite and nonnegative")

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
        _require_schema_version(self.schema_version, "uncertainty")
        object.__setattr__(self, "kind", _coerce_enum(self.kind, UncertaintyKind, "uncertainty kind"))
        _require_id(self.id, "uncertainty id")
        if (self.value is None) == (self.artifact_digest is None):
            raise ValueError("uncertainty requires exactly one scalar value or artifact digest")
        if self.value is not None and (not math.isfinite(self.value) or self.value < 0):
            raise ValueError("uncertainty scalar must be finite and nonnegative")
        if self.artifact_digest is not None and not _SHA256.fullmatch(self.artifact_digest):
            raise ValueError("uncertainty artifact digest must be SHA-256")

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> UncertaintyComponent:
        return cls(**{**value, "kind": _coerce_enum(value["kind"], UncertaintyKind, "uncertainty kind")})


def _require_dependency_dag(values: tuple[ValueSpec, ...], value_ids: set[str]) -> None:
    """Every derived value's dependencies must resolve inside the artifact, with no cycles.

    A per-value self-dependency check is not enough: ``a`` depending on ``b`` and ``b`` on ``a`` is
    equally ill-founded, and a dependency naming a value the artifact does not carry has no evaluation
    order at all. Both used to construct successfully and receive a stable posterior identity.
    """
    edges = {item.id: tuple(item.dependencies) for item in values}
    for value_id, dependencies in edges.items():
        unknown = [d for d in dependencies if d not in value_ids]
        if unknown:
            raise ValueError(f"derived value {value_id!r} depends on values this posterior does not carry: {unknown}")

    state: dict[str, int] = {}  # 0 = visiting, 1 = done

    def visit(node: str, trail: tuple[str, ...]) -> None:
        mark = state.get(node)
        if mark == 1:
            return
        if mark == 0:
            cycle = " -> ".join((*trail[trail.index(node) :], node))
            raise ValueError(f"posterior value dependencies form a cycle: {cycle}")
        state[node] = 0
        for dependency in edges.get(node, ()):
            visit(dependency, (*trail, node))
        state[node] = 1

    for value_id in edges:
        visit(value_id, ())


def _require_one_likelihood(likelihood: LikelihoodSpec, observations: tuple[ObservationSpec, ...]) -> None:
    """One evidence model per posterior: no observation may assert a different one for the same data.

    Each ``ObservationSpec`` embeds a likelihood and ``PosteriorArtifact`` embeds a second top-level
    one; the posterior only checked that the observation-ID sets closed, never that the two models
    agreed. A valid artifact could therefore be constructed whose observation declared a Poisson
    likelihood and whose posterior declared a Normal likelihood for that exact observation. The
    observation ids may legitimately be a subset (one observation's own likelihood need not list its
    siblings), but the model identity -- id, family, parameters, discrepancy reference -- must be the
    same model.
    """
    for observation in observations:
        own = observation.likelihood
        mismatch = [
            f"{name}: {theirs!r} != {ours!r}"
            for name, theirs, ours in (
                ("id", own.id, likelihood.id),
                ("family", own.family, likelihood.family),
                ("parameters", own.parameters, likelihood.parameters),
                ("discrepancy_ref", own.discrepancy_ref, likelihood.discrepancy_ref),
            )
            if theirs != ours
        ]
        if mismatch:
            raise ValueError(
                f"observation {observation.id!r} declares a different likelihood than the posterior "
                f"({'; '.join(mismatch)}); one posterior asserts exactly one evidence model per observation"
            )
        if observation.id not in likelihood.observation_ids:
            raise ValueError(f"posterior likelihood does not cover observation {observation.id!r}")


def _require_observation_units(values: tuple[ValueSpec, ...], observations: tuple[ObservationSpec, ...]) -> None:
    """An observation of a quantity is a measurement OF that quantity, so it shares that quantity's unit.

    Units used to be checked only for truthiness and never reconciled across the
    value/prior/observation chain, so an observation declaring seconds could enter a valid posterior
    for a metre-valued quantity. This is an identity comparison on canonical unit strings, not a
    dimensional algebra: ``"m"`` and ``"metre"`` are different units here, and the caller is
    responsible for canonicalizing and for converting magnitudes. Measurement uncertainty is carried
    in the observation's unit by the same contract.
    """
    units = {item.id: item.unit for item in values}
    for observation in observations:
        declared = observation.unit
        expected = units.get(observation.value_spec_id)
        if declared is not None and expected is not None and declared != expected:
            raise ValueError(
                f"observation {observation.id!r} is stated in {declared!r} but measures value "
                f"{observation.value_spec_id!r}, which is in {expected!r}"
            )


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
        _require_schema_version(self.schema_version, "posterior")
        # Own the evidence before anything below reads it: the checks that follow, and the identity
        # cached at the end, must describe a sequence the caller can no longer edit (MXR-080-1903).
        object.__setattr__(self, "values", _own_sequence(self.values, "posterior values"))
        object.__setattr__(self, "observations", _own_sequence(self.observations, "posterior observations"))
        object.__setattr__(self, "uncertainty", _own_sequence(self.uncertainty, "posterior uncertainty"))
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
        _require_dependency_dag(self.values, value_ids)
        _require_one_likelihood(self.likelihood, self.observations)
        _require_observation_units(self.values, self.observations)
        if self.sample_digest is not None and not _SHA256.fullmatch(self.sample_digest):
            raise ValueError("posterior sample digest must be SHA-256")
        object.__setattr__(self, "summary", _canonical_mapping(self.summary, "posterior summary"))
        object.__setattr__(self, "diagnostics", _canonical_mapping(self.diagnostics, "posterior diagnostics"))
        object.__setattr__(self, "_identity", semantic_digest(self))

    @property
    def identity(self) -> str:
        """The content digest fixed at construction -- a durable record cannot drift into a new identity."""
        return self._identity

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
        _require_schema_version(self.schema_version, "predictive")
        object.__setattr__(self, "target_value_ids", _own_sequence(self.target_value_ids, "predictive targets"))
        object.__setattr__(self, "uncertainty", _own_sequence(self.uncertainty, "predictive uncertainty"))
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
        _require_schema_version(self.schema_version, "calibration")
        _require_id(self.id, "calibration id")
        if not _SHA256.fullmatch(self.target_identity) or not self.method or not self.metrics:
            raise ValueError("calibration target, method, and metrics are required")
        object.__setattr__(self, "metrics", _canonical_mapping(self.metrics, "calibration metrics"))
        object.__setattr__(self, "slice", _canonical_mapping(self.slice, "calibration slice"))


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
        _require_schema_version(self.schema_version, "decision")
        object.__setattr__(self, "alternatives", _own_sequence(self.alternatives, "decision alternatives"))
        object.__setattr__(self, "assumptions", _own_sequence(self.assumptions, "decision assumptions"))
        _require_id(self.id, "decision id")
        if len(self.alternatives) < 2 or len(set(self.alternatives)) != len(self.alternatives):
            raise ValueError("decision alternatives must contain at least two unique values")
        if self.selected not in self.alternatives or set(self.utility) != set(self.alternatives):
            raise ValueError("decision selection and utility must close over the alternatives")
        if not _SHA256.fullmatch(self.posterior_identity) or not self.risk_measure:
            raise ValueError("decision posterior identity and risk measure are required")
        if any(not math.isfinite(value) for value in self.utility.values()):
            raise ValueError("decision utility values must be finite")
        object.__setattr__(self, "utility", _canonical_mapping(self.utility, "decision utility"))


@dataclass(frozen=True)
class CapabilityExtension:
    id: str
    owner_project: str
    input_schema_uri: str
    output_schema_uri: str
    maturity: str
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version, "capability extension")
        _require_id(self.id, "capability extension id")
        _require_id(self.owner_project, "capability extension owner")
        _require_id(self.maturity, "capability extension maturity")
        if not self.input_schema_uri.strip() or not self.output_schema_uri.strip():
            raise ValueError("capability extension input and output schema URIs are required")


@dataclass(frozen=True)
class TraceEvent:
    trace_id: str
    sequence: int
    event_type: str
    semantic_identity: str
    payload: Mapping[str, Any]
    occurred_at: str
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version, "trace event")
        _require_id(self.trace_id, "trace id")
        _require_id(self.event_type, "trace event type")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("trace sequence must be a nonnegative integer")
        if not _SHA256.fullmatch(self.semantic_identity):
            raise ValueError("trace semantic identity must be SHA-256")
        object.__setattr__(self, "payload", _canonical_mapping(self.payload, "trace payload"))
        try:
            timestamp = datetime.fromisoformat(self.occurred_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("trace occurred_at must be an RFC 3339 timestamp") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("trace occurred_at must include a timezone")


@runtime_checkable
class TraceSink(Protocol):
    def emit(self, event: TraceEvent) -> None: ...


def load_reference_fixture() -> dict[str, Any]:
    """Load the packaged cross-project value/prior/observation fixture."""
    return json.loads(files("mixle").joinpath("fixtures/quantitative-semantics-v1.json").read_text())
