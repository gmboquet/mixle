"""Canonical, dependency-free causal semantics for cross-project science contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "mixle.causal/v1"


class CausalContractError(ValueError):
    """Raised when a causal record would overstate what is identified."""


class CausalEvidenceKind(StrEnum):
    PREDICTION = "prediction"
    ASSOCIATION = "association"
    MECHANISM = "mechanism"
    INTERVENTION = "intervention"


class IdentificationStatus(StrEnum):
    IDENTIFIED = "identified"
    PARTIALLY_IDENTIFIED = "partially_identified"
    NOT_IDENTIFIED = "not_identified"


class AssumptionStatus(StrEnum):
    ASSERTED = "asserted"
    CHALLENGED = "challenged"
    FAILED = "failed"


def _required(value: str, label: str) -> None:
    if not value or not value.strip():
        raise CausalContractError(f"{label} must be non-empty")


def _json_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def semantic_id(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class Estimand:
    id: str
    treatment: str
    outcome: str
    target_population: str
    contrast: str
    time_horizon: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value, label in (
            (self.id, "estimand id"),
            (self.treatment, "treatment"),
            (self.outcome, "outcome"),
            (self.target_population, "target population"),
            (self.contrast, "contrast"),
        ):
            _required(value, label)
        if self.schema_version != SCHEMA_VERSION:
            raise CausalContractError("unsupported causal schema version")

    @property
    def identity(self) -> str:
        return semantic_id(self)

    def as_dict(self) -> dict[str, Any]:
        return _json_value(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Estimand:
        return cls(**value)


@dataclass(frozen=True)
class CausalAssumption:
    id: str
    kind: str
    statement: str
    status: AssumptionStatus = AssumptionStatus.ASSERTED
    testable: bool = False
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required(self.id, "assumption id")
        _required(self.kind, "assumption kind")
        _required(self.statement, "assumption statement")
        object.__setattr__(self, "status", AssumptionStatus(self.status))
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise CausalContractError("assumption evidence references must be unique")

    def as_dict(self) -> dict[str, Any]:
        return _json_value(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CausalAssumption:
        return cls(**{**value, "evidence_refs": tuple(value.get("evidence_refs", ()))})


@dataclass(frozen=True)
class InterventionSpec:
    id: str
    treatment: str
    value: float
    minimum: float
    maximum: float
    authority_ref: str | None
    safety_constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required(self.id, "intervention id")
        _required(self.treatment, "intervention treatment")
        if not all(math.isfinite(item) for item in (self.value, self.minimum, self.maximum)):
            raise CausalContractError("intervention range must be finite")
        if self.minimum > self.maximum or not self.minimum <= self.value <= self.maximum:
            raise CausalContractError("intervention value is outside its declared safe range")

    @property
    def authorized(self) -> bool:
        return bool(self.authority_ref and self.authority_ref.strip())

    def as_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True)
class CounterfactualQuery:
    id: str
    estimand_ref: str
    intervention: InterventionSpec
    evidence_kind: CausalEvidenceKind = CausalEvidenceKind.INTERVENTION

    def __post_init__(self) -> None:
        _required(self.id, "counterfactual query id")
        _required(self.estimand_ref, "estimand reference")
        object.__setattr__(self, "evidence_kind", CausalEvidenceKind(self.evidence_kind))
        if self.evidence_kind is not CausalEvidenceKind.INTERVENTION:
            raise CausalContractError("counterfactual queries require intervention semantics")

    def as_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True)
class IdentificationResult:
    estimand_ref: str
    status: IdentificationStatus
    assumptions: tuple[CausalAssumption, ...]
    evidence_kind: CausalEvidenceKind
    identifying_expression: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    diagnostics: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required(self.estimand_ref, "estimand reference")
        object.__setattr__(self, "status", IdentificationStatus(self.status))
        object.__setattr__(self, "evidence_kind", CausalEvidenceKind(self.evidence_kind))
        if self.schema_version != SCHEMA_VERSION:
            raise CausalContractError("unsupported causal schema version")
        if len({item.id for item in self.assumptions}) != len(self.assumptions):
            raise CausalContractError("identification assumptions must have unique ids")
        failed = [item.id for item in self.assumptions if item.status is AssumptionStatus.FAILED]
        if self.status is IdentificationStatus.IDENTIFIED:
            if not self.identifying_expression:
                raise CausalContractError("identified results require an identifying expression")
            if failed:
                raise CausalContractError("identified results cannot rely on failed assumptions")
            if self.lower_bound is not None or self.upper_bound is not None:
                raise CausalContractError("identified results cannot also claim partial bounds")
        elif self.status is IdentificationStatus.PARTIALLY_IDENTIFIED:
            bounds = (self.lower_bound, self.upper_bound)
            if any(item is None or not math.isfinite(item) for item in bounds):
                raise CausalContractError("partial identification requires finite bounds")
            if self.lower_bound > self.upper_bound:
                raise CausalContractError("partial identification bounds are reversed")
            if self.identifying_expression:
                raise CausalContractError("partial identification cannot claim a point expression")
        else:
            if self.identifying_expression or self.lower_bound is not None or self.upper_bound is not None:
                raise CausalContractError("not-identified results cannot claim an expression or bounds")
            if not self.diagnostics:
                raise CausalContractError("not-identified results require a diagnostic")

    @property
    def identity(self) -> str:
        return semantic_id(self)

    def as_dict(self) -> dict[str, Any]:
        return _json_value(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> IdentificationResult:
        assumptions = tuple(CausalAssumption.from_dict(item) for item in value.get("assumptions", ()))
        return cls(**{**value, "assumptions": assumptions, "diagnostics": tuple(value.get("diagnostics", ()))})


__all__ = [
    "SCHEMA_VERSION",
    "AssumptionStatus",
    "CausalAssumption",
    "CausalContractError",
    "CausalEvidenceKind",
    "CounterfactualQuery",
    "Estimand",
    "IdentificationResult",
    "IdentificationStatus",
    "InterventionSpec",
    "canonical_json",
    "semantic_id",
]
