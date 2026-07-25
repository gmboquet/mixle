"""Verified graduation records for mechanisms living in :mod:`mixle.experimental`.

A mechanism can graduate only when a digest-verified, matched-compute baseline comparison beats
the baseline and a digest-verified misfit measurement stays within its declared threshold. Empty
mappings and unverified observations are not evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _validated_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _validated_observed_at(value: Any) -> str:
    value = _validated_text(value, "observed_at")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("observed_at must be an ISO-8601 timestamp.") from exc
    if timestamp.tzinfo is None:
        raise ValueError("observed_at must include a timezone.")
    return value


def _validated_number(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _receipt_digest(kind: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"kind": kind, **payload},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validated_digest(value: Any) -> str:
    value = _validated_text(value, "digest")
    if not value.startswith("sha256:") or len(value) != len("sha256:") + 64:
        raise ValueError("digest must be a sha256 integrity digest.")
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError("digest must be a sha256 integrity digest.") from exc
    return value


@dataclass(frozen=True)
class BaselineComparisonReceipt:
    """Matched-compute comparison against the declared graduation baseline."""

    mechanism_name: str
    metric: str
    mechanism_value: float
    baseline_value: float
    matched_flops: float
    lower_is_better: bool
    observed_at: str
    producer: str
    artifact_ref: str
    digest: str
    schema_version: str = "mixle.graduation.baseline/v1"

    @classmethod
    def create(
        cls,
        *,
        mechanism_name: str,
        metric: str,
        mechanism_value: float,
        baseline_value: float,
        matched_flops: float,
        lower_is_better: bool,
        observed_at: str,
        producer: str,
        artifact_ref: str,
    ) -> BaselineComparisonReceipt:
        payload = {
            "mechanism_name": mechanism_name,
            "metric": metric,
            "mechanism_value": mechanism_value,
            "baseline_value": baseline_value,
            "matched_flops": matched_flops,
            "lower_is_better": lower_is_better,
            "observed_at": observed_at,
            "producer": producer,
            "artifact_ref": artifact_ref,
            "schema_version": "mixle.graduation.baseline/v1",
        }
        return cls(**payload, digest=_receipt_digest("baseline", payload))

    def __post_init__(self) -> None:
        _validated_text(self.mechanism_name, "mechanism_name")
        _validated_text(self.metric, "metric")
        _validated_number(self.mechanism_value, "mechanism_value")
        _validated_number(self.baseline_value, "baseline_value")
        _validated_number(self.matched_flops, "matched_flops", positive=True)
        if not isinstance(self.lower_is_better, bool):
            raise ValueError("lower_is_better must be boolean.")
        _validated_observed_at(self.observed_at)
        _validated_text(self.producer, "producer")
        _validated_text(self.artifact_ref, "artifact_ref")
        if self.schema_version != "mixle.graduation.baseline/v1":
            raise ValueError("unsupported baseline receipt schema_version.")
        _validated_digest(self.digest)
        if not self.verify():
            raise ValueError("baseline receipt digest does not match its contents.")

    def _payload(self) -> dict[str, Any]:
        return {
            "mechanism_name": self.mechanism_name,
            "metric": self.metric,
            "mechanism_value": self.mechanism_value,
            "baseline_value": self.baseline_value,
            "matched_flops": self.matched_flops,
            "lower_is_better": self.lower_is_better,
            "observed_at": self.observed_at,
            "producer": self.producer,
            "artifact_ref": self.artifact_ref,
            "schema_version": self.schema_version,
        }

    def verify(self) -> bool:
        """Independently verify receipt integrity."""
        return self.digest == _receipt_digest("baseline", self._payload())

    @property
    def beats_baseline(self) -> bool:
        return (
            self.mechanism_value < self.baseline_value
            if self.lower_is_better
            else self.mechanism_value > self.baseline_value
        )


@dataclass(frozen=True)
class MisfitReceipt:
    """Measured mechanism error and its graduation threshold."""

    mechanism_name: str
    metric: str
    value: float
    threshold: float
    observed_at: str
    producer: str
    artifact_ref: str
    digest: str
    schema_version: str = "mixle.graduation.misfit/v1"

    @classmethod
    def create(
        cls,
        *,
        mechanism_name: str,
        metric: str,
        value: float,
        threshold: float,
        observed_at: str,
        producer: str,
        artifact_ref: str,
    ) -> MisfitReceipt:
        payload = {
            "mechanism_name": mechanism_name,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "observed_at": observed_at,
            "producer": producer,
            "artifact_ref": artifact_ref,
            "schema_version": "mixle.graduation.misfit/v1",
        }
        return cls(**payload, digest=_receipt_digest("misfit", payload))

    def __post_init__(self) -> None:
        _validated_text(self.mechanism_name, "mechanism_name")
        _validated_text(self.metric, "metric")
        _validated_number(self.value, "value", nonnegative=True)
        _validated_number(self.threshold, "threshold", nonnegative=True)
        _validated_observed_at(self.observed_at)
        _validated_text(self.producer, "producer")
        _validated_text(self.artifact_ref, "artifact_ref")
        if self.schema_version != "mixle.graduation.misfit/v1":
            raise ValueError("unsupported misfit receipt schema_version.")
        _validated_digest(self.digest)
        if not self.verify():
            raise ValueError("misfit receipt digest does not match its contents.")

    def _payload(self) -> dict[str, Any]:
        return {
            "mechanism_name": self.mechanism_name,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "observed_at": self.observed_at,
            "producer": self.producer,
            "artifact_ref": self.artifact_ref,
            "schema_version": self.schema_version,
        }

    def verify(self) -> bool:
        """Independently verify receipt integrity."""
        return self.digest == _receipt_digest("misfit", self._payload())

    @property
    def passes(self) -> bool:
        return self.value <= self.threshold


@dataclass
class ExperimentalMechanism:
    """One entry in the experimental-mechanism graduation ledger."""

    name: str
    graduated: bool = False
    baseline_receipt: BaselineComparisonReceipt | None = None
    misfit_receipt: MisfitReceipt | None = None

    def __post_init__(self) -> None:
        self.name = _validated_text(self.name, "name")
        if not isinstance(self.graduated, bool):
            raise ValueError("graduated must be boolean.")
        if self.baseline_receipt is not None and not isinstance(self.baseline_receipt, BaselineComparisonReceipt):
            raise TypeError("baseline_receipt must be a BaselineComparisonReceipt.")
        if self.misfit_receipt is not None and not isinstance(self.misfit_receipt, MisfitReceipt):
            raise TypeError("misfit_receipt must be a MisfitReceipt.")
        if self.graduated and not self.is_eligible():
            raise ValueError("a mechanism cannot be marked graduated without passing verified receipts.")

    def is_eligible(self) -> bool:
        """Whether verified receipts prove both graduation gates."""
        baseline = self.baseline_receipt
        misfit = self.misfit_receipt
        return bool(
            isinstance(baseline, BaselineComparisonReceipt)
            and baseline.mechanism_name == self.name
            and baseline.verify()
            and baseline.beats_baseline
            and isinstance(misfit, MisfitReceipt)
            and misfit.mechanism_name == self.name
            and misfit.verify()
            and misfit.passes
        )


class DuplicateMechanismError(ValueError):
    """A different mechanism attempted to replace an existing registry identity."""


@dataclass
class _GraduationRegistry:
    """In-memory ledger of :class:`ExperimentalMechanism` entries, keyed by name."""

    _mechanisms: dict[str, ExperimentalMechanism] = field(default_factory=dict)

    def register(self, mechanism: ExperimentalMechanism) -> ExperimentalMechanism:
        """Register a new identity, allowing only idempotent re-registration of the same object."""
        if not isinstance(mechanism, ExperimentalMechanism):
            raise TypeError("mechanism must be an ExperimentalMechanism.")
        existing = self._mechanisms.get(mechanism.name)
        if existing is mechanism:
            return mechanism
        if existing is not None:
            raise DuplicateMechanismError(f"mechanism {mechanism.name!r} is already registered.")
        self._mechanisms[mechanism.name] = mechanism
        return mechanism

    def get(self, name: str) -> ExperimentalMechanism:
        """Look up a registered mechanism by name."""
        return self._mechanisms[_validated_text(name, "name")]

    def __iter__(self):
        return iter(self._mechanisms.values())

    def __len__(self) -> int:
        return len(self._mechanisms)


REGISTRY = _GraduationRegistry()
"""The process-wide graduation ledger. Track-E items register their ``ExperimentalMechanism`` here."""
