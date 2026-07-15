"""Governed lifecycle records for capabilities shared across Mixle projects.

Capability lifecycle state is deliberately factored into independent dimensions.
An evaluation result is not a maturity promotion, an available service is not an
authorized service, and a corroborated claim is not proof that an implementation
is operational.  :class:`CapabilityLifecycle` keeps those facts together without
collapsing them into one ambiguous status.

All records are frozen, JSON-compatible value objects.  A change creates a new
snapshot with a monotonically increasing revision, which makes transitions safe
to persist, audit, replay, and exchange between projects.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Mapping

__all__ = [
    "AuthorizationDecision",
    "AuthorizationOutcome",
    "AuthorizationStatus",
    "CapabilityIdentity",
    "CapabilityLifecycle",
    "CapabilityMaturity",
    "EpistemicStanding",
    "EvaluationState",
    "LifecycleTransitionError",
    "OperationalState",
]


class LifecycleTransitionError(ValueError):
    """Raised when a lifecycle transition would erase or contradict evidence."""


class CapabilityMaturity(StrEnum):
    """Product maturity; this is distinct from API stability in :mod:`mixle.maturity`."""

    CONCEPT = "concept"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    SUPPORTED = "supported"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class OperationalState(StrEnum):
    """Whether an implementation can currently accept work."""

    UNAVAILABLE = "unavailable"
    STARTING = "starting"
    AVAILABLE = "available"
    DEGRADED = "degraded"
    SUSPENDED = "suspended"
    RETIRED = "retired"


class EvaluationState(StrEnum):
    """State of the current evaluation, not a maturity or deployment verdict."""

    UNEVALUATED = "unevaluated"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    STALE = "stale"


class EpistemicStanding(StrEnum):
    """Standing of the claims supporting a capability."""

    UNASSESSED = "unassessed"
    HYPOTHESIS = "hypothesis"
    CORROBORATED = "corroborated"
    CONTESTED = "contested"
    REFUTED = "refuted"


class AuthorizationOutcome(StrEnum):
    """The outcome recorded by an authorization authority."""

    GRANTED = "granted"
    DENIED = "denied"


class AuthorizationStatus(StrEnum):
    """Effective status of a decision at a point in time."""

    NOT_REQUESTED = "not_requested"
    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"
    REVOKED = "revoked"


class _KeepAuthorization:
    pass


_KEEP_AUTHORIZATION = _KeepAuthorization()


def _require_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return _require_aware(value, "timestamp").isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    return _require_aware(parsed, field_name)


@dataclass(frozen=True, slots=True)
class CapabilityIdentity:
    """Immutable identity of one capability version.

    ``digest`` is optional for a purely semantic capability and should be set to
    the integrity digest of a concrete artifact when one exists.
    """

    capability_id: str
    version: str
    digest: str | None = None

    def __post_init__(self) -> None:
        if not self.capability_id.strip():
            raise ValueError("capability_id must not be empty")
        if not self.version.strip():
            raise ValueError("version must not be empty")
        if self.digest is not None and not self.digest.strip():
            raise ValueError("digest must be non-empty when supplied")

    def as_dict(self) -> dict[str, str | None]:
        return {"capability_id": self.capability_id, "version": self.version, "digest": self.digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityIdentity:
        return cls(
            capability_id=str(value["capability_id"]),
            version=str(value["version"]),
            digest=None if value.get("digest") is None else str(value["digest"]),
        )


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    """Scoped authorization issued by a named principal for one immutable version."""

    decision_id: str
    capability: CapabilityIdentity
    outcome: AuthorizationOutcome
    issued_by: str
    scopes: frozenset[str]
    decided_at: datetime
    expires_at: datetime | None = None
    reason: str = ""
    revoked_at: datetime | None = None
    revoked_by: str | None = None
    revocation_reason: str = ""

    def __post_init__(self) -> None:
        if not self.decision_id.strip():
            raise ValueError("decision_id must not be empty")
        if not self.issued_by.strip():
            raise ValueError("issued_by must not be empty")
        normalized_scopes = frozenset(scope.strip() for scope in self.scopes if scope.strip())
        if not normalized_scopes:
            raise ValueError("authorization requires at least one non-empty scope")
        object.__setattr__(self, "scopes", normalized_scopes)
        decided_at = _require_aware(self.decided_at, "decided_at")
        object.__setattr__(self, "decided_at", decided_at)
        if self.expires_at is not None:
            expires_at = _require_aware(self.expires_at, "expires_at")
            if expires_at <= decided_at:
                raise ValueError("expires_at must be later than decided_at")
            object.__setattr__(self, "expires_at", expires_at)
        if self.revoked_at is not None:
            revoked_at = _require_aware(self.revoked_at, "revoked_at")
            if self.outcome is not AuthorizationOutcome.GRANTED:
                raise ValueError("only a granted authorization can be revoked")
            if revoked_at < decided_at:
                raise ValueError("revoked_at must not precede decided_at")
            if self.expires_at is not None and revoked_at >= self.expires_at:
                raise ValueError("an expired authorization cannot subsequently be revoked")
            if not self.revoked_by or not self.revoked_by.strip():
                raise ValueError("revoked_by is required when revoked_at is set")
            object.__setattr__(self, "revoked_at", revoked_at)
        elif self.revoked_by is not None or self.revocation_reason:
            raise ValueError("revocation metadata requires revoked_at")

    def status_at(self, at: datetime) -> AuthorizationStatus:
        """Return the decision's effective status at ``at``."""
        at = _require_aware(at, "at")
        if at < self.decided_at:
            return AuthorizationStatus.NOT_REQUESTED
        if self.outcome is AuthorizationOutcome.DENIED:
            return AuthorizationStatus.DENIED
        if self.revoked_at is not None and at >= self.revoked_at:
            return AuthorizationStatus.REVOKED
        if self.expires_at is not None and at >= self.expires_at:
            return AuthorizationStatus.EXPIRED
        return AuthorizationStatus.GRANTED

    def allows(self, scope: str, *, at: datetime) -> bool:
        """Return whether this decision authorizes ``scope`` at ``at``."""
        return self.status_at(at) is AuthorizationStatus.GRANTED and (scope in self.scopes or "*" in self.scopes)

    def revoke(self, *, by: str, at: datetime, reason: str = "") -> AuthorizationDecision:
        """Return a revoked copy; the original audit record remains unchanged."""
        if self.outcome is not AuthorizationOutcome.GRANTED:
            raise LifecycleTransitionError("a denied authorization cannot be revoked")
        if self.revoked_at is not None:
            raise LifecycleTransitionError("authorization is already revoked")
        return replace(self, revoked_at=at, revoked_by=by, revocation_reason=reason)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "capability": self.capability.as_dict(),
            "outcome": self.outcome.value,
            "issued_by": self.issued_by,
            "scopes": sorted(self.scopes),
            "decided_at": _timestamp(self.decided_at),
            "expires_at": None if self.expires_at is None else _timestamp(self.expires_at),
            "reason": self.reason,
            "revoked_at": None if self.revoked_at is None else _timestamp(self.revoked_at),
            "revoked_by": self.revoked_by,
            "revocation_reason": self.revocation_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AuthorizationDecision:
        return cls(
            decision_id=str(value["decision_id"]),
            capability=CapabilityIdentity.from_dict(value["capability"]),
            outcome=AuthorizationOutcome(value["outcome"]),
            issued_by=str(value["issued_by"]),
            scopes=frozenset(str(scope) for scope in value["scopes"]),
            decided_at=_parse_timestamp(value["decided_at"], "decided_at"),
            expires_at=(
                None if value.get("expires_at") is None else _parse_timestamp(value["expires_at"], "expires_at")
            ),
            reason=str(value.get("reason", "")),
            revoked_at=(
                None if value.get("revoked_at") is None else _parse_timestamp(value["revoked_at"], "revoked_at")
            ),
            revoked_by=None if value.get("revoked_by") is None else str(value["revoked_by"]),
            revocation_reason=str(value.get("revocation_reason", "")),
        )


_MATURITY_TRANSITIONS = {
    CapabilityMaturity.CONCEPT: {CapabilityMaturity.CANDIDATE, CapabilityMaturity.RETIRED},
    CapabilityMaturity.CANDIDATE: {
        CapabilityMaturity.VALIDATED,
        CapabilityMaturity.DEPRECATED,
        CapabilityMaturity.RETIRED,
    },
    CapabilityMaturity.VALIDATED: {
        CapabilityMaturity.SUPPORTED,
        CapabilityMaturity.DEPRECATED,
        CapabilityMaturity.RETIRED,
    },
    CapabilityMaturity.SUPPORTED: {CapabilityMaturity.DEPRECATED, CapabilityMaturity.RETIRED},
    CapabilityMaturity.DEPRECATED: {CapabilityMaturity.RETIRED},
    CapabilityMaturity.RETIRED: set(),
}

_OPERATIONAL_TRANSITIONS = {
    OperationalState.UNAVAILABLE: {OperationalState.STARTING, OperationalState.AVAILABLE, OperationalState.RETIRED},
    OperationalState.STARTING: {
        OperationalState.AVAILABLE,
        OperationalState.DEGRADED,
        OperationalState.SUSPENDED,
        OperationalState.UNAVAILABLE,
        OperationalState.RETIRED,
    },
    OperationalState.AVAILABLE: {
        OperationalState.DEGRADED,
        OperationalState.SUSPENDED,
        OperationalState.UNAVAILABLE,
        OperationalState.RETIRED,
    },
    OperationalState.DEGRADED: {
        OperationalState.AVAILABLE,
        OperationalState.SUSPENDED,
        OperationalState.UNAVAILABLE,
        OperationalState.RETIRED,
    },
    OperationalState.SUSPENDED: {
        OperationalState.STARTING,
        OperationalState.AVAILABLE,
        OperationalState.UNAVAILABLE,
        OperationalState.RETIRED,
    },
    OperationalState.RETIRED: set(),
}

_EVALUATION_TRANSITIONS = {
    EvaluationState.UNEVALUATED: {EvaluationState.RUNNING},
    EvaluationState.RUNNING: {EvaluationState.PASSED, EvaluationState.FAILED, EvaluationState.INCONCLUSIVE},
    EvaluationState.PASSED: {EvaluationState.RUNNING, EvaluationState.STALE},
    EvaluationState.FAILED: {EvaluationState.RUNNING},
    EvaluationState.INCONCLUSIVE: {EvaluationState.RUNNING},
    EvaluationState.STALE: {EvaluationState.RUNNING},
}

_EPISTEMIC_TRANSITIONS = {
    EpistemicStanding.UNASSESSED: {EpistemicStanding.HYPOTHESIS},
    EpistemicStanding.HYPOTHESIS: {
        EpistemicStanding.CORROBORATED,
        EpistemicStanding.CONTESTED,
        EpistemicStanding.REFUTED,
    },
    EpistemicStanding.CORROBORATED: {EpistemicStanding.CONTESTED, EpistemicStanding.REFUTED},
    EpistemicStanding.CONTESTED: {
        EpistemicStanding.HYPOTHESIS,
        EpistemicStanding.CORROBORATED,
        EpistemicStanding.REFUTED,
    },
    EpistemicStanding.REFUTED: {EpistemicStanding.HYPOTHESIS, EpistemicStanding.CONTESTED},
}


def _check_transition(current: StrEnum, target: StrEnum, allowed: Mapping[StrEnum, set[StrEnum]]) -> None:
    if target != current and target not in allowed[current]:
        message = f"illegal {type(current).__name__} transition: {current.value} -> {target.value}"
        raise LifecycleTransitionError(message)


@dataclass(frozen=True, slots=True)
class CapabilityLifecycle:
    """An immutable, replayable snapshot of every independent lifecycle dimension."""

    capability: CapabilityIdentity
    maturity: CapabilityMaturity = CapabilityMaturity.CONCEPT
    operational: OperationalState = OperationalState.UNAVAILABLE
    evaluation: EvaluationState = EvaluationState.UNEVALUATED
    epistemic: EpistemicStanding = EpistemicStanding.UNASSESSED
    authorization: AuthorizationDecision | None = None
    revision: int = 0
    updated_at: datetime = datetime(1970, 1, 1, tzinfo=UTC)

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("revision must be non-negative")
        object.__setattr__(self, "updated_at", _require_aware(self.updated_at, "updated_at"))
        if self.authorization is not None and self.authorization.capability != self.capability:
            raise ValueError("authorization capability does not match lifecycle capability")
        if self.authorization is not None and self.authorization.decided_at > self.updated_at:
            raise ValueError("authorization decision cannot postdate its lifecycle snapshot")
        retired_maturity = self.maturity is CapabilityMaturity.RETIRED
        retired_operation = self.operational is OperationalState.RETIRED
        if retired_maturity != retired_operation:
            raise ValueError("maturity and operational state must retire together")
        if retired_maturity and self.authorization is not None:
            if self.authorization.status_at(self.updated_at) is AuthorizationStatus.GRANTED:
                raise ValueError("a retired capability cannot retain effective authorization")

    @property
    def authorization_status(self) -> AuthorizationStatus:
        if self.authorization is None:
            return AuthorizationStatus.NOT_REQUESTED
        return self.authorization.status_at(self.updated_at)

    def allows(self, scope: str, *, at: datetime | None = None) -> bool:
        """Return whether the current decision authorizes ``scope``; maturity is intentionally separate."""
        if self.authorization is None:
            return False
        return self.authorization.allows(scope, at=self.updated_at if at is None else at)

    def evolve(
        self,
        *,
        maturity: CapabilityMaturity | None = None,
        operational: OperationalState | None = None,
        evaluation: EvaluationState | None = None,
        epistemic: EpistemicStanding | None = None,
        authorization: AuthorizationDecision | None | _KeepAuthorization = _KEEP_AUTHORIZATION,
        at: datetime,
    ) -> CapabilityLifecycle:
        """Create the next snapshot after validating every requested transition.

        Passing ``authorization=None`` clears a decision; omitting it keeps the
        current decision.  Maturity promotion never happens automatically: a
        transition into ``validated`` or ``supported`` explicitly requires the
        evaluation dimension to be ``passed`` in the resulting snapshot.
        """
        at = _require_aware(at, "at")
        if at < self.updated_at:
            raise LifecycleTransitionError("transition timestamp precedes the current snapshot")
        next_maturity = self.maturity if maturity is None else CapabilityMaturity(maturity)
        next_operational = self.operational if operational is None else OperationalState(operational)
        next_evaluation = self.evaluation if evaluation is None else EvaluationState(evaluation)
        next_epistemic = self.epistemic if epistemic is None else EpistemicStanding(epistemic)
        next_authorization = self.authorization if authorization is _KEEP_AUTHORIZATION else authorization

        _check_transition(self.maturity, next_maturity, _MATURITY_TRANSITIONS)
        _check_transition(self.operational, next_operational, _OPERATIONAL_TRANSITIONS)
        _check_transition(self.evaluation, next_evaluation, _EVALUATION_TRANSITIONS)
        _check_transition(self.epistemic, next_epistemic, _EPISTEMIC_TRANSITIONS)
        if next_maturity != self.maturity and next_maturity in {
            CapabilityMaturity.VALIDATED,
            CapabilityMaturity.SUPPORTED,
        }:
            if next_evaluation is not EvaluationState.PASSED:
                raise LifecycleTransitionError(f"promotion to {next_maturity.value} requires a passed evaluation")
        if self.authorization is not None and next_authorization != self.authorization:
            if self.authorization.status_at(at) is AuthorizationStatus.GRANTED:
                raise LifecycleTransitionError("revoke an effective authorization before replacing or clearing it")

        return CapabilityLifecycle(
            capability=self.capability,
            maturity=next_maturity,
            operational=next_operational,
            evaluation=next_evaluation,
            epistemic=next_epistemic,
            authorization=next_authorization,
            revision=self.revision + 1,
            updated_at=at,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability.as_dict(),
            "maturity": self.maturity.value,
            "operational": self.operational.value,
            "evaluation": self.evaluation.value,
            "epistemic": self.epistemic.value,
            "authorization": None if self.authorization is None else self.authorization.as_dict(),
            "authorization_status": self.authorization_status.value,
            "revision": self.revision,
            "updated_at": _timestamp(self.updated_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityLifecycle:
        authorization = value.get("authorization")
        return cls(
            capability=CapabilityIdentity.from_dict(value["capability"]),
            maturity=CapabilityMaturity(value["maturity"]),
            operational=OperationalState(value["operational"]),
            evaluation=EvaluationState(value["evaluation"]),
            epistemic=EpistemicStanding(value["epistemic"]),
            authorization=None if authorization is None else AuthorizationDecision.from_dict(authorization),
            revision=int(value["revision"]),
            updated_at=_parse_timestamp(value["updated_at"], "updated_at"),
        )
