"""Versioned transactional proposal commit with canaries and verified rollback."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from types import MappingProxyType
from typing import Any

from mixle.experimental.typed_runtime._exact_controls import require_exact_bool, require_exact_int
from mixle.experimental.typed_runtime.contracts import (
    ArtifactKind,
    ObjectiveKind,
    StateSemantics,
    UpdateKind,
)
from mixle.experimental.typed_runtime.graph import UpdateGraph
from mixle.experimental.typed_runtime.proposal import (
    ProposalBatch,
    ProposalPacket,
    payload_fingerprint,
    proposal_conflicts,
)


@dataclass
class RuntimeVersions:
    """Coordinator-owned global and per-node versions."""

    model_version: int
    node_versions: dict[str, int]

    @classmethod
    def for_graph(cls, graph: UpdateGraph, *, model_version: int = 0) -> RuntimeVersions:
        """Create a zero-node-version vector for a compiled graph."""

        return cls(model_version, {node.node_id: 0 for node in graph.nodes})

    def __post_init__(self) -> None:
        """Make the version vector exact and coordinator-owned (MXR-080-1905).

        Two reproduced holes, both about a value that is a version everywhere except where it has to
        be recorded:

        * ``RuntimeVersions(0.0, {"node": 0})`` constructed, and the coordinator built on it
          committed once -- advancing ``model_version`` 0.0 -> 1.0 and the node version 0 -> 1 --
          and then raised out of ``CommitReceipt``, whose accepted-commit check is
          ``isinstance(before, int)``. The transaction's state moved and no receipt recorded it.
          ``np.int64`` constructed the same way and failed the same way, one layer earlier, in the
          receipt's serializability check.
        * ``node_versions`` was the caller's own dict, stored by reference. A caller that kept a
          handle to it could set ``owned["node"] = 999`` and rewrite the coordinator's version
          vector from outside, which is the exact state every preflight version check is there to
          compare against. It is copied now; ``as_dict`` already returned a detached view, so
          nothing downstream changes.

        NumPy integers are canonicalized rather than refused -- see
        :func:`~mixle.experimental.typed_runtime._exact_controls.require_exact_int`.
        """
        self.model_version = require_exact_int(self.model_version, "model_version", minimum=0)
        if not isinstance(self.node_versions, Mapping):
            raise TypeError(f"node_versions must be a mapping, got {type(self.node_versions).__name__}.")
        versions: dict[str, int] = {}
        for node_id, version in self.node_versions.items():
            if not isinstance(node_id, str) or not node_id.strip():
                raise ValueError(f"runtime node version names no node: {node_id!r}.")
            versions[node_id] = require_exact_int(version, f"node version for {node_id!r}", minimum=0)
        self.node_versions = versions

    def as_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible version vector."""

        return {"model_version": self.model_version, "node_versions": dict(self.node_versions)}


SnapshotFn = Callable[[], Any]
RestoreFn = Callable[[Any], None]
FingerprintFn = Callable[[], str]


@dataclass(frozen=True)
class TransactionParticipant:
    """Snapshot/restore adapter for one mutable state domain."""

    name: str
    semantics: frozenset[StateSemantics]
    snapshot_fn: SnapshotFn = field(repr=False, compare=False)
    restore_fn: RestoreFn = field(repr=False, compare=False)
    fingerprint_fn: FingerprintFn = field(repr=False, compare=False)
    artifacts: frozenset[ArtifactKind] = frozenset()

    def __post_init__(self) -> None:
        if not self.name or not self.semantics:
            raise ValueError("transaction participants need a name and mutable state semantics.")
        if StateSemantics.IMMUTABLE_RESULT in self.semantics:
            raise ValueError("immutable_result is not a transaction participant state domain.")
        if not self.artifacts:
            raise ValueError("transaction participants must declare snapshotted artifacts.")
        if any(not isinstance(artifact, ArtifactKind) for artifact in self.artifacts):
            raise TypeError("transaction participant artifacts must be ArtifactKind values.")

    def snapshot(self) -> Any:
        """Capture state before applying a proposal."""

        return self.snapshot_fn()

    def restore(self, snapshot: Any) -> None:
        """Restore a previously captured snapshot."""

        self.restore_fn(snapshot)

    def fingerprint(self) -> str:
        """Return a deterministic current-state fingerprint."""

        value = self.fingerprint_fn()
        if not isinstance(value, str) or not value:
            raise ValueError("participant fingerprints must be non-empty strings.")
        return value


@dataclass(frozen=True)
class ObjectiveGateEvidence:
    """Proposal-specific objective evidence used for mixed or multi-update gates."""

    objective_before: float
    objective_after: float
    lower_confidence_gain: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.objective_before,
            self.objective_after,
            self.lower_confidence_gain,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("proposal objective evidence must be finite.")

    def as_dict(self) -> dict[str, float | None]:
        """Return a JSON-compatible objective gate."""

        return {
            "objective_before": self.objective_before,
            "objective_after": self.objective_after,
            "lower_confidence_gain": self.lower_confidence_gain,
        }


@dataclass(frozen=True)
class CanaryVerdict:
    """Measured acceptance evidence after proposals have been tentatively applied."""

    accepted: bool
    reason: str
    objective_before: float | None = None
    objective_after: float | None = None
    lower_confidence_gain: float | None = None
    confidence_level: float | None = None
    sample_count: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    proposal_objectives: dict[str, ObjectiveGateEvidence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("canary verdict reason must be non-empty.")
        values = (self.objective_before, self.objective_after, self.lower_confidence_gain, self.confidence_level)
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("canary values must be finite when supplied.")
        if self.confidence_level is not None and not 0.0 <= self.confidence_level <= 1.0:
            raise ValueError("confidence_level must be in [0, 1].")
        if self.sample_count < 0:
            raise ValueError("canary sample_count must be non-negative.")
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("canary metrics must be finite.")
        if any(not proposal_id for proposal_id in self.proposal_objectives):
            raise ValueError("proposal objective evidence ids must be non-empty.")
        if any(not isinstance(row, ObjectiveGateEvidence) for row in self.proposal_objectives.values()):
            raise TypeError("proposal objective evidence must use ObjectiveGateEvidence.")
        if not isinstance(self.accepted, bool):
            raise TypeError(f"canary verdict accepted must be a Boolean, got {type(self.accepted).__name__}.")
        if any(not isinstance(key, str) for key in self.metrics):
            raise TypeError("canary metric names must be strings.")
        # Both were caller-owned dicts stored by reference on a frozen dataclass, so the measurement
        # that justified a commit could be rewritten after the commit was recorded (MXR-080-1874).
        object.__setattr__(
            self, "metrics", MappingProxyType({key: float(value) for key, value in self.metrics.items()})
        )
        object.__setattr__(self, "proposal_objectives", MappingProxyType(dict(self.proposal_objectives)))

    @property
    def objective_gain(self) -> float | None:
        """Measured objective difference when both values are available."""

        if self.objective_before is None or self.objective_after is None:
            return None
        return self.objective_after - self.objective_before

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible canary receipt."""

        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "objective_before": self.objective_before,
            "objective_after": self.objective_after,
            "objective_gain": self.objective_gain,
            "lower_confidence_gain": self.lower_confidence_gain,
            "confidence_level": self.confidence_level,
            "sample_count": self.sample_count,
            "metrics": dict(self.metrics),
            "proposal_objectives": {
                proposal_id: row.as_dict() for proposal_id, row in sorted(self.proposal_objectives.items())
            },
        }


class CommitStatus(StrEnum):
    """Terminal result of one commit attempt."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    ROLLBACK_FAILED = "rollback_failed"


def _finite_seconds(value: Any, label: str) -> float:
    """Return ``value`` as a finite non-negative float, or raise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number, got {value!r}.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{label} must be finite and non-negative, got {value!r}.")
    return numeric


def _frozen_receipt_value(value: Any, label: str) -> Any:
    """Return an immutable, JSON-expressible copy of one receipt value, recursively.

    The shallow copy this replaces severed only the TOP-level alias (MXR-080-1874). A version vector
    is ``{"model_version": int, "node_versions": {...}}``, so the nested ``node_versions`` dict was
    stored by reference behind a read-only proxy and the caller could still rewrite a committed
    version transition through their own copy. Freezing has to reach as deep as the structure does.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} must be finite to serialize, got {value!r}.")
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} keys must be strings, got {key!r}.")
            frozen[key] = _frozen_receipt_value(item, f"{label}[{key!r}]")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_receipt_value(item, f"{label}[{index}]") for index, item in enumerate(value))
    raise TypeError(
        f"{label} holds {type(value).__name__}, which is neither immutable nor JSON-expressible; a "
        "receipt that cannot serialize is not evidence."
    )


def _plain_receipt_value(value: Any) -> Any:
    """Undo :func:`_frozen_receipt_value`'s containers for ``as_dict``'s JSON-compatible output."""
    if isinstance(value, Mapping):
        return {key: _plain_receipt_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_receipt_value(item) for item in value]
    return value


def _immutable_mapping(value: Any, label: str, *, values_must_be_str: bool = False) -> MappingProxyType:
    """Return a detached, deeply read-only view of a caller-supplied receipt mapping (MXR-080-1865).

    Copied then wrapped: the copy severs the caller's alias so a later mutation cannot rewrite a
    recorded decision, and the proxy stops anyone editing the receipt through the field itself. The
    copy is deep (MXR-080-1874) -- see :func:`_frozen_receipt_value`.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping, got {type(value).__name__}.")
    if values_must_be_str:
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} keys must be strings, got {key!r}.")
            if not isinstance(item, str):
                raise TypeError(f"{label}[{key!r}] must be a string fingerprint, got {type(item).__name__}.")
    return _frozen_receipt_value(dict(value), label)


@dataclass(frozen=True)
class CommitReceipt:
    """Coordinator decision, state fingerprints, and version transition."""

    commit_id: str
    batch_id: str
    proposal_ids: tuple[str, ...]
    status: CommitStatus
    reason: str
    versions_before: dict[str, Any]
    versions_after: dict[str, Any]
    invalidated_nodes: tuple[str, ...] = ()
    canary: CanaryVerdict | None = None
    participant_fingerprints_before: dict[str, str] = field(default_factory=dict)
    participant_fingerprints_after: dict[str, str] = field(default_factory=dict)
    rollback_verified: bool | None = None
    error_type: str | None = None
    error_message: str | None = None
    run_id: str = ""
    model_id: str = ""
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Make the receipt a record of a decision rather than an annotation of one.

        Every field below was annotation-only (MXR-080-1865). ``status="accepted"`` -- the string,
        not the enum -- constructed, then reported ``accepted=False`` because ``is`` never matches a
        string, and crashed in ``as_dict()`` where ``.value`` does not exist. The version and
        fingerprint maps were stored by reference on a frozen dataclass, so a caller mutating its own
        dict afterwards rewrote a committed decision. A receipt that changes after the fact, or that
        cannot serialize, is not evidence of anything.
        """
        if not self.commit_id or not self.batch_id or not self.reason:
            raise ValueError("commit receipt identity must be complete.")
        if not self.run_id or not self.model_id:
            raise ValueError("commit receipt must bind run and model identity.")
        if not isinstance(self.status, CommitStatus):
            raise TypeError(
                f"commit receipt status must be a CommitStatus, got {type(self.status).__name__} "
                f"({self.status!r}). A string never matches the accepted check and has no .value."
            )
        if self.canary is not None and not isinstance(self.canary, CanaryVerdict):
            raise TypeError(f"commit receipt canary must be a CanaryVerdict, got {type(self.canary).__name__}.")
        if self.rollback_verified is not None and not isinstance(self.rollback_verified, bool):
            raise TypeError("commit receipt rollback_verified must be a boolean verdict or None.")
        _finite_seconds(self.elapsed_seconds, "commit receipt elapsed_seconds")
        for name in ("versions_before", "versions_after"):
            object.__setattr__(self, name, _immutable_mapping(getattr(self, name), f"commit receipt {name}"))
        for name in ("participant_fingerprints_before", "participant_fingerprints_after"):
            mapping = _immutable_mapping(getattr(self, name), f"commit receipt {name}", values_must_be_str=True)
            object.__setattr__(self, name, mapping)
        # An ACCEPTED receipt is the strongest claim this class makes -- that a proposal was measured,
        # passed its gate, and advanced the model -- and it was the one status nothing checked
        # (MXR-080-1874). ``CommitStatus.ACCEPTED`` with ``canary=None`` and an unchanged version
        # vector constructed and reported ``accepted=True``: an acceptance with no measurement behind
        # it and no state transition in front of it. The coordinator sets both before it builds this
        # receipt, so neither check can refuse a commit it produced.
        if self.status is CommitStatus.ACCEPTED:
            if self.canary is None:
                raise ValueError(
                    "an accepted commit receipt must carry the canary verdict that accepted it; "
                    "acceptance is a measured decision, not a status string."
                )
            if not self.canary.accepted:
                raise ValueError(
                    f"commit receipt reports status=accepted while its own canary rejected the batch "
                    f"({self.canary.reason!r})."
                )
            before = self.versions_before.get("model_version")
            after = self.versions_after.get("model_version")
            if not isinstance(before, int) or not isinstance(after, int):
                raise ValueError(
                    "an accepted commit receipt must record the model_version it moved from and to; "
                    f"got {before!r} -> {after!r}."
                )
            if after <= before:
                raise ValueError(
                    f"commit receipt reports status=accepted but its model_version did not advance "
                    f"({before} -> {after}); an accepted transaction is one that changed the model."
                )

    @property
    def accepted(self) -> bool:
        """Whether this transaction advanced the model version."""

        return self.status is CommitStatus.ACCEPTED

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible commit receipt."""

        return {
            "commit_id": self.commit_id,
            "batch_id": self.batch_id,
            "proposal_ids": list(self.proposal_ids),
            "status": self.status.value,
            "accepted": self.accepted,
            "reason": self.reason,
            "versions_before": _plain_receipt_value(self.versions_before),
            "versions_after": _plain_receipt_value(self.versions_after),
            "invalidated_nodes": list(self.invalidated_nodes),
            "canary": self.canary.as_dict() if self.canary is not None else None,
            "participant_fingerprints_before": dict(self.participant_fingerprints_before),
            "participant_fingerprints_after": dict(self.participant_fingerprints_after),
            "rollback_verified": self.rollback_verified,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "elapsed_seconds": self.elapsed_seconds,
        }


ApplyProposalFn = Callable[[ProposalPacket], None]
CanaryFn = Callable[[ProposalBatch], CanaryVerdict]


class TransactionalCoordinator:
    """Serialize proposal commits through preflight, snapshot, canary, and commit."""

    def __init__(
        self,
        graph: UpdateGraph,
        apply_proposal: ApplyProposalFn,
        canary: CanaryFn,
        *,
        run_id: str,
        model_id: str,
        participants: Iterable[TransactionParticipant] = (),
        versions: RuntimeVersions | None = None,
        objective_tolerance: float = 1.0e-9,
        enforce_monotone_objective: bool = True,
    ) -> None:
        if not run_id or not model_id:
            raise ValueError("transaction coordinator run_id and model_id must be non-empty.")
        if not math.isfinite(objective_tolerance) or objective_tolerance < 0.0:
            raise ValueError("objective_tolerance must be finite and non-negative.")
        self.graph = graph
        self.run_id = run_id
        self.model_id = model_id
        self.apply_proposal = apply_proposal
        self.canary = canary
        rows = tuple(participants)
        if len({row.name for row in rows}) != len(rows):
            raise ValueError("transaction participant names must be unique.")
        self.participants = rows
        self.versions = versions or RuntimeVersions.for_graph(graph)
        if set(self.versions.node_versions) != {node.node_id for node in graph.nodes}:
            raise ValueError("runtime node versions must exactly match the update graph.")
        self.objective_tolerance = objective_tolerance
        # An exact Boolean, because this flag alone decides whether an objective REGRESSION is
        # committed. ``enforce_monotone_objective=""`` was stored as ``""`` and read by a bare
        # ``if``, so a coordinator built from serialized configuration accepted a canary reporting
        # 5.0 -> 1.0 as ``CommitStatus.ACCEPTED`` (MXR-080-1905).
        self.enforce_monotone_objective = require_exact_bool(enforce_monotone_objective, "enforce_monotone_objective")
        # The ledgers are private with read-only views (MXR-080-1905). They were public lists, so
        # ``coordinator.receipts.clear()`` erased the committed history that ``ledger_fingerprint``
        # is computed from -- the fingerprint changed and nothing recorded that it had been rewritten.
        # A caller that wants a mutable copy takes ``list(coordinator.receipts)``.
        self._receipts: list[CommitReceipt] = []
        self._proposal_receipts: list[dict[str, Any]] = []
        self._poisoned = False
        self._commit_sequence = 0
        self._commit_ids: set[str] = set()
        self._seen_proposal_ids: set[str] = set()
        self._lock = RLock()

    @property
    def receipts(self) -> tuple[CommitReceipt, ...]:
        """Every terminal commit receipt this coordinator produced, oldest first.

        A detached tuple: the receipts themselves are frozen and sealed, and the sequence they form
        is the coordinator's own record, not a caller-editable list (MXR-080-1905).
        """
        with self._lock:
            return tuple(self._receipts)

    @property
    def proposal_receipts(self) -> tuple[dict[str, Any], ...]:
        """Serialized proposals seen by this coordinator, in arrival order.

        Each row is a fresh ``as_dict()`` payload built at commit time, so the tuple shares no
        structure a caller can reach (MXR-080-1905).
        """
        with self._lock:
            return tuple(self._proposal_receipts)

    @property
    def poisoned(self) -> bool:
        """Whether an unverified rollback has disabled further commits.

        Read-only: this was a plain attribute, so ``coordinator.poisoned = False`` un-poisoned a
        coordinator whose rollback could not be verified and let it accept commits on top of state
        nothing had confirmed was restored (MXR-080-1905). Only :meth:`commit` sets it, from the
        rollback verification itself.
        """
        return self._poisoned

    def _next_commit_id(self) -> str:
        while True:
            commit_id = "commit-%08d" % self._commit_sequence
            self._commit_sequence += 1
            if commit_id not in self._commit_ids:
                return commit_id

    def _required_semantics(self, batch: ProposalBatch) -> frozenset[StateSemantics]:
        required: set[StateSemantics] = set()
        for proposal in batch.proposals:
            states = self.graph.node(proposal.node_id).contract.state_semantics
            required.update(state for state in states if state is not StateSemantics.IMMUTABLE_RESULT)
        return frozenset(required)

    def _preflight_error(self, batch: ProposalBatch) -> str | None:
        if self._poisoned:
            return "coordinator-poisoned-by-unverified-rollback"
        known_nodes = {node.node_id for node in self.graph.nodes}
        for proposal in batch.proposals:
            if proposal.proposal_id in self._seen_proposal_ids:
                return "duplicate-proposal-id:%s" % proposal.proposal_id
            if proposal.run_id != self.run_id:
                return "run-id-mismatch:%s" % proposal.proposal_id
            if proposal.model_id != self.model_id:
                return "model-id-mismatch:%s" % proposal.proposal_id
            if proposal.node_id not in known_nodes:
                return "unknown-proposal-node:%s" % proposal.node_id
            try:
                current_payload_hash = payload_fingerprint(proposal.payload)
            except (TypeError, ValueError):
                return "invalid-proposal-payload:%s" % proposal.proposal_id
            if current_payload_hash != proposal.payload_hash:
                return "proposal-payload-mutated:%s" % proposal.proposal_id
        conflicts = proposal_conflicts(self.graph, batch)
        if conflicts:
            return "proposal-conflict:%s" % conflicts[0].reason
        for proposal in batch.proposals:
            node = self.graph.node(proposal.node_id)
            contract = node.contract
            if contract.update_kind is UpdateKind.FROZEN:
                return "frozen-node:%s" % proposal.node_id
            if proposal.update_kind is not contract.update_kind:
                return "update-kind-mismatch:%s" % proposal.node_id
            if proposal.objective_kind is not contract.objective_kind:
                return "objective-kind-mismatch:%s" % proposal.node_id
            if not proposal.writes.issubset(contract.writes):
                return "undeclared-write:%s" % proposal.node_id
            if proposal.base_model_version != self.versions.model_version:
                return "base-model-version-mismatch:%s" % proposal.node_id
            dependency_versions = proposal.dependency_version_map
            if proposal.node_id not in dependency_versions:
                return "missing-node-version:%s" % proposal.node_id
            for node_id, expected in dependency_versions.items():
                if node_id not in self.versions.node_versions:
                    return "unknown-dependency-version:%s" % node_id
                if expected != self.versions.node_versions[node_id]:
                    return "dependency-version-mismatch:%s" % node_id
            if any(node_id not in self.versions.node_versions for node_id in proposal.invalidates):
                return "unknown-explicit-invalidation:%s" % proposal.node_id

        covered = frozenset(state for participant in self.participants for state in participant.semantics)
        missing = self._required_semantics(batch) - covered
        if missing:
            return "missing-transaction-state:%s" % ",".join(sorted(state.value for state in missing))
        covered_artifacts = frozenset(
            artifact for participant in self.participants for artifact in participant.artifacts
        )
        required_artifacts = frozenset(artifact for proposal in batch.proposals for artifact in proposal.writes)
        missing_artifacts = required_artifacts - covered_artifacts
        if missing_artifacts:
            return "missing-transaction-artifact:%s" % ",".join(
                sorted(artifact.value for artifact in missing_artifacts)
            )
        return None

    def _ordered_proposals(self, batch: ProposalBatch) -> tuple[ProposalPacket, ...]:
        order = {node_id: index for index, node_id in enumerate(self.graph.topological_order())}
        return tuple(sorted(batch.proposals, key=lambda proposal: (order[proposal.node_id], proposal.proposal_id)))

    def _fingerprints(self) -> dict[str, str]:
        return {participant.name: participant.fingerprint() for participant in self.participants}

    def _objective_error(self, batch: ProposalBatch, verdict: CanaryVerdict) -> str | None:
        if not verdict.accepted:
            return "canary-rejected:%s" % verdict.reason
        proposal_ids = {proposal.proposal_id for proposal in batch.proposals}
        unknown_evidence = sorted(set(verdict.proposal_objectives) - proposal_ids)
        if unknown_evidence:
            return "unknown-proposal-objective-evidence:%s" % unknown_evidence[0]
        if verdict.lower_confidence_gain is not None and verdict.lower_confidence_gain < -self.objective_tolerance:
            return "negative-canary-lower-bound"
        strict_proposals = tuple(
            proposal.objective_kind in (ObjectiveKind.MLE, ObjectiveKind.MAP, ObjectiveKind.ELBO)
            for proposal in batch.proposals
        )
        if self.enforce_monotone_objective and any(strict_proposals):
            strict_rows = tuple(proposal for proposal, is_strict in zip(batch.proposals, strict_proposals) if is_strict)
            require_per_proposal = len(batch.proposals) > 1
            for proposal in strict_rows:
                evidence = verdict.proposal_objectives.get(proposal.proposal_id)
                if evidence is None and require_per_proposal:
                    return "strict-objective-evidence-missing:%s" % proposal.proposal_id
                if evidence is None:
                    if verdict.objective_before is None or verdict.objective_after is None:
                        return "strict-objective-canary-missing-values"
                    before = verdict.objective_before
                    after = verdict.objective_after
                    lower_bound = verdict.lower_confidence_gain
                else:
                    before = evidence.objective_before
                    after = evidence.objective_after
                    lower_bound = evidence.lower_confidence_gain
                if lower_bound is not None and lower_bound < -self.objective_tolerance:
                    return "negative-proposal-lower-bound:%s" % proposal.proposal_id
                if after + self.objective_tolerance < before:
                    return (
                        "strict-objective-regression"
                        if len(batch.proposals) == 1
                        else "strict-objective-regression:%s" % proposal.proposal_id
                    )
        return None

    def _rollback(
        self,
        snapshots: dict[str, Any],
        fingerprints_before: dict[str, str],
    ) -> tuple[bool, dict[str, str], BaseException | None]:
        restore_error: Exception | None = None
        for participant in reversed(self.participants):
            try:
                participant.restore(snapshots[participant.name])
            except Exception as error:  # noqa: BLE001 - receipt and poison any failed rollback
                restore_error = restore_error or error
        try:
            fingerprints_after = self._fingerprints()
        except Exception as error:  # noqa: BLE001 - failed verification poisons coordinator
            return False, {}, restore_error or error
        verified = restore_error is None and fingerprints_after == fingerprints_before
        return verified, fingerprints_after, restore_error

    def commit(self, batch: ProposalBatch | ProposalPacket, *, commit_id: str | None = None) -> CommitReceipt:
        """Attempt one atomic commit and always return a terminal receipt."""

        if isinstance(batch, ProposalPacket):
            batch = ProposalBatch("batch:%s" % batch.proposal_id, (batch,))
        with self._lock:
            started = time.perf_counter()
            commit_id = commit_id or self._next_commit_id()
            if commit_id in self._commit_ids:
                raise ValueError("commit id %s has already been used." % commit_id)
            self._commit_ids.add(commit_id)
            versions_before = self.versions.as_dict()
            proposal_ids = tuple(proposal.proposal_id for proposal in batch.proposals)
            self._proposal_receipts.extend(proposal.as_dict() for proposal in batch.proposals)
            error = self._preflight_error(batch)
            if error is not None:
                receipt = CommitReceipt(
                    commit_id,
                    batch.batch_id,
                    proposal_ids,
                    CommitStatus.REJECTED,
                    error,
                    versions_before,
                    self.versions.as_dict(),
                    run_id=self.run_id,
                    model_id=self.model_id,
                    elapsed_seconds=time.perf_counter() - started,
                )
                self._receipts.append(receipt)
                return receipt

            try:
                fingerprints_before = self._fingerprints()
                snapshots = {participant.name: participant.snapshot() for participant in self.participants}
            except Exception as apply_error:  # noqa: BLE001 - snapshot failure is a rejected transaction
                receipt = CommitReceipt(
                    commit_id,
                    batch.batch_id,
                    proposal_ids,
                    CommitStatus.REJECTED,
                    "snapshot-failed",
                    versions_before,
                    self.versions.as_dict(),
                    error_type=type(apply_error).__name__,
                    error_message=str(apply_error),
                    run_id=self.run_id,
                    model_id=self.model_id,
                    elapsed_seconds=time.perf_counter() - started,
                )
                self._receipts.append(receipt)
                return receipt

            verdict: CanaryVerdict | None = None
            apply_error: Exception | None = None
            rejection_reason: str | None = None
            tentative_fingerprints: dict[str, str] = {}
            try:
                for proposal in self._ordered_proposals(batch):
                    self.apply_proposal(proposal)
                verdict = self.canary(batch)
                if not isinstance(verdict, CanaryVerdict):
                    raise TypeError("canary callback must return CanaryVerdict.")
                rejection_reason = self._objective_error(batch, verdict)
                if rejection_reason is None:
                    tentative_fingerprints = self._fingerprints()
            except Exception as error:  # noqa: BLE001 - any tentative-apply error must roll back
                apply_error = error
                rejection_reason = "apply-or-canary-error"

            if rejection_reason is not None:
                verified, fingerprints_after, restore_error = self._rollback(snapshots, fingerprints_before)
                status = CommitStatus.ROLLED_BACK if verified else CommitStatus.ROLLBACK_FAILED
                self._poisoned = not verified
                error_value = restore_error or apply_error
                receipt = CommitReceipt(
                    commit_id,
                    batch.batch_id,
                    proposal_ids,
                    status,
                    rejection_reason,
                    versions_before,
                    self.versions.as_dict(),
                    canary=verdict,
                    participant_fingerprints_before=fingerprints_before,
                    participant_fingerprints_after=fingerprints_after,
                    rollback_verified=verified,
                    error_type=type(error_value).__name__ if error_value is not None else None,
                    error_message=str(error_value) if error_value is not None else None,
                    run_id=self.run_id,
                    model_id=self.model_id,
                    elapsed_seconds=time.perf_counter() - started,
                )
                self._receipts.append(receipt)
                return receipt

            invalidated_set = set()
            for proposal in batch.proposals:
                invalidated_set.update(self.graph.invalidated_by(proposal.node_id))
                invalidated_set.update(proposal.invalidates)
            invalidated = tuple(node_id for node_id in self.graph.topological_order() if node_id in invalidated_set)
            # Stage, receipt, THEN commit (MXR-080-1905). The version bump used to run first, so a
            # ``CommitReceipt`` that refused to construct left the coordinator advanced with nothing
            # recording it. The reproduction was a version vector the receipt could not express
            # (``RuntimeVersions(0.0, ...)``): ``commit`` raised after ``model_version`` had already
            # moved 0.0 -> 1.0, the node version 0 -> 1, and the proposal ids had been marked seen --
            # so the retry got ``duplicate-proposal-id`` for a transaction that produced no receipt.
            #
            # ``RuntimeVersions`` now refuses that value outright, so THAT input no longer reaches
            # here, and no other currently-reachable input makes this receipt fail late. The
            # reordering is kept anyway because it removes the class rather than the one instance:
            # building the receipt from the STAGED vector means the only way to reach the
            # assignments below is for the record of them to already exist.
            staged = RuntimeVersions(
                self.versions.model_version + 1,
                {
                    node_id: version + 1 if node_id in invalidated_set else version
                    for node_id, version in self.versions.node_versions.items()
                },
            )
            receipt = CommitReceipt(
                commit_id,
                batch.batch_id,
                proposal_ids,
                CommitStatus.ACCEPTED,
                "canary-accepted",
                versions_before,
                staged.as_dict(),
                invalidated_nodes=invalidated,
                canary=verdict,
                participant_fingerprints_before=fingerprints_before,
                participant_fingerprints_after=tentative_fingerprints,
                run_id=self.run_id,
                model_id=self.model_id,
                elapsed_seconds=time.perf_counter() - started,
            )
            # Applied in place rather than by rebinding ``self.versions``: callers reach the vector
            # through the coordinator (``coordinator.versions.model_version``), and keeping the same
            # object keeps any held reference correct.
            self.versions.model_version = staged.model_version
            self.versions.node_versions = dict(staged.node_versions)
            self._seen_proposal_ids.update(proposal_ids)
            self._receipts.append(receipt)
            return receipt

    def ledger_fingerprint(self) -> str:
        """Fingerprint receipt metadata for deterministic replay comparisons."""

        payloads = []
        for receipt in self._receipts:
            payload = receipt.as_dict()
            payload.pop("elapsed_seconds")
            payloads.append(payload)
        return payload_fingerprint(tuple(payloads))


__all__ = [
    "ApplyProposalFn",
    "CanaryFn",
    "CanaryVerdict",
    "CommitReceipt",
    "CommitStatus",
    "FingerprintFn",
    "ObjectiveGateEvidence",
    "RestoreFn",
    "RuntimeVersions",
    "SnapshotFn",
    "TransactionParticipant",
    "TransactionalCoordinator",
]
