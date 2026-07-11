"""Versioned transactional proposal commit with canaries and verified rollback."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from typing import Any

from mixle.experimental.typed_runtime.contracts import ObjectiveKind, StateSemantics, UpdateKind
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

        if model_version < 0:
            raise ValueError("model_version must be non-negative.")
        return cls(model_version, {node.node_id: 0 for node in graph.nodes})

    def __post_init__(self) -> None:
        if self.model_version < 0 or any(version < 0 for version in self.node_versions.values()):
            raise ValueError("runtime versions must be non-negative.")

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

    def __post_init__(self) -> None:
        if not self.name or not self.semantics:
            raise ValueError("transaction participants need a name and mutable state semantics.")
        if StateSemantics.IMMUTABLE_RESULT in self.semantics:
            raise ValueError("immutable_result is not a transaction participant state domain.")

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
        }


class CommitStatus(StrEnum):
    """Terminal result of one commit attempt."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    ROLLBACK_FAILED = "rollback_failed"


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
    elapsed_seconds: float = 0.0

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
            "versions_before": self.versions_before,
            "versions_after": self.versions_after,
            "invalidated_nodes": list(self.invalidated_nodes),
            "canary": self.canary.as_dict() if self.canary is not None else None,
            "participant_fingerprints_before": dict(self.participant_fingerprints_before),
            "participant_fingerprints_after": dict(self.participant_fingerprints_after),
            "rollback_verified": self.rollback_verified,
            "error_type": self.error_type,
            "error_message": self.error_message,
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
        participants: Iterable[TransactionParticipant] = (),
        versions: RuntimeVersions | None = None,
        objective_tolerance: float = 1.0e-9,
        enforce_monotone_objective: bool = True,
    ) -> None:
        if objective_tolerance < 0.0:
            raise ValueError("objective_tolerance must be non-negative.")
        self.graph = graph
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
        self.enforce_monotone_objective = enforce_monotone_objective
        self.receipts: list[CommitReceipt] = []
        self.proposal_receipts: list[dict[str, Any]] = []
        self.poisoned = False
        self._commit_sequence = 0
        self._commit_ids: set[str] = set()
        self._seen_proposal_ids: set[str] = set()
        self._lock = RLock()

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
        if self.poisoned:
            return "coordinator-poisoned-by-unverified-rollback"
        known_nodes = {node.node_id for node in self.graph.nodes}
        for proposal in batch.proposals:
            if proposal.proposal_id in self._seen_proposal_ids:
                return "duplicate-proposal-id:%s" % proposal.proposal_id
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
        return None

    def _ordered_proposals(self, batch: ProposalBatch) -> tuple[ProposalPacket, ...]:
        order = {node_id: index for index, node_id in enumerate(self.graph.topological_order())}
        return tuple(sorted(batch.proposals, key=lambda proposal: (order[proposal.node_id], proposal.proposal_id)))

    def _fingerprints(self) -> dict[str, str]:
        return {participant.name: participant.fingerprint() for participant in self.participants}

    def _objective_error(self, batch: ProposalBatch, verdict: CanaryVerdict) -> str | None:
        if not verdict.accepted:
            return "canary-rejected:%s" % verdict.reason
        if verdict.lower_confidence_gain is not None and verdict.lower_confidence_gain < -self.objective_tolerance:
            return "negative-canary-lower-bound"
        strict = all(
            proposal.objective_kind in (ObjectiveKind.MLE, ObjectiveKind.MAP, ObjectiveKind.ELBO)
            for proposal in batch.proposals
        )
        if self.enforce_monotone_objective and strict:
            if verdict.objective_before is None or verdict.objective_after is None:
                return "strict-objective-canary-missing-values"
            if verdict.objective_after + self.objective_tolerance < verdict.objective_before:
                return "strict-objective-regression"
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
            self.proposal_receipts.extend(proposal.as_dict() for proposal in batch.proposals)
            error = self._preflight_error(batch)
            self._seen_proposal_ids.update(proposal_ids)
            if error is not None:
                receipt = CommitReceipt(
                    commit_id,
                    batch.batch_id,
                    proposal_ids,
                    CommitStatus.REJECTED,
                    error,
                    versions_before,
                    self.versions.as_dict(),
                    elapsed_seconds=time.perf_counter() - started,
                )
                self.receipts.append(receipt)
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
                    elapsed_seconds=time.perf_counter() - started,
                )
                self.receipts.append(receipt)
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
                self.poisoned = not verified
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
                    elapsed_seconds=time.perf_counter() - started,
                )
                self.receipts.append(receipt)
                return receipt

            invalidated_set = set()
            for proposal in batch.proposals:
                invalidated_set.update(self.graph.invalidated_by(proposal.node_id))
                invalidated_set.update(proposal.invalidates)
            invalidated = tuple(node_id for node_id in self.graph.topological_order() if node_id in invalidated_set)
            self.versions.model_version += 1
            for node_id in invalidated:
                self.versions.node_versions[node_id] += 1
            receipt = CommitReceipt(
                commit_id,
                batch.batch_id,
                proposal_ids,
                CommitStatus.ACCEPTED,
                "canary-accepted",
                versions_before,
                self.versions.as_dict(),
                invalidated_nodes=invalidated,
                canary=verdict,
                participant_fingerprints_before=fingerprints_before,
                participant_fingerprints_after=tentative_fingerprints,
                elapsed_seconds=time.perf_counter() - started,
            )
            self.receipts.append(receipt)
            return receipt

    def ledger_fingerprint(self) -> str:
        """Fingerprint receipt metadata for deterministic replay comparisons."""

        payloads = []
        for receipt in self.receipts:
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
    "RestoreFn",
    "RuntimeVersions",
    "SnapshotFn",
    "TransactionParticipant",
    "TransactionalCoordinator",
]
