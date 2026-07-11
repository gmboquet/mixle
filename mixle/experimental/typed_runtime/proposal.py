"""Immutable update proposals, conflict analysis, and exact payload merging."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.contracts import ArtifactKind, MergeLaw, ObjectiveKind, UpdateKind
from mixle.experimental.typed_runtime.graph import UpdateGraph


def _hash_value(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"none")
    elif isinstance(value, bool):
        digest.update(b"bool:1" if value else b"bool:0")
    elif isinstance(value, (int, np.integer)):
        digest.update(b"int:" + str(int(value)).encode("ascii"))
    elif isinstance(value, (float, np.floating)):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError("proposal payloads cannot contain non-finite floating-point values.")
        digest.update(b"float:" + struct.pack("!d", scalar))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"str:" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
        digest.update(b"bytes:" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object-dtype arrays are not deterministic proposal payloads.")
        if np.issubdtype(value.dtype, np.inexact) and not np.all(np.isfinite(value)):
            raise ValueError("proposal payload arrays cannot contain non-finite values.")
        contiguous = np.ascontiguousarray(value)
        digest.update(b"ndarray:" + contiguous.dtype.str.encode("ascii"))
        _hash_value(digest, contiguous.shape)
        digest.update(contiguous.tobytes(order="C"))
    elif isinstance(value, Mapping):
        digest.update(b"mapping:")
        rows = []
        for key, item in value.items():
            key_digest = hashlib.sha256()
            _hash_value(key_digest, key)
            rows.append((key_digest.digest(), key, item))
        for _, key, item in sorted(rows, key=lambda row: row[0]):
            _hash_value(digest, key)
            _hash_value(digest, item)
    elif isinstance(value, tuple):
        digest.update(b"tuple:")
        for item in value:
            _hash_value(digest, item)
    elif isinstance(value, list):
        digest.update(b"list:")
        for item in value:
            _hash_value(digest, item)
    elif is_dataclass(value) and not isinstance(value, type):
        digest.update(("dataclass:%s.%s:" % (type(value).__module__, type(value).__qualname__)).encode("utf-8"))
        for spec in fields(value):
            _hash_value(digest, spec.name)
            _hash_value(digest, getattr(value, spec.name))
    else:
        as_dict = getattr(value, "as_dict", None)
        if callable(as_dict):
            digest.update(("as_dict:%s.%s:" % (type(value).__module__, type(value).__qualname__)).encode("utf-8"))
            _hash_value(digest, as_dict())
            return
        raise TypeError("unsupported deterministic proposal payload type: %s" % type(value).__name__)


def payload_fingerprint(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint without pickle or ``repr``."""

    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProposalPacket:
    """One worker's complete, immutable update proposal receipt."""

    proposal_id: str
    run_id: str
    model_id: str
    node_id: str
    shard_id: str
    base_model_version: int
    dependency_versions: tuple[tuple[str, int], ...]
    update_kind: UpdateKind
    objective_kind: ObjectiveKind
    payload: Any
    writes: frozenset[ArtifactKind] = frozenset({ArtifactKind.PARAMETERS})
    observations: float = 0.0
    tokens: int = 0
    responsibility_mass: float = 0.0
    local_objective_before: float | None = None
    local_objective_after: float | None = None
    global_objective_before: float | None = None
    global_objective_after: float | None = None
    predicted_gain: float | None = None
    measured_gain: float | None = None
    gain_standard_error: float | None = None
    optimizer_steps: int = 0
    effective_batch_size: float = 0.0
    wall_time_seconds: float = 0.0
    compute_units: float = 0.0
    communication_bytes: int = 0
    precision: str | None = None
    overflow_count: int = 0
    underflow_count: int = 0
    data_fingerprint: str | None = None
    ordering_fingerprint: str | None = None
    rng_fingerprint: str | None = None
    invalidates: tuple[str, ...] = ()
    rollback_reference: str | None = None
    surrogate_disclosed: bool = False
    payload_hash: str = ""

    def __post_init__(self) -> None:
        identifiers = (self.proposal_id, self.run_id, self.model_id, self.node_id, self.shard_id)
        if any(not value for value in identifiers):
            raise ValueError("proposal identifiers must be non-empty.")
        if self.base_model_version < 0:
            raise ValueError("base_model_version must be non-negative.")
        versions = self.dependency_versions
        if isinstance(versions, Mapping):
            versions = tuple(sorted((str(key), int(value)) for key, value in versions.items()))
            object.__setattr__(self, "dependency_versions", versions)
        if len({key for key, _ in versions}) != len(versions) or any(version < 0 for _, version in versions):
            raise ValueError("dependency versions must have unique keys and non-negative values.")
        nonnegative = (
            self.observations,
            self.tokens,
            self.responsibility_mass,
            self.optimizer_steps,
            self.effective_batch_size,
            self.wall_time_seconds,
            self.compute_units,
            self.communication_bytes,
            self.overflow_count,
            self.underflow_count,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("proposal work, mass, and telemetry values must be non-negative.")
        optional_numeric = (
            self.local_objective_before,
            self.local_objective_after,
            self.global_objective_before,
            self.global_objective_after,
            self.predicted_gain,
            self.measured_gain,
            self.gain_standard_error,
        )
        if any(value is not None and not math.isfinite(value) for value in optional_numeric):
            raise ValueError("proposal objective and gain values must be finite when supplied.")
        if self.gain_standard_error is not None and self.gain_standard_error < 0.0:
            raise ValueError("gain_standard_error must be non-negative.")
        if not self.writes:
            raise ValueError("an update proposal must declare at least one written artifact.")
        computed_hash = payload_fingerprint(self.payload)
        if self.payload_hash and self.payload_hash != computed_hash:
            raise ValueError("payload_hash does not match the proposal payload.")
        object.__setattr__(self, "payload_hash", computed_hash)
        surrogate = self.objective_kind in (
            ObjectiveKind.CONTRASTIVE,
            ObjectiveKind.PREFERENCE,
            ObjectiveKind.CONSTRAINT,
            ObjectiveKind.USER_SURROGATE,
        )
        if surrogate and not self.surrogate_disclosed:
            raise ValueError("surrogate proposals must set surrogate_disclosed=True.")

    @property
    def dependency_version_map(self) -> dict[str, int]:
        """Return dependency versions as a fresh mapping."""

        return dict(self.dependency_versions)

    def as_dict(self) -> dict[str, Any]:
        """Return metadata and payload fingerprint, never the runtime payload."""

        return {
            "proposal_id": self.proposal_id,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "node_id": self.node_id,
            "shard_id": self.shard_id,
            "base_model_version": self.base_model_version,
            "dependency_versions": dict(self.dependency_versions),
            "update_kind": self.update_kind.value,
            "objective_kind": self.objective_kind.value,
            "payload_hash": self.payload_hash,
            "writes": sorted(artifact.value for artifact in self.writes),
            "observations": self.observations,
            "tokens": self.tokens,
            "responsibility_mass": self.responsibility_mass,
            "local_objective_before": self.local_objective_before,
            "local_objective_after": self.local_objective_after,
            "global_objective_before": self.global_objective_before,
            "global_objective_after": self.global_objective_after,
            "predicted_gain": self.predicted_gain,
            "measured_gain": self.measured_gain,
            "gain_standard_error": self.gain_standard_error,
            "optimizer_steps": self.optimizer_steps,
            "effective_batch_size": self.effective_batch_size,
            "wall_time_seconds": self.wall_time_seconds,
            "compute_units": self.compute_units,
            "communication_bytes": self.communication_bytes,
            "precision": self.precision,
            "overflow_count": self.overflow_count,
            "underflow_count": self.underflow_count,
            "data_fingerprint": self.data_fingerprint,
            "ordering_fingerprint": self.ordering_fingerprint,
            "rng_fingerprint": self.rng_fingerprint,
            "invalidates": list(self.invalidates),
            "rollback_reference": self.rollback_reference,
            "surrogate_disclosed": self.surrogate_disclosed,
        }


@dataclass(frozen=True)
class ProposalBatch:
    """Proposals intended to commit as one versioned transaction."""

    batch_id: str
    proposals: tuple[ProposalPacket, ...]

    def __post_init__(self) -> None:
        if not self.batch_id or not self.proposals:
            raise ValueError("a proposal batch needs an id and at least one proposal.")
        if len({proposal.proposal_id for proposal in self.proposals}) != len(self.proposals):
            raise ValueError("proposal ids must be unique within a batch.")
        anchors = {(proposal.run_id, proposal.model_id, proposal.base_model_version) for proposal in self.proposals}
        if len(anchors) != 1:
            raise ValueError("batched proposals must share run, model, and base version.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible batch receipt."""

        return {"batch_id": self.batch_id, "proposals": [proposal.as_dict() for proposal in self.proposals]}


@dataclass(frozen=True)
class ProposalConflict:
    """A pair of proposals that cannot be committed concurrently."""

    left_proposal: str
    right_proposal: str
    reason: str


def proposal_conflicts(graph: UpdateGraph, batch: ProposalBatch) -> tuple[ProposalConflict, ...]:
    """Return same-state and dependency-order conflicts in deterministic order."""

    conflicts: list[ProposalConflict] = []
    proposals = sorted(batch.proposals, key=lambda proposal: proposal.proposal_id)
    for index, left in enumerate(proposals):
        graph.node(left.node_id)
        for right in proposals[index + 1 :]:
            graph.node(right.node_id)
            if left.node_id == right.node_id:
                conflicts.append(ProposalConflict(left.proposal_id, right.proposal_id, "overlapping-node-write"))
                continue
            left_closure = set(graph.invalidated_by(left.node_id, include_self=False))
            right_closure = set(graph.invalidated_by(right.node_id, include_self=False))
            if right.node_id in left_closure or left.node_id in right_closure:
                conflicts.append(ProposalConflict(left.proposal_id, right.proposal_id, "dependency-version-order"))
    return tuple(conflicts)


def _additive_payload(left: Any, right: Any) -> Any:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("additive array payloads must have matching shape and dtype.")
        return left + right
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        return left + right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if left.keys() != right.keys():
            raise ValueError("additive mapping payloads must have identical keys.")
        return {key: _additive_payload(left[key], right[key]) for key in left}
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return tuple(_additive_payload(a, b) for a, b in zip(left, right))
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [_additive_payload(a, b) for a, b in zip(left, right)]
    raise TypeError(
        "no exact additive merge for payload types %s and %s." % (type(left).__name__, type(right).__name__)
    )


PayloadMerger = Callable[[Any, Any], Any]


def merge_same_node_proposals(
    proposals: Sequence[ProposalPacket],
    *,
    merged_proposal_id: str,
    merge_law: MergeLaw,
    payload_merger: PayloadMerger | None = None,
) -> ProposalPacket:
    """Merge shard proposals for one node under an explicitly declared law."""

    rows = tuple(proposals)
    if not rows:
        raise ValueError("at least one proposal is required for merging.")
    anchor = rows[0]
    comparable = {
        (
            row.run_id,
            row.model_id,
            row.node_id,
            row.base_model_version,
            row.dependency_versions,
            row.update_kind,
            row.objective_kind,
            row.writes,
        )
        for row in rows
    }
    if len(comparable) != 1:
        raise ValueError("merged proposals must share node, versions, semantics, and write set.")
    if merge_law is MergeLaw.ADDITIVE:
        merger = payload_merger or _additive_payload
    elif payload_merger is not None and merge_law in (
        MergeLaw.ASSOCIATIVE_MONOID,
        MergeLaw.INVERTIBLE_GROUP,
        MergeLaw.WEIGHTED_SKETCH,
        MergeLaw.LOW_RANK,
    ):
        merger = payload_merger
    else:
        raise ValueError("merge law %s requires an explicit payload merger." % merge_law.value)

    ordered = sorted(rows, key=lambda row: row.proposal_id)
    payload = ordered[0].payload
    for row in ordered[1:]:
        payload = merger(payload, row.payload)

    def optional_sum(name: str) -> float | None:
        values = [getattr(row, name) for row in rows]
        return sum(values) if all(value is not None for value in values) else None

    combined_fingerprint = payload_fingerprint(tuple(row.data_fingerprint for row in ordered))
    combined_rng = payload_fingerprint(tuple(row.rng_fingerprint for row in ordered))
    return ProposalPacket(
        proposal_id=merged_proposal_id,
        run_id=anchor.run_id,
        model_id=anchor.model_id,
        node_id=anchor.node_id,
        shard_id="merged[%s]" % ",".join(row.shard_id for row in ordered),
        base_model_version=anchor.base_model_version,
        dependency_versions=anchor.dependency_versions,
        update_kind=anchor.update_kind,
        objective_kind=anchor.objective_kind,
        payload=payload,
        writes=anchor.writes,
        observations=sum(row.observations for row in rows),
        tokens=sum(row.tokens for row in rows),
        responsibility_mass=sum(row.responsibility_mass for row in rows),
        predicted_gain=optional_sum("predicted_gain"),
        measured_gain=optional_sum("measured_gain"),
        optimizer_steps=sum(row.optimizer_steps for row in rows),
        effective_batch_size=sum(row.effective_batch_size for row in rows),
        wall_time_seconds=sum(row.wall_time_seconds for row in rows),
        compute_units=sum(row.compute_units for row in rows),
        communication_bytes=sum(row.communication_bytes for row in rows),
        overflow_count=sum(row.overflow_count for row in rows),
        underflow_count=sum(row.underflow_count for row in rows),
        data_fingerprint=combined_fingerprint,
        ordering_fingerprint=payload_fingerprint(tuple(row.ordering_fingerprint for row in ordered)),
        rng_fingerprint=combined_rng,
        invalidates=tuple(sorted({node for row in rows for node in row.invalidates})),
        rollback_reference=anchor.rollback_reference,
        surrogate_disclosed=anchor.surrogate_disclosed,
    )


__all__ = [
    "PayloadMerger",
    "ProposalBatch",
    "ProposalConflict",
    "ProposalPacket",
    "merge_same_node_proposals",
    "payload_fingerprint",
    "proposal_conflicts",
]
