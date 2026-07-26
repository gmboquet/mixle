"""Immutable update proposals, conflict analysis, and exact payload merging."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from numbers import Integral, Real
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
    data_contribution_ids: tuple[str, ...] = ()
    ordering_fingerprint: str | None = None
    rng_fingerprint: str | None = None
    invalidates: tuple[str, ...] = ()
    rollback_reference: str | None = None
    surrogate_disclosed: bool = False
    staleness_semantics: str | None = None
    correction_fingerprint: str | None = None
    merge_error_bound: float | None = None
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
        for name in (
            "observations",
            "responsibility_mass",
            "effective_batch_size",
            "wall_time_seconds",
            "compute_units",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("%s must be a real number." % name)
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError("%s must be finite and non-negative." % name)
        for name in (
            "tokens",
            "optimizer_steps",
            "communication_bytes",
            "overflow_count",
            "underflow_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("%s must be an integer." % name)
            if value < 0:
                raise ValueError("%s must be non-negative." % name)
        optional_numeric = (
            self.local_objective_before,
            self.local_objective_after,
            self.global_objective_before,
            self.global_objective_after,
            self.predicted_gain,
            self.measured_gain,
            self.gain_standard_error,
            self.merge_error_bound,
        )
        if any(
            value is not None and (isinstance(value, bool) or not isinstance(value, Real)) for value in optional_numeric
        ):
            raise TypeError("proposal objective, gain, and error values must be real when supplied.")
        if any(value is not None and not math.isfinite(float(value)) for value in optional_numeric):
            raise ValueError("proposal objective, gain, and error values must be finite when supplied.")
        if self.gain_standard_error is not None and self.gain_standard_error < 0.0:
            raise ValueError("gain_standard_error must be non-negative.")
        if self.merge_error_bound is not None and self.merge_error_bound < 0.0:
            raise ValueError("merge_error_bound must be non-negative.")
        if not self.writes:
            raise ValueError("an update proposal must declare at least one written artifact.")
        contributions = self.data_contribution_ids
        if not contributions and self.data_fingerprint is not None:
            contributions = (self.data_fingerprint,)
            object.__setattr__(self, "data_contribution_ids", contributions)
        if any(not isinstance(value, str) or not value for value in contributions) or len(set(contributions)) != len(
            contributions
        ):
            raise ValueError("data contribution ids must be unique non-empty strings.")
        normalized_contributions = tuple(sorted(contributions))
        if contributions != normalized_contributions:
            object.__setattr__(self, "data_contribution_ids", normalized_contributions)
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
        allowed_staleness = {
            None,
            "bounded_stale_approximation",
            "corrected_statistical_approximation",
            "exact_rebase",
        }
        if self.staleness_semantics not in allowed_staleness:
            raise ValueError("unsupported proposal staleness_semantics.")
        if (self.correction_fingerprint is not None) != (
            self.staleness_semantics in {"corrected_statistical_approximation", "exact_rebase"}
        ):
            raise ValueError("corrected proposal semantics require exactly one correction_fingerprint.")

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
            "data_contribution_ids": list(self.data_contribution_ids),
            "ordering_fingerprint": self.ordering_fingerprint,
            "rng_fingerprint": self.rng_fingerprint,
            "invalidates": list(self.invalidates),
            "rollback_reference": self.rollback_reference,
            "surrogate_disclosed": self.surrogate_disclosed,
            "staleness_semantics": self.staleness_semantics,
            "correction_fingerprint": self.correction_fingerprint,
            "merge_error_bound": self.merge_error_bound,
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
        if np.issubdtype(left.dtype, np.bool_):
            raise TypeError("boolean arrays do not have a supported additive merge.")
        if np.issubdtype(left.dtype, np.integer):
            limits = np.iinfo(left.dtype)
            exact = left.astype(object) + right.astype(object)
            if np.any(exact < limits.min) or np.any(exact > limits.max):
                raise OverflowError("additive integer array payload overflow.")
        with np.errstate(over="ignore", invalid="ignore"):
            result = left + right
        if np.issubdtype(result.dtype, np.inexact) and not np.all(np.isfinite(result)):
            raise OverflowError("additive floating array payload produced a non-finite value.")
        return result
    if isinstance(left, (Integral, np.integer)) and isinstance(right, (Integral, np.integer)):
        if isinstance(left, bool) or isinstance(right, bool):
            raise TypeError("booleans do not have a supported additive merge.")
        return int(left) + int(right)
    if isinstance(left, (Real, np.floating)) and isinstance(right, (Real, np.floating)):
        result = float(left) + float(right)
        if not math.isfinite(result):
            raise OverflowError("additive floating payload produced a non-finite value.")
        return result
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if left.keys() != right.keys():
            raise ValueError("additive mapping payloads must have identical keys.")
        return {key: _additive_payload(left[key], right[key]) for key in left}
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return tuple(_additive_payload(a, b) for a, b in zip(left, right))
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [_additive_payload(a, b) for a, b in zip(left, right)]
    raise TypeError(
        "no supported additive merge for payload types %s and %s." % (type(left).__name__, type(right).__name__)
    )


def _additive_roundoff_bound(left: Any, right: Any) -> float:
    """Conservative absolute error introduced by one built-in floating addition."""

    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if np.issubdtype(left.dtype, np.inexact):
            epsilon = float(np.finfo(left.dtype).eps)
            magnitudes = np.abs(left.astype(np.complex128)) + np.abs(right.astype(np.complex128))
            return epsilon * float(np.max(magnitudes, initial=0.0))
        return 0.0
    if isinstance(left, (Integral, np.integer)) and isinstance(right, (Integral, np.integer)):
        return 0.0
    if isinstance(left, (Real, np.floating)) and isinstance(right, (Real, np.floating)):
        return math.ulp(1.0) * (abs(float(left)) + abs(float(right)))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return max((_additive_roundoff_bound(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return max((_additive_roundoff_bound(a, b) for a, b in zip(left, right)), default=0.0)
    return 0.0


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
    if len({row.proposal_id for row in rows}) != len(rows):
        raise ValueError("merged proposal ids must be unique.")
    if any(payload_fingerprint(row.payload) != row.payload_hash for row in rows):
        raise ValueError("a proposal payload changed after its evidence was created.")
    if any(not row.data_contribution_ids for row in rows):
        raise ValueError("merged proposals require explicit data contribution ids.")
    contribution_owners: dict[str, str] = {}
    for row in rows:
        for contribution_id in row.data_contribution_ids:
            previous = contribution_owners.setdefault(contribution_id, row.proposal_id)
            if previous != row.proposal_id:
                raise ValueError(
                    "data contribution %s appears in proposals %s and %s."
                    % (contribution_id, previous, row.proposal_id)
                )
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
            row.precision,
            row.invalidates,
            row.rollback_reference,
            row.surrogate_disclosed,
            row.staleness_semantics,
        )
        for row in rows
    }
    if len(comparable) != 1:
        raise ValueError("merged proposals must share their complete execution contract.")
    if anchor.staleness_semantics in {"corrected_statistical_approximation", "exact_rebase"} and len(
        {row.correction_fingerprint for row in rows}
    ) != len(rows):
        raise ValueError("corrected proposals must carry distinct source-bound correction evidence.")
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

    ordered = sorted(rows, key=lambda row: (row.data_contribution_ids, row.proposal_id))
    payload = ordered[0].payload
    built_in_addition = payload_merger is None and merge_law is MergeLaw.ADDITIVE
    merge_error_bound = (ordered[0].merge_error_bound or 0.0) if built_in_addition else math.nan
    for row in ordered[1:]:
        if built_in_addition:
            merge_error_bound += _additive_roundoff_bound(payload, row.payload)
        if not math.isnan(merge_error_bound):
            merge_error_bound += row.merge_error_bound or 0.0
        payload = merger(payload, row.payload)

    def optional_sum(name: str) -> float | None:
        values = [getattr(row, name) for row in rows]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError("%s must be present on every merged proposal or none." % name)
        return math.fsum(float(value) for value in values)

    def shared_optional(name: str) -> float | None:
        values = [getattr(row, name) for row in rows]
        if any(value is None for value in values):
            if all(value is None for value in values):
                return None
            raise ValueError("%s must be present on every merged proposal or none." % name)
        anchor_value = values[0]
        if any(value != anchor_value for value in values[1:]):
            raise ValueError("%s must agree across merged proposals." % name)
        return anchor_value

    combined_contributions = tuple(contribution_id for row in ordered for contribution_id in row.data_contribution_ids)
    combined_fingerprint = payload_fingerprint(
        tuple((row.data_contribution_ids, row.data_fingerprint) for row in ordered)
    )
    combined_rng = payload_fingerprint(tuple(row.rng_fingerprint for row in ordered))
    correction_fingerprint = (
        payload_fingerprint(tuple(row.correction_fingerprint for row in ordered))
        if anchor.staleness_semantics in {"corrected_statistical_approximation", "exact_rebase"}
        else None
    )
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
        observations=math.fsum(float(row.observations) for row in rows),
        tokens=sum(row.tokens for row in rows),
        responsibility_mass=math.fsum(float(row.responsibility_mass) for row in rows),
        local_objective_before=optional_sum("local_objective_before"),
        local_objective_after=optional_sum("local_objective_after"),
        global_objective_before=shared_optional("global_objective_before"),
        global_objective_after=shared_optional("global_objective_after"),
        predicted_gain=optional_sum("predicted_gain"),
        measured_gain=optional_sum("measured_gain"),
        gain_standard_error=optional_sum("gain_standard_error"),
        optimizer_steps=sum(row.optimizer_steps for row in rows),
        effective_batch_size=math.fsum(float(row.effective_batch_size) for row in rows),
        wall_time_seconds=max(float(row.wall_time_seconds) for row in rows),
        compute_units=math.fsum(float(row.compute_units) for row in rows),
        communication_bytes=sum(row.communication_bytes for row in rows),
        precision=anchor.precision,
        overflow_count=sum(row.overflow_count for row in rows),
        underflow_count=sum(row.underflow_count for row in rows),
        data_fingerprint=combined_fingerprint,
        data_contribution_ids=combined_contributions,
        ordering_fingerprint=payload_fingerprint(
            tuple((row.data_contribution_ids, row.ordering_fingerprint) for row in ordered)
        ),
        rng_fingerprint=combined_rng,
        invalidates=anchor.invalidates,
        rollback_reference=anchor.rollback_reference,
        surrogate_disclosed=anchor.surrogate_disclosed,
        staleness_semantics=anchor.staleness_semantics,
        correction_fingerprint=correction_fingerprint,
        merge_error_bound=None if math.isnan(merge_error_bound) else merge_error_bound,
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
