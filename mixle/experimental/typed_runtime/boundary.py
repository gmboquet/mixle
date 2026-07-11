"""Versioned, duplicate-safe messages across structured shard boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mixle.experimental.typed_runtime.graph import UpdateGraph
from mixle.experimental.typed_runtime.proposal import payload_fingerprint
from mixle.experimental.typed_runtime.transaction import RuntimeVersions


class BoundaryMessageKind(StrEnum):
    """Semantic payload crossing a model, graph, cache, or provider boundary."""

    SUFFICIENT_STATISTICS = "sufficient_statistics"
    FORWARD_MESSAGE = "forward_message"
    BACKWARD_MESSAGE = "backward_message"
    GRAPH_BOUNDARY = "graph_boundary"
    PROPOSAL_HANDOFF = "proposal_handoff"
    CACHE_BLOCK = "cache_block"


@dataclass(frozen=True)
class BoundaryMessage:
    """One immutable versioned shard-boundary payload."""

    message_id: str
    run_id: str
    model_id: str
    node_id: str
    source_shard: str
    target_shard: str
    model_version: int
    node_version: int
    target_dependency_version: int
    sequence_number: int
    kind: BoundaryMessageKind
    payload: Any
    observations: float = 0.0
    tokens: int = 0
    responsibility_mass: float = 0.0
    precision: str | None = None
    approximate: bool = False
    error_bound: float | None = None
    payload_hash: str = ""

    def __post_init__(self) -> None:
        identity = (
            self.message_id,
            self.run_id,
            self.model_id,
            self.node_id,
            self.source_shard,
            self.target_shard,
        )
        if any(not value for value in identity) or self.source_shard == self.target_shard:
            raise ValueError("boundary messages require non-empty identity and distinct shards.")
        versions = (self.model_version, self.node_version, self.target_dependency_version, self.sequence_number)
        if any(value < 0 for value in versions):
            raise ValueError("boundary message versions and sequence must be non-negative.")
        if self.observations < 0.0 or self.tokens < 0 or self.responsibility_mass < 0.0:
            raise ValueError("boundary message mass counters must be non-negative.")
        if self.error_bound is not None and (not math.isfinite(self.error_bound) or self.error_bound < 0.0):
            raise ValueError("boundary message error_bound must be finite and non-negative.")
        if self.approximate and self.error_bound is None:
            raise ValueError("approximate boundary messages must declare an error_bound.")
        computed = payload_fingerprint(self.payload)
        if self.payload_hash and self.payload_hash != computed:
            raise ValueError("boundary message payload_hash does not match payload.")
        object.__setattr__(self, "payload_hash", computed)

    @property
    def stream_key(self) -> tuple[str, str, str, BoundaryMessageKind]:
        """Ordering domain for this message."""

        return (self.source_shard, self.target_shard, self.node_id, self.kind)

    def as_dict(self) -> dict[str, Any]:
        """Return metadata and payload fingerprint, never the runtime payload."""

        return {
            "message_id": self.message_id,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "node_id": self.node_id,
            "source_shard": self.source_shard,
            "target_shard": self.target_shard,
            "model_version": self.model_version,
            "node_version": self.node_version,
            "target_dependency_version": self.target_dependency_version,
            "sequence_number": self.sequence_number,
            "kind": self.kind.value,
            "payload_hash": self.payload_hash,
            "observations": self.observations,
            "tokens": self.tokens,
            "responsibility_mass": self.responsibility_mass,
            "precision": self.precision,
            "approximate": self.approximate,
            "error_bound": self.error_bound,
        }


@dataclass(frozen=True)
class BoundaryReceipt:
    """Accepted/rejected boundary message with expected and observed versions."""

    message_id: str
    accepted: bool
    reason: str
    expected_sequence: int
    observed_sequence: int
    current_model_version: int
    current_node_version: int
    payload_hash: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible boundary receipt."""

        return {
            "message_id": self.message_id,
            "accepted": self.accepted,
            "reason": self.reason,
            "expected_sequence": self.expected_sequence,
            "observed_sequence": self.observed_sequence,
            "current_model_version": self.current_model_version,
            "current_node_version": self.current_node_version,
            "payload_hash": self.payload_hash,
        }


class BoundaryInbox:
    """Validate versions, order, approximation, and exactly-once delivery."""

    def __init__(self, graph: UpdateGraph) -> None:
        self.graph = graph
        self._seen_messages: set[str] = set()
        self._next_sequence: dict[tuple[str, str, str, BoundaryMessageKind], int] = {}
        self.receipts: list[BoundaryReceipt] = []

    def receive(self, message: BoundaryMessage, versions: RuntimeVersions) -> BoundaryReceipt:
        """Validate and consume one message without applying its runtime payload."""

        node = self.graph.node(message.node_id)
        expected_sequence = self._next_sequence.get(message.stream_key, 0)
        reason = "accepted"
        current_node_version = versions.node_versions[message.node_id]
        try:
            current_hash = payload_fingerprint(message.payload)
        except (TypeError, ValueError):
            current_hash = "invalid"
            reason = "invalid-payload"
        if reason == "accepted":
            if current_hash != message.payload_hash:
                reason = "payload-mutated"
            elif message.message_id in self._seen_messages:
                reason = "duplicate-message-id"
            elif message.model_version != versions.model_version:
                reason = "model-version-mismatch"
            elif message.node_version != current_node_version:
                reason = "node-version-mismatch"
            elif message.target_dependency_version != current_node_version:
                reason = "target-dependency-version-mismatch"
            elif message.sequence_number != expected_sequence:
                reason = "stale-sequence" if message.sequence_number < expected_sequence else "sequence-gap"
            elif message.approximate and node.contract.exact:
                reason = "approximation-for-exact-node"

        accepted = reason == "accepted"
        if accepted:
            self._seen_messages.add(message.message_id)
            self._next_sequence[message.stream_key] = expected_sequence + 1
        receipt = BoundaryReceipt(
            message.message_id,
            accepted,
            reason,
            expected_sequence,
            message.sequence_number,
            versions.model_version,
            current_node_version,
            message.payload_hash,
        )
        self.receipts.append(receipt)
        return receipt

    def next_sequence(self, message: BoundaryMessage) -> int:
        """Return the next expected sequence number for a message's stream."""

        return self._next_sequence.get(message.stream_key, 0)

    def as_dict(self) -> dict[str, Any]:
        """Return replay-relevant inbox state and receipts."""

        return {
            "seen_message_ids": sorted(self._seen_messages),
            "streams": [
                {
                    "source_shard": key[0],
                    "target_shard": key[1],
                    "node_id": key[2],
                    "kind": key[3].value,
                    "next_sequence": sequence,
                }
                for key, sequence in sorted(self._next_sequence.items(), key=lambda row: tuple(map(str, row[0])))
            ],
            "receipts": [receipt.as_dict() for receipt in self.receipts],
        }


__all__ = ["BoundaryInbox", "BoundaryMessage", "BoundaryMessageKind", "BoundaryReceipt"]
