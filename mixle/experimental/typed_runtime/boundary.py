"""Versioned, duplicate-safe messages across structured shard boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral, Real
from threading import RLock
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
    dependency_versions: tuple[tuple[str, int], ...]
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
        if any(not isinstance(value, str) or not value for value in identity) or self.source_shard == self.target_shard:
            raise ValueError("boundary messages require non-empty identity and distinct shards.")
        versions = (self.model_version, self.node_version, self.sequence_number)
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in versions):
            raise ValueError("boundary message versions and sequence must be non-negative.")
        dependencies = self.dependency_versions
        if isinstance(dependencies, Mapping):
            dependencies = tuple(dependencies.items())
            object.__setattr__(self, "dependency_versions", dependencies)
        if (
            not isinstance(dependencies, tuple)
            or not dependencies
            or any(
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not row[0]
                or isinstance(row[1], bool)
                or not isinstance(row[1], Integral)
                or row[1] < 0
                for row in dependencies
            )
            or len({node_id for node_id, _ in dependencies}) != len(dependencies)
        ):
            raise ValueError("boundary dependency_versions must be a non-empty unique non-negative version vector.")
        dependencies = tuple(sorted(dependencies))
        object.__setattr__(self, "dependency_versions", dependencies)
        dependency_map = dict(dependencies)
        if dependency_map.get(self.node_id) != self.node_version:
            raise ValueError("boundary node_version must equal its declared dependency version.")
        masses = (self.observations, self.responsibility_mass)
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0.0
            for value in masses
        ):
            raise ValueError("boundary message mass counters must be non-negative.")
        if isinstance(self.tokens, bool) or not isinstance(self.tokens, Integral) or self.tokens < 0:
            raise ValueError("boundary message tokens must be a non-negative integer.")
        if not isinstance(self.kind, BoundaryMessageKind):
            raise TypeError("boundary message kind must be BoundaryMessageKind.")
        if not isinstance(self.approximate, bool):
            raise TypeError("boundary message approximate flag must be boolean.")
        if self.precision is not None and (not isinstance(self.precision, str) or not self.precision):
            raise ValueError("boundary message precision must be a non-empty string when supplied.")
        if self.error_bound is not None and (
            isinstance(self.error_bound, bool)
            or not isinstance(self.error_bound, Real)
            or not math.isfinite(float(self.error_bound))
            or self.error_bound < 0.0
        ):
            raise ValueError("boundary message error_bound must be finite and non-negative.")
        if self.approximate and self.error_bound is None:
            raise ValueError("approximate boundary messages must declare an error_bound.")
        computed = payload_fingerprint(self.payload)
        if self.payload_hash and self.payload_hash != computed:
            raise ValueError("boundary message payload_hash does not match payload.")
        object.__setattr__(self, "payload_hash", computed)

    @property
    def stream_key(self) -> tuple[str, str, str, str, str, BoundaryMessageKind]:
        """Ordering domain for this message."""

        return (
            self.run_id,
            self.model_id,
            self.source_shard,
            self.target_shard,
            self.node_id,
            self.kind,
        )

    @property
    def dependency_version_map(self) -> dict[str, int]:
        """Return the declared dependency vector as a fresh mapping."""

        return dict(self.dependency_versions)

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
            "dependency_versions": dict(self.dependency_versions),
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
    run_id: str
    model_id: str
    node_id: str
    accepted: bool
    reason: str
    expected_sequence: int
    observed_sequence: int
    current_model_version: int
    current_node_version: int
    declared_dependency_versions: tuple[tuple[str, int], ...]
    current_dependency_versions: tuple[tuple[str, int], ...]
    payload_hash: str

    def __post_init__(self) -> None:
        identity = (self.message_id, self.run_id, self.model_id, self.node_id, self.reason, self.payload_hash)
        if any(not isinstance(value, str) or not value for value in identity):
            raise ValueError("boundary receipts require complete non-empty identity and reason fields.")
        if not isinstance(self.accepted, bool):
            raise TypeError("boundary receipt accepted must be boolean.")
        if self.accepted != (self.reason == "accepted"):
            raise ValueError("boundary receipt accepted flag must agree with its reason.")
        counters = (
            self.expected_sequence,
            self.observed_sequence,
            self.current_model_version,
            self.current_node_version,
        )
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in counters):
            raise ValueError("boundary receipt versions and sequences must be non-negative integers.")
        for name in ("declared_dependency_versions", "current_dependency_versions"):
            vector = getattr(self, name)
            if (
                not isinstance(vector, tuple)
                or not vector
                or any(
                    not isinstance(row, tuple)
                    or len(row) != 2
                    or not isinstance(row[0], str)
                    or not row[0]
                    or isinstance(row[1], bool)
                    or not isinstance(row[1], Integral)
                    or row[1] < 0
                    for row in vector
                )
                or len({node_id for node_id, _ in vector}) != len(vector)
            ):
                raise ValueError(f"boundary receipt {name} must be a complete version vector.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible boundary receipt."""

        return {
            "message_id": self.message_id,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "node_id": self.node_id,
            "accepted": self.accepted,
            "reason": self.reason,
            "expected_sequence": self.expected_sequence,
            "observed_sequence": self.observed_sequence,
            "current_model_version": self.current_model_version,
            "current_node_version": self.current_node_version,
            "declared_dependency_versions": dict(self.declared_dependency_versions),
            "current_dependency_versions": dict(self.current_dependency_versions),
            "payload_hash": self.payload_hash,
        }


class BoundaryInbox:
    """Validate versions, order, approximation, and exactly-once delivery."""

    def __init__(self, graph: UpdateGraph, *, run_id: str, model_id: str) -> None:
        if not isinstance(graph, UpdateGraph):
            raise TypeError("boundary inbox requires an UpdateGraph.")
        if not isinstance(run_id, str) or not run_id or not isinstance(model_id, str) or not model_id:
            raise ValueError("boundary inbox requires non-empty run_id and model_id.")
        self.graph = graph
        self.run_id = run_id
        self.model_id = model_id
        self._seen_messages: set[tuple[str, str, str]] = set()
        self._next_sequence: dict[tuple[str, str, str, str, str, BoundaryMessageKind], int] = {}
        self._receipts: list[BoundaryReceipt] = []
        self._lock = RLock()

    @property
    def receipts(self) -> tuple[BoundaryReceipt, ...]:
        """Return an immutable snapshot of receive decisions."""

        with self._lock:
            return tuple(self._receipts)

    def _required_dependencies(self, node_id: str) -> set[str]:
        return {node_id} | {edge.source_node for edge in self.graph.edges if edge.target_node == node_id}

    def receive(self, message: BoundaryMessage, versions: RuntimeVersions) -> BoundaryReceipt:
        """Validate and consume one message without applying its runtime payload."""

        if not isinstance(message, BoundaryMessage) or not isinstance(versions, RuntimeVersions):
            raise TypeError("receive requires a BoundaryMessage and RuntimeVersions.")
        with self._lock:
            node = self.graph.node(message.node_id)
            required_dependencies = self._required_dependencies(message.node_id)
            missing_runtime = sorted(required_dependencies - set(versions.node_versions))
            if missing_runtime:
                raise ValueError(
                    "runtime version vector is missing graph dependencies: %s" % ", ".join(missing_runtime)
                )
            current_dependencies = tuple(
                sorted((node_id, versions.node_versions[node_id]) for node_id in required_dependencies)
            )
            declared_dependencies = message.dependency_versions
            current_node_version = versions.node_versions[message.node_id]
            expected_sequence = self._next_sequence.get(message.stream_key, 0)
            message_identity = (message.run_id, message.model_id, message.message_id)
            reason = "accepted"
            try:
                current_hash = payload_fingerprint(message.payload)
            except (TypeError, ValueError):
                current_hash = "invalid"
                reason = "invalid-payload"
            if reason == "accepted":
                if message.run_id != self.run_id:
                    reason = "run-id-mismatch"
                elif message.model_id != self.model_id:
                    reason = "model-id-mismatch"
                elif current_hash != message.payload_hash:
                    reason = "payload-mutated"
                elif message_identity in self._seen_messages:
                    reason = "duplicate-message-id"
                elif message.model_version != versions.model_version:
                    reason = "model-version-mismatch"
                elif message.node_version != current_node_version:
                    reason = "node-version-mismatch"
                elif set(message.dependency_version_map) != required_dependencies:
                    reason = "dependency-vector-mismatch"
                elif declared_dependencies != current_dependencies:
                    reason = "dependency-version-mismatch"
                elif message.sequence_number != expected_sequence:
                    reason = "stale-sequence" if message.sequence_number < expected_sequence else "sequence-gap"
                elif message.approximate and node.contract.exact:
                    reason = "approximation-for-exact-node"

            accepted = reason == "accepted"
            if accepted:
                self._seen_messages.add(message_identity)
                self._next_sequence[message.stream_key] = expected_sequence + 1
            receipt = BoundaryReceipt(
                message.message_id,
                message.run_id,
                message.model_id,
                message.node_id,
                accepted,
                reason,
                expected_sequence,
                message.sequence_number,
                versions.model_version,
                current_node_version,
                declared_dependencies,
                current_dependencies,
                message.payload_hash,
            )
            self._receipts.append(receipt)
            return receipt

    def next_sequence(self, message: BoundaryMessage) -> int:
        """Return the next expected sequence number for a message's stream."""

        if not isinstance(message, BoundaryMessage):
            raise TypeError("next_sequence requires a BoundaryMessage.")
        with self._lock:
            return self._next_sequence.get(message.stream_key, 0)

    def as_dict(self) -> dict[str, Any]:
        """Return replay-relevant inbox state and receipts."""

        with self._lock:
            return {
                "run_id": self.run_id,
                "model_id": self.model_id,
                "seen_messages": [
                    {"run_id": run_id, "model_id": model_id, "message_id": message_id}
                    for run_id, model_id, message_id in sorted(self._seen_messages)
                ],
                "streams": [
                    {
                        "run_id": key[0],
                        "model_id": key[1],
                        "source_shard": key[2],
                        "target_shard": key[3],
                        "node_id": key[4],
                        "kind": key[5].value,
                        "next_sequence": sequence,
                    }
                    for key, sequence in sorted(self._next_sequence.items(), key=lambda row: tuple(map(str, row[0])))
                ],
                "receipts": [receipt.as_dict() for receipt in self._receipts],
            }


__all__ = ["BoundaryInbox", "BoundaryMessage", "BoundaryMessageKind", "BoundaryReceipt"]
