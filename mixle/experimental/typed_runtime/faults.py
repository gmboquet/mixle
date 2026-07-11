"""Deterministic boundary fault injection for distributed correctness tests."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from mixle.experimental.typed_runtime.boundary import BoundaryMessage


class FaultKind(StrEnum):
    """Injected delivery or worker failure."""

    DROP = "drop"
    DUPLICATE = "duplicate"
    DELAY = "delay"
    CORRUPT = "corrupt"
    STALE_VERSION = "stale_version"
    WORKER_LOSS = "worker_loss"


@dataclass(frozen=True)
class FaultEvent:
    """One scripted fault targeting a message id."""

    message_id: str
    kind: FaultKind
    release_step: int | None = None
    version_delta: int = 1

    def __post_init__(self) -> None:
        if not self.message_id:
            raise ValueError("fault event message_id must be non-empty.")
        if self.kind is FaultKind.DELAY and self.release_step is None:
            raise ValueError("delay faults require release_step.")
        if self.release_step is not None and self.release_step < 0:
            raise ValueError("release_step must be non-negative.")
        if self.version_delta < 1:
            raise ValueError("version_delta must be positive.")


@dataclass(frozen=True)
class FaultInjectionReceipt:
    """Scripted fault action and number of resulting deliveries."""

    message_id: str
    kind: FaultKind | None
    step: int
    emitted_messages: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible fault receipt."""

        return {
            "message_id": self.message_id,
            "kind": self.kind.value if self.kind is not None else None,
            "step": self.step,
            "emitted_messages": self.emitted_messages,
            "reason": self.reason,
        }


def _corrupt_payload(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        result = value.copy()
        if not result.size:
            raise ValueError("cannot corrupt an empty array payload.")
        result.reshape(-1)[0] += 1
        return result
    if isinstance(value, dict):
        if not value:
            raise ValueError("cannot corrupt an empty mapping payload.")
        result = copy.deepcopy(value)
        key = sorted(result, key=repr)[0]
        result[key] = _corrupt_payload(result[key])
        return result
    if isinstance(value, list):
        if not value:
            raise ValueError("cannot corrupt an empty list payload.")
        result = copy.deepcopy(value)
        result[0] = _corrupt_payload(result[0])
        return result
    if isinstance(value, tuple):
        if not value:
            raise ValueError("cannot corrupt an empty tuple payload.")
        return (_corrupt_payload(value[0]),) + value[1:]
    if isinstance(value, (int, float, np.number)):
        return value + 1
    raise TypeError("cannot inject deterministic corruption into payload type %s." % type(value).__name__)


class BoundaryFaultInjector:
    """Apply each scripted message fault at most once and retain delayed payloads."""

    def __init__(self, events: tuple[FaultEvent, ...]) -> None:
        if len({event.message_id for event in events}) != len(events):
            raise ValueError("fault script may contain only one event per message id.")
        self._events = {event.message_id: event for event in events}
        self._consumed: set[str] = set()
        self._delayed: list[tuple[int, BoundaryMessage]] = []
        self._dead_shards: set[str] = set()
        self.receipts: list[FaultInjectionReceipt] = []

    def intercept(self, message: BoundaryMessage, *, step: int) -> tuple[BoundaryMessage, ...]:
        """Return zero, one, or two messages after applying a scripted fault."""

        if step < 0:
            raise ValueError("fault injection step must be non-negative.")
        if message.source_shard in self._dead_shards:
            receipt = FaultInjectionReceipt(
                message.message_id,
                FaultKind.WORKER_LOSS,
                step,
                0,
                "source-shard-unavailable",
            )
            self.receipts.append(receipt)
            return ()
        event = self._events.get(message.message_id)
        if event is None or message.message_id in self._consumed:
            receipt = FaultInjectionReceipt(message.message_id, None, step, 1, "delivered")
            self.receipts.append(receipt)
            return (message,)
        self._consumed.add(message.message_id)

        emitted: tuple[BoundaryMessage, ...]
        reason: str
        if event.kind is FaultKind.DROP:
            emitted, reason = (), "scripted-drop"
        elif event.kind is FaultKind.DUPLICATE:
            emitted, reason = (message, copy.deepcopy(message)), "scripted-duplicate"
        elif event.kind is FaultKind.DELAY:
            if event.release_step is None or event.release_step <= step:
                raise ValueError("delay release_step must be greater than interception step.")
            self._delayed.append((event.release_step, copy.deepcopy(message)))
            emitted, reason = (), "queued-until-step-%d" % event.release_step
        elif event.kind is FaultKind.CORRUPT:
            corrupted = copy.deepcopy(message)
            object.__setattr__(corrupted, "payload", _corrupt_payload(corrupted.payload))
            emitted, reason = (corrupted,), "payload-corrupted-after-hash"
        elif event.kind is FaultKind.STALE_VERSION:
            stale = copy.deepcopy(message)
            object.__setattr__(stale, "model_version", max(0, stale.model_version - event.version_delta))
            emitted, reason = (stale,), "model-version-rewound"
        else:
            self._dead_shards.add(message.source_shard)
            emitted, reason = (), "source-shard-marked-unavailable"
        receipt = FaultInjectionReceipt(message.message_id, event.kind, step, len(emitted), reason)
        self.receipts.append(receipt)
        return emitted

    def release(self, *, step: int) -> tuple[BoundaryMessage, ...]:
        """Release delayed messages whose scripted step has arrived."""

        ready = [row for row in self._delayed if row[0] <= step]
        self._delayed = [row for row in self._delayed if row[0] > step]
        return tuple(message for _, message in sorted(ready, key=lambda row: (row[0], row[1].message_id)))

    def recover(self, shard_id: str) -> None:
        """Mark a failed source shard available for retried messages."""

        self._dead_shards.discard(shard_id)

    def as_dict(self) -> dict[str, Any]:
        """Return replay-relevant injector state and receipts."""

        return {
            "consumed_event_message_ids": sorted(self._consumed),
            "dead_shards": sorted(self._dead_shards),
            "delayed_message_ids": [message.message_id for _, message in self._delayed],
            "receipts": [receipt.as_dict() for receipt in self.receipts],
        }


__all__ = ["BoundaryFaultInjector", "FaultEvent", "FaultInjectionReceipt", "FaultKind"]
