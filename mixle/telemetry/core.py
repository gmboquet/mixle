"""Event schema and local-first recorder for platform telemetry.

An :class:`Event` is a typed, timestamped decision record containing the
decision kind, the features used, the selected choice, and an optional outcome.
A :class:`Telemetry` recorder buffers events and can append them to a JSONL log
for dashboards or learned policies.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# The decision kinds the ecosystem records. Extend freely -- consumers filter by kind. Kept as plain
# strings so a new decision type needs no code change here.
EVENT_KINDS = (
    "fit",  # a model was fit: features about data/model, choice = the estimation plan, outcome = ll/certificate
    "placement",  # a block ran local vs pool: features, choice, outcome = latency/cost
    "route",  # a request was routed across model versions/tiers: choice + realized cost/quality
    "escalation",  # a decide-or-escalate call: confident-local vs escalate-to-teacher, outcome = correct?
    "context",  # a ContextPacket was assembled: budget, what was included, outcome = answer quality
    "reason",  # a reasoning action (retrieve/compute/simulate/create/delegate/escalate) + its gain
    "pool_job",  # a PoolJob lifecycle event: submit/start/finish, cost, duration
    "drift",  # a drift alarm on a served artifact
    "em_round",  # one worker-rank's E/M-step timing+bytes+accumulator size for one distributed EM round
)

_TIME_KEY = "_wall_time"  # tests inject a deterministic clock via record(when=...)


@dataclass
class Event:
    """One typed, timestamped decision record. Features/choice describe the decision; outcome scores it."""

    kind: str
    features: dict[str, Any] = field(default_factory=dict)  # what the decision was made from
    choice: Any = None  # what was decided (a method name, a placement, a route, an action)
    outcome: dict[str, Any] = field(default_factory=dict)  # how it turned out (filled now or later)
    tags: dict[str, str] = field(default_factory=dict)  # scope/task/version labels for filtering
    ts: float = 0.0  # wall time; set by the recorder

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {self.kind!r}; expected one of {EVENT_KINDS}")

    def as_row(self) -> dict[str, Any]:
        """Return this event as a JSON-serializable row."""
        return asdict(self)


class Telemetry:
    """A local-first event recorder: buffer in memory, append to a JSONL log, read back for training.

    ``record(kind, features=..., choice=..., outcome=..., tags=...)`` appends one :class:`Event`.
    ``events(kind=...)`` yields the buffer (optionally filtered). ``training_rows(kind)`` yields the
    ``(features, choice, outcome)`` triples the learned-orchestration models consume.
    """

    def __init__(self, path: str | None = None, *, flush_every: int = 1) -> None:
        self.path = Path(path) if path is not None else None
        self.flush_every = int(flush_every)
        self._buffer: list[Event] = []
        self._unflushed: list[Event] = []
        self._lock = threading.Lock()
        self._clock = 0.0  # monotonic fallback clock when no wall time is supplied (deterministic)
        if self.path is not None and self.path.exists():
            self._load()

    def record(
        self,
        kind: str,
        *,
        features: dict[str, Any] | None = None,
        choice: Any = None,
        outcome: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        when: float | None = None,
    ) -> Event:
        """Record one decision event; returns it (mutate ``.outcome`` later to close the loop)."""
        with self._lock:
            self._clock += 1.0
            ev = Event(
                kind=kind,
                features=dict(features or {}),
                choice=choice,
                outcome=dict(outcome or {}),
                tags=dict(tags or {}),
                ts=float(when) if when is not None else self._clock,
            )
            self._buffer.append(ev)
            self._unflushed.append(ev)
            if self.path is not None and len(self._unflushed) >= self.flush_every:
                self._flush_locked()
        return ev

    def events(self, *, kind: str | None = None) -> Iterator[Event]:
        """Yield recorded events, optionally filtered by kind."""
        for ev in list(self._buffer):
            if kind is None or ev.kind == kind:
                yield ev

    def training_rows(self, kind: str) -> list[tuple[dict[str, Any], Any, dict[str, Any]]]:
        """The ``(features, choice, outcome)`` triples for a decision kind."""
        return [(ev.features, ev.choice, ev.outcome) for ev in self.events(kind=kind)]

    def __len__(self) -> int:
        return len(self._buffer)

    def flush(self) -> None:
        """Flush any buffered events to the configured JSONL log."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self.path is None or not self._unflushed:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            for ev in self._unflushed:
                f.write(json.dumps(ev.as_row()) + "\n")
        self._unflushed.clear()

    def _load(self) -> None:
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    self._buffer.append(Event(**row))


# --- a process-global default recorder so record(...) is a one-liner anywhere in the stack ---------

_DEFAULT: Telemetry | None = None
_DEFAULT_LOCK = threading.Lock()


def get_default_recorder() -> Telemetry:
    """The process-global recorder (a no-path in-memory buffer until one is configured)."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = Telemetry()
        return _DEFAULT


def set_default_recorder(recorder: Telemetry | None) -> None:
    """Install (or clear) the process-global recorder -- e.g. point it at the user's telemetry log."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        _DEFAULT = recorder


def record(kind: str, **kw: Any) -> Event:
    """Record an event on the process-global recorder (see :meth:`Telemetry.record`)."""
    return get_default_recorder().record(kind, **kw)
