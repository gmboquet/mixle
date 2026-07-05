"""Local-first telemetry for model, placement, routing, and reasoning decisions.

Telemetry records typed decision events with features, choices, outcomes, tags,
and timestamps. These records support dashboards, auditing, and learned
orchestration policies.

Events are append-only, JSONL-compatible, and designed to avoid raw user
content. A process-global recorder makes ``record(...)`` convenient, while
tests and isolated workflows can use their own :class:`Telemetry` instance.
"""

from __future__ import annotations

from mixle.telemetry.core import (
    Event,
    Telemetry,
    get_default_recorder,
    record,
    set_default_recorder,
)
from mixle.telemetry.dashboard import dashboard, render_dashboard

__all__ = [
    "dashboard",
    "render_dashboard",
    "Event",
    "Telemetry",
    "record",
    "get_default_recorder",
    "set_default_recorder",
]
