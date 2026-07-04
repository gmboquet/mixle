"""Telemetry -- the local-first event log that trains the ecosystem's learned orchestration.

Every decision the platform makes (a fit and its estimation plan, a placement local-vs-pool, a route
across model versions, a context assembly, a reasoning action, an escalation) is an EVENT. Recorded
from day one, these events become the training data for the learned-orchestration models (workstream
J): the static policies are the teachers, the learned routers/placers/schedulers earn traffic only
when receipted never-worse. This module is the schema + the local buffer; the sink to a shared
control plane is opt-in (workstream L2 / P).

Design constraints: PII-free by construction (events carry decision FEATURES and OUTCOMES, never raw
user content), append-only, JSONL on disk, and cheap enough to call in a hot loop. A process-global
default recorder makes ``record(...)`` a one-liner anywhere in the stack; tests and isolated runs use
their own :class:`Telemetry` instance.
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
