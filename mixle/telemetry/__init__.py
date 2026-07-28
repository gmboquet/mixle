"""Local-first telemetry for model, placement, routing, and reasoning decisions.

Telemetry records typed decision events with features, choices, outcomes, tags,
and timestamps. These records support dashboards, auditing, and learned
orchestration policies. Only ``kind`` is a closed vocabulary; the log itself is
JSONL, and a rewrite preserves every row already in it, including rows another
writer put there.

**Nothing here filters what a caller records, and the log is an ordinary local
file.** ``features``, ``choice``, ``outcome`` and ``tags`` are serialized as
given: there is no per-kind schema, no allowlist, no redaction, no size limit,
no classification, no encryption, no access control and no retention policy. A
prompt, an answer, an API key or a customer record passed through the normal API
therefore lands verbatim in a long-lived file on disk. Recording only derived,
non-identifying quantities -- counts, sizes, durations, costs, method names --
is a convention this package expects of its callers, not a property it enforces
on their behalf; a caller handling sensitive material must minimize or redact
*before* calling :func:`record`, and must govern the log file's location,
permissions and lifetime itself.

A process-global recorder makes ``record(...)`` convenient, while tests and
isolated workflows can use their own :class:`Telemetry` instance.
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
