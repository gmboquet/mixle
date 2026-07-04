"""Dashboards over receipts (L2) -- the telemetry stream summarized into one auditable view.

Every decision the platform makes (fit, placement, route, escalation, pool job, reasoner step) lands in
the telemetry stream; :func:`dashboard` folds that stream into a receipt summary -- per-kind counts,
choice distributions, cost/latency totals, abstention rates -- and :func:`render_dashboard` renders it
as plain markdown, so "what has the system been deciding, and what did it cost" is one call, not a
spelunking session. Pure fold over :meth:`Telemetry.events`; no display dependencies.
"""

from __future__ import annotations

from typing import Any


def dashboard(telemetry: Any) -> dict[str, Any]:
    """Fold the telemetry stream into a receipt summary (see module docstring)."""
    kinds: dict[str, int] = {}
    choices: dict[str, dict[str, int]] = {}
    cost_total = 0.0
    cost_n = 0
    abstain: dict[str, int] = {"answer": 0, "abstain": 0}
    n = 0
    for ev in telemetry.events():
        n += 1
        kinds[ev.kind] = kinds.get(ev.kind, 0) + 1
        if ev.choice is not None:
            choices.setdefault(ev.kind, {})
            choices[ev.kind][str(ev.choice)] = choices[ev.kind].get(str(ev.choice), 0) + 1
            if ev.kind == "reason" and str(ev.choice) in abstain:
                abstain[str(ev.choice)] += 1
        outcome = ev.outcome or {}
        for key in ("cost", "latency", "spent"):
            v = outcome.get(key)
            if isinstance(v, (int, float)):
                cost_total += float(v)
                cost_n += 1
                break
    answered = abstain["answer"]
    total_reason = answered + abstain["abstain"]
    return {
        "n_events": n,
        "by_kind": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "choices": choices,
        "cost_total": round(cost_total, 4),
        "n_costed": cost_n,
        "abstention_rate": round(abstain["abstain"] / total_reason, 4) if total_reason else None,
    }


def render_dashboard(telemetry: Any) -> str:
    """The dashboard as plain markdown -- printable in a terminal, embeddable in a report."""
    d = dashboard(telemetry)
    lines = [
        "# telemetry receipts",
        f"- events: {d['n_events']}",
        f"- total recorded cost/latency: {d['cost_total']} over {d['n_costed']} event(s)",
    ]
    if d["abstention_rate"] is not None:
        lines.append(f"- reasoner abstention rate: {d['abstention_rate']:.1%}")
    if d["by_kind"]:
        lines.append("\n## events by kind")
        for k, v in d["by_kind"].items():
            lines.append(f"- {k}: {v}")
    for kind, dist in d["choices"].items():
        lines.append(f"\n## {kind} choices")
        for c, v in sorted(dist.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {c}: {v}")
    return "\n".join(lines)
