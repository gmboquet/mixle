"""Summaries and markdown rendering for telemetry streams.

:func:`dashboard` folds telemetry events into per-kind counts, choice
distributions, per-unit cost and latency totals, and abstention rates.
:func:`render_dashboard` renders the same summary as plain markdown. The
implementation is a pure fold over :meth:`Telemetry.events` and has no display
dependencies.

Each aggregated quantity is kept in its own unit. Dollars (``outcome["cost"]``
or ``outcome["spent"]``) and seconds (``outcome["latency"]``) are separate
totals: adding a dollar to a second produces a number with no meaning, and
folding them into one accumulator also silently dropped whichever the event
carried second.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

# Aliases for one quantity, most specific first: an event names its dollar cost either way, so at
# most one per event is counted (summing both would double-count a single spend).
_COST_KEYS = ("cost", "spent")
_LATENCY_KEYS = ("latency",)


def _amount(value: Any) -> float | None:
    """``value`` as a finite non-negative float, or ``None`` when it cannot be aggregated.

    Booleans are excluded even though ``bool`` is an ``int``: ``cost=True`` is a flag someone put in
    the wrong field, not one dollar. Non-finite values are excluded because a single NaN turns the
    whole total into NaN, and negative values because a realized cost or latency is never a credit.
    Anything rejected here is *counted* as unaggregatable rather than quietly skipped.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    amount = float(value)
    if not math.isfinite(amount) or amount < 0.0:
        return None
    return amount


def dashboard(telemetry: Any) -> dict[str, Any]:
    """Fold the telemetry stream into a receipt summary (see module docstring)."""
    kinds: dict[str, int] = {}
    choices: dict[str, dict[str, int]] = {}
    totals = {"cost": 0.0, "latency": 0.0}
    counts = {"cost": 0, "latency": 0}
    n_unaggregatable = 0
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
        for metric, keys in (("cost", _COST_KEYS), ("latency", _LATENCY_KEYS)):
            for key in keys:
                if key not in outcome:
                    continue
                amount = _amount(outcome[key])
                if amount is None:
                    n_unaggregatable += 1
                else:
                    totals[metric] += amount
                    counts[metric] += 1
                break
    answered = abstain["answer"]
    total_reason = answered + abstain["abstain"]
    return {
        "n_events": n,
        "by_kind": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "choices": choices,
        "cost_total": round(totals["cost"], 4),
        "n_costed": counts["cost"],
        "latency_total": round(totals["latency"], 4),
        "n_latency": counts["latency"],
        "n_unaggregatable": n_unaggregatable,
        "abstention_rate": round(abstain["abstain"] / total_reason, 4) if total_reason else None,
    }


def render_dashboard(telemetry: Any) -> str:
    """The dashboard as plain markdown -- printable in a terminal, embeddable in a report."""
    d = dashboard(telemetry)
    lines = [
        "# telemetry receipts",
        f"- events: {d['n_events']}",
        f"- total recorded cost: {d['cost_total']} over {d['n_costed']} event(s)",
        f"- total recorded latency: {d['latency_total']} over {d['n_latency']} event(s)",
    ]
    if d["n_unaggregatable"]:
        lines.append(f"- unaggregatable cost/latency values skipped: {d['n_unaggregatable']}")
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
