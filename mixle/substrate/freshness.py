"""Freshness checks for substrate knowledge items.

Models drift; so does knowledge. An item can go stale three ways, each independently checkable:

* **moved data** -- the item references a file (``payload.ref`` / ``provenance.source``/``path``) that no
  longer exists, or whose content hash no longer matches the one recorded at ingest;
* **superseded** -- a newer item shares the same lineage parent or declares ``supersedes`` on this one,
  so this item is no longer the current version of its knowledge;
* **aged out** -- older than the caller's ``max_age_s`` policy for its kind (a soft signal: age alone is
  a review trigger, not proof of wrongness -- the finding says so).

:func:`check_freshness` audits one item and names every signal;
:func:`freshness_report` sweeps a store. These checks help monitoring workflows
surface stale citations and moved artifacts before they affect downstream
answers.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem


@dataclass
class Freshness:
    """One item's freshness verdict: fresh iff no staleness signal fired; every signal named."""

    item_id: str
    fresh: bool
    signals: list[str] = field(default_factory=list)
    age_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable freshness verdict."""
        return {"item_id": self.item_id, "fresh": self.fresh, "signals": self.signals, "age_s": round(self.age_s, 1)}


def _referenced_path(item: SubstrateItem) -> str | None:
    for key in ("ref", "path"):
        v = item.payload.get(key)
        if isinstance(v, str) and v:
            return v
    for key in ("path", "source"):
        v = item.provenance.get(key)
        if isinstance(v, str) and ("/" in v or "\\" in v):
            return v
    return None


def content_hash(path: str) -> str | None:
    """The sha256 (first 32 hex) of a file's bytes, or None if unreadable -- record this at ingest."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:32]
    except OSError:
        return None


def check_freshness(
    substrate: Substrate,
    item_id: str,
    *,
    max_age_s: float | None = None,
    now: float | None = None,
) -> Freshness:
    """Audit one item for the three staleness signals (see module docstring). Missing item -> stale."""
    item = substrate.get(item_id)
    if item is None:
        return Freshness(item_id=item_id, fresh=False, signals=["missing: item no longer exists"])
    now = time.time() if now is None else float(now)
    age = max(0.0, now - float(item.created_at))
    signals: list[str] = []

    # moved / changed data
    ref = _referenced_path(item)
    if ref is not None:
        if not Path(ref).exists():
            signals.append(f"moved: referenced path {ref!r} no longer exists")
        else:
            recorded = item.provenance.get("content_hash")
            if recorded:
                current = content_hash(ref)
                if current is not None and current != recorded:
                    signals.append(f"changed: content hash of {ref!r} no longer matches ingest")

    # superseded by a newer item
    for other in substrate.all():
        if other.id == item.id:
            continue
        declares = other.provenance.get("supersedes")
        if declares == item.id:
            signals.append(f"superseded: item {other.id!r} declares supersedes={item.id!r}")
        elif (
            other.kind == item.kind
            and other.created_at > item.created_at
            and item.id in other.links
            and other.provenance.get("replaces") == item.id
        ):
            signals.append(f"superseded: newer linked item {other.id!r} replaces this one")

    # aged out (a review trigger, not proof of wrongness)
    if max_age_s is not None and age > float(max_age_s):
        signals.append(f"aged: {age:.0f}s old exceeds the {float(max_age_s):.0f}s policy (review, not proof)")

    return Freshness(item_id=item_id, fresh=not signals, signals=signals, age_s=age)


def freshness_report(
    substrate: Substrate,
    *,
    max_age_s: float | None = None,
    scope: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Sweep a store for stale knowledge: ``{n_items, n_fresh, n_stale, stale: [...]}`` -- the monitor feed."""
    items = substrate.all(scope=scope)
    stale: list[dict[str, Any]] = []
    for it in items:
        f = check_freshness(substrate, it.id, max_age_s=max_age_s, now=now)
        if not f.fresh:
            stale.append(f.as_dict())
    return {"n_items": len(items), "n_fresh": len(items) - len(stale), "n_stale": len(stale), "stale": stale}
