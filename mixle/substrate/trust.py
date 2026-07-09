"""``verify_lineage()`` -- check that an item's provenance chain is intact (trust / provenance, N1).

Every substrate item can point at the items it derives from -- ``links`` are KG/lineage edges (parent
documents, the model that produced an artifact, the trace it came from). A citation or a merge is only
trustworthy if that chain actually resolves: a link to an item that no longer exists is a dangling
provenance edge, and any claim resting on it is unverifiable. :func:`verify_lineage` walks an item's
ancestry, reporting which links resolve and which dangle, and how deep the intact chain goes.
:func:`audit_substrate` runs it over the whole store -- a knowledge-integrity sweep, the same "trust is
re-derivable, not asserted" discipline the factuality receipts apply to answers, applied here to the
knowledge itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mixle.substrate.core import Substrate


@dataclass
class LineageReport:
    """Whether an item's provenance chain resolves end to end -- and where it breaks if it doesn't."""

    item_id: str
    intact: bool
    n_links: int
    dangling: list[str] = field(default_factory=list)  # link ids that resolve to nothing
    depth: int = 0  # how many levels of ancestry resolved before a leaf or a break
    visited: int = 0  # distinct ancestors reached (cycle-safe count)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable lineage report."""
        return {
            "item_id": self.item_id,
            "intact": self.intact,
            "n_links": self.n_links,
            "dangling": self.dangling,
            "depth": self.depth,
            "visited": self.visited,
        }


def verify_lineage(substrate: Substrate, item_id: str, *, max_depth: int = 20) -> LineageReport:
    """Walk ``item_id``'s ancestry via ``links``, reporting dangling edges and intact depth (cycle-safe).

    An item is ``intact`` iff every lineage link, transitively, resolves to an item that exists. Cycles
    are handled (each id is visited once). ``max_depth`` bounds pathological chains. A missing root item
    yields ``intact=False`` with itself recorded as dangling."""
    root = substrate.get(item_id)
    if root is None:
        return LineageReport(item_id=item_id, intact=False, n_links=0, dangling=[item_id], depth=0, visited=0)

    dangling: list[str] = []
    seen: set[str] = {item_id}
    max_reached = 0
    # BFS over lineage edges, tracking depth; each frontier level is one ancestry generation
    frontier = [(item_id, 0)]
    total_links = 0
    while frontier:
        cur_id, depth = frontier.pop()
        if depth >= max_depth:
            continue
        cur = substrate.get(cur_id)
        if cur is None:
            continue
        for link in cur.links:
            total_links += 1
            child = substrate.get(link)
            if child is None:
                if link not in dangling:
                    dangling.append(link)
                continue
            if link not in seen:
                seen.add(link)
                max_reached = max(max_reached, depth + 1)
                frontier.append((link, depth + 1))

    return LineageReport(
        item_id=item_id,
        intact=not dangling,
        n_links=total_links,
        dangling=dangling,
        depth=max_reached,
        visited=len(seen),
    )


def audit_substrate(substrate: Substrate, *, scope: str | None = None) -> dict[str, Any]:
    """A knowledge-integrity sweep: how many items have intact lineage, and every invalid link named.

    Returns ``{n_items, n_intact, n_broken, broken: [{item_id, dangling}, ...]}`` -- the store's trust
    surface at a glance, so an invalid provenance edge surfaces as a finding rather than an unreported inconsistency."""
    items = substrate.all(scope=scope)
    broken: list[dict[str, Any]] = []
    for it in items:
        report = verify_lineage(substrate, it.id)
        if not report.intact:
            broken.append({"item_id": it.id, "dangling": report.dangling})
    return {
        "n_items": len(items),
        "n_intact": len(items) - len(broken),
        "n_broken": len(broken),
        "broken": broken,
    }
