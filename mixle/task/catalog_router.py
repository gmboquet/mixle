"""IC-10 tool/model catalog -> :class:`~mixle.task.router.Router` wiring (M3-core).

Every routable capability -- a physics forward, an economic model, a climate projection, an
external domain model (IC-7) -- registers with the same uniform shape (:class:`CatalogEntry`, IC-10)
so a decomposer/router never special-cases a tool family. :func:`build_catalog_router` turns a flat
list of entries into one calibrated :class:`~mixle.task.router.Router` tier per entry (cheapest first),
falling through to a frontier/teacher callable exactly the way :class:`~mixle.task.cascade.Cascade`
already escalates uncertain calls.

Local convention (not part of the frozen IC-10 shape, documented here because M3 owns the wiring):
``CatalogEntry.schema`` is a plain ``dict[str, Any]`` (IC-10 imposes no further shape on it), so a
registrant MAY stash two optional, well-known keys on it: ``"output"`` (the JSON-schema of what this
entry produces, used for schema-compatibility matching in :mod:`mixle.task.knowledge_routing`) and
``"invoke"`` (a ``Callable[[dict], Any]`` that actually runs the tool/model given a gap dict). Neither
key is required; an entry with no ``"invoke"`` is still routable but can never produce a canonical
result on its own -- see :mod:`mixle.task.knowledge_routing` for how an unresolved gap is handled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mixle.task.calibrate import ESCALATE
from mixle.task.router import Router

__all__ = ["CatalogEntry", "build_catalog_router"]


@dataclass(frozen=True)
class CatalogEntry:
    """A routable capability: identity, its JSON `schema`, owning subsystem, `cost`, prior
    `reliability`, and an optional verifier (the IC-10 tuple).

    IC-10 (``notes/exec/contracts.md``) freezes ``verifier`` as ``str | None`` -- the *name* of the
    IC-6 verifier kind that gates this entry's output. M3's own algorithm (step 5: "attach each
    verifier; verified item ids resolve gaps") needs to actually CALL a verifier, not just look one up
    by name, so this field is typed ``Any | None`` here and accepts either an IC-10-style string tag
    or a live IC-6 ``Verifier``-shaped object (anything exposing ``.verify(claim, context)``) -- a
    thin, additive shim over the frozen contract, not a rename of it. Field names/order/defaults are
    otherwise identical to the frozen stub.
    """

    id: str
    schema: dict[str, Any]
    owner: str  # "physics" | "economic" | "climate" | "external" | ...
    cost: float = 0.0
    reliability: float = 1.0
    verifier: Any | None = None


class _ReliabilityGate:
    """Adapts one :class:`CatalogEntry` into a :class:`~mixle.task.router.Router` tier.

    ``decide(x)`` answers with this entry's own id when ``x`` names a domain this entry does not
    disagree with and the entry's prior reliability clears ``min_reliability``; otherwise it escalates
    to the next (costlier) tier, exactly like a calibrated task model's ``decide``.
    """

    def __init__(self, entry: CatalogEntry, *, min_reliability: float = 0.5) -> None:
        self.entry = entry
        self.min_reliability = float(min_reliability)

    def decide(self, x: Any) -> Any:
        domain = x.get("domain") if isinstance(x, dict) else None
        if domain is not None and domain != self.entry.owner:
            return ESCALATE
        if self.entry.reliability < self.min_reliability:
            return ESCALATE
        return self.entry.id


def build_catalog_router(catalog: list[CatalogEntry], teacher: Any) -> Router:
    """Build one ``(id, adapter, cost)`` tier per catalog entry, ascending cost, teacher last.

    ``teacher`` is the frontier fallback -- a BATCHED callable (``texts -> [labels]``, the same shape
    :class:`~mixle.task.router.Router`'s final tier already expects). Its per-request cost is not
    supplied by the caller (IC-10 entries carry the only costs this function knows about), so it is
    set just above the priciest registered entry -- the frontier is a last resort, never competitive
    with a cheaper registered tool on cost alone.
    """
    ordered = sorted(catalog, key=lambda e: (e.cost, e.id))
    tiers: list[tuple[str, Any, float]] = [(entry.id, _ReliabilityGate(entry), float(entry.cost)) for entry in ordered]
    frontier_cost = max((e.cost for e in ordered), default=0.0) + 1.0
    tiers.append(("frontier", teacher, frontier_cost))
    return Router(tiers)
