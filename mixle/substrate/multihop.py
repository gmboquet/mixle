"""Planned multi-hop retrieval over substrate links and content.

Single-shot :func:`~mixle.substrate.retrieve.retrieve` answers "what is most relevant to this query".
But real questions chain: a symptom points to a document, the document names an entity, the entity
links to a record, the record was produced by a model artifact. :func:`multihop` walks that chain --
starting from the best matches, then expanding along two kinds of hop:

  * LINK hops: follow an item's ``links`` (explicit KG edges / lineage the substrate already stores),
  * CONTENT hops: re-query the substrate with the frontier item's own text, surfacing neighbors it is
    about (the "this reminds me of" step),

up to ``max_hops``, under a per-expansion budget. The result is a
:class:`HopChain`: the items found and the provenance path by which each was
reached. Branches that surface no new items stop early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem


@dataclass
class HopStep:
    """One item in the chain plus how it was reached: the provenance of a retrieval decision."""

    item: SubstrateItem
    depth: int  # 0 = seed match, 1 = one hop out, ...
    via: str  # 'seed' | 'link' | 'content'
    parent_id: str | None  # the item this was reached from (None for seeds)
    score: float = 0.0


@dataclass
class HopChain:
    """A multi-hop retrieval result: the items found and the evidence PATH to each."""

    query: str
    steps: list[HopStep] = field(default_factory=list)

    @property
    def items(self) -> list[SubstrateItem]:
        """Retrieved items in hop-chain order."""
        return [s.item for s in self.steps]

    def by_depth(self) -> dict[int, list[SubstrateItem]]:
        """Group retrieved items by hop depth."""
        out: dict[int, list[SubstrateItem]] = {}
        for s in self.steps:
            out.setdefault(s.depth, []).append(s.item)
        return out

    def max_depth(self) -> int:
        """Return the deepest hop reached by the chain."""
        return max((s.depth for s in self.steps), default=0)

    def path_to(self, item_id: str) -> list[SubstrateItem]:
        """The evidence chain from a seed to ``item_id`` -- the trace the reasoner cites."""
        by_id = {s.item.id: s for s in self.steps}
        chain: list[SubstrateItem] = []
        cur = by_id.get(item_id)
        seen: set[str] = set()
        while cur is not None and cur.item.id not in seen:
            seen.add(cur.item.id)
            chain.append(cur.item)
            cur = by_id.get(cur.parent_id) if cur.parent_id else None
        return list(reversed(chain))

    def provenance(self) -> list[dict[str, Any]]:
        """Return compact provenance records for every hop step."""
        return [
            {
                "id": s.item.id,
                "kind": s.item.kind,
                "depth": s.depth,
                "via": s.via,
                "parent": s.parent_id,
                "score": round(float(s.score), 4),
            }
            for s in self.steps
        ]

    def to_context(self, task: str | None = None, **assemble_kw: Any) -> Any:
        """Assemble the hop-chain items into a context packet."""
        from mixle.substrate.context import assemble_context

        shard = Substrate()
        for it in self.items:
            shard.put(it)
        return assemble_context(shard, task or self.query, **assemble_kw)

    def __len__(self) -> int:
        return len(self.steps)


def _require_count(value: Any, name: str) -> int:
    """Validate ``value`` is an exact, non-negative :class:`int` count (MXR-080-0252).

    Mirrors :func:`mixle.substrate.core._require_count`'s contract for this module's own "how many"
    parameters (``max_hops``, ``seeds``, ``branch``, ``max_items``): a plain, non-negative Python
    ``int``, never a ``bool`` (a ``bool`` is an ``int`` subclass and would otherwise silently mean 0 or
    1) and never a float (a fractional value would otherwise be silently truncated by a bare ``int()``,
    and even a whole-valued one like ``2.0`` is not the exact type this module promises callers).
    Defined locally rather than imported so this module's own input validation does not depend on
    another module's private helper.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


def multihop(
    substrate: Substrate,
    query: str,
    *,
    max_hops: int = 2,
    seeds: int = 3,
    branch: int = 2,
    max_items: int = 12,
    min_score: float = 0.0,
    scope: str | None = None,
    telemetry: Any = None,
) -> HopChain:
    """Chain typed hops from ``query`` across the substrate, recording the evidence path (see docstring).

    Args:
        max_hops: how many hops out from the seeds to expand.
        seeds: how many top matches to start from (depth 0).
        branch: how many neighbors to expand per frontier item per hop.
        max_items: overall cap on the chain size.
        min_score: relevance floor for SEED and CONTENT hops -- a match must score strictly above it
            to enter the chain (LINK hops are explicit edges and always followed). Keeps a fuzzy
            retriever from chaining on near-zero-similarity noise; raise it for a dense embedder.
        scope: restrict to a team/access scope.

    Raises:
        TypeError: ``max_hops``/``seeds``/``branch``/``max_items`` is not a plain, exact ``int``
            (bools and fractional floats included).
        ValueError: any of them is negative.
    """
    # MXR-080-0252: max_hops/seeds/branch/max_items are all "how many" counts, and must mean exactly
    # that -- validated eagerly, at the entry boundary, before ANY search/hop runs. Before this check,
    # a negative `branch` fed straight into `links[:branch]` and `search(k=branch + len(seen))` below,
    # getting ordinary (and silent) Python slice/negative-k semantics instead of a rejection, and a
    # fractional/bool count was silently truncated or coerced (`range(1, True + 1)` == `range(1, 2)`).
    max_hops = _require_count(max_hops, "max_hops")
    seeds = _require_count(seeds, "seeds")
    branch = _require_count(branch, "branch")
    max_items = _require_count(max_items, "max_items")

    chain: list[HopStep] = []
    seen: set[str] = set()

    # MXR-080-0252: the item cap applies to seeding too -- previously unchecked here, so `seeds` alone
    # could exceed `max_items` before a single hop ran. Gated the same way expansion below gates every
    # candidate (`... or len(chain) >= max_items: continue`), so a capped seed set and a capped
    # expansion mean the same thing rather than two different policies for the same budget.
    for it, sc in substrate.search(query, k=seeds, scope=scope):
        if it.id in seen or len(chain) >= max_items or sc <= min_score:
            continue
        chain.append(HopStep(it, 0, "seed", None, sc))
        seen.add(it.id)

    frontier = list(chain)
    for depth in range(1, max_hops + 1):
        if len(chain) >= max_items:
            break
        next_frontier: list[HopStep] = []
        # LINK hops first (explicit, high-confidence lineage edges) across the whole frontier, then
        # opportunistic CONTENT hops -- so a reliable edge is never pre-empted by a fuzzy neighbor.
        for step in frontier:
            for lid in step.item.links[:branch]:
                if lid in seen or len(chain) >= max_items:
                    continue
                linked = substrate.get(lid)
                if linked is not None and (scope is None or linked.scope == scope):
                    s = HopStep(linked, depth, "link", step.item.id, 1.0)
                    chain.append(s)
                    next_frontier.append(s)
                    seen.add(lid)
        for step in frontier:
            added = 0
            for it, sc in substrate.search(step.item.text, k=branch + len(seen), scope=scope):
                if it.id in seen or len(chain) >= max_items or sc <= min_score:
                    continue
                s = HopStep(it, depth, "content", step.item.id, sc)
                chain.append(s)
                next_frontier.append(s)
                seen.add(it.id)
                added += 1
                if added >= branch:
                    break
        if not next_frontier:  # diminishing returns: this hop surfaced nothing new -> stop
            break
        frontier = next_frontier

    result = HopChain(query=query, steps=chain)
    _emit(telemetry, result, max_hops)
    return result


def _emit(telemetry: Any, result: HopChain, max_hops: int) -> None:
    try:
        from mixle.telemetry import record

        rec = telemetry.record if telemetry is not None else record
        rec(
            "reason",
            features={"action": "multihop", "max_hops": max_hops},
            choice=[s.item.id for s in result.steps],
            outcome={"n": len(result.steps), "reached_depth": result.max_depth()},
        )
    except Exception:  # noqa: BLE001 - telemetry must never break retrieval
        pass
