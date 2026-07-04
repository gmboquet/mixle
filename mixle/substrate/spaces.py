"""``Space`` -- a team-scoped view over the substrate, with explicit publish (multi-team sharing, P1).

The substrate tags every item with an access ``scope`` ("local", a team id, or a shared scope like
"public"), but the raw store filters on ONE scope at a time. A team's real visibility is a *union*: its
own items PLUS whatever has been shared into a common scope, and never another team's private items.
:class:`Space` is that view. Construct one for a team and it answers ``retrieve`` / ``all`` over exactly
the team's visible set, so two teams querying the same substrate see different, correctly-isolated
knowledge.

Sharing is EXPLICIT: nothing crosses a scope boundary until someone :func:`publish` es it. Publishing
re-scopes an item into a shared scope and stamps its provenance with who published it and from where --
an audit trail, so a shared item is always traceable to the act that shared it. This is the P1 seed
(spaces + ACL scopes + explicit publish); versioned merge and org governance (P2/P3) build on it.
"""

from __future__ import annotations

from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem

PUBLIC = "public"


def visible_scopes(team: str, *, shared: tuple[str, ...] = (PUBLIC,)) -> set[str]:
    """The scopes a team may read: its own id plus the shared scopes (never another team's private one)."""
    return {team, *shared}


def publish(
    substrate: Substrate,
    ids: list[str],
    *,
    to: str = PUBLIC,
    by: str | None = None,
    from_scope: str | None = None,
) -> list[str]:
    """Share items into a common scope -- the only way knowledge crosses a team boundary (audited).

    Re-scopes each item in ``ids`` to ``to`` and records ``published_by`` / ``published_from`` in its
    provenance. Returns the ids actually published (missing ids are skipped). ``from_scope``, if given,
    guards that only items currently in that scope are published (an ACL check the caller can enforce)."""
    published: list[str] = []
    for item_id in ids:
        item = substrate.get(item_id)
        if item is None:
            continue
        if from_scope is not None and item.scope != from_scope:
            continue
        prov = dict(item.provenance)
        prov["published_by"] = by
        prov["published_from"] = item.scope
        substrate.put(
            SubstrateItem(
                id=item.id,
                kind=item.kind,
                text=item.text,
                payload=item.payload,
                provenance=prov,
                tags=item.tags,
                links=item.links,
                scope=to,
                created_at=item.created_at,
            )
        )
        published.append(item_id)
    return published


class Space:
    """A team's scoped view over a shared substrate: its own items plus what has been shared to it."""

    def __init__(self, substrate: Substrate, team: str, *, shared: tuple[str, ...] = (PUBLIC,)) -> None:
        self.substrate = substrate
        self.team = team
        self.shared = shared

    @property
    def scopes(self) -> set[str]:
        return visible_scopes(self.team, shared=self.shared)

    def _visible_shard(self) -> Substrate:
        """A substrate of only the items this team may see -- the isolation boundary, made concrete."""
        shard = Substrate()
        for item in self.substrate.all():
            if item.scope in self.scopes:
                shard.put(item)
        return shard

    def all(self, *, kind: str | None = None) -> list[SubstrateItem]:
        """Every visible item (optionally of one kind) -- never another team's private knowledge."""
        return [i for i in self.substrate.all(kind=kind) if i.scope in self.scopes]

    def add(self, *, scope: str | None = None, **kw: Any) -> str:
        """Add an item to this team's own scope by default (pass ``scope=PUBLIC`` to share immediately)."""
        return self.substrate.add(scope=scope or self.team, **kw)

    def retrieve(self, query: str, *, k: int = 8, **kw: Any) -> Any:
        """Retrieve over exactly the team's visible set (own scope ∪ shared), with cross-kind diversity."""
        from mixle.substrate.retrieve import retrieve

        return retrieve(self._visible_shard(), query, k=k, **kw)

    def publish(self, ids: list[str], *, to: str = PUBLIC, by: str | None = None) -> list[str]:
        """Share this team's items into a common scope (audited). Only own-scope items are publishable."""
        return publish(self.substrate, ids, to=to, by=by or self.team, from_scope=self.team)
