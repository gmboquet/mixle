"""Team-scoped substrate views with explicit publishing.

The substrate tags every item with an access ``scope`` ("local", a team id, or a shared scope like
"public"), but the raw store filters on one scope at a time. A team's real visibility is a *union*: its
own items plus whatever has been shared into a common scope, and never another team's private items.
:class:`Space` is that view. Construct one for a team and it answers ``retrieve`` / ``all`` over exactly
the team's visible set, so two teams querying the same substrate see different, correctly-isolated
knowledge.

Sharing is explicit: nothing crosses a scope boundary until someone calls
:func:`publish`. Publishing re-scopes an item into a shared scope and records
who published it and from where.
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
    """Share items into a common scope.

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
        # versioned + audited: every share bumps the version and appends to the history, so a re-published
        # item never silently overwrites its predecessor -- the prior state is always recoverable (P2).
        version = int(prov.get("version", 0)) + 1
        history = list(prov.get("version_history", []))
        history.append({"version": version, "published_by": by, "published_from": item.scope, "to": to})
        prov["version"] = version
        prov["version_history"] = history
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


def version_of(item: Any) -> int:
    """The share version of an item (0 if never published) -- a monotonic counter bumped by each publish."""
    prov = getattr(item, "provenance", {}) or {}
    return int(prov.get("version", 0))


def history(substrate: Substrate, item_id: str) -> list[dict[str, Any]]:
    """The full publish history of an item: every version with who shared it, from where, to where."""
    item = substrate.get(item_id)
    if item is None:
        return []
    return list(item.provenance.get("version_history", []))


def merge_versions(
    substrate: Substrate, keep_id: str, other_id: str, *, by: str | None = None, prefer: str = "latest"
) -> str | None:
    """Reconcile two versions of the same knowledge into one, keeping full lineage (no silent loss, P2).

    Merges ``other_id`` into ``keep_id``: unions tags and links, keeps the text/payload of whichever has
    the higher version (``prefer="latest"``) or of ``keep`` (``prefer="keep"``), bumps the surviving
    item's version, records BOTH parents in the history, and removes the merged-away item. Returns the
    surviving id, or None if either is missing. Two teams that independently edited a shared item can be
    reconciled without either edit vanishing unrecorded."""
    keep = substrate.get(keep_id)
    other = substrate.get(other_id)
    if keep is None or other is None:
        return None

    take_other = prefer == "latest" and version_of(other) > version_of(keep)
    winner_text = other.text if take_other else keep.text
    winner_payload = other.payload if take_other else keep.payload

    prov = dict(keep.provenance)
    version = max(version_of(keep), version_of(other)) + 1
    history_list = list(prov.get("version_history", []))
    history_list.append(
        {
            "version": version,
            "merged_by": by,
            "merged_from": other_id,
            "parents": [
                {"id": keep_id, "version": version_of(keep)},
                {"id": other_id, "version": version_of(other)},
            ],
        }
    )
    prov["version"] = version
    prov["version_history"] = history_list
    prov["merged"] = True

    substrate.put(
        SubstrateItem(
            id=keep.id,
            kind=keep.kind,
            text=winner_text,
            payload=winner_payload,
            provenance=prov,
            tags=sorted(set(keep.tags) | set(other.tags)),
            links=sorted(set(keep.links) | set(other.links)),
            scope=keep.scope,
            created_at=keep.created_at,
        )
    )
    substrate.remove(other_id)
    return keep_id


class Space:
    """A team's scoped view over a shared substrate: its own items plus what has been shared to it."""

    def __init__(self, substrate: Substrate, team: str, *, shared: tuple[str, ...] = (PUBLIC,)) -> None:
        self.substrate = substrate
        self.team = team
        self.shared = shared

    @property
    def scopes(self) -> set[str]:
        """Scopes visible to this team space."""
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
