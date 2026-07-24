"""Team-scoped substrate views with explicit publishing, centralized access control, and immutable history.

The substrate tags every item with an access ``scope`` ("local", a team id, or a shared scope like
"public"), but the raw store filters on one scope at a time. A team's real visibility is a *union*: its
own items plus whatever has been shared into a common scope, and never another team's private items.
:class:`Space` is that view. Construct one for a team and it answers ``retrieve`` / ``all`` over exactly
the team's visible set, so two teams querying the same substrate see different, correctly-isolated
knowledge.

Sharing is explicit: nothing crosses a scope boundary until someone calls
:func:`publish`. Publishing re-scopes an item into a shared scope and records
who published it and from where.

Isolation is enforced, not caller convention (MXR-080-0264). Every read, write, and publish is checked
against an :class:`AccessPolicy` -- the single, centralized decision of whether a principal may touch a
scope. A principal always has full access to its own home scope (the scope sharing its name) and to
:data:`PUBLIC`, the explicit sharing commons; every OTHER scope, in particular another team's private
scope, is denied unless :meth:`AccessPolicy.grant` says otherwise. :class:`Space` validates ``shared``
against the policy at construction (and again, live, on every read, so a later revocation takes effect
immediately); :func:`publish` checks both the item's actual stored scope and the target; and
:func:`merge_versions` requires the merging principal to hold write access to both sides. None of these
trust a caller-supplied scope parameter as if it were already a validated permission.

Every publish and merge also persists an immutable, content-addressed snapshot of the state it is about
to overwrite or delete (MXR-080-0265/0266). ``version_history`` entries reference those snapshots by
digest; :func:`revision` recovers the complete prior item -- text, payload, provenance, tags, links,
scope, timestamp -- from one, so "the prior state is recoverable" is an actual, checkable property, not
just an audit label. :func:`merge_versions` additionally requires a typed :class:`MergeStrategy` and
authorized common lineage, and only performs its destructive step (deleting the merged-away item, whose
full state remains recoverable via its snapshot) when the caller explicitly passes ``confirm=True``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem

PUBLIC = "public"

# Reserved internal scope for immutable revision snapshots (MXR-080-0265). Never a legitimate team or
# shared scope: AccessPolicy refuses to grant it to anyone (see AccessPolicy.can_read/can_write), so no
# caller can join it via `shared=`/a grant and read every prior revision of every item in the store. The
# double-underscore wrapping mirrors Python's own "internal" dunder convention.
_REVISIONS_SCOPE = "__substrate_revisions__"


class AccessDeniedError(PermissionError):
    """Raised when a principal is not authorized for the scope it tried to read, write, or publish."""


@dataclass
class AccessPolicy:
    """The centralized access-policy decision point (MXR-080-0264): the one place that decides
    whether a principal may read or write a scope.

    A principal always has read+write access to the scope sharing its own name (:class:`Space` is
    constructed with ``team`` doubling as that team's principal id, so a team's own scope is always
    its home scope) and to :data:`PUBLIC`, the explicit sharing commons -- this module's whole design
    is that nothing else crosses a boundary until someone deliberately publishes into a common scope.
    Every OTHER scope, in particular another team's private scope, is denied until explicitly granted
    with :meth:`grant_read` / :meth:`grant_write` / :meth:`grant`. :class:`Space`, :func:`publish`, and
    :func:`merge_versions` all consult an ``AccessPolicy`` instead of trusting caller-supplied scope
    parameters (``shared=``, ``to=``, ``from_scope=``) as though they were already-validated
    permissions -- closing the concrete leak where a team-A space configured with
    ``shared=("team-b",)`` simply read team-B's private items with no check at all.
    """

    _read: dict[str, set[str]] = field(default_factory=dict)
    _write: dict[str, set[str]] = field(default_factory=dict)

    def grant_read(self, principal: str, scope: str) -> AccessPolicy:
        """Grant ``principal`` read access to ``scope`` beyond its home scope and PUBLIC (chainable)."""
        self._read.setdefault(principal, set()).add(scope)
        return self

    def grant_write(self, principal: str, scope: str) -> AccessPolicy:
        """Grant ``principal`` write access to ``scope`` beyond its home scope and PUBLIC (chainable)."""
        self._write.setdefault(principal, set()).add(scope)
        return self

    def grant(self, principal: str, scope: str) -> AccessPolicy:
        """Grant ``principal`` both read and write access to ``scope`` (chainable)."""
        return self.grant_read(principal, scope).grant_write(principal, scope)

    def revoke(self, principal: str, scope: str) -> AccessPolicy:
        """Revoke both read and write grants of ``scope`` from ``principal`` (chainable).

        A principal's home scope and PUBLIC are intrinsic, not grant-dependent, so this can never
        revoke those -- only a scope previously given via :meth:`grant_read`/:meth:`grant_write`."""
        self._read.get(principal, set()).discard(scope)
        self._write.get(principal, set()).discard(scope)
        return self

    def can_read(self, principal: str, scope: str) -> bool:
        """Whether ``principal`` may read items in ``scope``."""
        if not principal or scope == _REVISIONS_SCOPE:
            return False
        return scope == principal or scope == PUBLIC or scope in self._read.get(principal, ())

    def can_write(self, principal: str, scope: str) -> bool:
        """Whether ``principal`` may write (add or publish) items into ``scope``."""
        if not principal or scope == _REVISIONS_SCOPE:
            return False
        return scope == principal or scope == PUBLIC or scope in self._write.get(principal, ())

    def require_read(self, principal: str, scope: str) -> None:
        """Raise :class:`AccessDeniedError` unless ``principal`` may read ``scope``."""
        if not self.can_read(principal, scope):
            raise AccessDeniedError(f"{principal!r} is not authorized to read scope {scope!r}")

    def require_write(self, principal: str, scope: str) -> None:
        """Raise :class:`AccessDeniedError` unless ``principal`` may write ``scope``."""
        if not self.can_write(principal, scope):
            raise AccessDeniedError(f"{principal!r} is not authorized to write scope {scope!r}")


def visible_scopes(team: str, *, shared: tuple[str, ...] = (PUBLIC,)) -> set[str]:
    """The scopes a team may read: its own id plus the shared scopes (never another team's private one).

    A pure helper -- it does not itself enforce anything. :class:`Space` is the enforcement point: it
    validates ``shared`` against an :class:`AccessPolicy` before this ever runs."""
    return {team, *shared}


def _canonical_json(obj: Any) -> str:
    """A stable JSON encoding (sorted keys, no incidental whitespace), matching the canonicalization
    :mod:`mixle.substrate.context` uses for its IC-13 content hashes, so equal content hashes equal."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _item_digest(item: SubstrateItem) -> str:
    """Content-address ``item``'s full current state: sha256 hex over its kind, text, payload,
    provenance, tags, links, scope, and timestamp (MXR-080-0265) -- identical content always hashes
    identically, so re-snapshotting unchanged content is idempotent."""
    envelope = {
        "id": item.id,
        "kind": item.kind,
        "text": item.text,
        "payload": item.payload,
        "provenance": item.provenance,
        "tags": item.tags,
        "links": item.links,
        "scope": item.scope,
        "created_at": item.created_at,
    }
    return hashlib.sha256(_canonical_json(envelope).encode("utf-8")).hexdigest()


def _snapshot(substrate: Substrate, item: SubstrateItem) -> str:
    """Persist an immutable, content-addressed copy of ``item``'s CURRENT state and return its digest
    (MXR-080-0265/0266): the prior state that :func:`publish` and :func:`merge_versions` are about to
    overwrite or delete, preserved in full -- not just who/from/to bookkeeping.

    Snapshots live in a reserved internal scope that no :class:`AccessPolicy` can ever grant (see
    :data:`_REVISIONS_SCOPE`), with no text of their own, so they never surface in a team's
    :class:`Space` view, in retrieval, or in the text-embedding corpus. Content-addressed by
    :func:`_item_digest`, so writing the same content twice is a no-op past the first time -- a
    revision, once persisted, is never overwritten.
    """
    digest = _item_digest(item)
    revision_id = f"revision:{digest}"
    if substrate.get(revision_id) is None:
        substrate.put(
            SubstrateItem(
                id=revision_id,
                kind=item.kind,
                text="",  # never indexed or retrieved -- recovery only, never search
                payload={"snapshot": item.to_json()},
                provenance={"digest": digest, "snapshot_of": item.id},
                tags=[],
                links=[],
                scope=_REVISIONS_SCOPE,
                created_at=item.created_at,
            )
        )
    return digest


def revision(substrate: Substrate, digest: str) -> dict[str, Any] | None:
    """The complete immutable snapshot recorded under content hash ``digest`` (MXR-080-0265), or
    ``None`` if no such revision was ever persisted. Returns the full prior ``SubstrateItem`` state as
    a JSON-compatible dict (kind, text, payload, provenance, tags, links, scope, created_at) -- not
    just the who/from/to bookkeeping ``version_history`` alone would give you. The returned dict is
    independent of the stored snapshot (a deep copy), so a caller mutating it can never corrupt the
    persisted revision."""
    rev = substrate.get(f"revision:{digest}")
    if rev is None:
        return None
    return copy.deepcopy(rev.payload["snapshot"])


def publish(
    substrate: Substrate,
    ids: list[str],
    *,
    to: str = PUBLIC,
    by: str,
    from_scope: str | None = None,
    policy: AccessPolicy | None = None,
) -> list[str]:
    """Share items into a common scope, authorized (MXR-080-0264).

    Re-scopes each item in ``ids`` to ``to`` and records ``published_by`` / ``published_from`` in its
    provenance. Returns the ids actually published; missing ids, ids whose scope doesn't match
    ``from_scope`` (an additional, OPTIONAL caller assertion), and ids ``by`` is not authorized to move
    are all skipped rather than aborting the whole batch.

    ``by`` must be a non-empty, authenticated principal, and ``policy`` (an :class:`AccessPolicy`; a
    fresh, all-deny-but-home-scope-and-PUBLIC one when omitted) must authorize it to WRITE ``to`` --
    checked once, up front, for the whole call -- AND to READ each item's ACTUAL stored scope, checked
    per item. That second check is what closes the hole ``from_scope`` alone never did: it fires even
    when ``from_scope`` is omitted entirely, so a known item id from a scope ``by`` cannot read is never
    movable just by leaving the filter off.

    Every published item's pre-publish state is persisted as an immutable, content-addressed revision
    (MXR-080-0265) before it is overwritten; ``version_history`` entries carry that revision's digest,
    recoverable with :func:`revision` -- so "a re-published item never silently overwrites its
    predecessor, the prior state is always recoverable" is an enforced property, not just a claim.
    """
    if not by:
        raise AccessDeniedError("publish requires an authenticated principal (by=...)")
    policy = policy if policy is not None else AccessPolicy()
    policy.require_write(by, to)

    published: list[str] = []
    for item_id in ids:
        item = substrate.get(item_id)
        if item is None:
            continue
        if from_scope is not None and item.scope != from_scope:
            continue
        if not policy.can_read(by, item.scope):
            continue
        digest = _snapshot(substrate, item)
        prov = dict(item.provenance)
        # versioned + audited: every share bumps the version and appends to the history, so a re-published
        # item never silently overwrites its predecessor -- the prior state is always recoverable (P2),
        # now backed by an actual immutable snapshot (MXR-080-0265), not just this bookkeeping.
        version = int(prov.get("version", 0)) + 1
        history = list(prov.get("version_history", []))
        history.append(
            {
                "version": version,
                "published_by": by,
                "published_from": item.scope,
                "to": to,
                "revision": digest,
            }
        )
        prov["version"] = version
        prov["version_history"] = history
        prov["published_by"] = by
        prov["published_from"] = item.scope
        substrate.put(
            SubstrateItem(
                id=item.id,
                kind=item.kind,
                text=item.text,
                payload=dict(item.payload),
                provenance=prov,
                tags=list(item.tags),
                links=list(item.links),
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
    """The full publish/merge history of an item: every version with who changed it, from where, and
    to where, plus (MXR-080-0265/0266) a ``"revision"`` (publish entries) or ``parents[]["revision"]``
    (merge entries) content-hash digest that :func:`revision` resolves to the complete immutable prior
    item state -- not just this bookkeeping."""
    item = substrate.get(item_id)
    if item is None:
        return []
    return list(item.provenance.get("version_history", []))


class MergeStrategy(StrEnum):
    """Which side's text/payload survives a merge -- a closed, validated vocabulary (MXR-080-0266): an
    unrecognized strategy now raises instead of silently behaving like :attr:`KEEP`."""

    LATEST = "latest"
    KEEP = "keep"


def merge_versions(
    substrate: Substrate,
    keep_id: str,
    other_id: str,
    *,
    by: str | None = None,
    prefer: MergeStrategy | str = MergeStrategy.LATEST,
    policy: AccessPolicy | None = None,
    confirm: bool = False,
) -> str | None:
    """Reconcile two versions of the same knowledge into one, keeping full lineage (no silent loss, P2).

    Merges ``other_id`` into ``keep_id``: unions tags and links, keeps the text/payload of whichever
    ``prefer`` (a :class:`MergeStrategy`) selects, bumps the surviving item's version, records BOTH
    parents in the history, and removes the merged-away item. Returns the surviving id, or ``None`` if
    either id is missing (checked first, before any authorization -- a merge naming a nonexistent id is
    a no-op, not a permission question).

    Authorized common lineage, required (MXR-080-0266): ``by`` must be a non-empty, authenticated
    principal, and ``policy`` (an :class:`AccessPolicy`; a fresh, all-deny-but-home-scope-and-PUBLIC one
    when omitted) must authorize it to WRITE both ``keep``'s and ``other``'s current scopes -- a caller
    can no longer reconcile, and thereby delete, another principal's item merely by knowing its id.
    ``keep`` and ``other`` must also share the same ``kind``: merging e.g. a ``text`` item into an
    ``image`` item is never a legitimate reconciliation of the same knowledge, so this is a hard reject
    with no override. This is deliberately not a full lineage-graph check (no ``links``/common-ancestor
    requirement): P1's own cross-team-fork use case reconciles items that were never formally linked to
    each other; kind agreement plus dual-scope write authorization is the minimum coherent gate that
    rejects genuinely unrelated items (different kind, unauthorized principal) without breaking that
    legitimate use case.

    ``prefer`` (unrecognized values now raise :class:`ValueError` instead of silently behaving like
    ``"keep"``) and both items' pre-merge states are persisted as immutable, content-addressed revisions
    (MXR-080-0265) before ``keep`` is overwritten and ``other`` is deleted -- recoverable with
    :func:`revision` even though ``other`` no longer exists as a live item. The delete is real and
    permanent as a LIVE item, so it requires an explicit ``confirm=True`` acknowledgment; without it
    this raises :class:`ValueError` rather than silently performing the destructive step.
    """
    keep = substrate.get(keep_id)
    other = substrate.get(other_id)
    if keep is None or other is None:
        return None

    if not by:
        raise AccessDeniedError("merge_versions requires an authenticated principal (by=...)")
    policy = policy if policy is not None else AccessPolicy()
    policy.require_write(by, keep.scope)
    policy.require_write(by, other.scope)
    if keep.kind != other.kind:
        raise ValueError(
            f"cannot merge items of different kind ({keep.kind!r} vs {other.kind!r}) -- merge_versions "
            "reconciles independent edits of the SAME knowledge, not unrelated items"
        )
    strategy = MergeStrategy(prefer)  # raises ValueError on an unrecognized strategy
    if not confirm:
        raise ValueError(
            "merge_versions deletes the merged-away item; pass confirm=True to acknowledge the "
            "destructive step (its full state remains recoverable afterward via the immutable revision "
            "captured just before the merge -- see spaces.revision())"
        )

    take_other = strategy is MergeStrategy.LATEST and version_of(other) > version_of(keep)
    winner_text = other.text if take_other else keep.text
    winner_payload = other.payload if take_other else keep.payload

    # MXR-080-0265/0266: snapshot BOTH parents' full pre-merge state before either is touched, so the
    # merged-away item's content is never lost even though it is about to be deleted for real.
    keep_digest = _snapshot(substrate, keep)
    other_digest = _snapshot(substrate, other)

    prov = dict(keep.provenance)
    version = max(version_of(keep), version_of(other)) + 1
    history_list = list(prov.get("version_history", []))
    history_list.append(
        {
            "version": version,
            "merged_by": by,
            "merged_from": other_id,
            "parents": [
                {"id": keep_id, "version": version_of(keep), "revision": keep_digest},
                {"id": other_id, "version": version_of(other), "revision": other_digest},
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
            payload=dict(winner_payload),
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
    """A team's scoped view over a shared substrate: its own items plus what has been shared to it.

    ``team`` doubles as this space's authorization principal (MXR-080-0264): its home scope, per
    :class:`AccessPolicy`, is the scope of the same name. Construction validates every entry of
    ``shared`` against ``policy`` -- PUBLIC always passes; anything else, including another team's
    scope, needs an explicit grant, or construction raises :class:`AccessDeniedError` rather than
    silently trusting the caller-supplied tuple. Reads re-check live (see :attr:`scopes`), so revoking
    a grant after construction takes effect immediately rather than requiring a new ``Space``.
    """

    def __init__(
        self,
        substrate: Substrate,
        team: str,
        *,
        shared: tuple[str, ...] = (PUBLIC,),
        policy: AccessPolicy | None = None,
    ) -> None:
        if not team:
            raise AccessDeniedError("Space requires a non-empty team/principal id")
        self.substrate = substrate
        self.team = team
        self.policy = policy if policy is not None else AccessPolicy()
        for scope in shared:
            self.policy.require_read(team, scope)  # fail fast: a misconfigured shared= is rejected here
        self.shared = shared

    @property
    def scopes(self) -> set[str]:
        """Scopes actually visible to this team right now: its own scope plus whichever of ``shared``
        the policy currently authorizes -- re-checked live (not just once at construction), so a grant
        revoked later stops being visible immediately rather than requiring a new ``Space``."""
        return {self.team} | {s for s in self.shared if self.policy.can_read(self.team, s)}

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
        """Add an item to this team's own scope by default (pass ``scope=PUBLIC`` to share immediately,
        or any other scope this principal holds a write grant for). MXR-080-0264: no longer writes into
        an arbitrary scope -- in particular another team's -- on the caller's say-so alone."""
        target = scope or self.team
        self.policy.require_write(self.team, target)
        return self.substrate.add(scope=target, **kw)

    def retrieve(self, query: str, *, k: int = 8, **kw: Any) -> Any:
        """Retrieve over exactly the team's visible set (own scope ∪ shared), with cross-kind diversity."""
        from mixle.substrate.retrieve import retrieve

        return retrieve(self._visible_shard(), query, k=k, **kw)

    def publish(self, ids: list[str], *, to: str = PUBLIC, by: str | None = None) -> list[str]:
        """Share this team's items into a common scope (audited). Only own-scope items are publishable.

        ``by`` (default: ``self.team``) is also the authorization principal for this call (MXR-080-0264):
        overriding it to a different principal requires that principal to hold its own grants on
        ``self.policy``, since ``Space`` cannot vouch for an identity other than the team it represents.
        """
        acting = by or self.team
        return publish(self.substrate, ids, to=to, by=acting, from_scope=self.team, policy=self.policy)
