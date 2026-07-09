"""Promotion gates for shared substrate knowledge.

Teams can publish items into shared scopes, while curated scopes can require an
explicit approval step. :func:`propose` marks an item as pending promotion,
:func:`approve` promotes it when the approver is authorized for the target
scope, :func:`reject` records a refusal, and :func:`pending` lists items awaiting
review.

Approved promotion delegates the scope change to
:func:`~mixle.substrate.spaces.publish`, preserving the same provenance trail as
ordinary sharing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mixle.substrate.core import Substrate, SubstrateItem

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"


@dataclass
class Governance:
    """Who may approve promotions into which scope -- the org-governance ACL."""

    approvers: dict[str, set[str]] = field(default_factory=dict)  # scope -> {approver id, ...}

    def may_approve(self, who: str, scope: str) -> bool:
        """Return whether ``who`` is allowed to approve promotion into ``scope``."""
        return who in self.approvers.get(scope, set())

    def grant(self, who: str, scope: str) -> Governance:
        """Add ``who`` as an approver for ``scope`` (chainable)."""
        self.approvers.setdefault(scope, set()).add(who)
        return self


def _restamp(substrate: Substrate, item: SubstrateItem, prov: dict[str, Any], *, scope: str | None = None) -> None:
    substrate.put(
        SubstrateItem(
            id=item.id,
            kind=item.kind,
            text=item.text,
            payload=item.payload,
            provenance=prov,
            tags=item.tags,
            links=item.links,
            scope=scope if scope is not None else item.scope,
            created_at=item.created_at,
        )
    )


def propose(substrate: Substrate, ids: list[str], *, to: str, by: str | None = None) -> list[str]:
    """Mark items as pending promotion to scope ``to``; they are not yet visible there. Returns the ids."""
    proposed: list[str] = []
    for item_id in ids:
        item = substrate.get(item_id)
        if item is None:
            continue
        prov = dict(item.provenance)
        prov["proposal"] = {"to": to, "by": by, "status": PENDING}
        _restamp(substrate, item, prov)  # scope unchanged: proposing does not share
        proposed.append(item_id)
    return proposed


def pending(substrate: Substrate, *, to: str | None = None) -> list[SubstrateItem]:
    """Items awaiting approval (optionally only those proposed to scope ``to``)."""
    out: list[SubstrateItem] = []
    for item in substrate.all():
        prop = item.provenance.get("proposal")
        if prop and prop.get("status") == PENDING and (to is None or prop.get("to") == to):
            out.append(item)
    return out


def approve(substrate: Substrate, item_id: str, *, by: str, governance: Governance, to: str | None = None) -> bool:
    """Promote a pending item into its proposed scope -- IFF ``by`` may approve for that scope (the gate).

    On success the item is published into the target scope (via P1 :func:`~mixle.substrate.spaces.publish`,
    so it inherits the versioned/audited share) and its proposal is marked approved with the approver id.
    Returns False (no change) if the item has no pending proposal or ``by`` lacks approval rights."""
    item = substrate.get(item_id)
    if item is None:
        return False
    prop = item.provenance.get("proposal")
    if not prop or prop.get("status") != PENDING:
        return False
    target = to or prop.get("to")
    if not governance.may_approve(by, target):
        return False

    from mixle.substrate.spaces import publish

    publish(substrate, [item_id], to=target, by=by, from_scope=item.scope)
    promoted = substrate.get(item_id)
    prov = dict(promoted.provenance)
    prov["proposal"] = {**prop, "status": APPROVED, "approved_by": by, "to": target}
    _restamp(substrate, promoted, prov)  # keep the just-published scope
    return True


def reject(substrate: Substrate, item_id: str, *, by: str, reason: str = "") -> bool:
    """Refuse a pending promotion -- the item stays in its origin scope; the refusal is recorded."""
    item = substrate.get(item_id)
    if item is None:
        return False
    prop = item.provenance.get("proposal")
    if not prop or prop.get("status") != PENDING:
        return False
    prov = dict(item.provenance)
    prov["proposal"] = {**prop, "status": REJECTED, "rejected_by": by, "reason": reason}
    _restamp(substrate, item, prov)
    return True
