"""Promotion gates for shared substrate knowledge.

Teams can publish items into shared scopes, while curated scopes can require an
explicit approval step. :func:`propose` marks an item as pending promotion,
:func:`approve` promotes it when the approver is authorized for the target
scope, :func:`reject` refuses it, and :func:`pending` lists items awaiting
review.

Every transition names an explicit actor (``by=``) -- there is no implicit or
ambient identity -- and is checked against :class:`Governance`'s approver ACL.
:func:`approve` requires ``by`` to be an authorized approver of the target
scope; :func:`reject` requires ``by`` to be either the proposal's own proposer
(a self-withdrawal) or an authorized approver (a governance refusal); an actor
who is neither raises :class:`GovernanceAuthorizationError` rather than
silently succeeding or no-op'ing. :meth:`Governance.grant` is gated the same
way: only an existing approver of a scope may grant further approvers into it,
except to bootstrap a scope that has none yet.

Decisions are append-only. The full history of an item's proposals lives in
``item.provenance["proposal_history"]``; a transition is always recorded as a
NEW entry referencing (``supersedes``) the entry it decides, never as an
in-place overwrite, so a rejected or approved decision can never be silently
erased by a later :func:`propose` call. ``item.provenance["proposal"]`` always
mirrors the latest entry, for callers that only care about current state.

Approved promotion delegates the scope change to
:func:`~mixle.substrate.spaces.publish`, preserving the same provenance trail as
ordinary sharing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from mixle.substrate.core import Substrate, SubstrateItem

if TYPE_CHECKING:
    from mixle.capability_lifecycle import AuthorizationDecision

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"


class GovernanceAuthorizationError(PermissionError):
    """Raised when an actor attempts a governance transition they are not authorized to make."""


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _require_actor(by: str, action: str) -> str:
    """Every governance-mutating call must name an explicit actor -- no implicit/ambient identity."""
    if not isinstance(by, str) or not by.strip():
        raise ValueError(f"{action} requires an explicit, non-empty actor id (by=...)")
    return by


@dataclass
class Governance:
    """Who may approve promotions into which scope -- the org-governance ACL.

    ``approvers`` is the live, queryable ACL; ``grants`` is its append-only audit trail (who was
    granted, into which scope, and who authorized it). Both are READ-ONLY views: they were public
    mutable containers, so authority could be granted straight into the dict with no authorization
    check and no audit entry, and the trail recording that could itself be edited (MXR-080-1883).
    :meth:`grant` is the only way in, and it both checks and records.

    A scope with no approvers yet is unowned: the first grant into it may come from anyone, so an
    approver set can be bootstrapped from nothing. That remains a genuine hole -- anyone may claim an
    unowned scope -- and it is now marked ``bootstrap: True`` in the audit trail rather than being
    indistinguishable from an authorized grant. Rooting the bootstrap in a declared authority is
    outstanding. Once a scope has at least one approver, only an existing approver of THAT scope may
    grant further ones into it -- :meth:`grant` raises :class:`GovernanceAuthorizationError` for
    anyone else.
    """

    _approvers: dict[str, set[str]] = field(default_factory=dict, repr=False)  # scope -> {approver id}
    _grants: list[dict[str, Any]] = field(default_factory=list, repr=False)  # append-only audit trail

    @property
    def approvers(self) -> Mapping[str, frozenset[str]]:
        """The live ACL, as a read-only view.

        This was a public mutable dict, so ``governance.approvers.setdefault("secret", set()).add(...)``
        granted approval authority over a scope with no authorization check and no entry in the audit
        trail that exists to record exactly that (MXR-080-1883). :meth:`grant` is the only way in, and
        it both checks and records.
        """
        return MappingProxyType({scope: frozenset(who) for scope, who in self._approvers.items()})

    @property
    def grants(self) -> tuple[Mapping[str, Any], ...]:
        """The append-only grant audit trail, as a read-only view.

        An audit trail a caller can edit or truncate is not one; it was a public ``list``.
        """
        return tuple(MappingProxyType(dict(row)) for row in self._grants)

    def may_approve(self, who: str, scope: str) -> bool:
        """Return whether ``who`` is allowed to approve (or reject) promotion into ``scope``."""
        return who in self._approvers.get(scope, set())

    def grant(self, who: str, scope: str, *, by: str) -> Governance:
        """Add ``who`` as an approver for ``scope`` (chainable) -- audited, and gated once owned.

        ``by`` must already be an approver of ``scope``, unless ``scope`` currently has none at all (the
        bootstrap case: someone has to seed the first approver). Raises
        :class:`GovernanceAuthorizationError` otherwise.
        """
        _require_actor(by, "grant")
        _require_actor(who, "grant")
        incumbents = self._approvers.get(scope, set())
        if incumbents and by not in incumbents:
            raise GovernanceAuthorizationError(
                f"{by!r} is not an approver of {scope!r} and cannot grant new approvers into it"
            )
        self._approvers.setdefault(scope, set()).add(who)
        # `bootstrap` marks a grant that no incumbent authorized, because the scope had no approvers
        # yet and someone has to seed the first one. That is a real hole -- anyone may claim an
        # unowned scope -- and it is recorded rather than hidden, so an auditor reading the trail can
        # see which grants rest on no prior authority instead of having to infer it from ordering
        # (MXR-080-1883). Rooting the bootstrap in a declared authority is the remaining work.
        self._grants.append({"who": who, "scope": scope, "by": by, "granted_at": _now(), "bootstrap": not incumbents})
        return self


def _restamp(substrate: Substrate, item: SubstrateItem, prov: dict[str, Any], *, scope: str | None = None) -> None:
    substrate.put(
        SubstrateItem(
            id=item.id,
            kind=item.kind,
            text=item.text,
            payload=dict(item.payload),
            provenance=prov,
            tags=list(item.tags),
            links=list(item.links),
            derived_from=list(item.derived_from),
            scope=scope if scope is not None else item.scope,
            created_at=item.created_at,
        )
    )


def _latest_proposal(item: SubstrateItem) -> dict[str, Any] | None:
    """The most recent proposal-history entry for ``item`` (its current status), or ``None``."""
    history = item.provenance.get("proposal_history")
    if history:
        return dict(history[-1])
    legacy = item.provenance.get("proposal")  # tolerate pre-migration items with no history list yet
    return dict(legacy) if legacy else None


def propose(substrate: Substrate, ids: list[str], *, to: str, by: str) -> list[str]:
    """Mark items as pending promotion to scope ``to``; they are not yet visible there. Returns the ids.

    Each call appends a NEW entry to the item's append-only ``proposal_history`` -- it never overwrites
    an existing entry, so a prior decision's audit record (who rejected or approved it, and why) always
    survives, even when the item is proposed again later. An item that already has a PENDING proposal is
    skipped (left out of the returned ids): resolve that in-flight review with :func:`approve` or
    :func:`reject` first, rather than silently replacing it with a second one.
    """
    _require_actor(by, "propose")
    proposed: list[str] = []
    for item_id in ids:
        item = substrate.get(item_id)
        if item is None:
            continue
        latest = _latest_proposal(item)
        if latest is not None and latest.get("status") == PENDING:
            continue  # an in-flight review exists; propose() must not silently replace it
        history = list(item.provenance.get("proposal_history", []))
        record = {
            "seq": len(history) + 1,
            "supersedes": latest.get("seq") if latest else None,
            "to": to,
            "by": by,
            "status": PENDING,
            "proposed_at": _now(),
        }
        history.append(record)
        prov = dict(item.provenance)
        prov["proposal"] = record
        prov["proposal_history"] = history
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
    so it inherits the versioned/audited share) and a new APPROVED entry is appended to its
    ``proposal_history`` naming the approver -- the PENDING entry it decides is left untouched, never
    overwritten. Returns False (no change, not a security event) if the item has no pending proposal.
    Raises :class:`GovernanceAuthorizationError` if ``by`` is not an approver authorized for the target
    scope; no mutation happens in that case."""
    _require_actor(by, "approve")
    item = substrate.get(item_id)
    if item is None:
        return False
    latest = _latest_proposal(item)
    if not latest or latest.get("status") != PENDING:
        return False
    # The PROPOSAL declares the destination; the approver decides yes or no, not where (MXR-080-1883).
    # `to` used to SUBSTITUTE the target, and the authorization below was then checked against the
    # substituted value -- so the caller chose the destination and chose the scope they would be
    # checked against, which is not an authorization check at all. An approver for `public` could take
    # an item proposed from `secret` to `reviewed-secret` and publish it into `public` instead. `to` is
    # kept as an ASSERTION the caller may make about what they believe they are approving, and a
    # mismatch is refused rather than honoured.
    target = latest.get("to")
    if to is not None and to != target:
        raise GovernanceAuthorizationError(
            f"{by!r} attempted to approve item {item_id!r} into {to!r}, but its pending proposal "
            f"declares {target!r}. An approval decides a proposal; it does not redirect one. Reject "
            "this proposal and propose the intended target instead."
        )
    if not governance.may_approve(by, target):
        raise GovernanceAuthorizationError(f"{by!r} is not authorized to approve promotion into {target!r}")

    from mixle.substrate.spaces import AccessPolicy, publish

    # governance's OWN ACL just vetted that `by` may approve promotion into `target` (MXR-080-0263) --
    # make that decision legible to spaces' centralized access policy (MXR-080-0264) as a
    # narrowly-scoped grant for exactly this operation, rather than requiring `by` to separately hold
    # standing read/write rights over `item.scope`/`target`: an org-wide approver is not necessarily a
    # member of every team whose proposals it reviews, but a proposal it is authorized to approve is
    # exactly the authorization to move THIS item out of its current scope and into the approved target.
    policy = AccessPolicy().grant_read(by, item.scope).grant_write(by, target)
    publish(substrate, [item_id], to=target, by=by, from_scope=item.scope, policy=policy)
    promoted = substrate.get(item_id)
    history = list(promoted.provenance.get("proposal_history", []))
    decided = {
        "seq": len(history) + 1,
        "supersedes": latest.get("seq"),
        "to": target,
        "status": APPROVED,
        "approved_by": by,
        "decided_at": _now(),
    }
    history.append(decided)
    prov = dict(promoted.provenance)
    prov["proposal"] = decided
    prov["proposal_history"] = history
    _restamp(substrate, promoted, prov)  # keep the just-published scope
    return True


def reject(substrate: Substrate, item_id: str, *, by: str, governance: Governance, reason: str = "") -> bool:
    """Refuse a pending promotion -- the item stays in its origin scope; the refusal is recorded.

    ``by`` must be either the proposal's own proposer (a self-withdrawal) or an actor ``governance``
    authorizes to approve into the proposed scope (a governance refusal). Anyone else raises
    :class:`GovernanceAuthorizationError` -- an unrelated third party can no longer reject someone
    else's proposal. A new REJECTED entry is appended to the item's ``proposal_history``; the PENDING
    entry it decides is left untouched, never overwritten. Returns False (no change, not a security
    event) if the item has no pending proposal to refuse."""
    _require_actor(by, "reject")
    item = substrate.get(item_id)
    if item is None:
        return False
    latest = _latest_proposal(item)
    if not latest or latest.get("status") != PENDING:
        return False
    target = latest.get("to")
    is_proposer = by == latest.get("by")
    if not is_proposer and not governance.may_approve(by, target):
        raise GovernanceAuthorizationError(
            f"{by!r} is neither the proposer nor an approver authorized for {target!r} and cannot reject this proposal"
        )
    history = list(item.provenance.get("proposal_history", []))
    decided = {
        "seq": len(history) + 1,
        "supersedes": latest.get("seq"),
        "to": target,
        "status": REJECTED,
        "rejected_by": by,
        "reason": reason,
        "decided_at": _now(),
    }
    history.append(decided)
    prov = dict(item.provenance)
    prov["proposal"] = decided
    prov["proposal_history"] = history
    _restamp(substrate, item, prov)
    return True


def authorization_decision(
    substrate: Substrate,
    item_id: str,
    *,
    capability_id: str,
    version: str,
    digest: str | None = None,
) -> AuthorizationDecision:
    """Adapt a completed legacy scope proposal to the shared authorization contract.

    Pending or missing proposals have no decision and raise :class:`ValueError`.
    This adapter keeps the substrate API compatible while allowing orchestrators
    to exchange one authorization representation with other Mixle projects.
    """
    from mixle.capability_lifecycle import AuthorizationDecision, AuthorizationOutcome, CapabilityIdentity

    item = substrate.get(item_id)
    if item is None:
        raise ValueError(f"unknown substrate item: {item_id}")
    proposal = item.provenance.get("proposal")
    if not proposal or proposal.get("status") not in {APPROVED, REJECTED}:
        raise ValueError("proposal does not have a completed authorization decision")
    approved = proposal["status"] == APPROVED
    principal_field = "approved_by" if approved else "rejected_by"
    decided_at = proposal.get("decided_at")
    if not decided_at:
        raise ValueError("completed proposal is missing decided_at")
    return AuthorizationDecision(
        decision_id=f"substrate:{item_id}:{proposal['status']}",
        capability=CapabilityIdentity(capability_id, version, digest),
        outcome=AuthorizationOutcome.GRANTED if approved else AuthorizationOutcome.DENIED,
        issued_by=str(proposal[principal_field]),
        scopes=frozenset({str(proposal["to"])}),
        decided_at=datetime.fromisoformat(str(decided_at).replace("Z", "+00:00")),
        reason=str(proposal.get("reason", "")),
    )
