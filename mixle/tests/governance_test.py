"""Governance (P3): promotion gates at org/team scope — propose, approve (ACL-gated), reject.

MXR-080-0263 hardening: every mutating call names an explicit actor (no implicit/ambient identity),
``approve``/``reject``/``Governance.grant`` are authorization-checked (raising
:class:`GovernanceAuthorizationError` for an unauthorized actor rather than silently succeeding or
no-op'ing), and proposal/decision records are append-only (``item.provenance["proposal_history"]``) so
a completed decision can never be silently overwritten or erased by a later ``propose()``.
"""

import unittest

from mixle.substrate import Substrate
from mixle.substrate.governance import (
    APPROVED,
    PENDING,
    REJECTED,
    Governance,
    GovernanceAuthorizationError,
    approve,
    pending,
    propose,
    reject,
)


def _setup():
    s = Substrate()
    item = s.add(kind="artifact", text="company refund ontology term", scope="teamA")
    gov = Governance().grant("orgadmin", "org", by="root")  # "root" bootstraps the still-empty org ACL
    return s, item, gov


class ProposeTest(unittest.TestCase):
    def test_propose_does_not_share_yet(self):
        s, item, _ = _setup()
        propose(s, [item], to="org", by="alice")
        self.assertEqual(s.get(item).scope, "teamA")  # still private until approved
        self.assertEqual(s.get(item).provenance["proposal"]["status"], PENDING)

    def test_pending_lists_awaiting_items(self):
        s, item, _ = _setup()
        propose(s, [item], to="org", by="alice")
        pend = pending(s, to="org")
        self.assertEqual([i.id for i in pend], [item])
        self.assertEqual(pending(s, to="other"), [])  # scoped to the target

    def test_propose_requires_an_explicit_actor(self):
        s, item, _ = _setup()
        with self.assertRaises(ValueError):
            propose(s, [item], to="org", by="")

    def test_repropose_while_pending_is_skipped_not_overwritten(self):
        """A second propose() while the first is still pending must not silently replace it."""
        s, item, _ = _setup()
        propose(s, [item], to="org", by="alice")
        original = dict(s.get(item).provenance["proposal"])

        returned = propose(s, [item], to="org", by="eve")  # eve tries to re-propose over alice's pending item
        self.assertEqual(returned, [])  # skipped: not in the returned ids
        self.assertEqual(s.get(item).provenance["proposal"], original)  # untouched -- still alice's, still pending

    def test_repropose_after_decision_preserves_the_prior_decisions_audit_record(self):
        """MXR-080-0263: re-proposing over an already-decided item must not erase the decision."""
        s, item, gov = _setup()
        propose(s, [item], to="org", by="alice")
        self.assertTrue(reject(s, item, by="orgadmin", governance=gov, reason="duplicate"))
        rejected_record = dict(s.get(item).provenance["proposal"])
        self.assertEqual(rejected_record["status"], REJECTED)

        propose(s, [item], to="org", by="alice")  # alice tries again after the rejection

        current = s.get(item).provenance["proposal"]
        self.assertEqual(current["status"], PENDING)  # a fresh review is now in flight
        history = s.get(item).provenance["proposal_history"]
        self.assertIn(rejected_record, history)  # the prior rejection is still there, byte-for-byte
        self.assertEqual(
            [h["status"] for h in history],
            [PENDING, REJECTED, PENDING],
        )


class ApprovalGateTest(unittest.TestCase):
    def test_non_approver_cannot_promote(self):
        s, item, gov = _setup()
        propose(s, [item], to="org", by="alice")
        with self.assertRaises(GovernanceAuthorizationError):
            approve(s, item, by="alice", governance=gov)  # alice isn't an org approver
        self.assertEqual(s.get(item).scope, "teamA")  # unchanged
        self.assertEqual(s.get(item).provenance["proposal"]["status"], PENDING)  # still pending, not consumed

    def test_approve_requires_an_explicit_actor(self):
        s, item, gov = _setup()
        propose(s, [item], to="org", by="alice")
        with self.assertRaises(ValueError):
            approve(s, item, by="", governance=gov)

    def test_approver_promotes_and_audits(self):
        s, item, gov = _setup()
        propose(s, [item], to="org", by="alice")
        self.assertTrue(approve(s, item, by="orgadmin", governance=gov))
        promoted = s.get(item)
        self.assertEqual(promoted.scope, "org")  # now in the org scope
        self.assertEqual(promoted.provenance["proposal"]["status"], APPROVED)
        self.assertEqual(promoted.provenance["proposal"]["approved_by"], "orgadmin")
        self.assertEqual(promoted.provenance["published_by"], "orgadmin")  # inherits P1's audited share
        self.assertEqual(pending(s, to="org"), [])  # cleared from the queue

    def test_approve_appends_a_new_record_without_mutating_the_pending_one(self):
        s, item, gov = _setup()
        propose(s, [item], to="org", by="alice")
        pending_record = dict(s.get(item).provenance["proposal"])

        self.assertTrue(approve(s, item, by="orgadmin", governance=gov))

        history = s.get(item).provenance["proposal_history"]
        self.assertIn(pending_record, history)  # the original PENDING entry is untouched, not rewritten
        self.assertEqual(history[-1]["status"], APPROVED)
        self.assertEqual(history[-1]["supersedes"], pending_record["seq"])

    def test_promoted_item_becomes_org_visible(self):
        s, item, gov = _setup()
        propose(s, [item], to="org", by="alice")
        approve(s, item, by="orgadmin", governance=gov)
        # the item is now queryable in the org scope (approve() actually moved it, not just labeled it)
        self.assertIn(item, {i.id for i in s.all(scope="org")})

    def test_approve_without_proposal_is_noop(self):
        s, item, gov = _setup()
        self.assertFalse(approve(s, item, by="orgadmin", governance=gov))  # never proposed

    def test_approve_missing_item(self):
        s, _, gov = _setup()
        self.assertFalse(approve(s, "nope", by="orgadmin", governance=gov))

    def test_approve_does_not_alias_the_original_items_mutable_fields(self):
        s = Substrate()
        gov = Governance().grant("orgadmin", "org", by="root")
        item = s.add(kind="artifact", text="term", payload={"k": 1}, tags=["a"], links=["other"], scope="teamA")
        propose(s, [item], to="org", by="alice")
        original = s.get(item)  # fetched AFTER propose's restamp, BEFORE approve's publish + restamp

        self.assertTrue(approve(s, item, by="orgadmin", governance=gov))
        original.tags.append("MUTATED-TAG")
        original.payload["k"] = "MUTATED-PAYLOAD"
        original.links.append("MUTATED-LINK")

        stored = s.get(item)
        self.assertNotIn("MUTATED-TAG", stored.tags)
        self.assertEqual(stored.payload["k"], 1)
        self.assertNotIn("MUTATED-LINK", stored.links)


class RejectTest(unittest.TestCase):
    def test_reject_keeps_item_and_records_reason(self):
        s, item, gov = _setup()
        propose(s, [item], to="org", by="alice")
        self.assertTrue(reject(s, item, by="orgadmin", governance=gov, reason="duplicate"))
        self.assertEqual(s.get(item).scope, "teamA")  # stays put
        prop = s.get(item).provenance["proposal"]
        self.assertEqual(prop["status"], REJECTED)
        self.assertEqual(prop["reason"], "duplicate")

    def test_reject_requires_an_explicit_actor(self):
        s, item, gov = _setup()
        propose(s, [item], to="org", by="alice")
        with self.assertRaises(ValueError):
            reject(s, item, by=None, governance=gov)

    def test_unrelated_actor_cannot_reject_someone_elses_proposal(self):
        """MXR-080-0263's exact adversarial repro: mallory rejects alice's pending proposal."""
        s, item, gov = _setup()
        propose(s, [item], to="org", by="alice")
        with self.assertRaises(GovernanceAuthorizationError):
            reject(s, item, by="mallory", governance=gov, reason="mallory feels like it")
        # nothing changed: the proposal is still pending, still alice's, no rejection recorded anywhere
        prop = s.get(item).provenance["proposal"]
        self.assertEqual(prop["status"], PENDING)
        self.assertEqual(prop["by"], "alice")
        self.assertEqual(len(s.get(item).provenance["proposal_history"]), 1)

    def test_proposer_can_withdraw_their_own_pending_proposal(self):
        s = Substrate()
        item = s.add(kind="artifact", text="alice's own item", scope="teamA")
        propose(s, [item], to="org", by="alice")
        # alice needs no approver rights at all -- an empty ACL -- to withdraw her own proposal.
        self.assertTrue(reject(s, item, by="alice", governance=Governance(), reason="changed my mind"))
        prop = s.get(item).provenance["proposal"]
        self.assertEqual(prop["status"], REJECTED)
        self.assertEqual(prop["rejected_by"], "alice")

    def test_reject_appends_a_new_record_without_mutating_the_pending_one(self):
        s, item, gov = _setup()
        propose(s, [item], to="org", by="alice")
        pending_record = dict(s.get(item).provenance["proposal"])

        self.assertTrue(reject(s, item, by="orgadmin", governance=gov, reason="duplicate"))

        history = s.get(item).provenance["proposal_history"]
        self.assertIn(pending_record, history)  # the original PENDING entry is untouched, not rewritten
        self.assertEqual(history[-1]["status"], REJECTED)
        self.assertEqual(history[-1]["supersedes"], pending_record["seq"])

    def test_reject_does_not_alias_the_original_items_mutable_fields(self):
        """Regression test: _restamp() (shared by propose/approve/reject) built the "new"
        SubstrateItem with payload=item.payload / tags=item.tags / links=item.links -- BY
        REFERENCE, so mutating a previously-fetched copy of the item silently mutated the
        currently-stored, just-restamped item too."""
        s = Substrate()
        gov = Governance().grant("orgadmin", "org", by="root")
        item = s.add(kind="artifact", text="term", payload={"k": 1}, tags=["a"], links=["other"], scope="teamA")
        propose(s, [item], to="org", by="alice")
        original = s.get(item)  # fetched AFTER propose's restamp, BEFORE reject's

        self.assertTrue(reject(s, item, by="orgadmin", governance=gov, reason="duplicate"))
        original.tags.append("MUTATED-TAG")
        original.payload["k"] = "MUTATED-PAYLOAD"
        original.links.append("MUTATED-LINK")

        stored = s.get(item)
        self.assertNotIn("MUTATED-TAG", stored.tags)
        self.assertEqual(stored.payload["k"], 1)
        self.assertNotIn("MUTATED-LINK", stored.links)

    def test_rejected_item_not_in_pending(self):
        s, item, gov = _setup()
        propose(s, [item], to="org", by="alice")
        reject(s, item, by="orgadmin", governance=gov)
        self.assertEqual(pending(s, to="org"), [])


class GovernanceAclTest(unittest.TestCase):
    def test_grant_is_chainable_and_scoped(self):
        gov = Governance().grant("a", "org", by="root").grant("b", "team", by="root")
        self.assertTrue(gov.may_approve("a", "org"))
        self.assertFalse(gov.may_approve("a", "team"))  # scoped ACL
        self.assertFalse(gov.may_approve("c", "org"))

    def test_grant_bootstraps_an_unowned_scope_for_any_actor(self):
        gov = Governance()
        gov.grant("first-admin", "brand-new-scope", by="whoever-shows-up-first")
        self.assertTrue(gov.may_approve("first-admin", "brand-new-scope"))
        self.assertEqual(len(gov.grants), 1)  # audited: the bootstrap grant is not silently untracked
        record = gov.grants[0]
        self.assertEqual(record["who"], "first-admin")
        self.assertEqual(record["scope"], "brand-new-scope")
        self.assertEqual(record["by"], "whoever-shows-up-first")
        self.assertIn("granted_at", record)

    def test_grant_by_non_approver_raises_once_scope_is_owned(self):
        gov = Governance().grant("orgadmin", "org", by="root")
        with self.assertRaises(GovernanceAuthorizationError):
            gov.grant("mallory", "org", by="mallory")  # mallory is not an org approver
        self.assertFalse(gov.may_approve("mallory", "org"))

    def test_existing_approver_can_grant_a_peer(self):
        gov = Governance().grant("orgadmin", "org", by="root")
        gov.grant("orgadmin2", "org", by="orgadmin")  # an incumbent approver may extend the ACL
        self.assertTrue(gov.may_approve("orgadmin2", "org"))

    def test_grant_requires_an_explicit_actor(self):
        gov = Governance()
        with self.assertRaises(ValueError):
            gov.grant("someone", "scope", by="")


if __name__ == "__main__":
    unittest.main()
