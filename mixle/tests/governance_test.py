"""Governance (P3): promotion gates at org/team scope — propose, approve (ACL-gated), reject."""

import unittest

from mixle.substrate import Space, Substrate
from mixle.substrate.governance import (
    APPROVED,
    PENDING,
    REJECTED,
    Governance,
    approve,
    pending,
    propose,
    reject,
)


def _setup():
    s = Substrate()
    item = s.add(kind="artifact", text="company refund ontology term", scope="teamA")
    gov = Governance().grant("orgadmin", "org")
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


class ApprovalGateTest(unittest.TestCase):
    def test_non_approver_cannot_promote(self):
        s, item, gov = _setup()
        propose(s, [item], to="org", by="alice")
        self.assertFalse(approve(s, item, by="alice", governance=gov))  # alice isn't an org approver
        self.assertEqual(s.get(item).scope, "teamA")  # unchanged

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

    def test_promoted_item_becomes_org_visible(self):
        s, item, gov = _setup()
        propose(s, [item], to="org", by="alice")
        approve(s, item, by="orgadmin", governance=gov)
        # a team whose shared scopes include "org" now sees it
        space = Space(s, "teamB", shared=("public", "org"))
        self.assertIn(item, {i.id for i in space.all()})

    def test_approve_without_proposal_is_noop(self):
        s, item, gov = _setup()
        self.assertFalse(approve(s, item, by="orgadmin", governance=gov))  # never proposed

    def test_approve_missing_item(self):
        s, _, gov = _setup()
        self.assertFalse(approve(s, "nope", by="orgadmin", governance=gov))


class RejectTest(unittest.TestCase):
    def test_reject_keeps_item_and_records_reason(self):
        s, item, _ = _setup()
        propose(s, [item], to="org", by="alice")
        self.assertTrue(reject(s, item, by="orgadmin", reason="duplicate"))
        self.assertEqual(s.get(item).scope, "teamA")  # stays put
        prop = s.get(item).provenance["proposal"]
        self.assertEqual(prop["status"], REJECTED)
        self.assertEqual(prop["reason"], "duplicate")

    def test_rejected_item_not_in_pending(self):
        s, item, _ = _setup()
        propose(s, [item], to="org", by="alice")
        reject(s, item, by="orgadmin")
        self.assertEqual(pending(s, to="org"), [])


class GovernanceAclTest(unittest.TestCase):
    def test_grant_is_chainable_and_scoped(self):
        gov = Governance().grant("a", "org").grant("b", "team")
        self.assertTrue(gov.may_approve("a", "org"))
        self.assertFalse(gov.may_approve("a", "team"))  # scoped ACL
        self.assertFalse(gov.may_approve("c", "org"))


if __name__ == "__main__":
    unittest.main()
