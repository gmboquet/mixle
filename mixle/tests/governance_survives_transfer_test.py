"""E4: scope and governance survive transfer -- a ContextPacket assembled from a governed item still
carries its promotion history (propose/approve/reject provenance) and its current scope, so a receiver
can see who approved what, or that a promotion was refused, not just the bare text.
"""

import unittest

from mixle.substrate import Substrate
from mixle.substrate.context import ContextBudget, assemble_context
from mixle.substrate.governance import PENDING, REJECTED, Governance, approve, propose, reject


class GovernanceSurvivesTransferTest(unittest.TestCase):
    def test_approved_promotion_provenance_and_new_scope_both_survive_into_the_packet(self):
        s = Substrate()
        item_id = s.add(kind="artifact", text="the refund policy covers 30 days", scope="teamA")
        gov = Governance().grant("orgadmin", "org")

        propose(s, [item_id], to="org", by="alice")
        self.assertTrue(approve(s, item_id, by="orgadmin", governance=gov))

        packet = assemble_context(s, "refund policy", budget=ContextBudget(max_items=5), scope="org")

        self.assertEqual(len(packet.items), 1)
        transferred = packet.items[0]
        self.assertEqual(transferred.scope, "org")  # the promoted scope, not the origin teamA
        proposal = transferred.provenance["proposal"]
        self.assertEqual(proposal["status"], "approved")
        self.assertEqual(proposal["approved_by"], "orgadmin")
        self.assertEqual(proposal["to"], "org")

    def test_pending_item_is_not_visible_in_the_target_scopes_packet(self):
        s = Substrate()
        item_id = s.add(kind="artifact", text="an unreviewed claim about refunds", scope="teamA")
        propose(s, [item_id], to="org", by="alice")

        packet = assemble_context(s, "refunds", budget=ContextBudget(max_items=5), scope="org")

        self.assertEqual(packet.items, [])  # still teamA-scoped -- proposing alone does not share it
        origin_packet = assemble_context(s, "refunds", budget=ContextBudget(max_items=5), scope="teamA")
        self.assertEqual(len(origin_packet.items), 1)
        self.assertEqual(origin_packet.items[0].provenance["proposal"]["status"], PENDING)

    def test_rejected_promotion_leaves_the_item_in_its_origin_scope_with_the_refusal_recorded(self):
        s = Substrate()
        item_id = s.add(kind="artifact", text="a dubious refund shortcut", scope="teamA")
        propose(s, [item_id], to="org", by="alice")
        self.assertTrue(reject(s, item_id, by="orgadmin", reason="unverified"))

        org_packet = assemble_context(s, "refund shortcut", budget=ContextBudget(max_items=5), scope="org")
        self.assertEqual(org_packet.items, [])  # rejection never shares it into org

        origin_packet = assemble_context(s, "refund shortcut", budget=ContextBudget(max_items=5), scope="teamA")
        self.assertEqual(origin_packet.items[0].scope, "teamA")
        proposal = origin_packet.items[0].provenance["proposal"]
        self.assertEqual(proposal["status"], REJECTED)
        self.assertEqual(proposal["rejected_by"], "orgadmin")
        self.assertEqual(proposal["reason"], "unverified")

    def test_governance_survives_a_second_hop_receiver_reading_the_same_packet(self):
        """A different receiver reading packet.items sees the exact same governance state -- transfer
        does not launder or drop it on a second hop."""
        s = Substrate()
        item_id = s.add(kind="artifact", text="the refund policy covers 30 days", scope="teamA")
        gov = Governance().grant("orgadmin", "org")
        propose(s, [item_id], to="org", by="alice")
        approve(s, item_id, by="orgadmin", governance=gov)

        packet = assemble_context(s, "refund policy", budget=ContextBudget(max_items=5), scope="org")
        receiver_view = packet.items[0]  # a second model receiving just this packet, not the substrate
        self.assertEqual(receiver_view.provenance["proposal"]["status"], "approved")
        self.assertEqual(receiver_view.scope, "org")


if __name__ == "__main__":
    unittest.main()
