"""verify_lineage / audit_substrate (N1): provenance-chain integrity over the substrate.

MXR-080-0260/0261: verify_lineage() traverses `derived_from` (typed ancestry) only, never `links`
(generic KG relations), enforces the caller's scope through the whole traversal via AccessPolicy, and
returns a three-state LineageState (INTACT/BROKEN/UNVERIFIED) rather than a bare boolean. The dedicated
test classes below (DepthCapTruncationTest, CycleAsIntactTest, CrossScopeLineageLeakTest,
TypedProvenanceEdgeTest) are the adversarial regressions for those two findings; VerifyLineageTest and
AuditTest cover the base mechanics.
"""

import unittest

from mixle.substrate import AccessPolicy, Substrate, SubstrateItem, audit_substrate, verify_lineage
from mixle.substrate.trust import LineageReport, LineageState


class VerifyLineageTest(unittest.TestCase):
    def test_intact_chain_resolves(self):
        s = Substrate()
        doc = s.add(kind="text", text="source doc")
        model = s.add(kind="artifact", text="model", derived_from=[doc])
        deploy = s.add(kind="artifact", text="deployment", derived_from=[model])
        r = verify_lineage(s, deploy)
        self.assertIsInstance(r, LineageReport)
        self.assertEqual(r.state, LineageState.INTACT)
        self.assertTrue(r.intact)
        self.assertEqual(r.depth, 2)  # deploy -> model -> doc
        self.assertEqual(r.visited, 3)

    def test_dangling_link_is_flagged(self):
        s = Substrate()
        orphan = s.add(kind="trace", text="orphan", derived_from=["ghost-id"])
        r = verify_lineage(s, orphan)
        self.assertEqual(r.state, LineageState.BROKEN)
        self.assertFalse(r.intact)
        self.assertEqual(r.dangling, ["ghost-id"])

    def test_missing_root_is_not_intact(self):
        r = verify_lineage(Substrate(), "nope")
        self.assertEqual(r.state, LineageState.BROKEN)
        self.assertFalse(r.intact)
        self.assertIn("nope", r.dangling)

    def test_no_links_is_trivially_intact(self):
        s = Substrate()
        leaf = s.add(kind="text", text="leaf")
        r = verify_lineage(s, leaf)
        self.assertEqual(r.state, LineageState.INTACT)
        self.assertEqual(r.n_links, 0)

    def test_cycles_are_reported_broken_not_silently_intact(self):
        """MXR-080-0261: a lineage cycle is a confirmed structural defect (an item cannot legitimately
        derive, even transitively, from itself) -- it must never read as INTACT just because cycle-safe
        traversal terminates without a dangling edge. See CycleAsIntactTest for the dedicated coverage."""
        s = Substrate()
        c1 = s.add(kind="text", text="c1")
        c2 = s.add(kind="text", text="c2", derived_from=[c1])
        # make c1 point back at c2 -> a genuine cycle
        c1_item = s.get(c1)
        s.put(SubstrateItem(**{**c1_item.to_json(), "derived_from": [c2]}))
        r = verify_lineage(s, c1)
        self.assertEqual(r.state, LineageState.BROKEN)
        self.assertFalse(r.intact)
        self.assertEqual(r.cycles, [c1])
        self.assertEqual(r.visited, 2)  # still terminates despite the cycle

    def test_deep_break_is_caught(self):
        s = Substrate()
        a = s.add(kind="text", text="a", derived_from=["gone"])
        b = s.add(kind="artifact", text="b", derived_from=[a])
        r = verify_lineage(s, b)
        self.assertEqual(r.state, LineageState.BROKEN)  # the break two levels down still fails the chain
        self.assertIn("gone", r.dangling)


class DepthCapTruncationTest(unittest.TestCase):
    """MXR-080-0260 (Critical): reaching max_depth must never silently certify what lies beyond it.
    Pre-fix, verify_lineage() simply skipped a node at the cap without recording anything, so a chain
    whose cap-boundary node linked to a genuinely missing ancestor one level beyond the cap was
    reported intact=True -- confirmed against the pre-fix baseline (see the negative-control repro)."""

    def test_item_beyond_the_cap_is_never_certified_intact(self):
        s = Substrate()
        # root -> a -> b -> c (AT the cap when max_depth=3) -> missing (one level beyond, never visited)
        c = s.add(kind="text", text="c", derived_from=["ghost-beyond-cap"])
        b = s.add(kind="text", text="b", derived_from=[c])
        a = s.add(kind="text", text="a", derived_from=[b])
        root = s.add(kind="artifact", text="root", derived_from=[a])

        r = verify_lineage(s, root, max_depth=3)
        self.assertEqual(r.state, LineageState.UNVERIFIED)  # never INTACT -- the tail was never checked
        self.assertFalse(r.intact)
        self.assertIn(c, r.truncated)  # c is the node at the cap whose OWN links were never inspected
        self.assertEqual(r.dangling, [])  # honest: we don't know it's broken, only that we stopped looking

    def test_a_leaf_exactly_at_the_cap_is_not_truncated(self):
        """The cap boundary alone must not manufacture a false truncated finding -- a node that happens
        to sit exactly at max_depth but has no further edges is a genuine leaf, not a hidden tail."""
        s = Substrate()
        b = s.add(kind="text", text="b")  # a genuine leaf, no derived_from
        a = s.add(kind="text", text="a", derived_from=[b])
        root = s.add(kind="artifact", text="root", derived_from=[a])

        r = verify_lineage(s, root, max_depth=2)
        self.assertEqual(r.state, LineageState.INTACT)
        self.assertEqual(r.truncated, [])

    def test_raising_max_depth_resolves_the_same_chain_to_broken_not_unverified(self):
        """Confirms `truncated` was really hiding a real break, not manufacturing one: the SAME chain,
        given enough depth to actually reach the missing ancestor, reports BROKEN with it named."""
        s = Substrate()
        c = s.add(kind="text", text="c", derived_from=["ghost-beyond-cap"])
        b = s.add(kind="text", text="b", derived_from=[c])
        a = s.add(kind="text", text="a", derived_from=[b])
        root = s.add(kind="artifact", text="root", derived_from=[a])

        r = verify_lineage(s, root, max_depth=20)
        self.assertEqual(r.state, LineageState.BROKEN)
        self.assertEqual(r.dangling, ["ghost-beyond-cap"])
        self.assertEqual(r.truncated, [])


class CycleAsIntactTest(unittest.TestCase):
    """MXR-080-0261 (Critical, security): a lineage cycle must be reported as a confirmed structural
    defect (BROKEN), never silently absorbed by cycle-safe traversal and reported intact=True. Pre-fix,
    verify_lineage()'s visited-set only prevented infinite recursion; it never flagged the cycle itself."""

    def test_mutual_two_node_cycle_is_broken_not_intact(self):
        s = Substrate()
        c1 = s.add(kind="text", text="c1")
        c2 = s.add(kind="text", text="c2", derived_from=[c1])
        c1_item = s.get(c1)
        s.put(SubstrateItem(**{**c1_item.to_json(), "derived_from": [c2]}))  # c1 -> c2 -> c1
        r = verify_lineage(s, c1)
        self.assertEqual(r.state, LineageState.BROKEN)
        self.assertEqual(r.cycles, [c1])
        self.assertEqual(r.visited, 2)  # still terminates despite the cycle

    def test_self_referencing_item_is_broken(self):
        s = Substrate()
        x = s.add(kind="text", text="x")
        x_item = s.get(x)
        s.put(SubstrateItem(**{**x_item.to_json(), "derived_from": [x]}))  # x derives from itself
        r = verify_lineage(s, x)
        self.assertEqual(r.state, LineageState.BROKEN)
        self.assertEqual(r.cycles, [x])

    def test_a_diamond_shared_ancestor_is_not_a_cycle(self):
        """Two independent derivation paths converging on the same real ancestor (a -> b, a -> c,
        b -> d, c -> d) is ordinary DAG structure, not a cycle -- must stay INTACT, not over-flagged."""
        s = Substrate()
        d = s.add(kind="text", text="d")
        b = s.add(kind="text", text="b", derived_from=[d])
        c = s.add(kind="text", text="c", derived_from=[d])
        a = s.add(kind="artifact", text="a", derived_from=[b, c])
        r = verify_lineage(s, a)
        self.assertEqual(r.state, LineageState.INTACT)
        self.assertEqual(r.cycles, [])
        self.assertEqual(r.visited, 4)  # a, b, c, d -- d counted once despite two incoming paths


class CrossScopeLineageLeakTest(unittest.TestCase):
    """MXR-080-0261 (Critical, security): a scope-restricted caller's lineage verdict for their OWN
    item must never depend on -- or be certifiable using -- another, inaccessible scope's content.
    Pre-fix, verify_lineage() took no scope/policy argument at all: it walked every scope unconditionally,
    so a team-a-scoped caller's own audit was silently validated (or invalidated) by team-b's private
    ancestry, leaking whether team-b's internal lineage happened to be intact. This is the same class of
    leak MXR-080-0237 closed for semantic search; built with the same rigor -- two distinguishable
    scopes, a concrete measurable before/after.
    """

    def _build(self, *, team_b_ancestor_intact: bool) -> tuple[Substrate, str]:
        s = Substrate()
        if team_b_ancestor_intact:
            b_leaf = s.add(kind="text", text="team-b private source material", scope="team-b")
            b_ancestor = s.add(kind="record", text="team-b private record", scope="team-b", derived_from=[b_leaf])
        else:
            b_ancestor = s.add(
                kind="record",
                text="team-b private record",
                scope="team-b",
                derived_from=["team-b-ghost-ancestor"],
            )
        a_item = s.add(kind="artifact", text="team-a deliverable", scope="team-a", derived_from=[b_ancestor])
        return s, a_item

    def test_a_scoped_caller_never_gets_intact_from_an_inaccessible_ancestor(self):
        """The direct finding-level proof: team-a's own item derives from a team-b ancestor that
        genuinely, verifiably exists and is itself intact -- yet team-a, scoped to its own view, must
        report UNVERIFIED (not INTACT), since it has no authorized way to confirm that for itself."""
        s, a_item = self._build(team_b_ancestor_intact=True)
        r_unscoped = verify_lineage(s, a_item)  # scope=None: an unrestricted caller legitimately can see it
        self.assertEqual(r_unscoped.state, LineageState.INTACT)

        r_scoped = verify_lineage(s, a_item, scope="team-a")
        self.assertEqual(r_scoped.state, LineageState.UNVERIFIED)  # never INTACT via inaccessible content
        self.assertEqual(len(r_scoped.unverified), 1)
        self.assertEqual(r_scoped.dangling, [])  # honestly unverifiable, never falsely "confirmed broken"

    def test_team_as_own_verdict_does_not_depend_on_team_bs_internal_integrity(self):
        """The black-box, measurable check (mirrors MXR-080-0237's methodology): team-a's scoped report
        for the SAME shape of team-a item is identical in state and every count whether team-b's own
        private ancestry is internally intact or itself broken -- team-a has no authorized visibility
        into team-b at all, so team-b's internal state is completely invisible to it, not just its raw
        content. (The exact dangling/unverified id VALUES differ only because the two substrates mint
        different item ids; the shapes -- state, counts, and the fact team-a's dangling stays empty
        either way -- are what must be, and are, identical.)"""
        s_clean, a_clean = self._build(team_b_ancestor_intact=True)
        s_broken, a_broken = self._build(team_b_ancestor_intact=False)

        r_clean = verify_lineage(s_clean, a_clean, scope="team-a")
        r_broken = verify_lineage(s_broken, a_broken, scope="team-a")

        self.assertEqual(r_clean.state, LineageState.UNVERIFIED)
        self.assertEqual(r_broken.state, LineageState.UNVERIFIED)
        self.assertEqual(r_clean.dangling, [])
        self.assertEqual(r_broken.dangling, [])  # team-b's OWN dangling ref never even reached team-a's view
        self.assertEqual(r_clean.cycles, r_broken.cycles)
        self.assertEqual(len(r_clean.unverified), len(r_broken.unverified))
        self.assertEqual(r_clean.visited, r_broken.visited)
        self.assertEqual(r_clean.n_links, r_broken.n_links)

    def test_a_grant_extends_authorized_visibility_and_can_then_certify_intact(self):
        """Confirms this is real, working authorization -- not a hardcoded deny -- by showing the
        positive case: once team-a is explicitly GRANTED read access to team-b (AccessPolicy.grant_read),
        the same chain that was UNVERIFIED now legitimately resolves all the way through to INTACT."""
        s, a_item = self._build(team_b_ancestor_intact=True)
        policy = AccessPolicy().grant_read("team-a", "team-b")
        r = verify_lineage(s, a_item, scope="team-a", policy=policy)
        self.assertEqual(r.state, LineageState.INTACT)

    def test_scoped_audit_substrate_is_not_validated_by_an_inaccessible_scope(self):
        """The audit_substrate()-level version of the same property: MXR-080-0261 explicitly calls out
        that 'a scoped audit can thus be validated by inaccessible items' -- confirms audit_substrate now
        threads scope through verify_lineage rather than just filtering the top-level item list."""
        s, a_item = self._build(team_b_ancestor_intact=True)
        report = audit_substrate(s, scope="team-a")
        self.assertEqual(report["n_items"], 1)  # only team-a's own item is IN the audited domain
        self.assertEqual(report["n_broken"], 0)
        self.assertEqual(report["n_unverified"], 1)  # the team-a item itself, not silently "intact"
        self.assertEqual(report["unverified"][0]["item_id"], a_item)


class TypedProvenanceEdgeTest(unittest.TestCase):
    """MXR-080-0261: `links` (generic, untyped KG-relation edges -- kg_rag/multihop's "related to")
    must never be treated as ancestry. Pre-fix, verify_lineage() read `item.links` directly, so ANY
    KG relation was silently certifiable as a derivation parent just by existing."""

    def test_a_kg_relation_link_is_never_traversed_as_ancestry(self):
        s = Substrate()
        related_entity = s.add(kind="record", text="an unrelated KG entity, not an ancestor")
        subject = s.add(kind="record", text="subject entity", links=[related_entity])
        r = verify_lineage(s, subject)
        self.assertEqual(r.state, LineageState.INTACT)  # no derived_from at all -> trivially intact
        self.assertEqual(r.n_links, 0)  # the KG relation was never even examined
        self.assertEqual(r.visited, 1)  # only the subject itself -- related_entity was never reached

    def test_a_dangling_kg_relation_link_does_not_break_lineage(self):
        """The flip side: a `links` entry to something that doesn't even exist must not fail lineage
        either -- it was never ancestry to begin with, so its dangling-ness is not this function's
        concern (kg_rag/multihop's own consumers are responsible for that surface, unchanged)."""
        s = Substrate()
        subject = s.add(kind="record", text="subject entity", links=["nonexistent-related-entity"])
        r = verify_lineage(s, subject)
        self.assertEqual(r.state, LineageState.INTACT)
        self.assertEqual(r.dangling, [])

    def test_derived_from_and_links_are_independently_readable_on_the_same_item(self):
        """Both fields coexist on one item without interfering -- links for KG association, derived_from
        for genuine ancestry -- confirming the fix is additive, not a repurposing of the old field."""
        s = Substrate()
        ancestor = s.add(kind="text", text="genuine ancestor")
        related = s.add(kind="record", text="merely related entity")
        subject = s.add(kind="record", text="subject", links=[related], derived_from=[ancestor])
        stored = s.get(subject)
        self.assertEqual(stored.links, [related])
        self.assertEqual(stored.derived_from, [ancestor])
        r = verify_lineage(s, subject)
        self.assertEqual(r.state, LineageState.INTACT)
        self.assertEqual(r.visited, 2)  # subject + ancestor -- related is invisible to lineage


class AuditTest(unittest.TestCase):
    def test_audit_counts_intact_and_broken(self):
        s = Substrate()
        doc = s.add(kind="text", text="doc")
        s.add(kind="artifact", text="model", derived_from=[doc])  # intact
        s.add(kind="trace", text="orphan", derived_from=["ghost"])  # broken
        report = audit_substrate(s)
        self.assertEqual(report["n_items"], 3)
        self.assertEqual(report["n_broken"], 1)
        self.assertEqual(report["n_unverified"], 0)
        self.assertEqual(report["n_intact"], 2)
        self.assertEqual(report["broken"][0]["dangling"], ["ghost"])

    def test_clean_store_has_no_broken(self):
        s = Substrate()
        a = s.add(kind="text", text="a")
        s.add(kind="artifact", text="b", derived_from=[a])
        report = audit_substrate(s)
        self.assertEqual(report["n_broken"], 0)
        self.assertEqual(report["n_unverified"], 0)


if __name__ == "__main__":
    unittest.main()
