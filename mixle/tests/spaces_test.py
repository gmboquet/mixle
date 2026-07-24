"""Space (P1): team-scoped views over the substrate with explicit, audited publish.

MXR-080-0264 hardening: every read/write/publish is checked against a centralized
:class:`AccessPolicy` rather than trusting caller-supplied scope parameters (``AccessPolicyTest``,
``CrossTenantLeakTest``).
"""

import unittest

from mixle.substrate import (
    PUBLIC,
    AccessDeniedError,
    AccessPolicy,
    Space,
    Substrate,
    history,
    merge_versions,
    publish,
    version_of,
    visible_scopes,
)


def _shared_store():
    s = Substrate()
    ids = {
        "a": s.add(kind="text", text="alpha teamA secret roadmap", scope="teamA"),
        "b": s.add(kind="text", text="beta teamB secret pricing", scope="teamB"),
        "p": s.add(kind="text", text="gamma public onboarding guide", scope=PUBLIC),
    }
    return s, ids


def _staffed_policy() -> AccessPolicy:
    """alice staffs teamA, bob staffs teamB, carol is an org-level reconciler authorized for both --
    the standing every direct (non-Space) publish call below needs to move something OUT of a
    non-public, non-PUBLIC scope it doesn't itself share a name with."""
    return AccessPolicy().grant("alice", "teamA").grant("bob", "teamB").grant("carol", "teamA").grant("carol", "teamB")


class VisibilityTest(unittest.TestCase):
    def test_visible_scopes_is_own_plus_shared(self):
        self.assertEqual(visible_scopes("teamA"), {"teamA", "public"})
        self.assertEqual(visible_scopes("teamA", shared=("public", "org")), {"teamA", "public", "org"})

    def test_a_team_sees_its_own_and_public_not_another_teams(self):
        s, _ = _shared_store()
        a_texts = {i.text[:10] for i in Space(s, "teamA").all()}
        self.assertIn("alpha team", a_texts)
        self.assertIn("gamma publ", a_texts)
        self.assertNotIn("beta teamB", a_texts)  # teamB's private item is invisible to teamA

    def test_retrieve_respects_the_boundary(self):
        s, _ = _shared_store()
        # both teams have a "secret"; teamA must never retrieve teamB's
        hits = {i.text[:10] for i in Space(s, "teamA").retrieve("secret", k=5).items}
        self.assertNotIn("beta teamB", hits)


class AccessPolicyTest(unittest.TestCase):
    """MXR-080-0264: AccessPolicy is the single, centralized read/write authorization decision."""

    def test_home_scope_and_public_need_no_grant(self):
        policy = AccessPolicy()
        self.assertTrue(policy.can_read("teamA", "teamA"))
        self.assertTrue(policy.can_write("teamA", "teamA"))
        self.assertTrue(policy.can_read("teamA", PUBLIC))
        self.assertTrue(policy.can_write("teamA", PUBLIC))

    def test_another_teams_scope_is_denied_by_default(self):
        policy = AccessPolicy()
        self.assertFalse(policy.can_read("teamA", "teamB"))
        self.assertFalse(policy.can_write("teamA", "teamB"))

    def test_explicit_read_grant_does_not_imply_write(self):
        policy = AccessPolicy().grant_read("teamA", "teamB")
        self.assertTrue(policy.can_read("teamA", "teamB"))
        self.assertFalse(policy.can_write("teamA", "teamB"))

    def test_grant_gives_both_read_and_write(self):
        policy = AccessPolicy().grant("teamA", "teamB")
        self.assertTrue(policy.can_read("teamA", "teamB"))
        self.assertTrue(policy.can_write("teamA", "teamB"))

    def test_revoke_removes_a_previously_granted_scope(self):
        policy = AccessPolicy().grant("teamA", "teamB")
        policy.revoke("teamA", "teamB")
        self.assertFalse(policy.can_read("teamA", "teamB"))
        self.assertFalse(policy.can_write("teamA", "teamB"))

    def test_empty_principal_is_never_authorized_even_for_its_own_name(self):
        self.assertFalse(AccessPolicy().can_read("", ""))

    def test_require_read_raises_access_denied(self):
        with self.assertRaises(AccessDeniedError):
            AccessPolicy().require_read("teamA", "teamB")

    def test_require_write_raises_access_denied(self):
        with self.assertRaises(AccessDeniedError):
            AccessPolicy().require_write("teamA", "teamB")


class CrossTenantLeakTest(unittest.TestCase):
    """MXR-080-0264's concrete reproductions: a team-A space configured with team-B's scope must not
    read team-B's private items; publish() must not move a known item to an arbitrary scope just
    because from_scope was omitted; Space.add() must not write directly into another team's scope."""

    def test_space_configured_with_another_teams_scope_is_rejected_at_construction(self):
        """The audit's exact scenario: 'a team-A space configured with (\"team-b\",) read team-B's
        secrets'. Construction itself must now refuse the unauthorized scope, not silently allow it
        and leak on every subsequent read."""
        s, _ = _shared_store()
        with self.assertRaises(AccessDeniedError):
            Space(s, "teamA", shared=("teamB",))

    def test_an_explicitly_granted_cross_team_read_does_cross_and_nothing_else_does(self):
        """The positive control for the test above: with a REAL, explicit grant (not caller
        convention), the same shared= configuration legitimately works, and only that scope crosses."""
        s, _ = _shared_store()
        policy = AccessPolicy().grant_read("teamA", "teamB")
        space = Space(s, "teamA", shared=("teamB",), policy=policy)
        texts = {i.text[:10] for i in space.all()}
        self.assertIn("beta teamB", texts)
        self.assertNotIn("gamma publ", texts)  # PUBLIC was never in shared= this time -- still isolated

    def test_publish_cannot_move_a_known_item_to_an_arbitrary_scope_without_from_scope(self):
        """'The exported publish() can move any known item to any scope when from_scope is omitted,
        with no ACL.' alice (teamA) knows teamB's private item's id and tries to publish it straight
        to PUBLIC with NO from_scope filter at all -- must still be denied."""
        s, ids = _shared_store()
        published = publish(s, [ids["b"]], to=PUBLIC, by="alice", policy=_staffed_policy())
        self.assertEqual(published, [])  # denied: alice has no read grant on teamB
        self.assertEqual(s.get(ids["b"]).scope, "teamB")  # untouched, never moved, never exposed

    def test_publish_requires_an_authenticated_principal(self):
        s, ids = _shared_store()
        with self.assertRaises(AccessDeniedError):
            publish(s, [ids["a"]], to=PUBLIC, by="")

    def test_space_add_cannot_write_directly_into_another_teams_scope(self):
        """'Space.add(scope=...) can write directly to another team's scope.'"""
        s, _ = _shared_store()
        space = Space(s, "teamA")
        with self.assertRaises(AccessDeniedError):
            space.add(kind="text", text="planted by teamA", scope="teamB")
        # nothing was written into teamB as a result
        self.assertEqual({i.text for i in s.all(scope="teamB")}, {"beta teamB secret pricing"})

    def test_space_add_to_public_still_works_with_no_extra_grant(self):
        """The one caller-chosen scope override this module has always sanctioned (see Space.add's
        docstring: "pass scope=PUBLIC to share immediately") keeps working with zero new ceremony."""
        s, _ = _shared_store()
        iid = Space(s, "teamA").add(kind="text", text="shared immediately", scope=PUBLIC)
        self.assertEqual(s.get(iid).scope, PUBLIC)

    def test_space_requires_a_non_empty_team(self):
        s, _ = _shared_store()
        with self.assertRaises(AccessDeniedError):
            Space(s, "")

    def test_revoking_a_grant_after_construction_hides_the_scope_immediately(self):
        """Space reads re-check the policy live, not just once at construction, so a later revocation
        is not a stale, unenforced permission."""
        s, _ = _shared_store()
        policy = AccessPolicy().grant_read("teamA", "teamB")
        space = Space(s, "teamA", shared=("teamB",), policy=policy)
        self.assertIn("teamB", space.scopes)
        policy.revoke("teamA", "teamB")
        self.assertNotIn("teamB", space.scopes)
        self.assertNotIn("beta teamB", {i.text[:10] for i in space.all()})


class PublishTest(unittest.TestCase):
    def test_publish_shares_and_audits(self):
        s, ids = _shared_store()
        published = publish(s, [ids["a"]], to=PUBLIC, by="alice", policy=_staffed_policy())
        self.assertEqual(published, [ids["a"]])
        item = s.get(ids["a"])
        self.assertEqual(item.scope, PUBLIC)  # re-scoped
        self.assertEqual(item.provenance["published_by"], "alice")  # audit trail
        self.assertEqual(item.provenance["published_from"], "teamA")

    def test_published_item_becomes_visible_to_other_teams(self):
        s, ids = _shared_store()
        self.assertNotIn("alpha team", {i.text[:10] for i in Space(s, "teamB").all()})
        Space(s, "teamA").publish([ids["a"]])  # by defaults to "teamA" -- its own home scope, no grant needed
        self.assertIn("alpha team", {i.text[:10] for i in Space(s, "teamB").all()})  # now shared

    def test_from_scope_guards_publishing(self):
        s, ids = _shared_store()
        # "bob" (teamB) cannot publish teamA's item: the from_scope guard skips it
        published = publish(s, [ids["a"]], to=PUBLIC, by="bob", from_scope="teamB", policy=_staffed_policy())
        self.assertEqual(published, [])
        self.assertEqual(s.get(ids["a"]).scope, "teamA")  # unchanged

    def test_space_publish_only_touches_own_items(self):
        s, ids = _shared_store()
        # teamB's Space.publish over teamA's id is a no-op (own-scope guard)
        self.assertEqual(Space(s, "teamB").publish([ids["a"]]), [])

    def test_missing_ids_are_skipped(self):
        s, ids = _shared_store()
        self.assertEqual(publish(s, ["nonexistent", ids["a"]], by="teamA"), [ids["a"]])

    def test_space_add_defaults_to_team_scope(self):
        s, _ = _shared_store()
        space = Space(s, "teamA")
        iid = space.add(kind="text", text="new teamA note")
        self.assertEqual(s.get(iid).scope, "teamA")

    def test_publish_does_not_alias_the_original_items_mutable_fields(self):
        """Regression test: publish() built the "new" published SubstrateItem with
        payload=item.payload / tags=item.tags / links=item.links -- BY REFERENCE, so mutating a
        previously-fetched copy of the item silently mutated the currently-stored published item
        too (provenance was already copied via dict(item.provenance); payload/tags/links were not)."""
        s = Substrate()
        iid = s.add(kind="text", text="alpha", payload={"k": 1}, tags=["a"], links=["other"], scope="teamA")
        original = s.get(iid)  # fetched BEFORE publish

        publish(s, [iid], to=PUBLIC, by="alice", policy=_staffed_policy())

        # mutate the pre-publish object a caller might still be holding
        original.tags.append("MUTATED-TAG")
        original.payload["k"] = "MUTATED-PAYLOAD"
        original.links.append("MUTATED-LINK")

        stored = s.get(iid)
        self.assertNotIn("MUTATED-TAG", stored.tags)
        self.assertEqual(stored.payload["k"], 1)
        self.assertNotIn("MUTATED-LINK", stored.links)


class VersioningTest(unittest.TestCase):
    def test_each_publish_bumps_version_and_records_history(self):
        s, ids = _shared_store()
        policy = _staffed_policy()
        publish(s, [ids["a"]], by="alice", policy=policy)
        publish(s, [ids["a"]], by="alice", policy=policy)
        self.assertEqual(version_of(s.get(ids["a"])), 2)  # monotonic
        self.assertEqual(len(history(s, ids["a"])), 2)  # every share recorded, no silent overwrite
        self.assertEqual(history(s, ids["a"])[0]["published_by"], "alice")

    def test_history_of_unknown_item_is_empty(self):
        s, _ = _shared_store()
        self.assertEqual(history(s, "nope"), [])

    def test_merge_keeps_higher_version_text_and_unions_metadata(self):
        s = Substrate()
        a = s.add(kind="text", text="v-a", scope="teamA", tags=["plan"], links=["x"])
        b = s.add(kind="text", text="v-b", scope="teamB", tags=["price"], links=["y"])
        policy = _staffed_policy()
        publish(s, [a], by="alice", policy=policy)  # a -> v1
        publish(s, [b], by="bob", policy=policy)
        publish(s, [b], by="bob", policy=policy)  # b -> v2 (higher)
        keep = merge_versions(s, a, b, by="carol")
        merged = s.get(keep)
        self.assertEqual(merged.text, "v-b")  # higher version wins
        self.assertEqual(merged.tags, ["plan", "price"])  # unioned
        self.assertEqual(merged.links, ["x", "y"])
        self.assertIsNone(s.get(b))  # merged-away item removed
        self.assertGreater(version_of(merged), 2)  # bumped past both

    def test_merge_records_both_parents(self):
        s = Substrate()
        a = s.add(kind="text", text="a", scope="teamA")
        b = s.add(kind="text", text="b", scope="teamB")
        policy = _staffed_policy()
        publish(s, [a], by="alice", policy=policy)
        publish(s, [b], by="bob", policy=policy)
        keep = merge_versions(s, a, b, by="carol")
        last = history(s, keep)[-1]
        self.assertEqual(last["merged_by"], "carol")
        parent_ids = {p["id"] for p in last["parents"]}
        self.assertEqual(parent_ids, {a, b})  # lineage names both edits

    def test_merge_prefer_keep_wins_regardless_of_version(self):
        s = Substrate()
        a = s.add(kind="text", text="keep-me", scope="teamA")
        b = s.add(kind="text", text="newer", scope="teamB")
        policy = _staffed_policy()
        publish(s, [b], by="bob", policy=policy)
        publish(s, [b], by="bob", policy=policy)  # b higher version
        keep = merge_versions(s, a, b, prefer="keep")
        self.assertEqual(s.get(keep).text, "keep-me")  # keep wins despite lower version

    def test_merge_missing_item_returns_none(self):
        s, ids = _shared_store()
        self.assertIsNone(merge_versions(s, ids["a"], "nope"))

    def test_merge_versions_does_not_alias_the_winners_payload(self):
        """Regression test: merge_versions() passed winner_payload (aliased directly from
        other.payload or keep.payload, whichever version wins) straight into the surviving
        SubstrateItem BY REFERENCE."""
        s = Substrate()
        a = s.add(kind="text", text="v-a", scope="teamA", payload={"k": "a-payload"})
        b = s.add(kind="text", text="v-b", scope="teamB", payload={"k": "b-payload"})
        policy = _staffed_policy()
        publish(s, [b], by="bob", policy=policy)  # b -> v1, higher than a's v0: b's payload wins under prefer="latest"
        original_b = s.get(b)  # fetched BEFORE the merge

        keep = merge_versions(s, a, b, by="carol")
        original_b.payload["k"] = "MUTATED-PAYLOAD"

        self.assertEqual(s.get(keep).payload["k"], "b-payload")


if __name__ == "__main__":
    unittest.main()
