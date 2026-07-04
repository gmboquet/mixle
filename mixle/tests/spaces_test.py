"""Space (P1): team-scoped views over the substrate with explicit, audited publish."""

import unittest

from mixle.substrate import (
    PUBLIC,
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


class PublishTest(unittest.TestCase):
    def test_publish_shares_and_audits(self):
        s, ids = _shared_store()
        published = publish(s, [ids["a"]], to=PUBLIC, by="alice")
        self.assertEqual(published, [ids["a"]])
        item = s.get(ids["a"])
        self.assertEqual(item.scope, PUBLIC)  # re-scoped
        self.assertEqual(item.provenance["published_by"], "alice")  # audit trail
        self.assertEqual(item.provenance["published_from"], "teamA")

    def test_published_item_becomes_visible_to_other_teams(self):
        s, ids = _shared_store()
        self.assertNotIn("alpha team", {i.text[:10] for i in Space(s, "teamB").all()})
        Space(s, "teamA").publish([ids["a"]], by="alice")
        self.assertIn("alpha team", {i.text[:10] for i in Space(s, "teamB").all()})  # now shared

    def test_from_scope_guards_publishing(self):
        s, ids = _shared_store()
        # teamB cannot publish teamA's item: the from_scope guard skips it
        published = publish(s, [ids["a"]], to=PUBLIC, from_scope="teamB")
        self.assertEqual(published, [])
        self.assertEqual(s.get(ids["a"]).scope, "teamA")  # unchanged

    def test_space_publish_only_touches_own_items(self):
        s, ids = _shared_store()
        # teamB's Space.publish over teamA's id is a no-op (own-scope guard)
        self.assertEqual(Space(s, "teamB").publish([ids["a"]]), [])

    def test_missing_ids_are_skipped(self):
        s, ids = _shared_store()
        self.assertEqual(publish(s, ["nonexistent", ids["a"]], by="x"), [ids["a"]])

    def test_space_add_defaults_to_team_scope(self):
        s, _ = _shared_store()
        space = Space(s, "teamA")
        iid = space.add(kind="text", text="new teamA note")
        self.assertEqual(s.get(iid).scope, "teamA")


class VersioningTest(unittest.TestCase):
    def test_each_publish_bumps_version_and_records_history(self):
        s, ids = _shared_store()
        publish(s, [ids["a"]], by="alice")
        publish(s, [ids["a"]], by="alice")
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
        publish(s, [a], by="alice")  # a -> v1
        publish(s, [b], by="bob")
        publish(s, [b], by="bob")  # b -> v2 (higher)
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
        publish(s, [a], by="x")
        publish(s, [b], by="y")
        keep = merge_versions(s, a, b, by="carol")
        last = history(s, keep)[-1]
        self.assertEqual(last["merged_by"], "carol")
        parent_ids = {p["id"] for p in last["parents"]}
        self.assertEqual(parent_ids, {a, b})  # lineage names both edits

    def test_merge_prefer_keep_wins_regardless_of_version(self):
        s = Substrate()
        a = s.add(kind="text", text="keep-me", scope="teamA")
        b = s.add(kind="text", text="newer", scope="teamB")
        publish(s, [b], by="y")
        publish(s, [b], by="y")  # b higher version
        keep = merge_versions(s, a, b, prefer="keep")
        self.assertEqual(s.get(keep).text, "keep-me")  # keep wins despite lower version

    def test_merge_missing_item_returns_none(self):
        s, ids = _shared_store()
        self.assertIsNone(merge_versions(s, ids["a"], "nope"))


if __name__ == "__main__":
    unittest.main()
