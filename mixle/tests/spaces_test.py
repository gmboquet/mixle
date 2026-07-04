"""Space (P1): team-scoped views over the substrate with explicit, audited publish."""

import unittest

from mixle.substrate import PUBLIC, Space, Substrate, publish, visible_scopes


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


if __name__ == "__main__":
    unittest.main()
