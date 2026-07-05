"""KG-RAG (D3): entity linking + typed triple retrieval as a reasoner action."""

import unittest

from mixle.reason.ontology import Ontology
from mixle.substrate.act import investigate
from mixle.substrate.kg_rag import kg_action, link_entities, retrieve_triples

TRIPLES = [
    ("ada", "lives_in", "paris"),
    ("ada", "works_at", "acme"),
    ("bob", "lives_in", "lyon"),
    ("paris", "lives_in", "ada"),  # dirty: type-violating
]


def _ont():
    return (
        Ontology()
        .add_class("Person")
        .add_class("City")
        .add_class("Organization")
        .add_relation("lives_in", "Person", "City")
        .add_relation("works_at", "Person", "Organization")
    )


TYPES = {"ada": "Person", "bob": "Person", "paris": "City", "lyon": "City", "acme": "Organization"}


class LinkEntitiesTest(unittest.TestCase):
    def test_links_mentioned_entities_only(self):
        self.assertEqual(link_entities("where does ada live", ["ada", "bob", "paris"]), ["ada"])

    def test_longest_name_wins_over_substrings(self):
        linked = link_entities("facts about new york city", ["york", "new york city"])
        self.assertEqual(linked, ["new york city"])  # the substring does not double-link

    def test_no_mention_links_nothing(self):
        self.assertEqual(link_entities("boiling point of xenon", ["ada", "paris"]), [])


class RetrieveTriplesTest(unittest.TestCase):
    def test_returns_facts_touching_linked_entities(self):
        out = retrieve_triples(TRIPLES, "where does ada live", ontology=_ont(), types=TYPES)
        self.assertEqual(out["entities"], ["ada"])
        self.assertIn(("ada", "lives_in", "paris"), out["facts"])
        self.assertIn(("ada", "works_at", "acme"), out["facts"])
        self.assertNotIn(("bob", "lives_in", "lyon"), out["facts"])  # bob wasn't asked about

    def test_ontology_excludes_dirty_triples_from_evidence(self):
        out = retrieve_triples(TRIPLES, "where does ada live", ontology=_ont(), types=TYPES)
        self.assertNotIn(("paris", "lives_in", "ada"), out["facts"])  # never served
        self.assertEqual([r["triple"] for r in out["rejected"]], [("paris", "lives_in", "ada")])

    def test_without_ontology_everything_matching_is_served(self):
        out = retrieve_triples(TRIPLES, "where does ada live")
        self.assertIn(("paris", "lives_in", "ada"), out["facts"])  # no schema, no filter (honest default)

    def test_k_caps_the_fact_count(self):
        out = retrieve_triples(TRIPLES, "where does ada live", k=1)
        self.assertEqual(len(out["facts"]), 1)


class KgActionTest(unittest.TestCase):
    def test_reasoner_answers_from_typed_facts(self):
        act = kg_action(TRIPLES, ontology=_ont(), types=TYPES, description="where people live and work: ada bob paris")
        inv = investigate("where does ada live", [act], lambda q, ctx: ctx.splitlines()[0], min_confidence=0.1)
        self.assertFalse(inv.abstained)
        self.assertEqual(inv.answer, "ada lives_in paris")

    def test_unlinked_question_yields_no_evidence_and_abstains(self):
        act = kg_action(TRIPLES, ontology=_ont(), types=TYPES, description="people and cities")
        inv = investigate("boiling point of xenon", [act], lambda q, ctx: "x", min_confidence=0.3)
        self.assertTrue(inv.abstained)  # no forced match; honest fall-through


if __name__ == "__main__":
    unittest.main()
