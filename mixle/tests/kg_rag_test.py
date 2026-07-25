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


def _ada_triples_gen():
    """A ONE-SHOT generator -- exhausted after a single full iteration, unlike ``TRIPLES`` (a list,
    replayable forever). ``ada`` is a head (subject), so an inventory pass that only gets partway
    through a generator can still see it -- the audit's own reproduction shape."""
    yield ("ada", "lives_in", "paris")
    yield ("paris", "capital_of", "france")


class OneShotGeneratorTest(unittest.TestCase):
    """MXR-080-0253: kg_action()/retrieve_triples() must consume a ``triples`` iterable exactly once.

    Before the fix, ``kg_action`` walked ``triples`` twice while building its advertised entity
    inventory (a head-entity set comprehension, then a separate tail-entity one) and again later
    inside the fired action's ``_run``. A one-shot generator is exhausted by the first walk, so the
    second inventory pass silently contributed nothing and the later ``_run`` call saw nothing at
    all: the reproduced action advertised entity ``ada`` (from the first, partially-successful pass)
    but returned no fact for it, ever -- not stale evidence, a structural guarantee that was never
    met for any non-replayable iterable.
    """

    def test_audit_scenario_generator_input_advertises_and_answers_ada(self):
        """The audit's own exact reproduction: a one-shot generator including a fact about ``ada``,
        fed straight to kg_action(). Pre-fix this failed at the final assertion -- the description
        advertised 'ada' (from the surviving first pass) while action.run(...) came back []."""
        act = kg_action(_ada_triples_gen())
        self.assertIn("ada", act.description)  # advertised from the (now single, safe) inventory pass
        result = act.run("where does ada live")
        self.assertEqual(result, ["ada lives_in paris"])  # must ACTUALLY be answerable, not just advertised

    def test_advertised_inventory_includes_tail_only_entities(self):
        # pre-fix, the tail-entity comprehension always ran against an already-exhausted generator
        # and contributed nothing -- 'france' (only ever an object, never a subject) never appeared.
        act = kg_action(_ada_triples_gen())
        self.assertIn("france", act.description)

    def test_action_run_is_repeatable_across_multiple_calls(self):
        # a materialized-once triples set must serve every call, not just whichever call happened to
        # run first -- guards against a fix that merely defers exhaustion by one call instead of
        # removing it. Second query names 'france' only (not 'paris'/'ada') so it links to exactly
        # the second triple, keeping the two assertions unambiguous.
        act = kg_action(_ada_triples_gen())
        first = act.run("where does ada live")
        second = act.run("where is france located")
        self.assertEqual(first, ["ada lives_in paris"])
        self.assertEqual(second, ["paris capital_of france"])

    def test_retrieve_triples_accepts_a_one_shot_generator_directly(self):
        out = retrieve_triples((t for t in TRIPLES), "where does ada live", ontology=_ont(), types=TYPES)
        self.assertEqual(out["entities"], ["ada"])
        self.assertIn(("ada", "lives_in", "paris"), out["facts"])
        self.assertIn(("ada", "works_at", "acme"), out["facts"])

    def test_investigate_answers_from_a_generator_backed_action(self):
        # end-to-end through the reasoner, not just the unit-level run() -- the module's own stated
        # contract (investigate() buying typed evidence) must hold for a one-shot source too.
        act = kg_action((t for t in TRIPLES), ontology=_ont(), types=TYPES, description="people and cities: ada")
        inv = investigate("where does ada live", [act], lambda q, ctx: ctx.splitlines()[0], min_confidence=0.1)
        self.assertFalse(inv.abstained)
        self.assertEqual(inv.answer, "ada lives_in paris")


class TripleArityValidationTest(unittest.TestCase):
    """MXR-080-0253: every triple must have exactly 3 elements, checked once at the boundary.

    Before the fix, arity was unchecked: a short triple raised an opaque IndexError wherever its
    missing slot first got read (not necessarily anywhere near construction), and a long triple was
    silently accepted -- its extra field quietly dropped by every consumer that only ever reads
    ``t[0]``/``t[2]`` -- until something finally unpacked all of it and raised a confusing
    ``ValueError: too many values to unpack``."""

    def test_retrieve_triples_rejects_a_short_triple(self):
        with self.assertRaises(ValueError):
            retrieve_triples([("ada", "lives_in")], "where does ada live")

    def test_retrieve_triples_rejects_a_long_triple(self):
        with self.assertRaises(ValueError):
            retrieve_triples([("ada", "lives_in", "paris", "confidence=0.9")], "where does ada live")

    def test_kg_action_rejects_malformed_triples_at_construction(self):
        # fails eagerly, when the action is built -- not lazily, inside a fired run() where
        # investigate() would otherwise swallow it into an opaque "failed" step.
        with self.assertRaises(ValueError):
            kg_action([("ada", "lives_in")])
        with self.assertRaises(ValueError):
            kg_action([("ada", "lives_in", "paris", "extra")])

    def test_well_formed_triples_are_unaffected(self):
        out = retrieve_triples(TRIPLES, "where does ada live")
        self.assertIn(("ada", "lives_in", "paris"), out["facts"])


class RankLimitValidationTest(unittest.TestCase):
    """MXR-080-0253: k must be an exact, non-negative int (mirrors MXR-080-0236's
    ``_require_count`` contract, shared via mixle.substrate.core) -- a negative k used to fall
    through to ordinary Python slicing (``hits[:-1]``, silently dropping just the last hit) instead
    of being rejected."""

    def test_retrieve_triples_rejects_negative_k(self):
        with self.assertRaises(ValueError):
            retrieve_triples(TRIPLES, "tell me about ada", k=-1)

    def test_retrieve_triples_rejects_bool_k(self):
        with self.assertRaises(TypeError):
            retrieve_triples(TRIPLES, "tell me about ada", k=True)

    def test_retrieve_triples_rejects_fractional_k(self):
        with self.assertRaises(TypeError):
            retrieve_triples(TRIPLES, "tell me about ada", k=2.5)

    def test_retrieve_triples_k_zero_returns_nothing(self):
        out = retrieve_triples(TRIPLES, "tell me about ada", k=0)
        self.assertEqual(out["facts"], [])

    def test_negative_k_no_longer_silently_drops_the_last_hit(self):
        """The audit's own reproduction: k=-1 used to return len(hits)-1 facts (Python's ``[:-1]``)
        instead of raising, for a query with more than one matching fact."""
        with self.assertRaises(ValueError):
            retrieve_triples(TRIPLES, "where does ada live and work", k=-1)

    def test_kg_action_rejects_negative_k_at_construction(self):
        with self.assertRaises(ValueError):
            kg_action(TRIPLES, k=-1)

    def test_kg_action_rejects_bool_and_fractional_k_at_construction(self):
        with self.assertRaises(TypeError):
            kg_action(TRIPLES, k=True)
        with self.assertRaises(TypeError):
            kg_action(TRIPLES, k=1.5)


if __name__ == "__main__":
    unittest.main()
