"""Ontology (D1): classes/relations/axioms as typed constraints + the ontology-typed KG distribution."""

import unittest

import numpy as np

from mixle.reason.ontology import Ontology, OntologyConstrainedKG


def _ont():
    return (
        Ontology()
        .add_class("Agent")
        .add_class("Person", "Agent")
        .add_class("Organization", "Agent")
        .add_class("City")
        .add_relation("employs", "Organization", "Person")
        .add_relation("lives_in", "Person", "City", "functional")
        .add_relation("married_to", "Person", "Person", "symmetric", "irreflexive")
        .add_disjoint("Person", "Organization")
    )


TYPES = {"acme": "Organization", "ada": "Person", "bob": "Person", "paris": "City", "lyon": "City"}


class TripleCheckTest(unittest.TestCase):
    def test_conforming_triple_has_no_violations(self):
        self.assertEqual(_ont().check_triple("acme", "employs", "ada", TYPES), [])

    def test_range_violation_is_named(self):
        probs = _ont().check_triple("acme", "employs", "paris", TYPES)
        self.assertEqual(len(probs), 1)
        self.assertIn("range", probs[0])

    def test_domain_violation_is_named(self):
        probs = _ont().check_triple("ada", "employs", "bob", TYPES)
        self.assertIn("domain", probs[0])

    def test_hierarchy_conformance(self):
        # a relation requiring Agent accepts a Person (subclass)
        ont = _ont().add_relation("controls", "Agent", "Agent")
        self.assertEqual(ont.check_triple("ada", "controls", "acme", TYPES), [])

    def test_irreflexive_axiom(self):
        probs = _ont().check_triple("ada", "married_to", "ada", TYPES)
        self.assertTrue(any("irreflexive" in p for p in probs))

    def test_unknown_relation_is_a_violation(self):
        self.assertIn("unknown relation", _ont().check_triple("ada", "eats", "paris", TYPES)[0])

    def test_untyped_entities_pass_signature_checks(self):
        # no type claim -> no domain/range violation to name (honest: unconstrained, not asserted-valid)
        self.assertEqual(_ont().check_triple("mystery", "employs", "enigma", {}), [])

    def test_unknown_axiom_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            _ont().add_relation("x", "Person", "Person", "sparkly")


class GraphAuditTest(unittest.TestCase):
    def test_functional_relation_with_two_tails_is_flagged(self):
        g = [("ada", "lives_in", "paris"), ("ada", "lives_in", "lyon")]
        rep = _ont().check_graph(g, TYPES)
        self.assertFalse(rep["consistent"])
        self.assertTrue(any("functional" in v["problems"][0] for v in rep["violations"]))

    def test_asymmetric_both_directions_flagged(self):
        ont = _ont().add_relation("manages", "Person", "Person", "asymmetric")
        g = [("ada", "manages", "bob"), ("bob", "manages", "ada")]
        rep = ont.check_graph(g, TYPES)
        self.assertFalse(rep["consistent"])

    def test_consistent_graph_passes(self):
        g = [("acme", "employs", "ada"), ("ada", "lives_in", "paris")]
        self.assertTrue(_ont().check_graph(g, TYPES)["consistent"])

    def test_filter_splits_kept_and_rejected(self):
        kept, rejected = _ont().filter_triples([("acme", "employs", "ada"), ("acme", "employs", "paris")], TYPES)
        self.assertEqual(kept, [("acme", "employs", "ada")])
        self.assertEqual(rejected[0]["triple"], ("acme", "employs", "paris"))


class ConstrainedKGTest(unittest.TestCase):
    def _ckg(self):
        from mixle.inference import optimize
        from mixle.stats.graphs.knowledge_graph import KnowledgeGraphEstimator

        ents = ["acme", "ada", "bob", "paris", "lyon"]
        rels = ["employs", "lives_in", "married_to"]
        e = {x: i for i, x in enumerate(ents)}
        r = {x: i for i, x in enumerate(rels)}
        data = [
            (e["acme"], r["employs"], e["ada"]),
            (e["acme"], r["employs"], e["bob"]),
            (e["ada"], r["lives_in"], e["paris"]),
            (e["bob"], r["lives_in"], e["lyon"]),
        ] * 20
        kg = optimize(
            data,
            KnowledgeGraphEstimator(num_entities=5, num_relations=3, dim=8),
            out=None,
            max_its=40,
            rng=np.random.RandomState(0),
        )
        return OntologyConstrainedKG(kg, _ont(), entities=ents, relations=rels, types=TYPES)

    def test_tail_posterior_masses_only_range_conforming_entities(self):
        post = self._ckg().tail_posterior("acme", "employs")
        self.assertEqual(set(post), {"ada", "bob"})  # only Persons; cities/orgs get ZERO mass
        self.assertAlmostEqual(sum(post.values()), 1.0, places=6)  # renormalized

    def test_complete_returns_the_learned_consistent_tail(self):
        tail, p = self._ckg().complete("ada", "lives_in")
        self.assertIn(tail, {"paris", "lyon"})  # a City, never a Person/Org
        self.assertGreater(p, 0.5)


class ConstrainedDecodeTest(unittest.TestCase):
    def _decode(self, floor=0.5):
        from mixle.reason.graph_llm import GraphLLM
        from mixle.reason.ontology import constrained_decode

        ont = (
            Ontology()
            .add_class("Person")
            .add_class("City")
            .add_relation("lives_in", "Person", "City")
            .add_relation("born_in", "Person", "City")
        )
        types = {"ada": "Person", "paris": "City", "lyon": "City"}
        rng = np.random.RandomState(0)

        def generate(prompt):
            out = ["ada|lives_in|paris"]  # reliable fact
            if rng.rand() < 0.5:
                out.append("paris|lives_in|ada")  # schema-violating hallucination
            if rng.rand() < 0.2:
                out.append("ada|born_in|lyon")  # under-confident fact
            return ";".join(out)

        def parse(s):
            return [tuple(t.split("|")) for t in s.split(";") if t]

        llm = GraphLLM(generate, parse, n=25)
        return constrained_decode(llm, "facts about ada", ont, types, floor=floor)

    def test_reliable_consistent_fact_is_asserted(self):
        dec = self._decode()
        self.assertIn(("ada", "lives_in", "paris"), dec.asserted())
        self.assertEqual(dec.facts[0][1], 1.0)  # asserted in every sample

    def test_schema_violating_hallucination_is_rejected_with_reason(self):
        dec = self._decode()
        rejected = {tuple(r["triple"]) for r in dec.rejected}
        self.assertIn(("paris", "lives_in", "ada"), rejected)
        self.assertNotIn(("paris", "lives_in", "ada"), dec.asserted())  # never asserted
        reason = next(r for r in dec.rejected if tuple(r["triple"]) == ("paris", "lives_in", "ada"))
        self.assertIn("domain", reason["problems"][0])  # the WHY is named

    def test_underconfident_fact_is_withheld_not_silently_dropped(self):
        dec = self._decode()
        withheld = {t for t, _ in dec.below_floor}
        self.assertIn(("ada", "born_in", "lyon"), withheld)
        self.assertNotIn(("ada", "born_in", "lyon"), dec.asserted())

    def test_floor_zero_asserts_all_consistent_facts(self):
        dec = self._decode(floor=0.0)
        self.assertIn(("ada", "born_in", "lyon"), dec.asserted())  # now above the (zero) floor
        self.assertEqual(dec.below_floor, [])


if __name__ == "__main__":
    unittest.main()
