"""Ontology (D1): classes/relations/axioms as typed constraints + the ontology-typed KG distribution."""

import unittest

import numpy as np

from mixle.reason.ontology import Ontology, OntologyConstrainedKG, constrained_decode


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

    def test_symmetric_one_directional_edge_is_flagged(self):
        # married_to is declared symmetric (see _ont()); asserting only one direction must be named.
        g = [("ada", "married_to", "bob")]
        rep = _ont().check_graph(g, TYPES)
        self.assertFalse(rep["consistent"])
        self.assertTrue(any("symmetric" in v["problems"][0] for v in rep["violations"]))

    def test_symmetric_both_directions_passes(self):
        g = [("ada", "married_to", "bob"), ("bob", "married_to", "ada")]
        self.assertTrue(_ont().check_graph(g, TYPES)["consistent"])

    def test_transitive_closure_gap_is_flagged(self):
        ont = _ont().add_relation("ancestor_of", "Person", "Person", "transitive")
        g = [("ada", "ancestor_of", "bob"), ("bob", "ancestor_of", "acme")]
        rep = ont.check_graph(g, {**TYPES, "acme": "Person"})
        self.assertFalse(rep["consistent"])
        self.assertTrue(any("transitive" in v["problems"][0] for v in rep["violations"]))

    def test_transitive_closed_graph_passes(self):
        ont = _ont().add_relation("ancestor_of", "Person", "Person", "transitive")
        g = [
            ("ada", "ancestor_of", "bob"),
            ("bob", "ancestor_of", "acme"),
            ("ada", "ancestor_of", "acme"),  # closure edge present
        ]
        rep = ont.check_graph(g, {**TYPES, "acme": "Person"})
        self.assertTrue(rep["consistent"])

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


class ConstructionSafetyTest(unittest.TestCase):
    """MXR-080-0297: duplicate/cyclic/invalid construction must be rejected, not silently accepted."""

    def test_add_class_rejects_duplicate_name(self):
        ont = Ontology().add_class("A").add_class("B", "A").add_class("C", "B")
        with self.assertRaises(ValueError):
            ont.add_class("C", "A")  # would have silently overwritten C's parent B -> A
        self.assertEqual(ont.classes["C"], "B")  # unchanged by the rejected call

    def test_add_class_duplicate_self_reference_rejected(self):
        # a duplicate add_class("X", "X") is rejected as a duplicate name before self-parentage
        # would even be considered -- it can never sneak through as a "new" class.
        ont = Ontology().add_class("X")
        with self.assertRaises(ValueError):
            ont.add_class("X", "X")
        self.assertIsNone(ont.classes["X"])

    def test_add_class_unknown_parent_rejected(self):
        with self.assertRaises(ValueError):
            Ontology().add_class("X", "NoSuchParent")

    def test_replace_class_requires_existing_name(self):
        with self.assertRaises(KeyError):
            Ontology().add_class("A").replace_class("NeverAdded", "A")

    def test_replace_class_unknown_parent_rejected(self):
        with self.assertRaises(ValueError):
            Ontology().add_class("A").replace_class("A", "NoSuchParent")

    def test_replace_class_self_parent_rejected(self):
        ont = Ontology().add_class("X")
        with self.assertRaises(ValueError):
            ont.replace_class("X", "X")

    def test_replace_class_cycle_through_replacement_rejected(self):
        # P -> Q is fine; reparenting P to Q (Q's current ancestor chain already leads back to P)
        # would close a 2-cycle P -> Q -> P. This can ONLY happen through replacement, since a
        # freshly add_class'd name can never already be its own ancestor.
        ont = Ontology().add_class("P").add_class("Q", "P")
        with self.assertRaises(ValueError):
            ont.replace_class("P", "Q")
        self.assertEqual(ont.classes["P"], None)  # unchanged by the rejected call
        self.assertEqual(ont.classes["Q"], "P")

    def test_replace_class_deeper_cycle_through_replacement_rejected(self):
        ont = Ontology().add_class("A").add_class("B", "A").add_class("C", "B")  # C -> B -> A
        with self.assertRaises(ValueError):
            ont.replace_class("A", "C")  # would close A -> C -> B -> A

    def test_legitimate_replace_class_reparents_without_error(self):
        ont = Ontology().add_class("P").add_class("Q", "P").add_class("R")
        ont.replace_class("Q", "R")  # Q moves from under P to under R -- no cycle, must succeed
        self.assertEqual(ont.classes["Q"], "R")
        self.assertTrue(ont.is_a("Q", "R"))
        self.assertFalse(ont.is_a("Q", "P"))

    def test_add_relation_rejects_duplicate_name(self):
        ont = Ontology().add_class("Person").add_class("City").add_relation("lives_in", "Person", "City")
        with self.assertRaises(ValueError):
            ont.add_relation("lives_in", "Person", "City", "functional")
        self.assertEqual(ont.axioms["lives_in"], set())  # unchanged by the rejected call

    def test_replace_relation_requires_existing_name(self):
        ont = Ontology().add_class("Person").add_class("City")
        with self.assertRaises(KeyError):
            ont.replace_relation("lives_in", "Person", "City")

    def test_replace_relation_redefines_signature_and_axioms(self):
        ont = Ontology().add_class("Person").add_class("City").add_relation("lives_in", "Person", "City")
        ont.replace_relation("lives_in", "Person", "City", "functional")
        self.assertEqual(ont.axioms["lives_in"], {"functional"})

    def test_add_disjoint_rejects_unknown_class(self):
        with self.assertRaises(ValueError):
            Ontology().add_class("Person").add_disjoint("Person", "Nonexistent")

    def test_add_disjoint_rejects_identical_class(self):
        with self.assertRaises(ValueError):
            Ontology().add_class("Person").add_disjoint("Person", "Person")

    def test_is_a_raises_loudly_on_cycle_instead_of_silent_partial_answer(self):
        # Construction now makes a cycle unreachable through the public API; this exercises is_a's
        # own defense-in-depth guard by installing a cycle directly on .classes, bypassing the API
        # (e.g. as if some future code path or direct attribute access did so) -- MXR-080-0297 says
        # a cycle must never again look like an ordinary, silently-stopped "not an ancestor" False.
        ont = Ontology().add_class("P").add_class("Q", "P")
        ont.classes["P"] = "Q"  # direct mutation: P -> Q -> P
        with self.assertRaises(RuntimeError):
            ont.is_a("P", "SomethingElse")


class CrossTripleConflictResolutionTest(unittest.TestCase):
    """MXR-080-0298: filter_triples/constrained_decode must validate the ACCEPTED SET as a graph."""

    def test_functional_relation_two_tails_no_longer_both_kept(self):
        # The audit's own reproduction: a functional relation asserted with two different tails for
        # the same head was previously returned ENTIRELY in `kept`.
        ont = Ontology().add_class("Person").add_class("City").add_relation("lives_in", "Person", "City", "functional")
        types = {"ada": "Person", "paris": "City", "lyon": "City"}
        triples = [("ada", "lives_in", "paris"), ("ada", "lives_in", "lyon")]
        kept, rejected = ont.filter_triples(triples, types)
        self.assertEqual(len(kept), 1, "a functional relation must never keep two tails for one head")
        self.assertEqual(kept, [("ada", "lives_in", "paris")])  # first-declared wins, deterministically
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["triple"], ("ada", "lives_in", "lyon"))
        self.assertIn("functional", rejected[0]["problems"][0])

    def test_inverse_functional_two_heads_resolved(self):
        ont = Ontology().add_class("Person").add_relation("has_bio_mother", "Person", "Person", "inverse_functional")
        types = {"a": "Person", "b": "Person", "m": "Person"}
        kept, rejected = ont.filter_triples([("a", "has_bio_mother", "m"), ("b", "has_bio_mother", "m")], types)
        self.assertEqual(kept, [("a", "has_bio_mother", "m")])
        self.assertEqual(rejected[0]["triple"], ("b", "has_bio_mother", "m"))
        self.assertIn("inverse_functional", rejected[0]["problems"][0])

    def test_asymmetric_both_directions_resolved_to_first_declared(self):
        ont = Ontology().add_class("Person").add_relation("manages", "Person", "Person", "asymmetric")
        types = {"ada": "Person", "bob": "Person"}
        kept, rejected = ont.filter_triples([("ada", "manages", "bob"), ("bob", "manages", "ada")], types)
        self.assertEqual(kept, [("ada", "manages", "bob")])
        self.assertEqual(rejected[0]["triple"], ("bob", "manages", "ada"))

    def test_symmetric_unpaired_edge_rejected_paired_edges_kept(self):
        ont = Ontology().add_class("Person").add_relation("married_to", "Person", "Person", "symmetric")
        types = {"ada": "Person", "bob": "Person", "carl": "Person"}
        triples = [("ada", "married_to", "bob"), ("ada", "married_to", "carl"), ("carl", "married_to", "ada")]
        kept, rejected = ont.filter_triples(triples, types)
        self.assertNotIn(("ada", "married_to", "bob"), kept)  # unpaired -> cannot stand alone
        self.assertIn(("ada", "married_to", "carl"), kept)
        self.assertIn(("carl", "married_to", "ada"), kept)
        self.assertEqual(rejected[0]["triple"], ("ada", "married_to", "bob"))

    def test_transitive_premises_survive_missing_closure_reported_not_stripped(self):
        ont = Ontology().add_class("Person").add_relation("ancestor_of", "Person", "Person", "transitive")
        types = {"ada": "Person", "bob": "Person", "carl": "Person"}
        triples = [("ada", "ancestor_of", "bob"), ("bob", "ancestor_of", "carl")]
        kept, rejected = ont.filter_triples(triples, types)
        # the two premises did nothing wrong individually and must NOT be discarded
        self.assertEqual(set(kept), set(triples))
        self.assertEqual(rejected[0]["triple"], ("ada", "ancestor_of", "carl"))
        self.assertIn("transitive", rejected[0]["problems"][0])

    def test_constrained_decode_resolves_functional_conflict_across_different_samples(self):
        # Per-sample filtering alone cannot catch this: each individual sampled graph asserts only
        # ONE tail (so each sample is, by itself, functional-clean), but DIFFERENT samples pick
        # different tails. If both tails' marginals independently clear the floor, the aggregate
        # `facts` set is globally inconsistent unless it is re-validated as a graph.
        ont = Ontology().add_class("Person").add_class("City").add_relation("lives_in", "Person", "City", "functional")
        types = {"ada": "Person", "paris": "City", "london": "City"}
        calls = {"n": 0}

        def generate(prompt):
            calls["n"] += 1
            # 3/5 samples say paris, 2/5 say london -- both would clear a floor of 0.3
            return "ada|lives_in|paris" if calls["n"] % 5 in (1, 2, 3) else "ada|lives_in|london"

        def parse(s):
            return [tuple(t.split("|")) for t in s.split(";") if t]

        from mixle.reason.graph_llm import GraphLLM

        llm = GraphLLM(generate, parse, n=5)
        dec = constrained_decode(llm, "facts about ada", ont, types, floor=0.3)
        asserted = dec.asserted()
        self.assertEqual(len(asserted), 1, f"functional relation must assert exactly one tail, got {asserted}")
        self.assertEqual(asserted[0], ("ada", "lives_in", "paris"))  # higher marginal (0.6 > 0.4) wins
        self.assertTrue(any(r["triple"] == ("ada", "lives_in", "london") for r in dec.rejected))


class CompletionMaskTest(unittest.TestCase):
    """MXR-080-0299: KG completion must mask the FULL triple contract, not range alone."""

    class _FixedScoreKG:
        """A deterministic fake KG: hands back a hand-chosen log-posterior vector."""

        def __init__(self, raw_scores):
            raw = np.asarray(raw_scores, dtype=float)
            self._lp = raw - np.log(np.sum(np.exp(raw)))  # a genuine log-softmax

        def tail_log_posterior(self, h, r):
            return self._lp

    def test_irreflexive_self_completion_gets_zero_mass_not_the_max(self):
        # The audit's own reproduction: for an irreflexive `knows` relation, completing head `a`
        # with candidate `a` was assigned the LARGEST probability of any candidate (0.6652 in the
        # audit's run) because only the candidate's range class was checked. Self-completion must
        # instead receive NO mass at all.
        ont = Ontology().add_class("Person").add_relation("knows", "Person", "Person", "irreflexive")
        types = {"a": "Person", "b": "Person", "c": "Person"}
        # deliberately give the self-candidate ('a', index 0) the HIGHEST raw score, so an unmasked
        # softmax would make it the argmax -- reproducing the audit's "assigned the largest
        # probability" shape without depending on a specific fitted model's exact numbers.
        kg = self._FixedScoreKG([2.0, 0.5, 0.3])
        unmasked = np.exp(kg._lp) / np.exp(kg._lp).sum()
        self.assertEqual(np.argmax(unmasked), 0)  # sanity: 'a' (self) WOULD be the argmax if unmasked

        ckg = OntologyConstrainedKG(kg, ont, entities=["a", "b", "c"], relations=["knows"], types=types)
        post = ckg.tail_posterior("a", "knows")
        self.assertNotIn("a", post)  # zero (i.e. absent) mass, not merely "not the max"
        self.assertEqual(set(post), {"b", "c"})
        self.assertAlmostEqual(sum(post.values()), 1.0, places=9)

    def test_head_domain_violation_yields_empty_posterior(self):
        ont = Ontology().add_class("Person").add_class("Organization").add_relation("employs", "Organization", "Person")
        types = {"ada": "Person", "bob": "Person", "acme": "Organization"}
        kg = self._FixedScoreKG([0.1, 0.2, 0.3])
        # 'ada' is a Person, but `employs` requires an Organization head -- no tail should be
        # proposed no matter how the range mask alone would have scored it.
        ckg = OntologyConstrainedKG(kg, ont, entities=["ada", "bob", "acme"], relations=["employs"], types=types)
        self.assertEqual(ckg.tail_posterior("ada", "employs"), {})

    def test_disjoint_candidate_excluded_even_when_range_conforming(self):
        # A pathological but legal schema: Organization is declared disjoint with its OWN ancestor
        # Agent, so every Organization instance is simultaneously both halves of that pair. The
        # head (a Person, uninvolved in the disjoint pair) is fine; the completion mask must still
        # exclude the Organization CANDIDATE even though it range-conforms (controls: Agent -> Agent).
        ont = (
            Ontology()
            .add_class("Agent")
            .add_class("Person", "Agent")
            .add_class("Organization", "Agent")
            .add_relation("controls", "Agent", "Agent")
            .add_disjoint("Agent", "Organization")
        )
        types = {"ada": "Person", "bob": "Person", "acme": "Organization"}
        kg = self._FixedScoreKG([0.1, 0.2, 0.9])  # acme (index 2) would win an unmasked softmax
        ckg = OntologyConstrainedKG(kg, ont, entities=["ada", "bob", "acme"], relations=["controls"], types=types)
        post = ckg.tail_posterior("ada", "controls")
        self.assertIn("bob", post)
        self.assertNotIn("acme", post)  # Organization is disjoint-excluded despite range-conforming


class ScoreNormalizationTest(unittest.TestCase):
    """MXR-080-0300: invalid KG scores must never normalize into NaN "probabilities"."""

    class _RawKG:
        def __init__(self, lp):
            self._lp = np.asarray(lp)

        def tail_log_posterior(self, h, r):
            return self._lp

    def _ckg(self, kg, entities=("a", "b", "c")):
        ont = Ontology().add_class("Person").add_relation("knows", "Person", "Person")
        types = {e: "Person" for e in set(entities)}
        return OntologyConstrainedKG(kg, ont, entities=list(entities), relations=["knows"], types=types)

    def test_all_neg_inf_log_posterior_fails_closed_instead_of_nan(self):
        # The audit's own reproduction: subtracting an all(-inf) maximum produces NaN, and
        # `total <= 0` is False for NaN, so a dictionary of NaN "probabilities" was returned.
        ckg = self._ckg(self._RawKG([-np.inf, -np.inf, -np.inf]))
        with self.assertRaises(ValueError):
            ckg.tail_posterior("a", "knows")

    def test_nan_log_posterior_rejected(self):
        ckg = self._ckg(self._RawKG([0.1, np.nan, 0.2]))
        with self.assertRaises(ValueError):
            ckg.tail_posterior("a", "knows")

    def test_positive_inf_log_posterior_rejected(self):
        ckg = self._ckg(self._RawKG([0.1, np.inf, 0.2]))
        with self.assertRaises(ValueError):
            ckg.tail_posterior("a", "knows")

    def test_wrong_length_log_posterior_rejected(self):
        ckg = self._ckg(self._RawKG([0.1, 0.2]))  # 2 entries for 3 entities
        with self.assertRaises(ValueError):
            ckg.tail_posterior("a", "knows")

    def test_non_vector_log_posterior_rejected(self):
        ckg = self._ckg(self._RawKG([[0.1, 0.2, 0.3]]))  # 2-D, not a vector
        with self.assertRaises(ValueError):
            ckg.tail_posterior("a", "knows")

    def test_duplicate_entity_name_rejected_at_construction(self):
        ont = Ontology().add_class("Person").add_relation("knows", "Person", "Person")
        types = {"a": "Person", "b": "Person"}
        with self.assertRaises(ValueError):
            OntologyConstrainedKG(
                self._RawKG([0.1, 0.2, 0.3]), ont, entities=["a", "a", "b"], relations=["knows"], types=types
            )

    def test_duplicate_relation_name_rejected_at_construction(self):
        ont = Ontology().add_class("Person").add_relation("knows", "Person", "Person")
        types = {"a": "Person", "b": "Person"}
        with self.assertRaises(ValueError):
            OntologyConstrainedKG(
                self._RawKG([0.1, 0.2]),
                ont,
                entities=["a", "b"],
                relations=["knows", "knows"],
                types=types,
            )

    def test_ordinary_case_still_normalizes_to_a_valid_distribution(self):
        # a mix of finite and -inf entries (the latter admissible-but-zero-mass) must still
        # normalize cleanly over the admissible subset -- this fix must not break the happy path.
        ckg = self._ckg(self._RawKG([1.0, -np.inf, 0.5]))
        post = ckg.tail_posterior("a", "knows")
        self.assertAlmostEqual(sum(post.values()), 1.0, places=9)
        self.assertAlmostEqual(post["b"], 0.0, places=9)  # -inf candidate: admissible, zero mass
        self.assertGreater(post["a"], post["c"])  # higher raw score keeps a higher share


if __name__ == "__main__":
    unittest.main()
