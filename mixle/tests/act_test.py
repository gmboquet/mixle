"""investigate() (S3): the reasoner's widened action space -- retrieve / compute / simulate."""

import unittest

import numpy as np

from mixle.inference import create as create_model
from mixle.inference import learn_bayesian_network, simulate
from mixle.inference.skill import SkillRegistry, skill
from mixle.substrate import Substrate
from mixle.substrate.act import (
    Action,
    Investigation,
    compute_action,
    create_action,
    delegate_action,
    investigate,
    retrieve_action,
    score_action,
    simulate_action,
)


def _echo(question, ctx):
    return f"ANSWER[{ctx[:60]}]"


def _plan_spend(n, seed):
    rng = np.random.RandomState(seed)
    return [(["free", "pro"][i % 2], float(20 + 80 * (i % 2) + 3 * rng.randn())) for i in range(n)]


class ScoreTest(unittest.TestCase):
    def test_eig_per_cost_favors_relevant_cheap_actions(self):
        a = Action("a", "compute", run=lambda q: ["x"], cost=1.0, description="convert temperature units")
        b = Action("b", "compute", run=lambda q: ["y"], cost=4.0, description="convert temperature units")
        q = "convert temperature"
        self.assertGreater(score_action(a, q), score_action(b, q))  # cheaper same-relevance wins

    def test_retrieve_has_a_base_floor(self):
        r = retrieve_action(Substrate())
        self.assertGreater(score_action(r, "anything at all"), 0.0)  # always weakly informative


class InvestigateTest(unittest.TestCase):
    def _actions(self):
        s = Substrate()
        s.add(kind="text", text="refunds are processed within 30 days of request")
        reg = SkillRegistry()
        sk = skill("temp", lambda q: "100C is 212F", description="convert temperature celsius fahrenheit", registry=reg)
        net = learn_bayesian_network(_plan_spend(400, 0), max_parents=1)
        sim = simulate(net).scenario("pro", {0: "pro"})
        return [
            retrieve_action(s),
            compute_action(sk, cost=1.0),
            simulate_action(sim, 1, "pro", description="forecast spend under the pro plan", cost=2.0),
        ]

    def test_compute_action_answers_a_computational_question(self):
        inv = investigate("convert the temperature", self._actions(), _echo)
        self.assertIsInstance(inv, Investigation)
        self.assertFalse(inv.abstained)
        self.assertTrue(any(s.kind == "compute" and s.fragments for s in inv.steps))

    def test_retrieve_action_carries_a_knowledge_question(self):
        inv = investigate("when are refunds processed", self._actions(), _echo)
        self.assertFalse(inv.abstained)
        self.assertIn("refunds", " ".join(inv.evidence))

    def test_simulate_action_reports_a_whatif(self):
        inv = investigate("forecast spend under the pro plan", self._actions(), _echo)
        self.assertTrue(any(s.kind == "simulate" and s.fragments for s in inv.steps))
        self.assertIn("mean of field", " ".join(inv.evidence))

    def test_abstains_when_no_action_is_informative(self):
        s = Substrate()  # empty substrate, only irrelevant compute/simulate
        net = learn_bayesian_network(_plan_spend(200, 0), max_parents=1)
        sim = simulate(net).scenario("pro", {0: "pro"})
        acts = [
            compute_action(skill("t", lambda q: "x", description="temperature", registry=SkillRegistry())),
            simulate_action(sim, 1, "pro", description="spend"),
        ]
        inv = investigate("quantum chromodynamics lagrangian", acts, _echo, min_confidence=0.5)
        self.assertTrue(inv.abstained)
        self.assertIsNone(inv.answer)

    def test_cost_budget_caps_actions(self):
        acts = self._actions()
        inv = investigate("forecast spend under the pro plan", acts, _echo, budget_cost=1.0)
        self.assertLessEqual(inv.spent, 1.0)  # the cost=2 simulate cannot fire under a 1.0 budget

    def test_broken_action_does_not_sink_the_investigation(self):
        def _boom(q):
            raise RuntimeError("nope")

        s = Substrate()
        s.add(kind="text", text="refunds within 30 days")
        acts = [Action("boom", "compute", run=_boom, cost=1.0, description="refunds policy"), retrieve_action(s)]
        inv = investigate("refunds policy", acts, _echo)
        self.assertFalse(inv.abstained)  # retrieve still carried it despite the broken action

    def test_trace_is_ordered_provenance(self):
        inv = investigate("convert the temperature", self._actions(), _echo)
        trace = inv.trace()
        self.assertTrue(all("action" in t and "kind" in t for t in trace))


class CreateAndDelegateTest(unittest.TestCase):
    def test_create_action_reports_a_built_models_guarantee(self):
        def _build(q):
            return create_model([float(x) for x in np.random.RandomState(0).normal(5, 2, 200)], seed=0)

        act = create_action(_build, description="build a spend model from data", cost=4.0)
        inv = investigate("build a spend model from data", [act], _echo, min_confidence=0.0)
        self.assertFalse(inv.abstained)
        self.assertIn("guarantee", " ".join(inv.evidence))

    def test_create_is_costlier_than_retrieve(self):
        r = retrieve_action(Substrate())
        c = create_action(lambda q: [1, 2, 3], description="x")
        self.assertGreater(c.cost, r.cost)  # creation is the expensive action

    def test_delegate_action_marks_priced_escalation(self):
        act = delegate_action(lambda q: "remote says 42", description="ask the remote solver", priced=True)
        inv = investigate("ask the remote solver please", [act], _echo, min_confidence=0.0)
        self.assertIn("priced", " ".join(inv.evidence))
        self.assertEqual(inv.steps[0].kind, "delegate")

    def test_delegate_is_the_most_expensive_by_default(self):
        self.assertGreater(
            delegate_action(lambda q: "x").cost,
            create_action(lambda q: "y").cost,
        )  # escalation of last resort under the 99%-local topology


class EarlyStopTest(unittest.TestCase):
    def test_stops_after_the_cheapest_sufficient_action(self):
        cheap = Action("cheap", "compute", run=lambda q: ["hit"], cost=1.0, description="answer the question")
        pricey = Action("pricey", "delegate", run=lambda q: ["also"], cost=8.0, description="answer the question")
        inv = investigate("answer the question", [cheap, pricey], _echo)
        self.assertEqual([s.action for s in inv.steps], ["cheap"])  # never fired the pricey one
        self.assertEqual(inv.spent, 1.0)

    def test_escalates_only_when_cheaper_actions_return_nothing(self):
        empty = Action("empty", "compute", run=lambda q: [], cost=1.0, description="answer the question")
        pricey = Action("pricey", "delegate", run=lambda q: ["the answer"], cost=8.0, description="answer the question")
        inv = investigate("answer the question", [empty, pricey], _echo)
        self.assertFalse(inv.abstained)
        self.assertIn("the answer", " ".join(inv.evidence))  # forced to the expensive action, correctly

    def test_confidence_tracks_relevance_not_cost(self):
        # a costly but perfectly on-topic action earns full confidence (it is merely tried last)
        pricey = Action("p", "delegate", run=lambda q: ["x"], cost=100.0, description="proprietary tax rule")
        inv = investigate("proprietary tax rule", [pricey], _echo, min_confidence=0.4)
        self.assertFalse(inv.abstained)
        self.assertGreaterEqual(inv.confidence, 0.9)

    def test_retrieve_min_score_filters_false_positives(self):
        s = Substrate()
        s.add(kind="text", text="Refunds are processed within 30 days.")
        s.add(kind="text", text="Support is staffed during business hours.")
        act = retrieve_action(s, min_score=0.2)
        # an unrelated query yields weak scores below the floor -> no false evidence
        self.assertEqual(act.run("proprietary tax rule"), [])
        self.assertTrue(act.run("when are refunds processed"))  # a real match still comes through

    def test_stopwords_do_not_manufacture_overlap(self):
        a = Action("a", "compute", run=lambda q: ["x"], cost=1.0, description="forecast the spend")
        # "what is the" shares only stopwords with the description -> zero relevance
        self.assertEqual(score_action(a, "what is the tax"), 0.0)


if __name__ == "__main__":
    unittest.main()
