"""investigate() (S3): the reasoner's widened action space -- retrieve / compute / simulate."""

import unittest

import numpy as np

from mixle.inference import learn_bayesian_network, simulate
from mixle.inference.skill import SkillRegistry, skill
from mixle.substrate import Substrate
from mixle.substrate.act import (
    Action,
    Investigation,
    compute_action,
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


if __name__ == "__main__":
    unittest.main()
