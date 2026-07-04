"""Reasoner (R): the deployable shell bundling substrate + skills + actions behind .ask()."""

import unittest

import numpy as np

from mixle.inference import learn_action_policy, learn_bayesian_network, simulate
from mixle.inference.skill import SkillRegistry, skill
from mixle.substrate import Reasoner, simulate_action
from mixle.substrate.core import Substrate


def _echo(q, ctx):
    return f"A[{ctx[:40]}]"


def _sub():
    s = Substrate()
    s.add(kind="text", text="Refunds are processed within 30 days.")
    return s


def _skills():
    reg = SkillRegistry()
    skill("temp", lambda q: "100C=212F", description="convert temperature celsius fahrenheit", registry=reg)
    return reg


def _sim():
    net = learn_bayesian_network(
        [(["free", "pro"][i % 2], float(20 + 80 * (i % 2) + 3 * np.random.RandomState(0).randn())) for i in range(400)],
        max_parents=1,
    )
    return simulate(net).scenario("pro", {0: "pro"})


class ReasonerTest(unittest.TestCase):
    def test_wires_the_standard_action_space(self):
        r = Reasoner(_echo, substrate=_sub(), skills=_skills())
        kinds = [a.kind for a in r.actions]
        self.assertIn("retrieve", kinds)  # one retrieve over the store
        self.assertIn("compute", kinds)  # one compute per skill

    def test_ask_routes_to_the_right_action(self):
        r = Reasoner(_echo, substrate=_sub(), skills=_skills(), retrieve_min_score=0.2)
        self.assertEqual(r.ask("when are refunds processed").steps[0].kind, "retrieve")
        self.assertEqual(r.ask("convert temperature").steps[0].kind, "compute")

    def test_add_action_extends_the_space_and_chains(self):
        r = Reasoner(_echo, substrate=_sub(), skills=_skills(), retrieve_min_score=0.2)
        out = r.add_action(simulate_action(_sim(), 1, "pro", description="forecast spend under the pro plan", cost=2.0))
        self.assertIs(out, r)  # chainable
        inv = r.ask("forecast spend under the pro plan")
        self.assertIn("simulate", [s.kind for s in inv.steps if s.fragments])

    def test_abstains_on_the_unanswerable(self):
        r = Reasoner(_echo, substrate=_sub(), skills=_skills(), retrieve_min_score=0.2)
        self.assertTrue(r.ask("airspeed of an unladen swallow", min_confidence=0.5).abstained)

    def test_use_policy_swaps_the_scorer(self):
        r = Reasoner(_echo, substrate=_sub(), skills=_skills())
        rows = [({"kind": "compute", "cost": 1.0, "overlap": 0.5}, "compute", {"value": 1.0}) for _ in range(8)]
        policy = learn_action_policy(rows)
        self.assertIs(r.use_policy(policy), r)
        self.assertIs(r.scorer, policy)

    def test_ask_overrides_pass_through(self):
        r = Reasoner(_echo, substrate=_sub(), skills=_skills(), retrieve_min_score=0.2)
        inv = r.ask("when are refunds processed", max_actions=1)
        self.assertLessEqual(len(inv.steps), 1)

    def test_actions_only_no_substrate(self):
        r = Reasoner(_echo, skills=_skills())
        self.assertTrue(all(a.kind != "retrieve" for a in r.actions))  # no store -> no retrieve action

    def test_verify_attaches_a_grounded_factuality_receipt(self):
        # an answerer that echoes the retrieved evidence produces a fully grounded answer
        r = Reasoner(lambda q, ctx: ctx.splitlines()[0] if ctx else "none", substrate=_sub(), retrieve_min_score=0.2)
        inv = r.ask("when are refunds processed", verify=True)
        self.assertIsNotNone(inv.factuality)
        self.assertTrue(inv.factuality.is_grounded())

    def test_no_verify_leaves_factuality_none(self):
        r = Reasoner(_echo, substrate=_sub(), retrieve_min_score=0.2)
        self.assertIsNone(r.ask("when are refunds processed").factuality)


if __name__ == "__main__":
    unittest.main()
