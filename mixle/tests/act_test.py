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


class ComputeActionDispatchTest(unittest.TestCase):
    """compute_action's skill(question) / skill() dispatch, independent of investigate()'s own
    (separate, legitimate) handling of an action raising -- see test_broken_action_does_not_sink_
    the_investigation above for that layer."""

    def test_a_bug_inside_skill_is_not_masked_by_a_retry_without_question(self):
        # a skill accepting question must be called with it exactly once; a TypeError from inside
        # its own body must propagate, not be swallowed and silently retried as skill() -- which
        # would re-run the skill with an entirely different (missing) argument. question=None has a
        # default specifically so a naive try/except TypeError fallback's skill() retry is itself
        # syntactically valid and reaches the body (appending to calls) -- otherwise that retry
        # would fail at argument-binding before ever calling in, and this test could not tell a
        # single correct call apart from a silent duplicate retry.
        calls = []

        def buggy_skill(question=None):
            calls.append(question)
            return None + 1  # an internal bug unrelated to whether question is accepted

        action = compute_action(buggy_skill, name="buggy")
        with self.assertRaises(TypeError):
            action.run("convert 100C")
        self.assertEqual(calls, ["convert 100C"])  # called once, with question -- never retried

    def test_skill_without_question_support_falls_back_correctly(self):
        action = compute_action(lambda: "42", name="const")
        self.assertEqual(action.run("anything"), ["const => 42"])


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


class CostValidationTest(unittest.TestCase):
    """MXR-080-0254: a cost of -5 against a zero budget used to pass the budget check, drag spend to
    -5, and score a priority near a billion (base_score + overlap divided by a clamped 1e-9); a NaN
    cost defeated every comparison meant to catch it. Costs are now validated finite and non-negative
    up front, and zero cost gets explicit (not clamped-division) handling."""

    def test_negative_cost_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as cm:
            Action("neg_cost_action", "compute", run=lambda q: ["x"], cost=-5.0, description="d")
        msg = str(cm.exception)
        self.assertIn("-5", msg)
        self.assertIn("neg_cost_action", msg)

    def test_nan_cost_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            Action("a", "compute", run=lambda q: ["x"], cost=float("nan"), description="d")

    def test_infinite_cost_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            Action("a", "compute", run=lambda q: ["x"], cost=float("inf"), description="d")
        with self.assertRaises(ValueError):
            Action("a", "compute", run=lambda q: ["x"], cost=float("-inf"), description="d")

    def test_investigate_also_rejects_a_cost_mutated_negative_after_construction(self):
        # defense in depth: Action is a mutable dataclass, so __post_init__ alone can't guarantee a
        # cost stays valid for the object's lifetime -- investigate() re-validates before any ranking
        # or accounting runs, so a bypass via direct attribute mutation is still caught.
        a = Action("a", "compute", run=lambda q: ["x"], cost=1.0, description="d")
        a.cost = -5.0
        with self.assertRaises(ValueError):
            investigate("d", [a], _echo)

    def test_zero_cost_action_is_allowed(self):
        a = Action("free", "compute", run=lambda q: ["x"], cost=0.0, description="d")
        self.assertEqual(a.cost, 0.0)

    def test_zero_cost_relevant_action_scores_infinite_and_fires_first(self):
        free = Action("free", "compute", run=lambda q: ["free hit"], cost=0.0, description="answer the question")
        priced = Action("priced", "compute", run=lambda q: ["paid hit"], cost=1.0, description="answer the question")
        self.assertEqual(score_action(free, "answer the question"), float("inf"))
        inv = investigate("answer the question", [priced, free], _echo)
        self.assertEqual(inv.steps[0].action, "free")  # free and relevant is maximally preferred, not paid

    def test_zero_cost_irrelevant_action_scores_zero_not_infinite(self):
        free_but_irrelevant = Action("free", "compute", run=lambda q: ["x"], cost=0.0, description="")
        self.assertEqual(score_action(free_but_irrelevant, "totally unrelated topic xyz"), 0.0)


class EarnedConfidenceTest(unittest.TestCase):
    """MXR-080-0255: retrieve's fixed 0.35 relevance floor, combined with the default min_score=0
    admitting exact-zero-score hits, let a store with nothing relevant answer confidently anyway.
    Confidence must be earned from what was actually retrieved, not the action type's routing prior."""

    def test_quantum_chromodynamics_in_a_cat_store_abstains(self):
        # the audit's own reproduction: a store containing only unrelated content must not produce a
        # confident, non-abstaining answer just because retrieve() always fires.
        s = Substrate()
        s.add(kind="text", text="cats sleep on mats")
        act = retrieve_action(s)  # default min_score=0.0
        inv = investigate("quantum chromodynamics lagrangian", [act], _echo)
        self.assertTrue(inv.abstained)
        self.assertIsNone(inv.answer)
        self.assertEqual(inv.confidence, 0.0)

    def test_zero_score_retrieval_is_not_returned_as_a_fragment(self):
        s = Substrate()
        s.add(kind="text", text="cats sleep on mats")
        act = retrieve_action(s)  # default min_score=0.0: must still exclude an exact-zero score
        self.assertEqual(act.run("quantum chromodynamics lagrangian"), [])

    def test_genuine_match_still_answers_with_earned_confidence(self):
        # control: a real match must still clear the bar, and now earns confidence from its actual
        # score rather than being capped at the old fixed 0.35 floor.
        s = Substrate()
        s.add(kind="text", text="refunds are processed within 30 days of a written request")
        act = retrieve_action(s)
        inv = investigate("when are refunds processed", [act], _echo)
        self.assertFalse(inv.abstained)
        self.assertIn("refunds", " ".join(inv.evidence))
        self.assertGreater(inv.confidence, 0.35)  # a real, strong match earns MORE than the old fixed floor

    def test_confidence_reflects_earned_relevance_not_a_fixed_floor(self):
        # a direct unit test of the earned_relevance wiring, independent of the substrate's own scorer.
        def _make(rel):
            return Action(
                "r",
                "retrieve",
                run=lambda q: ["some fragment"],
                cost=1.0,
                description="",
                base_score=0.35,
                earned_relevance=lambda: rel,
            )

        high = investigate("q", [_make(0.9)], _echo, min_confidence=0.0)
        low = investigate("q", [_make(0.05)], _echo, min_confidence=0.0)
        self.assertGreater(high.confidence, low.confidence)
        self.assertAlmostEqual(high.confidence, 0.9, places=2)
        self.assertAlmostEqual(low.confidence, 0.05, places=2)
        self.assertNotEqual(low.confidence, 0.35)  # never the old fixed routing-prior floor

    def test_retrieve_still_carries_a_routing_prior_for_ordering(self):
        # base_score keeps retrieval "always at least weakly worth trying" for ORDERING (score_action);
        # only the CONFIDENCE-bearing relevance changed. See ScoreTest.test_retrieve_has_a_base_floor.
        r = retrieve_action(Substrate())
        self.assertGreater(score_action(r, "anything at all"), 0.0)


class TraceIntegrityTest(unittest.TestCase):
    """MXR-080-0256: the scorer used to be evaluated once for sorting and again for execution, so a
    stateful/stochastic policy could fire in an order inconsistent with its own recorded score; action
    exceptions were coerced into empty "successful" steps; answerer exceptions left no Investigation
    at all. All three are now explicit."""

    def test_scorer_is_evaluated_exactly_once_per_action(self):
        calls = []

        def counting_scorer(action, question):
            calls.append(action.name)
            return score_action(action, question)

        a = Action("a", "compute", run=lambda q: ["x"], cost=1.0, description="answer the question")
        b = Action("b", "compute", run=lambda q: ["y"], cost=2.0, description="answer the question")
        investigate("answer the question", [a, b], _echo, scorer=counting_scorer, min_confidence=0.0, min_evidence=2)
        self.assertEqual(sorted(calls), ["a", "b"])  # each action scored once, not twice

    def test_stateful_scorer_cannot_diverge_between_sort_and_execution(self):
        # a scorer whose return value changes on every call would, under double-invocation, sort by one
        # pass of values and then re-score (filter + record) by a DIFFERENT pass -- inconsistent with
        # what was recorded. With single evaluation, the recorded Step.score is exactly what ranked it.
        call_n = {"n": 0}
        returns = [10.0, 1.0, 9.0, 2.0]  # a second pass would hand out different values entirely

        def flaky_scorer(action, question):
            v = returns[call_n["n"] % len(returns)]
            call_n["n"] += 1
            return v

        a = Action("a", "compute", run=lambda q: ["x"], cost=1.0, description="d")
        b = Action("b", "compute", run=lambda q: ["y"], cost=1.0, description="d")
        inv = investigate("d", [a, b], _echo, scorer=flaky_scorer, min_confidence=0.0, min_evidence=2)
        self.assertEqual(call_n["n"], 2)  # exactly one call per action, not one for sort + one for execution
        # the recorded scores are exactly the single pass's first two values, unperturbed by a phantom re-score
        self.assertEqual({s.action: s.score for s in inv.steps}, {"a": 10.0, "b": 1.0})

    def test_action_exception_is_recorded_as_an_explicit_failed_step(self):
        def _boom(q):
            raise RuntimeError("boom detail")

        boom = Action("boom", "compute", run=_boom, cost=1.0, description="answer the question")
        ok = Action("ok", "compute", run=lambda q: ["fine"], cost=1.0, description="answer the question")
        inv = investigate("answer the question", [boom, ok], _echo)

        boom_step = next(s for s in inv.steps if s.action == "boom")
        self.assertTrue(boom_step.failed)
        self.assertIn("RuntimeError", boom_step.error)
        self.assertIn("boom detail", boom_step.error)
        self.assertEqual(boom_step.fragments, [])

        ok_step = next(s for s in inv.steps if s.action == "ok")
        self.assertFalse(ok_step.failed)
        self.assertEqual(ok_step.error, "")

        # the trace() dict view surfaces the failure too, not just the Step object
        trace_row = next(t for t in inv.trace() if t["action"] == "boom")
        self.assertTrue(trace_row["failed"])
        self.assertIn("RuntimeError", trace_row["error"])

    def test_broken_action_still_does_not_sink_the_investigation(self):
        # the pre-existing guarantee (see InvestigateTest) must survive: a failed action is recorded,
        # not swallowed, but the investigation still proceeds on the actions that succeeded.
        def _boom(q):
            raise RuntimeError("nope")

        s = Substrate()
        s.add(kind="text", text="refunds within 30 days")
        acts = [Action("boom", "compute", run=_boom, cost=1.0, description="refunds policy"), retrieve_action(s)]
        inv = investigate("refunds policy", acts, _echo)
        self.assertFalse(inv.abstained)

    def test_answerer_failure_produces_an_explicit_failed_investigation(self):
        def _boom_answerer(q, ctx):
            raise RuntimeError("answerer exploded")

        act = Action("a", "compute", run=lambda q: ["evidence"], cost=1.0, description="answer the question")
        inv = investigate("answer the question", [act], _boom_answerer, min_confidence=0.0)

        self.assertIsInstance(inv, Investigation)  # did not propagate -- a record came back instead
        self.assertIsNone(inv.answer)
        self.assertTrue(inv.abstained)
        self.assertTrue(inv.failed)
        self.assertIn("RuntimeError", inv.error)
        self.assertIn("answerer exploded", inv.error)


if __name__ == "__main__":
    unittest.main()
