"""fit_plan_model / PlanModel: plans as fitted Markov-chain models over agent traces (CARD C1-a).

A known 3-tool workflow grammar (search -> fetch -> summarize, fetch sometimes skipped) generates the
synthetic traces; a held-out plan from the SAME grammar should score as typical, a shuffled/alien
sequence should not, samples should only ever emit known tools, and fitting is deterministic given seed.
"""

import unittest

import numpy as np

from mixle.task.plan_model import PlanModel, fit_plan_model
from mixle.task.traces import AgentTrace, AgentTraces

_KNOWN_TOOLS = {"search", "fetch", "summarize"}


def _grammar_trace(i: int, rng: np.random.RandomState) -> AgentTrace:
    """search -> [fetch] -> summarize: fetch is skipped ~30% of the time, order is otherwise fixed."""
    plan = [{"tool": "search", "args": {}}]
    if rng.rand() < 0.7:
        plan.append({"tool": "fetch", "args": {}})
    plan.append({"tool": "summarize", "args": {}})
    return AgentTrace(request=f"request {i}", plan=plan, reply="done")


def _traces(n: int, seed: int = 0) -> AgentTraces:
    rng = np.random.RandomState(seed)
    return AgentTraces(traces=[_grammar_trace(i, rng) for i in range(n)])


class FitPlanModelTest(unittest.TestCase):
    def setUp(self):
        self.model = fit_plan_model(_traces(50, seed=0), smoothing=0.5)

    def test_returns_a_plan_model(self):
        self.assertIsInstance(self.model, PlanModel)

    def test_held_out_grammar_plans_score_as_typical(self):
        """(a) held-out plans from the SAME grammar score above the quantile flag."""
        held_out = _traces(20, seed=99)  # different seed -> genuinely unseen draws from the same grammar
        for t in held_out.traces:
            self.assertTrue(self.model.is_typical(t.plan), f"held-out grammar plan flagged atypical: {t.plan}")

    def test_alien_tool_sequence_is_flagged_atypical(self):
        """(b) a shuffled/alien tool sequence is flagged atypical."""
        alien = ["summarize", "search", "fetch", "fetch", "fetch"]  # wrong order, repeated tool
        self.assertFalse(self.model.is_typical(alien))
        # and it scores strictly worse than a well-formed grammar plan
        typical = ["search", "fetch", "summarize"]
        self.assertLess(self.model.log_prob(alien), self.model.log_prob(typical))

    def test_samples_only_emit_known_tools(self):
        """(c) samples only emit known tools."""
        rng = np.random.RandomState(7)
        for _ in range(30):
            sample = self.model.sample(rng)
            for tool in sample:
                self.assertIn(str(tool), _KNOWN_TOOLS)

    def test_fitting_is_deterministic_given_seed(self):
        """(d) round-trip determinism given seed."""
        model_a = fit_plan_model(_traces(50, seed=0), smoothing=0.5)
        model_b = fit_plan_model(_traces(50, seed=0), smoothing=0.5)
        plan = ["search", "fetch", "summarize"]
        self.assertEqual(model_a.log_prob(plan), model_b.log_prob(plan))
        self.assertEqual(list(model_a.training_log_probs), list(model_b.training_log_probs))

    def test_accepts_either_plan_shape(self):
        dict_shaped = [{"tool": "search", "args": {}}, {"tool": "summarize", "args": {}}]
        str_shaped = ["search", "summarize"]
        self.assertEqual(self.model.log_prob(dict_shaped), self.model.log_prob(str_shaped))

    def test_is_typical_quantile_is_tunable(self):
        # the less-common (but still grammar-valid) fetch-skipped path: fetch appears ~70% of training
        # traces, so this pattern sits at the bottom of the training log-prob distribution -- typical
        # under a permissive floor, atypical under a strict one. quantile is a LOWER bound on the
        # training log-prob distribution: quantile=0.0 -> the most permissive floor (the training
        # minimum); quantile close to 1.0 -> the strictest (near the training maximum).
        less_common = ["search", "summarize"]
        self.assertTrue(self.model.is_typical(less_common, quantile=0.05))
        self.assertFalse(self.model.is_typical(less_common, quantile=0.9))


if __name__ == "__main__":
    unittest.main()
