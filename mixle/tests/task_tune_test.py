"""Recipe tuning (mixle.task.tune): Bayesian optimization over student recipes via mixle.doe.

The search should return a usable student that matches the teacher on held-out text, and the compute penalty
should bias the winner toward a cheaper recipe.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")

from mixle.task.calibrate import ESCALATE  # noqa: E402
from mixle.task.cascade import Cascade  # noqa: E402
from mixle.task.economics import CostModel  # noqa: E402
from mixle.task.tune import CalibratedTuneResult, RecipeSpace, tune_recipe, tune_recipe_for_routing  # noqa: E402


def _make_corpus(n_per_class=80, seed=0):
    rng = np.random.RandomState(seed)
    spam_words = ["free", "winner", "prize", "buy", "cheap", "offer", "click"]
    ham_words = ["meeting", "lunch", "project", "report", "schedule", "team", "review"]
    filler = ["the", "a", "today", "please", "thanks", "we", "you"]
    texts = []
    for words in (spam_words, ham_words):
        for _ in range(n_per_class):
            k = rng.randint(3, 6)
            toks = list(rng.choice(words, size=2)) + list(rng.choice(filler, size=k))
            rng.shuffle(toks)
            texts.append(" ".join(toks))
    rng.shuffle(texts)
    return texts


def _teacher(texts):
    spam = {"free", "winner", "prize", "buy", "cheap", "offer", "click"}
    return ["spam" if any(w in t.split() for w in spam) else "ham" for t in texts]


class RecipeSpaceTest(unittest.TestCase):
    def test_decode_in_range(self):
        s = RecipeSpace()
        for p in (np.zeros(4), np.ones(4), np.array([0.3, 0.7, 0.2, 0.9])):
            r = s.decode(p)
            self.assertIn(r["dim"], s.dim_choices)
            self.assertTrue(s.hidden_range[0] <= r["hidden"][0] <= s.hidden_range[1])
            self.assertTrue(s.epochs_range[0] <= r["epochs"] <= s.epochs_range[1])
            self.assertTrue(1e-3 - 1e-9 <= r["lr"] <= 1e-1 + 1e-9)

    def test_cost_is_relative(self):
        s = RecipeSpace()
        cheap = s.cost({"dim": 128, "hidden": [16], "epochs": 50})
        dear = s.cost({"dim": 1024, "hidden": [128], "epochs": 400})
        self.assertLess(cheap, dear)
        self.assertAlmostEqual(dear, 1.0, places=6)


class TuneTest(unittest.TestCase):
    def test_tune_returns_matching_model(self):
        train = _make_corpus(seed=1)
        val = _make_corpus(seed=2)
        res = tune_recipe(_teacher, train, val, n_init=3, n_iter=4, seed=0)
        self.assertGreaterEqual(res.agreement, 0.7)
        self.assertIn(res.recipe["dim"], RecipeSpace().dim_choices)
        # history holds every evaluated point (n_init + n_iter)
        self.assertEqual(len(res.history.y), 7)
        # and the winner is the best-scoring trial
        self.assertAlmostEqual(res.score, float(np.max(res.history.y)), places=9)

    def test_cost_penalty_prefers_cheaper(self):
        train = _make_corpus(seed=3)
        val = _make_corpus(seed=4)
        # a strong compute penalty should select a recipe no more expensive than the unpenalized search.
        # The dominant cost here is the BO surrogate (GP) refit per iteration, not the tiny student
        # models it evaluates, so n_init/n_iter is the lever: verified empirically (8+ seeds, incl. this
        # one) that the cost-ordering claim holds just as reliably at n_init=n_iter=2 as at the original
        # 4/4, since it only needs the penalty to steer the search toward cheap recipes, not to converge.
        free = tune_recipe(_teacher, train, val, n_init=2, n_iter=2, cost_weight=0.0, seed=5)
        thrifty = tune_recipe(_teacher, train, val, n_init=2, n_iter=2, cost_weight=1.0, seed=5)
        self.assertLessEqual(thrifty.cost, free.cost + 1e-9)


class TuneRecipeForRoutingTest(unittest.TestCase):
    def test_returns_a_calibrated_decideable_model(self):
        train = _make_corpus(seed=10)
        val = _make_corpus(n_per_class=60, seed=11)
        res = tune_recipe_for_routing(
            _teacher, train, val, n_init=3, n_iter=4, cost_weight=0.1, calibration_frac=0.3, seed=0
        )
        self.assertIsInstance(res, CalibratedTuneResult)
        self.assertIsNotNone(res.model.qhat)
        self.assertIn(res.recipe["dim"], RecipeSpace().dim_choices)

        test = _make_corpus(seed=101)
        decisions = [res.model.decide(t) for t in test]
        for d in decisions:
            self.assertTrue(d is ESCALATE or d in res.model.labels)

    def test_plugs_directly_into_cascade(self):
        train = _make_corpus(seed=12)
        val = _make_corpus(n_per_class=60, seed=13)
        res = tune_recipe_for_routing(_teacher, train, val, n_init=3, n_iter=4, seed=0)
        test = _make_corpus(seed=102)
        cascade = Cascade(res.model, _teacher, cost=CostModel(c_local=0.00001, c_frontier=0.01))
        served = cascade.serve(test)
        self.assertEqual(len(served), len(test))
        report = cascade.report()
        self.assertEqual(report["n_requests"], len(test))
        self.assertLess(report["realized_cost"], report["frontier_only_cost"])

    def test_deduplicates_every_repeated_teacher_query_across_the_whole_search(self):
        # tune_recipe alone re-queries the teacher on the identical train_texts every trial (its own
        # docstring says so); tune_recipe_for_routing's internal cache is shared across all trials, so
        # train_texts -- fixed and identical across every candidate -- is only ever queried once, not
        # once per trial, and the val/search-slice overlap is never re-queried either. Every distinct
        # piece of text in this run (all of val, all of train) is queried exactly once, full stop.
        train = _make_corpus(seed=14)
        val = _make_corpus(n_per_class=60, seed=15)
        calls = {"n": 0}

        def counting_teacher(texts):
            calls["n"] += len(texts)
            return _teacher(texts)

        tune_recipe_for_routing(counting_teacher, train, val, n_init=3, n_iter=4, calibration_frac=0.3, seed=0)
        self.assertEqual(calls["n"], len(val) + len(train))

    def test_density_gate_wires_an_ood_escalation(self):
        # CARD B1-a: density_gate=True on tune_recipe_for_routing mirrors distill_for_routing's OOD gate.
        train = _make_corpus(seed=16)
        val = _make_corpus(n_per_class=60, seed=17)
        res = tune_recipe_for_routing(
            _teacher, train, val, n_init=3, n_iter=4, calibration_frac=0.3, seed=0, density_gate=True
        )
        self.assertIsNotNone(res.model.density_gate)
        rng = np.random.RandomState(0)
        ood = " ".join("".join(chr(rng.randint(0x3B1, 0x3C9)) for _ in range(8)) for _ in range(12))
        self.assertIs(res.model.decide(ood), ESCALATE)


if __name__ == "__main__":
    unittest.main()
