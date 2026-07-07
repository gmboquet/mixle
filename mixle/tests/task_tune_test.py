"""Recipe tuning (mixle.task.tune): Bayesian optimization over student recipes via mixle.doe.

The search should return a usable student that matches the teacher on held-out text, and the compute penalty
should bias the winner toward a cheaper recipe.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")

from mixle.task.tune import RecipeSpace, tune_recipe  # noqa: E402


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
        # a strong compute penalty should select a recipe no more expensive than the unpenalized search
        free = tune_recipe(_teacher, train, val, n_init=4, n_iter=4, cost_weight=0.0, seed=5)
        thrifty = tune_recipe(_teacher, train, val, n_init=4, n_iter=4, cost_weight=1.0, seed=5)
        self.assertLessEqual(thrifty.cost, free.cost + 1e-9)


if __name__ == "__main__":
    unittest.main()
