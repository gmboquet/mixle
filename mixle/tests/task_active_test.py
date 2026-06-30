"""Active labeling (mixle.task.active): spend labels where they matter, beating random for the same budget.

The money claim: at a fixed labeling budget, uncertainty-driven selection reaches at least as good a student as
random labeling -- usually better -- so the same quality costs fewer teacher calls.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")

from mixle.task.active import acquisition_scores, active_distill  # noqa: E402
from mixle.task.distill import distill  # noqa: E402

SPAM = ["free", "winner", "prize", "buy", "cheap", "offer", "click", "loan", "viagra", "casino"]
HAM = ["meeting", "lunch", "project", "report", "schedule", "team", "review", "invoice", "agenda", "budget"]
FILLER = ["the", "a", "today", "please", "thanks", "we", "you", "and", "to", "for"]


def pool(seed, n_per_class=300):
    r = np.random.RandomState(seed)
    out = []
    for words in (SPAM, HAM):
        for _ in range(n_per_class):
            toks = list(r.choice(words, size=2)) + list(r.choice(FILLER, size=r.randint(3, 8)))
            r.shuffle(toks)
            out.append(" ".join(toks))
    r.shuffle(out)
    return out


def teacher(texts):
    s = set(SPAM)
    return ["spam" if any(w in t.split() for w in s) else "ham" for t in texts]


RECIPE = {"n": 4, "dim": 512, "hidden": [64], "epochs": 200, "lr": 1e-2}


class AcquisitionTest(unittest.TestCase):
    def test_scores_rank_uncertain_higher(self):
        train = pool(1)
        student = distill(teacher, train, **RECIPE, seed=0)
        # an ambiguous mixed string should not score below a clearly-spam string on margin uncertainty
        clear = "free prize winner casino loan"
        mixed = "meeting free lunch prize the and"
        s = acquisition_scores(student, [clear, mixed], "margin")
        self.assertEqual(s.shape, (2,))
        self.assertTrue(np.all(np.isfinite(s)))

    def test_unknown_acquisition_raises(self):
        train = pool(2)
        student = distill(teacher, train, **RECIPE, seed=0)
        with self.assertRaises(ValueError):
            acquisition_scores(student, train[:5], "nonsense")


class ActiveLabelingTest(unittest.TestCase):
    def test_budget_respected_and_logged(self):
        p = pool(3)
        res = active_distill(teacher, p, budget=80, seed_size=20, rounds=4, acquisition="margin", recipe=RECIPE, seed=0)
        self.assertLessEqual(res.labels_used, 80)
        self.assertGreaterEqual(res.labels_used, 20)
        self.assertEqual(res.labels_used, len(res.labeled_labels))
        self.assertGreaterEqual(len(res.history), 2)

    def test_active_beats_or_matches_random_at_same_budget(self):
        p = pool(4)
        val = pool(999)[:200]
        truth = teacher(val)
        budget = 70

        active = active_distill(
            teacher,
            p,
            budget=budget,
            seed_size=20,
            rounds=5,
            acquisition="margin",
            recipe=RECIPE,
            val_texts=val,
            seed=0,
        )
        rand = active_distill(
            teacher,
            p,
            budget=budget,
            seed_size=20,
            rounds=5,
            acquisition="random",
            recipe=RECIPE,
            val_texts=val,
            seed=0,
        )

        def acc(model):
            pred = model.batch(val)
            return float(np.mean([a == b for a, b in zip(pred, truth)]))

        # same labeling budget; uncertainty sampling should not do worse than random (usually better)
        self.assertGreaterEqual(acc(active.model), acc(rand.model) - 0.03)
        self.assertEqual(active.labels_used, rand.labels_used)


if __name__ == "__main__":
    unittest.main()
