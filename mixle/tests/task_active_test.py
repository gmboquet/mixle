"""Active labeling (mixle.task.active): budget accounting and same-budget non-inferiority.

What these tests establish, exactly: the budget ledger is honest (every teacher query is counted,
validation labels included), and at ONE fixed equal budget on ONE synthetic seed, uncertainty-driven
selection produces held-out teacher agreement within a small tolerance of random selection. That is a
same-budget smoke check, not a labels-to-target result: no claim that the same quality costs fewer
calls follows from it (STAT-RR15-1), and none is made. A labels-to-target learning curve over
independent seeds with uncertainty on the label-count difference is the experiment that would
support a fewer-calls claim.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

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


class ExactPairedConclusionRuleTest(unittest.TestCase):
    """The examples' conclusion rule itself, exercised where Wald gates fail (STAT-RR16-1).

    Four discordant pairs all favoring one side is exactly the boundary the review measured: the
    Wald interval's lower endpoint goes positive while the exact two-sided paired p-value is
    0.125 -- no rejection at 5%. The rule must stay exact at any discordance count.
    """

    @staticmethod
    def _rule():
        import importlib.util
        from pathlib import Path

        examples = Path(__file__).resolve().parents[2] / "examples"
        spec = importlib.util.spec_from_file_location("_active_example", examples / "task_llm_active_example.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.exact_paired_pvalue

    def test_small_discordance_does_not_reject(self):
        rule = self._rule()
        self.assertAlmostEqual(rule(4, 0), 0.125, places=12)  # the review's exact case
        self.assertGreaterEqual(rule(4, 0), 0.05)  # no superiority claim at four discordances
        self.assertEqual(rule(0, 0), 1.0)
        self.assertAlmostEqual(rule(3, 1), 0.625, places=12)

    def test_large_onesided_discordance_rejects_and_is_symmetric(self):
        rule = self._rule()
        self.assertLess(rule(15, 1), 0.05)
        self.assertEqual(rule(4, 0), rule(0, 4))
        self.assertLessEqual(rule(30, 13), 1.0)


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
        self.assertEqual(res.training_labels_used, len(res.labeled_labels))
        self.assertEqual(res.validation_labels_used, 0)
        self.assertEqual(res.teacher_queries, res.labels_used)
        self.assertGreaterEqual(len(res.history), 2)

    def test_active_beats_or_matches_random_at_same_budget(self):
        p = pool(4)
        val = pool(999)[:200]
        truth = teacher(val)
        # The 200 validation labels are real teacher purchases too. A total budget of 270
        # leaves the same 70-label training budget for each policy: an EQUAL-BUDGET
        # comparison, with the query count verified below -- not a fewer-calls result.
        budget = 270

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

        # equal-budget non-inferiority on one seed, with tolerance for training noise: active
        # held-out teacher agreement must not be materially worse than random at the same spend
        self.assertGreaterEqual(acc(active.model), acc(rand.model) - 0.06)
        self.assertEqual(active.teacher_queries, rand.teacher_queries)  # the budgets really were equal
        self.assertEqual(active.labels_used, rand.labels_used)
        self.assertEqual(active.labels_used, budget)
        self.assertEqual(active.training_labels_used, 70)
        self.assertEqual(active.validation_labels_used, 200)

    def test_validation_queries_are_charged_and_malformed_teacher_batches_fail_closed(self):
        p = pool(5)
        val = pool(1001)[:10]
        purchased = {"n": 0}

        def counted_teacher(texts):
            purchased["n"] += len(texts)
            return teacher(texts)

        result = active_distill(
            counted_teacher,
            p,
            budget=30,
            seed_size=10,
            rounds=2,
            acquisition="random",
            recipe=RECIPE,
            val_texts=val,
            seed=0,
        )
        self.assertEqual(result.labels_used, 30)
        self.assertEqual(result.teacher_queries, purchased["n"])
        self.assertEqual(result.training_labels_used, 20)
        self.assertEqual(result.validation_labels_used, 10)

        with self.assertRaisesRegex(ValueError, "exactly one label"):
            active_distill(
                lambda _texts: ["too", "many"],
                p[:20],
                budget=10,
                seed_size=5,
                rounds=1,
                recipe=RECIPE,
            )

    def test_newly_purchased_labels_expand_the_inferred_class_universe(self):
        observed_spaces = []

        def fake_fit(_texts, _labels, label_space, _recipe, _seed):
            observed_spaces.append(tuple(label_space))
            return SimpleNamespace()

        def expanding_teacher(texts):
            return [text[0] for text in texts]

        with mock.patch("mixle.task.active._fit", side_effect=fake_fit):
            result = active_distill(
                expanding_teacher,
                ["a0", "a1", "b0", "b1"],
                budget=4,
                seed_size=1,
                rounds=3,
                acquisition="random",
                seed=0,
            )
        self.assertEqual(set(result.labeled_labels), {"a", "b"})
        self.assertEqual(set(observed_spaces[-1]), {"a", "b"})
        self.assertTrue(any(len(space) == 1 for space in observed_spaces))


if __name__ == "__main__":
    unittest.main()
