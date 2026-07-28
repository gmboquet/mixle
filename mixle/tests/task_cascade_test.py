"""Cascade serving (mixle.task.cascade): cheap-local-then-teacher routing with realized savings and harvest.

The cascade must answer accurately (local singletons are covered, escalations defer to the teacher), call the
teacher only on escalated requests, report positive savings versus frontier-only, and harvest the escalated
items as labeled training data for re-distillation.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")

from mixle.task.calibrate import CalibratedTaskModel  # noqa: E402
from mixle.task.cascade import Cascade  # noqa: E402
from mixle.task.distill import distill  # noqa: E402
from mixle.task.economics import CostModel  # noqa: E402

SPAM = ["free", "winner", "prize", "buy", "cheap", "offer", "click"]
HAM = ["meeting", "lunch", "project", "report", "schedule", "team", "review"]
FILLER = ["the", "a", "today", "please", "thanks", "we", "you"]


def corpus(seed, n_per_class=120):
    r = np.random.RandomState(seed)
    out = []
    for w in (SPAM, HAM):
        for _ in range(n_per_class):
            toks = list(r.choice(w, size=2)) + list(r.choice(FILLER, size=r.randint(3, 7)))
            r.shuffle(toks)
            out.append(" ".join(toks))
    r.shuffle(out)
    return out


class CountingTeacher:
    """Ground-truth labeler that records how many times it was called (to prove the cascade saves calls)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += len(texts)
        s = set(SPAM)
        return ["spam" if any(w in t.split() for w in s) else "ham" for t in texts]


def _calibrated(alpha=0.1, seed=0):
    train, cal = corpus(seed), corpus(seed + 50)
    student = distill(CountingTeacher(), train, n=4, dim=512, hidden=[64], epochs=250, seed=0)
    return CalibratedTaskModel(student, alpha=alpha).calibrate(cal, CountingTeacher()(cal))


class CascadeServeTest(unittest.TestCase):
    def test_accurate_and_teacher_called_only_on_escalations(self):
        model = _calibrated(seed=1)
        teacher = CountingTeacher()
        cost = CostModel(c_frontier=1.0, c_local=0.0)
        casc = Cascade(model, teacher, cost=cost)

        test = corpus(seed=900)
        truth = CountingTeacher()(test)
        preds = casc.serve(test)

        acc = np.mean([p == t for p, t in zip(preds, truth)])
        self.assertGreaterEqual(acc, 0.88)  # covered locally + exact on escalations
        # the teacher only saw escalated requests (one call per escalation)
        self.assertEqual(teacher.calls, casc.stats.n_escalated)
        self.assertLess(casc.stats.n_escalated, len(test))  # genuinely offloaded work locally

    def test_report_shows_savings(self):
        model = _calibrated(seed=2)
        cost = CostModel(c_frontier=1.0, c_local=0.0)
        casc = Cascade(model, CountingTeacher(), cost=cost)
        casc.serve(corpus(seed=901))
        rep = casc.report()
        self.assertGreater(rep["savings_vs_frontier"], 0.0)
        self.assertAlmostEqual(rep["realized_cost"], rep["n_escalated"] * 1.0)
        self.assertAlmostEqual(rep["frontier_only_cost"], rep["n_requests"] * 1.0)

    def test_harvest_feeds_redistillation_and_lowers_escalation(self):
        model = _calibrated(seed=3)
        teacher = CountingTeacher()
        casc = Cascade(model, teacher)
        served = corpus(seed=902)
        casc.serve(served)

        htexts, hlabels = casc.harvested()
        self.assertEqual(len(htexts), casc.stats.n_escalated)
        self.assertEqual(len(hlabels), len(htexts))
        self.assertTrue(all(label in ("spam", "ham") for label in hlabels))

        # re-distill including harvested escalations; escalation on fresh traffic should not increase
        base_train = corpus(seed=3)
        student2 = distill(teacher, base_train + htexts, n=4, dim=512, hidden=[64], epochs=250, seed=0)
        cal = corpus(seed=53)
        model2 = CalibratedTaskModel(student2, alpha=0.1).calibrate(cal, teacher(cal))
        fresh = corpus(seed=903)
        self.assertLessEqual(model2.escalation_rate(fresh), model.escalation_rate(fresh) + 0.05)


class _AlwaysEscalate:
    """Minimal stand-in for a CalibratedTaskModel that defers every request to the teacher."""

    def decide(self, text):
        from mixle.task.calibrate import ESCALATE

        return ESCALATE


class TeacherBatchNormalizationTest(unittest.TestCase):
    """MXR-080-1664: a one-element teacher batch is unwrapped whatever container it arrives in."""

    def test_numpy_batch_output_is_unwrapped_to_its_element(self):
        casc = Cascade(_AlwaysEscalate(), lambda batch: np.array(["teacher-label"]))
        answer = casc("anything")
        self.assertEqual(answer, "teacher-label")
        self.assertNotIsInstance(answer, np.ndarray)
        _texts, labels = casc.harvested()
        self.assertEqual(labels, ["teacher-label"])

    def test_list_and_tuple_batches_still_unwrap(self):
        for out in (["a"], ("a",)):
            casc = Cascade(_AlwaysEscalate(), lambda batch, out=out: out)
            self.assertEqual(casc("x"), "a")

    def test_scalar_teacher_output_is_rejected_under_the_batch_contract(self):
        casc = Cascade(_AlwaysEscalate(), lambda batch: "bare-label")
        with self.assertRaises(TypeError):
            casc("x")

    def test_item_mode_takes_the_scalar_answer_verbatim(self):
        seen = []

        def teacher(text):
            seen.append(text)
            return "bare-label"

        casc = Cascade(_AlwaysEscalate(), teacher, teacher_mode="item")
        self.assertEqual(casc("x"), "bare-label")
        self.assertEqual(seen, ["x"])  # called with the request itself, not a one-element batch

    def test_wrong_cardinality_batch_is_rejected(self):
        casc = Cascade(_AlwaysEscalate(), lambda batch: ["a", "b"])
        with self.assertRaises(ValueError):
            casc("x")

    def test_unknown_teacher_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            Cascade(_AlwaysEscalate(), lambda batch: ["a"], teacher_mode="guess")


if __name__ == "__main__":
    unittest.main()
