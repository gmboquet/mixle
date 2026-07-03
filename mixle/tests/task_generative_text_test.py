"""The generative text student: exact posteriors, shared-vocab smoothing, built-in typicality."""

import tempfile
import unittest

import numpy as np

from mixle.task import TaskModel, distill_text_generative, solve


def _intent(t):
    if "refund" in t or "money back" in t:
        return "refund"
    if "hello" in t or "hi " in t:
        return "greeting"
    return "other"


def _texts():
    return (
        [f"hi there friend {i}" for i in range(40)]
        + [f"i want a refund for order {i}" for i in range(40)]
        + [f"question about item {i}" for i in range(40)]
    )


class GenerativeTextStudentTest(unittest.TestCase):
    def test_classifies_and_smooths_cross_class_tokens(self):
        m = distill_text_generative(_intent, _texts())
        self.assertEqual(m("question about item 999"), "other")
        # the killer case: a vocab token seen in OTHER classes but never in this one must not veto —
        # numbers appear across all three templates, so class-conditional zero counts abound
        self.assertEqual(m("hi there friend 15"), "greeting")
        z = m.adapter.logits_batch(m.model, ["question about item 15"])
        self.assertTrue(np.all(np.isfinite(z)))  # no -inf vetoes anywhere

    def test_per_token_evidence_orders_typicality(self):
        m = distill_text_generative(_intent, _texts())
        ev = m.adapter.log_evidence(m.model, ["i want a refund for order 7", "zzq flurble xkcd wibble glorp"])
        self.assertGreater(ev[0], ev[1])  # in-domain more typical per token than gibberish

    def test_solve_student_generative_end_to_end(self):
        sol = solve(_intent, _texts(), student="generative", ood=None, seed=0)
        self.assertGreater(sol.holdout_agreement, 0.9)
        self.assertEqual(sol("hi there my good friend"), "greeting")
        self.assertEqual(sol("i want my money back refund please"), "refund")
        # improve() re-distills with the SAME student family (the knob rides distill_kw)
        for t in ["give me a refund now 77", "hello hello friend"]:
            sol(t)
        sol.improve()
        self.assertGreater(sol.holdout_agreement, 0.9)

    def test_save_load_round_trip(self):
        m = distill_text_generative(_intent, _texts())
        with tempfile.TemporaryDirectory() as d:
            path = m.save(d + "/gen")
            back = TaskModel.load(path)
        for t in ["hi there pal", "refund my order 3", "question about item 4"]:
            self.assertEqual(back(t), m(t))
        ev_a = m.adapter.log_evidence(m.model, ["refund my order 3"])
        ev_b = back.adapter.log_evidence(back.model, ["refund my order 3"])
        self.assertAlmostEqual(float(ev_a[0]), float(ev_b[0]), places=10)


if __name__ == "__main__":
    unittest.main()
