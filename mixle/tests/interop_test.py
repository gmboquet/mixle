"""Interop (Q): external models wrapped with UQ, entering the reasoner as self-doubting delegates."""

import unittest

from mixle.substrate import ExternalAnswer, ExternalModel, external_action
from mixle.substrate.act import investigate


def _echo(q, ctx):
    return f"A[{ctx[:40]}]"


def _constant_gen(answer="the tax rate is 8.25 percent"):
    return lambda prompt: answer


def _flaky_gen():
    answers = ["answer A", "answer B", "answer C", "answer D"]
    state = {"i": 0}

    def gen(prompt):
        state["i"] += 1
        return answers[state["i"] % len(answers)]

    return gen


class ExternalModelTest(unittest.TestCase):
    def test_consistent_model_is_confident(self):
        m = ExternalModel(_constant_gen(), max_entropy=0.5, samples=6)
        a = m.answer("what is the tax rate")
        self.assertIsInstance(a, ExternalAnswer)
        self.assertLessEqual(a.entropy, 0.5)
        self.assertTrue(a.confident)

    def test_self_contradicting_model_is_uncertain(self):
        m = ExternalModel(_flaky_gen(), max_entropy=0.5, samples=8)
        a = m.answer("what is the tax rate")
        self.assertGreater(a.entropy, 0.5)  # many meaning classes -> high entropy
        self.assertFalse(a.confident)

    def test_answer_carries_the_generated_text(self):
        m = ExternalModel(_constant_gen("forty two"), max_entropy=1.0)
        self.assertEqual(m.answer("q").answer, "forty two")


class ExternalActionTest(unittest.TestCase):
    def test_confident_external_contributes_evidence(self):
        m = ExternalModel(_constant_gen(), max_entropy=0.5, samples=6)
        act = external_action(m, description="tax rate external solver")
        inv = investigate("what is the tax rate", [act], _echo, min_confidence=0.0)
        self.assertFalse(inv.abstained)
        self.assertIn("confident", " ".join(inv.evidence))
        self.assertIn("8.25", " ".join(inv.evidence))

    def test_uncertain_external_withholds_and_reasoner_abstains(self):
        m = ExternalModel(_flaky_gen(), max_entropy=0.5, samples=8)
        act = external_action(m, description="tax rate external solver")
        inv = investigate("what is the tax rate", [act], _echo, min_confidence=0.3)
        self.assertTrue(inv.abstained)  # a self-contradicting external answer is treated as no answer

    def test_trust_uncertain_overrides_the_gate(self):
        m = ExternalModel(_flaky_gen(), max_entropy=0.5, samples=8)
        act = external_action(m, description="tax rate external solver", trust_uncertain=True)
        inv = investigate("what is the tax rate", [act], _echo, min_confidence=0.0)
        self.assertFalse(inv.abstained)
        self.assertIn("uncertain", " ".join(inv.evidence))  # flagged, but included

    def test_external_action_is_a_costly_delegate(self):
        m = ExternalModel(_constant_gen(), max_entropy=1.0)
        act = external_action(m)
        self.assertEqual(act.kind, "delegate")
        self.assertGreaterEqual(act.cost, 8.0)  # escalation of last resort


if __name__ == "__main__":
    unittest.main()
