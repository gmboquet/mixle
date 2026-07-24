"""Interop (Q): external models wrapped with UQ, entering the reasoner as self-doubting delegates."""

import unittest

from mixle.substrate import ExternalAnswer, ExternalModel, external_action
from mixle.substrate.act import investigate
from mixle.substrate.interop import Confidence  # white-box: MXR-080-0270


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


def _disagreeing_gen():
    """Call #1 (the draw ``answer()`` returns) is 'Paris'; every later call in the same batch is
    unanimously 'London' -- the shape MXR-080-0270's repro needs: a returned draw that is a
    minority within its own measured cluster, not a separately generated, unassessed call."""
    calls = {"n": 0}

    def gen(prompt):
        calls["n"] += 1
        return "Paris" if calls["n"] == 1 else "London"

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

    # -- MXR-080-0270 (Critical): the returned draw used to come from a separate `generate()` call,
    # outside the sampled cluster used to measure confidence -- it could disagree with every sampled
    # answer while still inheriting THEIR entropy/confidence. Fixed: the returned answer is now the
    # first draw of the very batch entropy is measured over. ------------------------------------------
    def test_mxr_080_0270_returned_answer_is_assessed_not_a_separate_call(self):
        m = ExternalModel(_disagreeing_gen(), max_entropy=0.1, samples=8)
        a = m.answer("what is the capital of France?")
        self.assertEqual(a.answer, "Paris")  # the returned draw: call #1
        # 'Paris' is a 1-in-8 minority against 7x 'London' in its OWN batch -- entropy must be > 0,
        # not the 0.0 a separate, London-only sampled cluster would (wrongly) report.
        self.assertGreater(a.entropy, 0.0)
        self.assertIs(a.state, Confidence.UNCERTAIN)
        self.assertFalse(a.confident)  # must NOT inherit confidence from a cluster it disagrees with

    # -- MXR-080-0270 (Critical): with neither calibration_prompts nor an explicit max_entropy, the
    # cutoff used to silently default to +inf, so any finite entropy -- however high -- read as
    # "confident". Fixed: no calibration policy now fails closed to Confidence.UNCALIBRATED. ----------
    def test_mxr_080_0270_no_calibration_policy_fails_closed(self):
        m = ExternalModel(_flaky_gen(), samples=8)  # NOTE: no calibration_prompts, no max_entropy
        self.assertIsNone(m.max_entropy)
        a = m.answer("what is the tax rate")
        self.assertGreater(a.entropy, 0.5)  # genuinely high entropy -- self-contradicting
        self.assertIs(a.state, Confidence.UNCALIBRATED)
        self.assertFalse(a.confident)  # fails closed: never "confident" with nothing to compare against
        self.assertFalse(a.calibrated)
        self.assertFalse(m.confident("what is the tax rate"))  # the standalone probe also fails closed

    def test_mxr_080_0270_calibration_prompts_yield_a_genuinely_confident_verdict(self):
        # Control: a real calibration policy (via calibration_prompts, not just an explicit
        # max_entropy) plus a genuinely self-consistent model IS reported confident -- the fail-closed
        # fix must not make every answer uncalibrated/unconfident by construction.
        m = ExternalModel(_constant_gen(), calibration_prompts=["q1", "q2", "q3"], samples=6)
        self.assertIsNotNone(m.max_entropy)
        a = m.answer("what is the tax rate")
        self.assertIs(a.state, Confidence.CONFIDENT)
        self.assertTrue(a.confident)
        self.assertTrue(a.calibrated)

    def test_mxr_080_0270_explicit_non_finite_max_entropy_rejected(self):
        with self.assertRaises(ValueError):
            ExternalModel(_constant_gen(), max_entropy=float("inf"))
        with self.assertRaises(ValueError):
            ExternalModel(_constant_gen(), max_entropy=float("nan"))
        with self.assertRaises(ValueError):
            ExternalModel(_constant_gen(), max_entropy=-0.5)


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

    # -- MXR-080-0270 (Critical): an uncalibrated model must withhold by default exactly like a
    # genuinely self-contradicting one -- both are "cannot vouch for this", not "confident" -- and,
    # opted in via trust_uncertain, must be labeled "uncalibrated" rather than lumped into "uncertain"
    # (which would wrongly imply a calibration policy existed and was exceeded). ----------------------
    def test_mxr_080_0270_uncalibrated_external_withholds_by_default(self):
        m = ExternalModel(_constant_gen(), samples=6)  # no calibration policy at all
        act = external_action(m, description="tax rate external solver")
        inv = investigate("what is the tax rate", [act], _echo, min_confidence=0.0)
        self.assertTrue(inv.abstained)  # withheld even though the underlying model is self-consistent

    def test_mxr_080_0270_trust_uncertain_labels_uncalibrated_distinctly(self):
        m = ExternalModel(_constant_gen(), samples=6)  # no calibration policy at all
        act = external_action(m, description="tax rate external solver", trust_uncertain=True)
        inv = investigate("what is the tax rate", [act], _echo, min_confidence=0.0)
        self.assertFalse(inv.abstained)
        evidence = " ".join(inv.evidence)
        self.assertIn("uncalibrated", evidence)
        self.assertNotIn("external[confident", evidence)

    def test_external_action_is_a_costly_delegate(self):
        m = ExternalModel(_constant_gen(), max_entropy=1.0)
        act = external_action(m)
        self.assertEqual(act.kind, "delegate")
        self.assertGreaterEqual(act.cost, 8.0)  # escalation of last resort


if __name__ == "__main__":
    unittest.main()
