"""The assembled laptop scientist: real encoders (CLIP/MiniLM/SmolLM2) + certified heads + verified QA.

Marked optional+slow: needs the open-weight models in the local HF cache. Excluded from the fast gate;
part of the full correctness run. Each test is a RECEIPT that a frontier-relevant claim actually holds
on a laptop with no network -- not an assertion.
"""

import unittest

import numpy as np
import pytest

pytestmark = [pytest.mark.optional, pytest.mark.slow, pytest.mark.integration]

transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")


def _cifar(n_train=1500, n_test=600):
    from datasets import load_dataset

    tr = load_dataset("cifar10", split=f"train[:{n_train}]")
    te = load_dataset("cifar10", split=f"test[:{n_test}]")
    return tr, te


class CertifiedPerceptionTest(unittest.TestCase):
    """CLIP image latents + a closed-form mixle head: accurate, CERTIFIED, and calibrated on real CIFAR-10."""

    @classmethod
    def setUpClass(cls):
        from mixle.scientist import Scientist, encode_images

        tr, te = _cifar()
        cls.ztr = encode_images([r["img"] for r in tr])
        cls.zte = encode_images([r["img"] for r in te])
        cls.ytr = [r["label"] for r in tr]
        cls.yte = np.array([r["label"] for r in te])
        cls.model = Scientist.study(cls.ztr, cls.ytr, alpha=0.1, seed=0)

    def test_accuracy_is_high_and_fit_is_closed_form(self):
        acc = float((self.model.predict(self.zte) == self.yte).mean())
        self.assertGreater(acc, 0.85)  # real CLIP + a closed-form head, no gradient descent
        self.assertEqual(self.model.certificate.guarantee.name, "GLOBAL_UNIQUE")
        self.assertEqual(len(self.model.certificate.gradient_blocks), 0)

    def test_conformal_sets_cover_at_the_stated_level(self):
        sets = self.model.prediction_sets(self.zte)
        coverage = float(np.mean([y in s for y, s in zip(self.yte, sets)]))
        self.assertGreater(coverage, 0.85)  # 90% target, honest sampling slack

    def test_confident_predictions_are_more_accurate_than_overall(self):
        pred = self.model.predict(self.zte)
        confident = ~self.model.abstains(self.zte)
        acc_all = float((pred == self.yte).mean())
        acc_conf = float((pred[confident] == self.yte[confident]).mean())
        self.assertGreater(acc_conf, acc_all)  # abstention buys accuracy where it matters

    def test_trains_in_seconds(self):
        self.assertLess(self.model.train_seconds, 5.0)  # the head fit itself is near-instant


class VerifiedReasoningTest(unittest.TestCase):
    """Grounded QA through the local LLM: answers only what the substrate supports, abstains otherwise."""

    def setUp(self):
        from mixle.scientist import Scientist

        self.sci = Scientist()
        self.sci.learn(
            [
                "Uranium-238 decays to lead-206 with a half-life of 4.468 billion years, the basis of U-Pb dating.",
                "The Cretaceous-Paleogene (K-Pg) boundary is dated to approximately 66.0 million years ago.",
                "Carbon-14 has a half-life of 5730 years, useful for dating materials younger than 50,000 years.",
            ]
        )

    def test_answers_supported_questions_and_grounds_them(self):
        inv = self.sci.ask("what is the half-life of uranium-238")
        self.assertFalse(inv.abstained)
        self.assertGreaterEqual(inv.factuality.grounded_fraction, 0.5)
        self.assertIn("4.468", inv.answer)  # the real number, extracted from evidence

    def test_abstains_on_unsupported_questions(self):
        # raw SmolLM2 confidently hallucinates these; the scientist refuses without provenance
        for q in ["what is the boiling point of tungsten", "who discovered the electron"]:
            self.assertTrue(self.sci.ask(q).abstained, q)

    def test_every_answer_carries_citations(self):
        inv = self.sci.ask("when is the K-Pg boundary dated to")
        self.assertFalse(inv.abstained)
        self.assertTrue(inv.factuality.verdicts)  # per-claim receipt exists


class ProposeAndWonderTest(unittest.TestCase):
    """The don't-know-but-here's-how half: abstention becomes a plan, and curiosity generates conjectures."""

    def setUp(self):
        from mixle.scientist import Scientist
        from mixle.substrate.act import Action

        self.sci = Scientist()
        self.sci.learn(
            [
                "Uranium-238 decays to lead-206 with a half-life of 4.468 billion years, the basis of U-Pb dating.",
                "Zircon crystals incorporate uranium but reject lead at crystallization, so lead is radiogenic.",
                "The Cretaceous-Paleogene (K-Pg) boundary is dated to approximately 66.0 million years ago.",
            ]
        )
        self.sci.add_action(
            Action(
                "halflife_calc",
                "compute",
                run=lambda q: ["x"],
                cost=1.0,
                description="compute decay ages from isotope half-life measurements",
            )
        )

    def test_abstention_returns_a_ranked_research_proposal(self):
        inv = self.sci.investigate("what is the half-life of potassium-40")
        self.assertTrue(inv.abstained)
        self.assertIsNotNone(inv.proposal)
        self.assertTrue(inv.proposal.options)  # concrete ways to find out
        # the mounted, topically-relevant compute capability is ranked at the top (cheapest relevant)
        self.assertEqual(inv.proposal.best()["kind"], "compute")
        self.assertIn("half-life", inv.proposal.render())

    def test_proposal_names_the_nearest_knowledge_as_the_gap(self):
        prop = self.sci.propose("what is the half-life of potassium-40")
        self.assertTrue(prop.nearest_knowledge)  # it says what it ALMOST knows
        self.assertIn("don't know", prop.render().lower())

    def test_answered_question_needs_no_proposal(self):
        inv = self.sci.investigate("when is the K-Pg boundary dated to")
        self.assertFalse(inv.abstained)
        self.assertIsNone(getattr(inv, "proposal", None))

    def test_wonder_generates_labeled_conjectures_it_does_not_already_know(self):
        conjectures = self.sci.wonder(topic="dating", n=2, seed=1)
        self.assertTrue(conjectures)  # curiosity produced something
        for c in conjectures:
            self.assertEqual(c.status, "conjecture")  # never asserted as fact
            self.assertTrue(self.sci.ask(c.question).abstained)  # genuinely open, not rediscovery
            self.assertIsNotNone(c.proposal)  # each carries a proposed test


if __name__ == "__main__":
    unittest.main()
