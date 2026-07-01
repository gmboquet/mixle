"""Tests for LLM-output UQ (mixle.reason.llm) — semantic entropy + conformal answer-or-abstain."""

import unittest

import numpy as np

from mixle.inference import cluster_samples, semantic_entropy
from mixle.reason import LLMUncertainty


class MockLLM:
    """A stochastic 'LLM': for each prompt it knows the gold answer with probability ``1 - noise``,
    otherwise emits a random wrong answer. ``noise`` controls how much it actually knows.
    """

    def __init__(self, gold: dict[str, str], noise: dict[str, float], seed: int = 0):
        self.gold = gold
        self.noise = noise
        self.rng = np.random.RandomState(seed)
        self.wrong = ["cat", "dog", "moon", "seven", "blue", "iron", "delta", "north"]

    def __call__(self, prompt: str) -> str:
        if self.rng.random() < self.noise[prompt]:
            return self.wrong[self.rng.randint(len(self.wrong))]  # confabulate
        return self.gold[prompt]


class SemanticEntropyPrimitiveTest(unittest.TestCase):
    def test_clustering_collapses_equivalent_phrasings(self):
        samples = ["Paris", "it's paris", "PARIS", "Lyon"]
        eq = lambda a, b: a.strip().lower() == b.strip().lower()  # noqa: E731
        c = cluster_samples(samples, eq)
        self.assertEqual(len(c.representatives), 3)  # paris (x3 phrasings) + lyon
        # entropy over 2 unequal clusters (3 vs 1) is positive but modest
        self.assertGreater(semantic_entropy(samples, eq), 0.0)

    def test_unanimous_samples_have_zero_entropy(self):
        self.assertAlmostEqual(semantic_entropy(["a", "a", "a", "a"]), 0.0, places=12)

    def test_all_distinct_maximizes_entropy(self):
        se = semantic_entropy(["a", "b", "c", "d"])
        self.assertAlmostEqual(se, np.log(4), places=10)


class LLMUncertaintyTest(unittest.TestCase):
    def setUp(self):
        self.gold = {"easy": "paris", "hard": "quux"}
        # 'easy': model almost always right (low entropy); 'hard': mostly guessing (high entropy)
        self.llm = MockLLM(self.gold, noise={"easy": 0.05, "hard": 0.85}, seed=1)
        self.uq = LLMUncertainty(self.llm, n=25)

    def test_semantic_entropy_separates_known_from_guessed(self):
        easy = self.uq.assess("easy")
        hard = self.uq.assess("hard")
        # the model is far more uncertain on the question it does not know
        self.assertLess(easy.semantic_entropy, hard.semantic_entropy)
        self.assertGreater(easy.confidence, hard.confidence)
        self.assertEqual(easy.answer, "paris")  # confident answer is the correct one

    def test_epistemic_decompose_across_paraphrases(self):
        # three paraphrases of the SAME hard question -> members; expect nonzero total uncertainty
        d = self.uq.decompose(["hard", "hard", "hard"])
        self.assertEqual(d.kind, "entropy")
        self.assertGreaterEqual(d.epistemic, 0.0)
        self.assertGreater(d.total, 0.0)

    def test_answer_or_abstain_needs_calibration(self):
        with self.assertRaises(RuntimeError):
            self.uq.answer("easy")

    def test_conformal_selective_prediction_controls_error(self):
        # Build a calibration + test set mixing easy (knowable) and hard (not) questions.
        rng = np.random.RandomState(3)
        vocab = ["paris", "rome", "tokyo", "cairo", "lima", "oslo"]
        gold, noise = {}, {}
        kinds = {}
        for i in range(120):
            p = f"q{i}"
            gold[p] = vocab[rng.randint(len(vocab))]
            knowable = rng.random() < 0.5
            kinds[p] = knowable
            noise[p] = 0.05 if knowable else 0.9  # knowable => low noise, else near-random
        llm = MockLLM(gold, noise, seed=5)
        uq = LLMUncertainty(llm, n=25)

        prompts = list(gold)
        cal = [(p, gold[p]) for p in prompts[:70]]
        test = prompts[70:]
        alpha = 0.15
        uq.calibrate(cal, alpha=alpha)

        answered, errors, abstained_hard = 0, 0, 0
        for p in test:
            out = uq.answer(p)
            if out is None:
                if not kinds[p]:
                    abstained_hard += 1
                continue
            answered += 1
            if out.answer != gold[p]:
                errors += 1
        # when it answers, the error rate respects the guarantee (with finite-sample slack)
        self.assertGreater(answered, 0)
        self.assertLessEqual(errors / answered, alpha + 0.12)
        # and it abstains on at least some of the questions it cannot know
        self.assertGreater(abstained_hard, 0)


if __name__ == "__main__":
    unittest.main()
