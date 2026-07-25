"""Tests for LLM-output UQ (mixle.reason.llm) — semantic entropy + conformal answer-or-abstain."""

import unittest

import numpy as np

from mixle.inference import cluster_samples, semantic_entropy
from mixle.reason import LLMUncertainty
from mixle.reason.llm import (  # white-box: MXR-080-0293, MXR-080-0296
    Generation,
    _clopper_pearson_upper,
    _coerce_generation,
    _require_positive_n,
    _selective_risk_threshold,
)


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

    @staticmethod
    def _easy_hard_data(n_total: int, seed_data: int = 3):
        rng = np.random.RandomState(seed_data)
        vocab = ["paris", "rome", "tokyo", "cairo", "lima", "oslo"]
        gold, noise, kinds = {}, {}, {}
        for i in range(n_total):
            p = f"q{i}"
            gold[p] = vocab[rng.randint(len(vocab))]
            knowable = rng.random() < 0.5
            kinds[p] = knowable
            noise[p] = 0.05 if knowable else 0.9  # knowable => low noise, else near-random
        return gold, noise, kinds

    def test_conformal_selective_prediction_controls_error(self):
        # Build a calibration + test set mixing easy (knowable) and hard (not) questions.
        #
        # MXR-080-0293: calibrate() now certifies a threshold via an exact Clopper-Pearson bound,
        # Bonferroni-corrected across every candidate threshold (see _selective_risk_threshold) --
        # a genuine finite-sample (alpha, delta)-PAC guarantee, not a same-sample point estimate. That
        # correction legitimately needs more calibration data than the old (unsound) same-sample
        # selection did to certify a non-trivial threshold at this alpha: 200 questions (120
        # calibration / 80 test) is comfortably past that point for this signal -- see
        # test_small_calibration_set_correctly_refuses_to_certify below for what happens with too
        # little data (the OLD code would have "certified" a threshold there too; the new code
        # honestly refuses).
        gold, noise, kinds = self._easy_hard_data(200)
        llm = MockLLM(gold, noise, seed=5)
        uq = LLMUncertainty(llm, n=20)

        prompts = list(gold)
        cal = [(p, gold[p]) for p in prompts[:120]]
        test = prompts[120:]
        alpha = 0.15
        uq.calibrate(cal, alpha=alpha, delta=0.05)

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
        self.assertLessEqual(errors / answered, alpha + 0.10)
        # and it abstains on at least some of the questions it cannot know
        self.assertGreater(abstained_hard, 0)

    def test_small_calibration_set_correctly_refuses_to_certify(self):
        # MXR-080-0293 (Critical): the OLD same-sample selection would happily "certify" a threshold
        # from a 70-example calibration set split across two difficulty regimes (that was this exact
        # test's calibration set before this fix). A valid finite-sample (alpha, delta) procedure
        # cannot: with this little data per confidence bucket, no threshold's Clopper-Pearson upper
        # bound clears alpha=0.15, so calibrate() must refuse everything (threshold = +inf) rather than
        # deploy a threshold with no real statistical backing.
        gold, noise, _ = self._easy_hard_data(120)
        llm = MockLLM(gold, noise, seed=5)
        uq = LLMUncertainty(llm, n=20)
        cal = [(p, gold[p]) for p in list(gold)[:70]]
        uq.calibrate(cal, alpha=0.15, delta=0.05)
        self.assertEqual(uq._threshold, float("inf"))  # white-box: MXR-080-0293
        self.assertIsNone(uq.answer("q71"))  # refuses to answer anything

    def test_calibrate_rejects_invalid_alpha_delta_and_empty_examples(self):
        uq = LLMUncertainty(MockLLM({"q": "a"}, {"q": 0.0}), n=3)
        with self.assertRaises(ValueError):
            uq.calibrate([("q", "a")], alpha=1.5)
        with self.assertRaises(ValueError):
            uq.calibrate([("q", "a")], delta=0.0)  # delta must be in the OPEN interval (0, 1)
        with self.assertRaises(ValueError):
            uq.calibrate([("q", "a")], delta=1.0)
        with self.assertRaises(ValueError):
            uq.calibrate([])


class GenerationNormalizationTest(unittest.TestCase):
    """White-box coverage for MXR-080-0296's sample()-boundary normalization."""

    def test_plain_string_and_tuple_generations_coerce(self):
        self.assertEqual(_coerce_generation("paris"), Generation(text="paris"))
        g = _coerce_generation(("paris", -0.5))
        self.assertEqual(g.text, "paris")
        self.assertAlmostEqual(g.logprob, -0.5)

    def test_generation_rejects_non_finite_logprob(self):
        with self.assertRaises(ValueError):
            Generation(text="x", logprob=float("nan"))
        with self.assertRaises(ValueError):
            Generation(text="x", logprob=float("inf"))

    def test_coerce_generation_rejects_wrong_arity_tuple(self):
        with self.assertRaises(ValueError):
            _coerce_generation(("paris", -0.5, "extra"))

    def test_sample_rejects_mixed_tuple_and_plain_shapes(self):
        calls = iter(["paris", ("lyon", -0.2)])
        uq = LLMUncertainty(lambda p: next(calls), n=2)
        with self.assertRaises(ValueError):
            uq.sample("q")

    def test_sample_returns_generation_records(self):
        uq = LLMUncertainty(lambda p: ("paris", -0.3), n=3)
        samples = uq.sample("q")
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(isinstance(s, Generation) for s in samples))
        self.assertTrue(all(s.text == "paris" and s.logprob == -0.3 for s in samples))

    def test_decompose_clusters_generator_text_not_raw_tuples(self):
        # MXR-080-0296: a (text, logprob) generator used to have its raw tuples clustered (every
        # logprob differs, so every draw looked like its own distinct "meaning"). With only 2 distinct
        # underlying texts, clustering on text must collapse to (at most) 2 clusters, giving the SAME
        # total entropy as an equivalent plain-string generator -- not ln(n_samples).
        counter = {"i": 0}

        def tuple_gen(prompt):
            counter["i"] += 1
            text = "paris" if counter["i"] % 2 == 0 else "lyon"
            return (text, -0.01 * counter["i"])  # logprob is unique on every call

        def plain_gen(prompt):
            counter["i"] += 1
            return "paris" if counter["i"] % 2 == 0 else "lyon"

        uq_tuple = LLMUncertainty(tuple_gen, n=8)
        d_tuple = uq_tuple.decompose(["q", "q"])
        counter["i"] = 0
        uq_plain = LLMUncertainty(plain_gen, n=8)
        d_plain = uq_plain.decompose(["q", "q"])
        self.assertAlmostEqual(d_tuple.total, d_plain.total, places=9)
        self.assertLess(d_tuple.total, np.log(16))  # nowhere near "every draw is its own cluster"


class PositiveSampleCountTest(unittest.TestCase):
    """White-box + behavioral coverage for MXR-080-0296's sample-count validation."""

    def test_require_positive_n_accepts_positive_int(self):
        self.assertEqual(_require_positive_n(5), 5)
        self.assertEqual(_require_positive_n(np.int64(3)), 3)

    def test_require_positive_n_rejects_zero(self):
        # 0 used to silently mean "use the default" (`n or self.n`, and 0 is falsy).
        with self.assertRaises(ValueError):
            _require_positive_n(0)

    def test_require_positive_n_rejects_negative(self):
        # a negative n used to silently produce zero samples (range(negative) is empty).
        with self.assertRaises(ValueError):
            _require_positive_n(-3)

    def test_require_positive_n_rejects_bool_and_non_int(self):
        with self.assertRaises(TypeError):
            _require_positive_n(True)
        with self.assertRaises(TypeError):
            _require_positive_n(2.5)

    def test_constructor_rejects_non_positive_n(self):
        with self.assertRaises(ValueError):
            LLMUncertainty(lambda p: "x", n=0)
        with self.assertRaises(ValueError):
            LLMUncertainty(lambda p: "x", n=-1)

    def test_sample_n_zero_raises_instead_of_defaulting(self):
        uq = LLMUncertainty(lambda p: "x", n=10)
        with self.assertRaises(ValueError):
            uq.sample("q", n=0)

    def test_sample_negative_n_raises_instead_of_returning_empty(self):
        uq = LLMUncertainty(lambda p: "x", n=10)
        with self.assertRaises(ValueError):
            uq.sample("q", n=-3)

    def test_sample_none_n_uses_constructor_default(self):
        uq = LLMUncertainty(lambda p: "x", n=4)
        self.assertEqual(len(uq.sample("q", None)), 4)


class SelectiveRiskThresholdTest(unittest.TestCase):
    """White-box coverage for MXR-080-0293's replacement threshold-selection procedure."""

    def test_lucky_single_example_no_longer_certifies_a_threshold(self):
        # The audit's adversarial pattern: 10 examples at confidence 0.5 with a genuinely poor 40%
        # error rate (well above alpha=0.10), plus ONE example at confidence 0.95 that happens, purely
        # by chance, to be correct. The OLD same-sample selection picked tau=0.95 (the lone example's
        # same-sample error is 0.0 <= alpha) and reported it as meeting "correct with probability >=
        # 0.9" -- a claim with a sample size of ONE behind it. A valid finite-sample procedure cannot
        # certify anything from n=1: no threshold's Clopper-Pearson bound clears alpha here.
        confs = np.array([0.5] * 10 + [0.95])
        errs = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(errs.sum(), 4.0)  # the 0.5 bucket really is 40% wrong, not a strawman
        threshold = _selective_risk_threshold(confs, errs, alpha=0.10, delta=0.05)
        self.assertEqual(threshold, float("inf"))  # refuses, rather than "certifying" from n=1

    def test_larger_bucket_of_the_same_true_rate_does_certify(self):
        # Same underlying 0/0.95-confidence generative story, but the high-confidence bucket now has
        # enough examples (all correct) for the Clopper-Pearson bound to actually clear alpha -- so the
        # fix is not simply "always refuse", it correctly certifies once there is real evidence.
        confs = np.array([0.5] * 10 + [0.95] * 40)
        errs = np.array([1.0] * 4 + [0.0] * 6 + [0.0] * 40)
        threshold = _selective_risk_threshold(confs, errs, alpha=0.10, delta=0.05)
        self.assertEqual(threshold, 0.95)

    def test_clopper_pearson_upper_bound_matches_closed_form_zero_errors(self):
        # k=0 errors out of n trials: the exact one-sided Clopper-Pearson upper bound has the closed
        # form 1 - delta ** (1/n) (solving (1-p)^n = delta for p). Confirms _clopper_pearson_upper
        # isn't just "looks plausible" but matches the textbook exact formula.
        n, delta = 30, 0.05
        expected = 1.0 - delta ** (1.0 / n)
        self.assertAlmostEqual(_clopper_pearson_upper(0, n, delta), expected, places=9)

    def test_clopper_pearson_all_wrong_is_vacuous(self):
        self.assertEqual(_clopper_pearson_upper(5, 5, 0.05), 1.0)

    def test_no_candidates_refuses(self):
        self.assertEqual(_selective_risk_threshold(np.array([]), np.array([]), alpha=0.1, delta=0.05), float("inf"))


if __name__ == "__main__":
    unittest.main()
