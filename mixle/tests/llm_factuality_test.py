"""Calibrated information likelihood for LLM answers (mixle.reason.llm.fit_factuality)."""

import unittest

import numpy as np

from mixle.inference import expected_calibration_error
from mixle.reason import FactualityModel, LLMUncertainty
from mixle.reason.llm import _auc  # white-box: MXR-080-0294


class KnowsSomeLLM:
    """Confidence (self-consistency) is informative but MISCALIBRATED: the model is right with a
    probability that rises with how much it knows, yet its raw agreement rate over-states it."""

    def __init__(self, prompts, truth_prob, seed=0):
        self.truth_prob = truth_prob  # prompt -> P(correct)
        self.gold = {p: "yes" for p in prompts}
        self.rng = np.random.RandomState(seed)

    def __call__(self, prompt):
        return "yes" if self.rng.random() < self.truth_prob[prompt] else "no"


class FactualityTest(unittest.TestCase):
    def _data(self, n=160, seed=0):
        rng = np.random.RandomState(seed)
        prompts = [f"q{i}" for i in range(n)]
        truth = {p: float(rng.uniform(0.5, 1.0)) for p in prompts}  # varying knowledge
        return prompts, truth

    def test_calibrated_probability_tracks_correctness(self):
        prompts, truth = self._data()
        llm = KnowsSomeLLM(prompts, truth, seed=1)
        uq = LLMUncertainty(llm, n=20)
        cal, test = prompts[:100], prompts[100:]
        fm = uq.fit_factuality([(p, "yes") for p in cal])
        self.assertIsInstance(fm, FactualityModel)
        # the raw signal genuinely discriminates right-vs-wrong (AUC well above chance)
        self.assertGreater(fm.discrimination, 0.6)
        # on held-out prompts, the calibrated P(correct) is well-calibrated against actual correctness
        probs, ys = [], []
        for p in test:
            probs.append(fm.probability(p))
            a = uq.assess(p)
            ys.append(1.0 if a.answer == "yes" else 0.0)
        ece = float(expected_calibration_error(np.array(probs), np.array(ys)))
        self.assertLess(ece, 0.18)  # calibrated probabilities, not raw confidence

    def test_signal_unrelated_to_truth_reports_chance_discrimination(self):
        # The honest readout of "the likelihood coming out of the LLM has no relationship to the truth
        # of the information": a signal that is unrelated to correctness gets AUC ~ 0.5 and a FLAT
        # calibration map (its value tells you nothing beyond the base rate). (Note: self-consistency
        # itself is NOT such a signal -- higher agreement correlates with the modal, more-likely-correct
        # answer -- so we use an explicitly random signal to make the point.)
        rng = np.random.RandomState(4)
        prompts = [f"q{i}" for i in range(220)]
        truth = {p: float(rng.uniform(0.55, 0.95)) for p in prompts}
        uq = LLMUncertainty(KnowsSomeLLM(prompts, truth, seed=2), n=15)
        noise = {p: float(rng.random()) for p in prompts}  # signal with no relationship to correctness
        fm = uq.fit_factuality([(p, "yes") for p in prompts], signal=lambda p: noise[p])
        self.assertLess(abs(fm.discrimination - 0.5), 0.12)  # ~ chance AUC
        # calibrated P(correct) is ~flat regardless of the (meaningless) signal value
        vals = [fm.probability(p) for p in prompts[:25]]
        self.assertLess(float(np.std(vals)), 0.12)

    def test_custom_signal(self):
        # a caller-supplied signal (e.g. a token logprob) can be calibrated the same way
        prompts = [f"q{i}" for i in range(80)]
        truth = {p: 0.5 + 0.5 * (i % 2) for i, p in enumerate(prompts)}  # alternating easy/hard
        uq = LLMUncertainty(KnowsSomeLLM(prompts, truth, seed=3), n=10)
        fm = uq.fit_factuality(
            [(p, "yes") for p in prompts],
            signal=lambda p: truth[p],  # oracle-ish signal for the test
        )
        self.assertGreater(fm.discrimination, 0.7)
        self.assertGreater(fm.probability(prompts[1]), fm.probability(prompts[0]))


class TieCorrectAucTest(unittest.TestCase):  # white-box: MXR-080-0294
    def test_four_identical_scores_balanced_outcomes_is_exactly_half(self):
        # The audit's exact scenario: 4 identical scores, 2 correct / 2 incorrect. The old
        # argsort-of-argsort ranking gave arbitrary distinct ranks to the tied scores and reported
        # 0.25 for this exact (score, outcome) pairing instead of the mathematically required 0.5 (a
        # tied, non-discriminating signal must score as chance, exactly 0.5).
        scores = np.array([0.7, 0.7, 0.7, 0.7])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(_auc(scores, outcomes), 0.5, places=12)

    def test_auc_is_order_invariant_under_ties(self):
        # AUC depends only on the (score, outcome) multiset, never on array order. The old ranking
        # broke this for tied scores (see test_four_identical_scores_balanced_outcomes_is_exactly_half);
        # every arrangement of the same multiset must now agree.
        scores = np.array([0.7, 0.7, 0.7, 0.7])
        for outcomes in (
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ):
            self.assertAlmostEqual(_auc(scores, np.array(outcomes)), 0.5, places=12)

    def test_partial_ties_still_use_average_rank(self):
        # 2 and 3 are tied (rank 2.5 each); 1 is rank 1, 4 is rank 4. Positive outcome on the tied
        # pair (index 1) and the top score (index 3); negative on rank 1 and one of the tied pair.
        scores = np.array([1.0, 2.0, 2.0, 4.0])
        outcomes = np.array([0.0, 1.0, 0.0, 1.0])
        # rank-sum of positives = 2.5 (score 2.0) + 4 (score 4.0) = 6.5; pos=2, neg=2
        # AUC = (6.5 - 2*3/2) / (2*2) = (6.5 - 3) / 4 = 0.875
        self.assertAlmostEqual(_auc(scores, outcomes), 0.875, places=12)

    def test_no_ties_matches_original_argsort_formula(self):
        # With no ties, average-rank and argsort-of-argsort ranks coincide -- the fix must not change
        # behavior on the (already-correct) no-ties case.
        rng = np.random.RandomState(0)
        scores = rng.uniform(size=50)
        outcomes = (rng.uniform(size=50) < 0.5).astype(float)
        ranks_old = np.argsort(np.argsort(scores)) + 1.0
        pos, neg = outcomes.sum(), (1.0 - outcomes).sum()
        expected = (ranks_old[outcomes == 1.0].sum() - pos * (pos + 1) / 2.0) / (pos * neg)
        self.assertAlmostEqual(_auc(scores, outcomes), float(expected), places=9)

    def test_degenerate_all_one_class_returns_half(self):
        self.assertEqual(_auc(np.array([0.1, 0.9]), np.array([1.0, 1.0])), 0.5)
        self.assertEqual(_auc(np.array([0.1, 0.9]), np.array([0.0, 0.0])), 0.5)

    def test_rejects_non_binary_outcomes(self):
        with self.assertRaises(ValueError):
            _auc(np.array([0.1, 0.9]), np.array([0.0, 0.5]))

    def test_rejects_non_finite_scores(self):
        with self.assertRaises(ValueError):
            _auc(np.array([0.1, float("nan")]), np.array([0.0, 1.0]))


class FactualityTargetIntegrityTest(unittest.TestCase):
    """STAT-RR23-07: truthiness coercion let a string-returning oracle turn an always-wrong
    generator (actual correctness 0.0) into probability(...) == 1.0."""

    def test_string_verdicts_refuse_and_wrong_generators_calibrate_low(self):
        rng = np.random.RandomState(0)
        u = LLMUncertainty(lambda p: f"wrong-{rng.randint(10**9)}", equivalent=lambda a, b: a == b, n=2)
        with self.assertRaisesRegex(ValueError, "bool or 0/1"):
            u.fit_factuality([(f"p{i}", "right") for i in range(10)], correct=lambda a, g: "false")
        model = u.fit_factuality([(f"p{i}", "right") for i in range(10)], correct=lambda a, g: False)
        self.assertLess(model.probability("fresh"), 0.5)


if __name__ == "__main__":
    unittest.main()
