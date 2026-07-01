"""Calibrated information likelihood for LLM answers (mixle.reason.llm.fit_factuality)."""

import unittest

import numpy as np

from mixle.inference import expected_calibration_error
from mixle.reason import FactualityModel, LLMUncertainty


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


if __name__ == "__main__":
    unittest.main()
