"""Tests for Schwarz (BIC) model weights on marginal field profiles (WS-F)."""

import math
import unittest

import numpy as np

from pysp.utils.automatic import analyze_structure
from pysp.utils.automatic.profiling import _bic_weights


class BicWeightsTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_bic_weights({}, 100), {})

    def test_weights_sum_to_one_and_favor_the_best(self):
        scores = {"gaussian": 2.0, "lognormal": 2.1, "gamma": 2.5}
        w = _bic_weights(scores, nobs=50)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=12)
        self.assertGreater(w["gaussian"], w["lognormal"])
        self.assertGreater(w["lognormal"], w["gamma"])

    def test_matches_schwarz_formula(self):
        scores = {"a": 1.0, "b": 1.3}
        n = 40
        w = _bic_weights(scores, n)
        # w_b / w_a = exp(-n*ln2*(1.3-1.0))
        ratio = w["b"] / w["a"]
        self.assertAlmostEqual(ratio, math.exp(-n * math.log(2.0) * 0.3), places=10)

    def test_concentrates_on_winner_as_n_grows(self):
        scores = {"gaussian": 2.0, "lognormal": 2.2}
        small = _bic_weights(scores, 10)
        large = _bic_weights(scores, 2000)
        self.assertGreater(large["gaussian"], small["gaussian"])
        self.assertGreater(large["gaussian"], 0.999)

    def test_profile_exposes_weights_consistent_with_recommendation(self):
        rng = np.random.RandomState(0)
        data = list(rng.lognormal(mean=0.7, sigma=0.6, size=600))
        field = analyze_structure(data, pairwise=False).fields[0]
        weights = field.model_weights()
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
        # The recommended model should carry the largest weight.
        self.assertEqual(max(weights, key=weights.get), field.recommendation)


if __name__ == "__main__":
    unittest.main()
