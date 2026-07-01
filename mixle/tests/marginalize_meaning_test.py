"""Marginalizing the string distribution over meaning equivalence classes (semantic entropy done right)."""
import unittest

import numpy as np

from mixle.inference import Clustering, marginalize_meaning, semantic_entropy
from mixle.inference.calibration import ProbabilityCalibrator, expected_calibration_error


class MarginalizeMeaningTest(unittest.TestCase):
    def test_counting_form_matches_cluster_frequencies(self):
        # no probs/weights -> P(c) = count_c / N (Monte-Carlo marginal)
        items = ["paris", "paris", "paris", "lyon"]
        m = marginalize_meaning(items)
        d = dict(zip(m.representatives, m.probs))
        self.assertAlmostEqual(d["paris"], 0.75, places=12)
        self.assertAlmostEqual(d["lyon"], 0.25, places=12)

    def test_probability_marginalization_differs_from_counting(self):
        # THE crux: meaning A is expressed by TWO distinct strings, meaning B by ONE. Counting the
        # three distinct strings gives A=2/3; but marginalizing the actual string probabilities
        # (sum within class) gives P(A)=0.3+0.2=0.5, P(B)=0.5. Counting's hidden "equiprobable
        # strings" assumption is simply wrong.
        strings = ["A said one way", "A said another way", "B"]
        eq = lambda a, b: a[0] == b[0]  # noqa: E731  -- same first char == same meaning
        p = np.array([0.3, 0.2, 0.5])
        m = marginalize_meaning(strings, eq, log_probs=np.log(p))
        by = dict(zip([r[0] for r in m.representatives], m.probs))
        self.assertAlmostEqual(by["A"], 0.5, places=10)
        self.assertAlmostEqual(by["B"], 0.5, places=10)
        # the counting form would have said A=2/3
        count = marginalize_meaning(strings, eq)
        self.assertAlmostEqual(dict(zip([r[0] for r in count.representatives], count.probs))["A"], 2 / 3, places=10)

    def test_weights_sum_within_class(self):
        m = marginalize_meaning(["x", "y", "x"], weights=[1.0, 2.0, 3.0])
        d = dict(zip(m.representatives, m.probs))
        self.assertAlmostEqual(d["x"], 4.0 / 6.0, places=12)  # 1 + 3
        self.assertAlmostEqual(d["y"], 2.0 / 6.0, places=12)

    def test_semantic_entropy_uses_marginal(self):
        # entropy of the probability-marginalized meaning distribution, not the string counts
        strings = ["A1", "A2", "B"]
        eq = lambda a, b: a[0] == b[0]  # noqa: E731
        se_prob = semantic_entropy(strings, eq, log_probs=np.log([0.3, 0.2, 0.5]))
        self.assertAlmostEqual(se_prob, np.log(2), places=10)  # P(A)=P(B)=0.5 -> ln 2
        se_count = semantic_entropy(strings, eq)  # A=2/3, B=1/3
        self.assertNotAlmostEqual(se_prob, se_count, places=3)

    def test_returns_clustering(self):
        m = marginalize_meaning(["a", "b"], log_probs=[0.0, 0.0])
        self.assertIsInstance(m, Clustering)
        np.testing.assert_allclose(m.probs.sum(), 1.0)

    def test_bad_length_raises(self):
        with self.assertRaises(ValueError):
            marginalize_meaning(["a", "b", "c"], log_probs=[0.0, 0.0])


class ProbabilityCalibratorTest(unittest.TestCase):
    def test_isotonic_lowers_ece_on_miscalibrated_scores(self):
        rng = np.random.RandomState(0)
        truth = rng.uniform(0, 1, 3000)
        y = (rng.uniform(size=3000) < truth).astype(float)
        raw = np.clip(truth**0.4, 0, 1)  # monotone but overconfident
        cal = ProbabilityCalibrator("isotonic").fit(raw[:1500], y[:1500])
        p = cal.predict(raw[1500:])
        raw_ece = float(expected_calibration_error(raw[1500:], y[1500:]))
        cal_ece = float(expected_calibration_error(p, y[1500:]))
        self.assertLess(cal_ece, raw_ece)
        self.assertLess(cal_ece, 0.06)

    def test_platt_lowers_ece(self):
        rng = np.random.RandomState(1)
        truth = rng.uniform(0, 1, 3000)
        y = (rng.uniform(size=3000) < truth).astype(float)
        raw = 1.0 / (1.0 + np.exp(-3.0 * (truth - 0.5)))  # miscalibrated logistic
        cal = ProbabilityCalibrator("platt").fit(raw[:1500], y[:1500])
        p = cal.predict(raw[1500:])
        self.assertLess(
            float(expected_calibration_error(p, y[1500:])),
            float(expected_calibration_error(raw[1500:], y[1500:])),
        )

    def test_flat_fit_when_score_is_unrelated_to_outcome(self):
        # a score with NO relationship to the outcome -> calibrated prob ~ base rate everywhere
        rng = np.random.RandomState(2)
        raw = rng.uniform(0, 1, 4000)  # random score
        y = (rng.uniform(size=4000) < 0.3).astype(float)  # outcome independent of score
        cal = ProbabilityCalibrator("isotonic").fit(raw[:2000], y[:2000])
        p = cal.predict(raw[2000:])
        # calibrated probabilities collapse toward the 0.3 base rate (score carried no information)
        self.assertLess(p.std(), 0.08)
        self.assertAlmostEqual(p.mean(), 0.3, delta=0.05)

    def test_monotone_and_needs_fit(self):
        cal = ProbabilityCalibrator("isotonic")
        with self.assertRaises(RuntimeError):
            cal.predict([0.5])
        cal.fit([0.1, 0.2, 0.3, 0.9], [0, 0, 1, 1])
        out = cal.predict([0.0, 0.15, 0.5, 1.0])
        self.assertTrue(np.all(np.diff(out) >= -1e-9))


if __name__ == "__main__":
    unittest.main()
