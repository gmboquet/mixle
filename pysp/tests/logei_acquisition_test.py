"""WS-11/WS-12: numerically stable log Expected Improvement acquisition (Ament et al. 2023)."""

import unittest

import numpy as np

from pysp.doe import available_acquisitions, expected_improvement, log_expected_improvement, propose_next


class LogEITest(unittest.TestCase):
    def test_registered(self):
        for name in ("log_expected_improvement", "logei", "log_ei"):
            self.assertIn(name, available_acquisitions())

    def test_matches_log_ei_when_well_conditioned(self):
        mean = np.array([0.0, 0.5, 1.0, 1.5])
        std = np.array([1.0, 1.0, 1.0, 1.0])
        self.assertTrue(
            np.allclose(log_expected_improvement(mean, std, 1.0), np.log(expected_improvement(mean, std, 1.0)))
        )

    def test_finite_in_both_tails_where_ei_underflows(self):
        # deep no-improvement tail: EI underflows to 0 (log -> -inf) but logEI stays finite
        bad = log_expected_improvement(np.array([50.0]), np.array([1.0]), 0.0)[0]
        self.assertTrue(np.isfinite(bad))
        self.assertEqual(expected_improvement(np.array([50.0]), np.array([1.0]), 0.0)[0], 0.0)  # underflow
        # milder tail: EI is negligibly small while logEI is still informative
        self.assertLess(expected_improvement(np.array([20.0]), np.array([1.0]), 0.0)[0], 1.0e-80)
        self.assertTrue(np.isfinite(log_expected_improvement(np.array([20.0]), np.array([1.0]), 0.0)[0]))
        # excellent point (z >> 0): the erfcx branch is avoided, no overflow
        great = log_expected_improvement(np.array([-50.0]), np.array([1.0]), 0.0)[0]
        self.assertTrue(np.isfinite(great))
        self.assertAlmostEqual(great, np.log(50.0), delta=0.1)  # h(z) ~ z for large z

    def test_zero_std_is_neg_inf(self):
        self.assertEqual(log_expected_improvement(np.array([0.5]), np.array([0.0]), 1.0)[0], -np.inf)

    def test_usable_in_propose_next(self):
        rng = np.random.RandomState(0)
        x = rng.rand(10, 1)
        y = (x[:, 0] - 0.7) ** 2
        nxt = propose_next(x, y, bounds=[(0.0, 1.0)], acq="logei", seed=1)
        self.assertTrue(0.0 <= nxt[0] <= 1.0)


if __name__ == "__main__":
    unittest.main()
