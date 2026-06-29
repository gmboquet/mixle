"""Global sensitivity analysis: Sobol indices vs the Ishigami analytic values, Morris screening (Phase 4)."""

import unittest

import numpy as np

from mixle.doe import morris_screening, sobol_indices


def ishigami(x, a=7.0, b=0.1):
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


BOUNDS = [(-np.pi, np.pi)] * 3


class SobolTest(unittest.TestCase):
    def setUp(self):
        self.res = sobol_indices(ishigami, BOUNDS, n=16384, seed=0, names=["x1", "x2", "x3"])

    def test_first_order_matches_analytic_ishigami(self):
        np.testing.assert_allclose(self.res["S1"], [0.314, 0.442, 0.0], atol=0.03)

    def test_total_order_matches_analytic_ishigami(self):
        np.testing.assert_allclose(self.res["ST"], [0.557, 0.442, 0.244], atol=0.03)

    def test_x3_is_pure_interaction(self):
        self.assertLess(self.res["S1"][2], 0.05)  # no main effect
        self.assertGreater(self.res["ST"][2] - self.res["S1"][2], 0.1)  # but interacts (with x1)

    def test_additive_linear_model(self):
        res = sobol_indices(lambda x: x[:, 0] + 2 * x[:, 1] + 3 * x[:, 2], [(0, 1)] * 3, n=8192)
        self.assertAlmostEqual(res["S1"].sum(), 1.0, delta=0.02)  # additive -> first orders partition
        np.testing.assert_allclose(res["S1"], res["ST"], atol=0.02)  # no interactions
        np.testing.assert_allclose(res["S1"] / res["S1"][0], [1.0, 4.0, 9.0], atol=0.2)  # variance ~ coef^2

    def test_constant_output_is_all_zero(self):
        res = sobol_indices(lambda x: np.ones(len(x)), [(0, 1)] * 2, n=512)
        np.testing.assert_array_equal(res["S1"], [0.0, 0.0])
        np.testing.assert_array_equal(res["ST"], [0.0, 0.0])


class MorrisTest(unittest.TestCase):
    def test_ranks_influential_inputs(self):
        m = morris_screening(ishigami, BOUNDS, trajectories=60, seed=1, names=["x1", "x2", "x3"])
        self.assertEqual(m["mu_star"].shape, (3,))
        self.assertTrue(np.all(m["mu_star"] > 0))  # all three move the output (x3 via interaction)


if __name__ == "__main__":
    unittest.main()
