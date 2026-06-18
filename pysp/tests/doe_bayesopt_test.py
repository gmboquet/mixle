"""Tests for the DoE Bayesian-optimization loop (WS-E).

``expected_improvement`` is torch-free and tested directly; the GP-surrogate ``propose_next`` /
``minimize`` paths require torch and are skipped when it is unavailable.
"""

import importlib.util
import unittest

import numpy as np

from pysp.doe import expected_improvement, minimize, propose_next

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ExpectedImprovementTest(unittest.TestCase):
    def test_zero_std_gives_zero_ei(self):
        ei = expected_improvement(mean=np.array([0.0, 1.0]), std=np.array([0.0, 0.0]), best=0.5)
        np.testing.assert_array_equal(ei, np.zeros(2))

    def test_minimize_rewards_lower_mean(self):
        # Equal std: a smaller predicted mean is more improving under minimization.
        ei = expected_improvement(mean=np.array([-1.0, 0.0, 1.0]), std=np.array([1.0, 1.0, 1.0]), best=0.0)
        self.assertGreater(ei[0], ei[1])
        self.assertGreater(ei[1], ei[2])

    def test_maximize_rewards_higher_mean(self):
        ei = expected_improvement(
            mean=np.array([-1.0, 0.0, 1.0]), std=np.array([1.0, 1.0, 1.0]), best=0.0, maximize=True
        )
        self.assertGreater(ei[2], ei[1])
        self.assertGreater(ei[1], ei[0])

    def test_nonnegative_and_increases_with_uncertainty(self):
        # At the incumbent mean (improve=0), EI grows with predictive std.
        low = expected_improvement(mean=np.array([0.0]), std=np.array([0.3]), best=0.0)
        high = expected_improvement(mean=np.array([0.0]), std=np.array([1.5]), best=0.0)
        self.assertTrue(np.all(low >= 0.0) and np.all(high >= 0.0))
        self.assertGreater(high[0], low[0])


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class BayesOptLoopTest(unittest.TestCase):
    def test_propose_next_in_bounds(self):
        bounds = [(-2.0, 2.0), (0.0, 5.0)]
        rng = np.random.RandomState(0)
        x = rng.uniform([-2.0, 0.0], [2.0, 5.0], size=(6, 2))
        y = np.sum((x - np.array([0.5, 2.0])) ** 2, axis=1)
        nxt = np.asarray(propose_next(x, y, bounds, n_candidates=128, seed=1, fit_kwargs={"max_its": 60}))
        self.assertEqual(nxt.shape, (2,))
        self.assertTrue(np.all(nxt >= [-2.0, 0.0]) and np.all(nxt <= [2.0, 5.0]))

    def test_minimize_finds_near_optimum_of_a_bowl(self):
        target = np.array([0.5, -1.0])
        bounds = [(-3.0, 3.0), (-3.0, 3.0)]

        def objective(p):
            return float(np.sum((p - target) ** 2))

        result = minimize(objective, bounds, n_init=6, n_iter=20, seed=0, n_candidates=256, fit_kwargs={"max_its": 60})
        self.assertEqual(result.x.shape[0], result.y.shape[0])
        self.assertEqual(result.x.shape[0], 26)
        # BO should beat the best of the initial random design and land near the optimum.
        best_init = float(np.min(result.y[:6]))
        self.assertLessEqual(result.best_y, best_init)
        self.assertLess(result.best_y, 0.5)


if __name__ == "__main__":
    unittest.main()
