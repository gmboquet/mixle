"""Tests for the ask-tell BayesianOptimizer (WS-E).

The ask-tell mechanics (initial space-filling design, tell/best bookkeeping, validation) are
torch-free; the GP-acquisition phase requires torch and is skipped without it.
"""

import importlib.util
import unittest

import numpy as np

from pysp.doe import BayesianOptimizer

HAS_TORCH = importlib.util.find_spec("torch") is not None


class AskTellMechanicsTest(unittest.TestCase):
    bounds = [(-2.0, 2.0), (0.0, 5.0)]

    def _in_bounds(self, x):
        b = np.asarray(self.bounds, dtype=float)
        return bool(np.all(x >= b[:, 0] - 1e-9) and np.all(x <= b[:, 1] + 1e-9))

    def test_ask_returns_point_in_bounds(self):
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        x = opt.ask()
        self.assertEqual(x.shape, (2,))
        self.assertTrue(self._in_bounds(x))

    def test_initial_asks_are_space_filling_and_distinct(self):
        # Within the n_init budget, asks come from a Latin-hypercube design (no GP, no torch needed).
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        pts = np.array([opt.ask() for _ in range(4)])
        self.assertEqual(pts.shape, (4, 2))
        self.assertEqual(len({tuple(p) for p in pts}), 4)  # all distinct
        self.assertTrue(all(self._in_bounds(p) for p in pts))

    def test_batch_ask_in_init_phase(self):
        opt = BayesianOptimizer(self.bounds, n_init=6, seed=1)
        batch = opt.ask(q=3)
        self.assertEqual(batch.shape, (3, 2))
        self.assertTrue(all(self._in_bounds(p) for p in batch))

    def test_tell_records_and_best_tracks_minimum(self):
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        opt.tell([0.0, 1.0], 5.0).tell([1.0, 2.0], 2.0).tell([-1.0, 4.0], 9.0)
        self.assertEqual(opt.n_observations, 3)
        self.assertEqual(opt.x.shape, (3, 2))
        self.assertEqual(opt.y.shape, (3,))
        self.assertEqual(opt.best.best_y, 2.0)
        np.testing.assert_array_equal(opt.best.best_x, [1.0, 2.0])

    def test_tell_accepts_batches(self):
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        opt.tell([[0.0, 1.0], [1.0, 2.0]], [3.0, 1.0])
        self.assertEqual(opt.n_observations, 2)
        self.assertEqual(opt.best.best_y, 1.0)

    def test_maximize_flips_incumbent(self):
        opt = BayesianOptimizer(self.bounds, maximize=True, n_init=4, seed=0)
        opt.tell([[0.0, 1.0], [1.0, 2.0]], [3.0, 1.0])
        self.assertEqual(opt.best.best_y, 3.0)

    def test_validation(self):
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        with self.assertRaises(ValueError):
            opt.ask(q=0)
        with self.assertRaises(ValueError):
            opt.tell([0.0], 1.0)  # wrong dimension
        with self.assertRaises(ValueError):
            opt.tell([[0.0, 1.0], [1.0, 2.0]], [1.0])  # x/y length mismatch
        with self.assertRaises(ValueError):
            _ = BayesianOptimizer(self.bounds, n_init=4, seed=0).best  # no observations yet


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class AskTellOptimizationTest(unittest.TestCase):
    def test_loop_converges_on_a_bowl(self):
        target = np.array([0.5, -1.0])
        bounds = [(-3.0, 3.0), (-3.0, 3.0)]

        def objective(p):
            return float(np.sum((p - target) ** 2))

        opt = BayesianOptimizer(bounds, n_init=6, n_candidates=256, seed=0, fit_kwargs={"max_its": 60})
        for _ in range(24):
            x = opt.ask()
            opt.tell(x, objective(x))
        self.assertEqual(opt.n_observations, 24)
        self.assertLess(opt.best.best_y, 0.5)  # beats the coarse initial design, nears the optimum

    def test_batch_ask_after_data_returns_distinct_points(self):
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]

        def objective(p):
            return float(np.sum(p**2))

        opt = BayesianOptimizer(bounds, n_init=5, n_candidates=128, seed=0, fit_kwargs={"max_its": 50})
        for _ in range(5):  # exhaust the init design
            x = opt.ask()
            opt.tell(x, objective(x))
        batch = opt.ask(q=3)  # now a GP kriging-believer batch
        self.assertEqual(batch.shape, (3, 2))
        self.assertGreater(len({tuple(np.round(p, 6)) for p in batch}), 1)


if __name__ == "__main__":
    unittest.main()
