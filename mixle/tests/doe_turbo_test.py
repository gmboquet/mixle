"""Trust-region Bayesian optimization, TuRBO (mixle.doe.trust_region)."""

import importlib.util
import unittest
import warnings

import numpy as np

from mixle.doe.trust_region import TrustRegion

HAS_TORCH = importlib.util.find_spec("torch") is not None


class TrustRegionStateTest(unittest.TestCase):
    def test_expands_on_success_shrinks_on_failure(self):
        tr = TrustRegion(dim=4)
        start = tr.length
        for _ in range(tr.success_tol):
            tr.update(True)
        self.assertGreater(tr.length, start)  # doubled after consecutive successes

        tr = TrustRegion(dim=4)
        start = tr.length
        for _ in range(tr.failure_tol):
            tr.update(False)
        self.assertLess(tr.length, start)  # halved after consecutive failures

    def test_collapses(self):
        tr = TrustRegion(dim=2)
        for _ in range(200):
            tr.update(False)
        self.assertTrue(tr.collapsed)


@unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
class TurboOptimizeTest(unittest.TestCase):
    def test_finds_quadratic_optimum(self):
        from mixle.doe import turbo_minimize

        opt = np.array([0.3, -0.7, 1.2, -1.5, 0.0, 0.9])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = turbo_minimize(
                lambda x: float(np.sum((x - opt) ** 2)),
                [(-2.0, 2.0)] * 6,
                n_init=12,
                max_evals=120,
                batch_size=2,
                seed=0,
            )
        self.assertLess(np.linalg.norm(res["x"] - opt), 0.5)
        self.assertEqual(res["X"].shape[1], 6)

    def test_beats_random_search_in_high_dim(self):
        from mixle.doe import turbo_minimize

        def sphere(x):
            return float(np.sum(x**2))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = turbo_minimize(sphere, [(-3.0, 3.0)] * 10, n_init=20, max_evals=200, batch_size=4, seed=1)
        rand_best = min(sphere(np.random.RandomState(s).uniform(-3, 3, 10)) for s in range(200))
        self.assertLess(res["y"], rand_best)


if __name__ == "__main__":
    unittest.main()
