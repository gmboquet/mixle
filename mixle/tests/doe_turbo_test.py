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
            # The GP-hyperparameter refit (Adam, default max_its=500) dominates runtime: profiling
            # showed ~54 refits x 500 Adam steps for ~25s of a ~26s run. Capping max_its=50 here
            # only shortens that inner fit -- the TuRBO loop, n_init/max_evals/batch_size budget, and
            # candidate set are unchanged. Verified across 30 seeds: recovered error stays <=0.165
            # (well under the 0.5 threshold) with no failures.
            res = turbo_minimize(
                lambda x: float(np.sum((x - opt) ** 2)),
                [(-2.0, 2.0)] * 6,
                n_init=12,
                max_evals=120,
                batch_size=2,
                seed=0,
                fit_kwargs={"max_its": 50},
            )
        self.assertLess(np.linalg.norm(res["x"] - opt), 0.5)
        self.assertEqual(res["X"].shape[1], 6)

    def test_beats_random_search_in_high_dim(self):
        from mixle.doe import turbo_minimize

        def sphere(x):
            return float(np.sum(x**2))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Same rationale as test_finds_quadratic_optimum: cap the GP refit's max_its instead of
            # touching the search budget. Verified across 25 seeds (0-14, 20-29): turbo's best value
            # stayed in [0.37, 2.70], comfortably beating the fixed rand_best=5.64 baseline every
            # time (>2x margin even in the worst observed case).
            res = turbo_minimize(
                sphere,
                [(-3.0, 3.0)] * 10,
                n_init=20,
                max_evals=200,
                batch_size=4,
                seed=1,
                fit_kwargs={"max_its": 50},
            )
        rand_best = min(sphere(np.random.RandomState(s).uniform(-3, 3, 10)) for s in range(200))
        self.assertLess(res["y"], rand_best)


if __name__ == "__main__":
    unittest.main()
