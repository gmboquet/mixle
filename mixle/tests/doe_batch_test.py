"""Rigorous batch (multi-point) Bayesian optimization (mixle.doe.batch)."""

import importlib.util
import unittest
import warnings

import numpy as np
from scipy.stats import norm

from mixle.doe.batch import monte_carlo_qei

HAS_TORCH = importlib.util.find_spec("torch") is not None  # the proposal drivers fit the torch GP surrogate


class MonteCarloQeiTest(unittest.TestCase):
    """The q-EI estimator is pure NumPy and checkable in closed form."""

    def test_q1_matches_analytic_ei(self):
        best, mu, sigma = 1.0, 0.5, 0.8
        z = (best - mu) / sigma
        analytic = (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)  # EI, minimization
        mc = monte_carlo_qei([mu], [[sigma**2]], best, maximize=False, samples=200000, seed=0)
        self.assertAlmostEqual(mc, analytic, places=2)

    def test_correlated_duplicate_gives_no_batch_gain(self):
        # two perfectly-correlated identical points have the q-EI of a single point (the property
        # kriging-believer violates): batching duplicates is worthless.
        best, mu, var = 1.0, 0.5, 0.64
        single = monte_carlo_qei([mu], [[var]], best, samples=200000, seed=0)
        dup = monte_carlo_qei([mu, mu], [[var, var], [var, var]], best, samples=200000, seed=0)
        self.assertAlmostEqual(dup, single, places=2)

    def test_independent_points_increase_qei(self):
        best, mu, var = 1.0, 0.5, 0.64
        single = monte_carlo_qei([mu], [[var]], best, samples=200000, seed=0)
        indep = monte_carlo_qei([mu, mu], [[var, 0.0], [0.0, var]], best, samples=200000, seed=0)
        self.assertGreater(indep, single)


@unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
class BatchProposalTest(unittest.TestCase):
    def _problem(self):
        rng = np.random.RandomState(0)
        x = rng.uniform(-3, 3, (12, 1))
        y = np.sin(3 * x[:, 0]) + 0.3 * x[:, 0] ** 2  # two basins
        return x, y, [(-3.0, 3.0)]

    def test_qei_batch_is_diverse(self):
        from mixle.doe import propose_qei_batch

        x, y, bounds = self._problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = propose_qei_batch(x, y, bounds, q=3, n_candidates=120, mc_samples=128, seed=1)
        self.assertEqual(b.shape, (3, 1))
        self.assertTrue((b >= -3).all() and (b <= 3).all())
        pdist = [abs(b[i, 0] - b[j, 0]) for i in range(3) for j in range(i + 1, 3)]
        self.assertGreater(min(pdist), 0.1)  # not near-duplicate points

    def test_local_penalization_is_diverse(self):
        from mixle.doe import propose_local_penalization

        x, y, bounds = self._problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = propose_local_penalization(x, y, bounds, q=3, n_candidates=400, seed=1)
        self.assertEqual(b.shape, (3, 1))
        pdist = [abs(b[i, 0] - b[j, 0]) for i in range(3) for j in range(i + 1, 3)]
        self.assertGreater(min(pdist), 0.3)  # floored spacing keeps the batch spread


if __name__ == "__main__":
    unittest.main()
