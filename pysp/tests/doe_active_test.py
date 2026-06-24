"""Active learning and Bayesian optimal design (pysp.doe.active)."""

import importlib.util
import unittest
import warnings

import numpy as np

from pysp.doe.active import expected_information_gain_linear, expected_information_gain_nmc

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ExpectedInformationGainTest(unittest.TestCase):
    def test_linear_eig_prefers_spread_design(self):
        spread = np.array([[1, -1.0], [1, -0.33], [1, 0.33], [1, 1.0]])
        clustered = np.array([[1, -0.05], [1, 0.0], [1, 0.02], [1, 0.05]])
        self.assertGreater(
            expected_information_gain_linear(spread, noise=0.5),
            expected_information_gain_linear(clustered, noise=0.5),
        )

    def test_nmc_matches_linear_gaussian_closed_form(self):
        f, sigma = 1.5, 0.7
        analytic = 0.5 * np.log(1 + f**2 / sigma**2)

        def prior(rng, n):
            return rng.standard_normal((n, 1))

        def loglik(thetas, y):
            return -0.5 * ((y - thetas[:, 0] * f) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

        def sim(theta, rng):
            return np.array([theta[0] * f + sigma * rng.standard_normal()])

        nmc = expected_information_gain_nmc(prior, loglik, sim, n_outer=4000, n_inner=4000, seed=0)
        self.assertAlmostEqual(nmc, analytic, delta=0.05)


@unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
class ActiveLearningTest(unittest.TestCase):
    def test_alm_proposes_into_the_data_gap(self):
        from pysp.doe import propose_active_learning

        x = np.array([[-2.0], [-1.8], [-1.6], [1.6], [1.8], [2.0]])  # gap in the middle
        y = np.sin(x[:, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xn = propose_active_learning(x, y, [(-2.0, 2.0)], method="alm", n_candidates=200, seed=1)
        self.assertLess(abs(xn[0]), 1.0)  # placed where uncertainty is highest

    def test_active_learning_loop_runs(self):
        from pysp.doe import active_learning_design

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            des = active_learning_design(
                lambda x: float(np.sin(3 * x[0])), [(-2.0, 2.0)], n_init=6, max_evals=14, method="alc", seed=2
            )
        self.assertEqual(des["X"].shape[0], 14)


if __name__ == "__main__":
    unittest.main()
