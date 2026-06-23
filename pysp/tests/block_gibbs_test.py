"""Block Gibbs with per-block inference dispatch: conjugate + Metropolis in one model (Phase 7)."""

import unittest

import numpy as np

from pysp.inference.block_gibbs import BlockGibbs, ConjugateBlock, MetropolisBlock


class BlockGibbsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.y = rng.normal(2.0, 1.5, 200)
        self.n = len(self.y)

    def _model(self):
        y, n = self.y, self.n

        def draw_mu(state, r):  # conjugate Gaussian full conditional (prior N(0, 100))
            s2 = state["sigma"] ** 2
            pv = 1.0 / (n / s2 + 1 / 100.0)
            return r.normal(pv * (y.sum() / s2), np.sqrt(pv))

        def logp_sigma(sigma, state):  # non-conjugate (half-Cauchy prior) -> Metropolis
            if sigma <= 0:
                return -np.inf
            return -0.5 * np.sum((y - state["mu"]) ** 2) / sigma**2 - n * np.log(sigma) - np.log(1 + sigma**2)

        return ConjugateBlock("mu", draw_mu), MetropolisBlock("sigma", logp_sigma, scale=0.2)

    def test_recovers_the_posterior(self):
        mu_b, sig_b = self._model()
        ch = BlockGibbs([mu_b, sig_b], init={"mu": 0.0, "sigma": 1.0}).run(4000, burn=1000, seed=1)
        self.assertAlmostEqual(ch["mu"].mean(), self.y.mean(), delta=0.05)
        self.assertAlmostEqual(ch["sigma"].mean(), self.y.std(), delta=0.1)

    def test_dispatches_distinct_update_kinds(self):
        mu_b, sig_b = self._model()
        self.assertEqual(mu_b.kind, "conjugate")
        self.assertEqual(sig_b.kind, "metropolis")
        BlockGibbs([mu_b, sig_b], init={"mu": 0.0, "sigma": 1.0}).run(2000, burn=500, seed=1)
        self.assertGreater(sig_b.acceptance_rate, 0.1)  # the Metropolis block actually moves
        self.assertLess(sig_b.acceptance_rate, 0.9)

    def test_conjugate_block_matches_analytic_spread(self):
        mu_b, sig_b = self._model()
        ch = BlockGibbs([mu_b, sig_b], init={"mu": 0.0, "sigma": 1.0}).run(4000, burn=1000, seed=1)
        self.assertAlmostEqual(ch["mu"].std(), 1.5 / np.sqrt(self.n), delta=0.03)  # Gaussian conditional spread

    def test_mixed_matches_all_metropolis(self):
        mu_b, sig_b = self._model()
        mixed = BlockGibbs([mu_b, sig_b], init={"mu": 0.0, "sigma": 1.0}).run(4000, burn=1000, seed=1)
        y, n = self.y, self.n

        def logp_mu(mu, state):
            return -0.5 * np.sum((y - mu) ** 2) / state["sigma"] ** 2 - 0.5 * mu**2 / 100.0

        _, sig_b2 = self._model()
        allmh = BlockGibbs([MetropolisBlock("mu", logp_mu, 0.2), sig_b2], init={"mu": 0.0, "sigma": 1.0}).run(
            4000, burn=1000, seed=2
        )
        self.assertAlmostEqual(mixed["mu"].mean(), allmh["mu"].mean(), delta=0.05)
        self.assertAlmostEqual(mixed["sigma"].mean(), allmh["sigma"].mean(), delta=0.05)


if __name__ == "__main__":
    unittest.main()
