"""WS-7: dense mass-matrix HMC -- warmup-adapted Euclidean metric for correlated posteriors."""

import unittest

import numpy as np

from mixle.inference import ess_bulk
from mixle.inference.mcmc import dense_mass_hmc, hamiltonian_monte_carlo


def _ess(samples):
    s = np.array([np.atleast_1d(v) for v in samples])
    return np.array([ess_bulk(s[None, :, [k]])[0] for k in range(s.shape[1])]), s


class DenseMassHMCTest(unittest.TestCase):
    def test_samples_correct_target_and_beats_identity(self):
        c = np.array([[1.0, 0.99], [0.99, 1.0]])  # strongly correlated, ill-conditioned
        cinv = np.linalg.inv(c)
        lp = lambda x: -0.5 * np.asarray(x) @ cinv @ np.asarray(x)  # noqa: E731
        g = lambda x: -cinv @ np.asarray(x)  # noqa: E731

        dense = dense_mass_hmc(lp, g, [0.0, 0.0], 4000, 0.2, 20, warmup=1000, rng=np.random.RandomState(0))
        dense_ess, s = _ess(dense.samples)
        # recovers the target (mean ~ 0, the strong correlation)
        self.assertTrue(np.allclose(s.mean(axis=0), 0.0, atol=0.1))
        self.assertAlmostEqual(np.cov(s.T)[0, 1], 0.99, delta=0.1)

        ident = hamiltonian_monte_carlo(lp, g, [0.0, 0.0], 4000, 0.2, 20, rng=np.random.RandomState(0))
        ident_ess, _ = _ess(ident.samples)
        # the adapted metric decorrelates the target -> dramatically higher effective sample size
        self.assertGreater(dense_ess.min(), 5.0 * ident_ess.max())


if __name__ == "__main__":
    unittest.main()
