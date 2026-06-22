"""WS-2: WrappedNormalDistribution -- a Gaussian wrapped onto the circle (directional)."""

import unittest

import numpy as np
from scipy.integrate import quad

from pysp.stats import WrappedNormalDistribution as WN


class WrappedNormalTest(unittest.TestCase):
    def test_density_integrates_to_one_and_positive(self):
        for mu, s2 in [(0.0, 0.3), (1.0, 1.0), (-1.5, 4.0), (0.5, 15.0)]:
            d = WN(mu, s2)
            integral, _ = quad(lambda t, dd=d: dd.density(t), -np.pi, np.pi)
            grid_min = min(d.density(t) for t in np.linspace(-np.pi, np.pi, 400))
            with self.subTest(mu=mu, s2=s2):
                self.assertAlmostEqual(integral, 1.0, places=5)
                self.assertGreater(grid_min, 0.0)  # truncated spatial sum stays strictly positive

    def test_seq_matches_scalar(self):
        d = WN(1.0, 1.0)
        th = np.array([-2.5, -0.3, 0.5, 1.2, 3.0])
        scalar = np.array([d.log_density(t) for t in th])
        self.assertTrue(np.allclose(scalar, d.seq_log_density(th)))

    def test_sampler_matches_density(self):
        d = WN(0.7, 0.8)
        s = np.asarray(d.sampler(seed=0).sample(400_000))
        hist, edges = np.histogram(s, bins=40, range=(-np.pi, np.pi), density=True)
        mids = 0.5 * (edges[:-1] + edges[1:])
        dens = np.array([d.density(m) for m in mids])
        self.assertLess(float(np.max(np.abs(hist - dens))), 0.02)

    def test_estimator_recovers_params(self):
        true = WN(1.2, 0.7)
        data = np.asarray(true.sampler(seed=2).sample(80_000))
        est = true.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(data), np.ones(len(data)), None)
        m = est.estimate(len(data), acc.value())
        self.assertAlmostEqual(m.mu, 1.2, delta=0.05)
        self.assertAlmostEqual(m.sigma2, 0.7, delta=0.05)

    def test_first_moment_matches_rho(self):
        # E[cos(theta - mu)] = exp(-sigma2/2) = rho
        d = WN(0.0, 1.0)
        s = np.asarray(d.sampler(seed=3).sample(400_000))
        self.assertAlmostEqual(float(np.mean(np.cos(s))), np.exp(-0.5), delta=0.01)


if __name__ == "__main__":
    unittest.main()
