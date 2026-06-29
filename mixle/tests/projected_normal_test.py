"""WS-2: ProjectedNormalDistribution -- a circular law from projecting a 2-D Gaussian (directional)."""

import unittest

import numpy as np
from scipy.integrate import quad

import mixle
from mixle.capability import HasCDF
from mixle.stats import ProjectedNormalDistribution as PN


class ProjectedNormalTest(unittest.TestCase):
    def test_density_integrates_to_one(self):
        for mu in [(0.0, 0.0), (2.0, 0.0), (1.5, -1.0), (3.0, 2.0)]:
            integral, _ = quad(lambda t, d=PN(*mu): d.density(t), -np.pi, np.pi)
            with self.subTest(mu=mu):
                self.assertAlmostEqual(integral, 1.0, places=5)

    def test_seq_matches_scalar(self):
        d = PN(1.5, -1.0)
        th = np.array([-2.5, -0.3, 0.5, 1.2, 3.0])
        scalar = np.array([d.log_density(t) for t in th])
        seq = d.seq_log_density((np.cos(th), np.sin(th)))
        self.assertTrue(np.allclose(scalar, seq))

    def test_sampler_matches_density(self):
        d = PN(2.0, 1.0)
        s = np.asarray(d.sampler(seed=0).sample(400_000))
        hist, edges = np.histogram(s, bins=40, range=(-np.pi, np.pi), density=True)
        mids = 0.5 * (edges[:-1] + edges[1:])
        dens = np.array([d.density(m) for m in mids])
        self.assertLess(float(np.max(np.abs(hist - dens))), 0.03)

    def test_uniform_at_zero(self):
        d = PN(0.0, 0.0)
        for t in (-2.0, 0.0, 1.0, 3.0):
            self.assertAlmostEqual(d.density(t), 1.0 / (2.0 * np.pi), places=9)

    def test_em_recovers_mu(self):
        true = PN(2.0, -1.0)
        data = np.asarray(true.sampler(seed=1).sample(50_000))
        enc = true.dist_to_encoder().seq_encode(data)
        w = np.ones(len(data))
        est = true.estimator()
        model = None
        for _ in range(40):  # EM: latent-radius E-step + closed-form M-step
            acc = est.accumulator_factory().make()
            acc.seq_update(enc, w, model)
            model = est.estimate(len(data), acc.value())
        self.assertAlmostEqual(model.mu_x, 2.0, delta=0.1)
        self.assertAlmostEqual(model.mu_y, -1.0, delta=0.1)
        # fitted log-likelihood beats the uniform (mu=0) baseline
        self.assertGreater(float(np.sum(model.seq_log_density(enc))), float(np.sum(PN(0.0, 0.0).seq_log_density(enc))))

    def test_not_a_cdf_family(self):
        self.assertFalse(mixle.supports(PN(1.0, 0.0), HasCDF))  # circular: no scalar cdf/quantile


if __name__ == "__main__":
    unittest.main()
