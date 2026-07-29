"""WS-2: ProjectedNormalDistribution -- a circular law from projecting a 2-D Gaussian (directional)."""

import unittest

import numpy as np
from scipy.integrate import quad

import mixle
from mixle.capability import HasCDF
from mixle.stats import ProjectedNormalDistribution as PN
from mixle.stats.directional.projected_normal import ProjectedNormalFitError


class ProjectedNormalTest(unittest.TestCase):
    def test_density_integrates_to_one(self):
        for mu in [(0.0, 0.0), (2.0, 0.0), (1.5, -1.0), (3.0, 2.0)]:
            integral, _ = quad(lambda t, d=PN(*mu): d.density(t), -np.pi, np.pi)
            with self.subTest(mu=repr(mu)):
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

    def test_large_finite_mean_has_finite_density_and_em_radius(self):
        from mixle.engines import NUMPY_ENGINE

        for mu_x, theta in ((40.0, 0.0), (40.0, np.pi)):
            with self.subTest(mu_x=repr(mu_x), theta=repr(theta)):
                d = PN(mu_x, 0.0)
                encoded = d.dist_to_encoder().seq_encode([theta])
                scalar = d.log_density(theta)
                sequence = d.seq_log_density(encoded)[0]
                backend = d.backend_seq_log_density(encoded, NUMPY_ENGINE)[0]
                self.assertTrue(np.isfinite(scalar))
                self.assertAlmostEqual(sequence, scalar, places=10)
                self.assertAlmostEqual(backend, scalar, places=10)
                stats = d.backend_legacy_sufficient_statistics(
                    encoded,
                    {"mu_x": mu_x, "mu_y": 0.0},
                    NUMPY_ENGINE,
                )
                self.assertTrue(all(np.all(np.isfinite(value)) for value in stats))

    def test_parameters_and_angles_must_be_finite(self):
        for parameters in ((np.nan, 0.0), (0.0, np.inf), (np.inf, 0.0)):
            with self.subTest(parameters=repr(parameters)), self.assertRaises(ValueError):
                PN(*parameters)
        d = PN(1.0, 0.0)
        with self.assertRaises(ValueError):
            d.log_density(np.nan)
        with self.assertRaises(ValueError):
            d.dist_to_encoder().seq_encode([0.0, np.nan])

    def test_forged_encoded_observations_are_rejected(self):
        from mixle.engines import NUMPY_ENGINE

        d = PN(1.0, 0.0)
        for encoded in (
            (np.asarray([1.0]), np.asarray([1.0])),
            (np.asarray([1.0, 0.0]), np.asarray([0.0])),
            (np.asarray([np.nan]), np.asarray([0.0])),
        ):
            with self.subTest(encoded=repr(encoded)), self.assertRaises(ValueError):
                d.seq_log_density(encoded)
            with self.assertRaises(ValueError):
                d.backend_seq_log_density(encoded, NUMPY_ENGINE)

    def test_accumulator_and_estimator_fail_closed(self):
        d = PN(1.0, 0.0)
        estimator = d.estimator()
        accumulator = estimator.accumulator_factory().make()
        accumulator.update(0.0, 1.0, None)
        before = accumulator.value()
        for weight in (-1.0, np.nan, np.inf):
            with self.subTest(weight=repr(weight)), self.assertRaises(ValueError):
                accumulator.update(1.0, weight, None)
            self.assertEqual(accumulator.value(), before)
        with self.assertRaises(ProjectedNormalFitError):
            estimator.estimate(None, (0.0, 0.0, 0.0))
        for statistics in (
            (np.nan, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
        ):
            with self.subTest(statistics=repr(statistics)), self.assertRaises(ValueError):
                estimator.estimate(None, statistics)
        with self.assertRaises(ValueError):
            d.estimator(pseudo_count=1.0)

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
