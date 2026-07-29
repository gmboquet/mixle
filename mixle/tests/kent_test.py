"""WS-2: Kent (FB5) distribution on S^2; normalizer verified exactly via mpmath sphere integration."""

import unittest

import numpy as np

import mixle
from mixle.capability import Fittable
from mixle.stats import KentDistribution as Kent
from mixle.stats.directional.kent import KentFitError, _log_kent_norm


class KentTest(unittest.TestCase):
    def test_normalizer_integrates_to_one_mpmath(self):
        import mpmath as mp

        # dps=10 keeps a comfortable margin below the assertion tolerance (1e-9): the true
        # numerical error at this precision is ~4.4e-11 (verified against a correct formula,
        # ~23x margin), and an injected 0.1% normalization bug is still caught cleanly at this
        # precision. This cuts the double-nested sphere quadrature's runtime by roughly 3x.
        mp.mp.dps = 10
        g = np.eye(3)
        for kappa, beta in [(5.0, 1.0), (10.0, 3.0), (2.0, 0.5)]:
            d = Kent(g, kappa, beta)
            integ = mp.e ** mp.mpf(-d._log_c) * mp.quad(
                lambda th, k=kappa, b=beta: mp.quad(
                    lambda ph: (
                        mp.e ** (k * mp.cos(th) + b * ((mp.sin(th) * mp.cos(ph)) ** 2 - (mp.sin(th) * mp.sin(ph)) ** 2))
                        * mp.sin(th)
                    ),
                    [0, 2 * mp.pi],
                ),
                [0, mp.pi],
            )
            with self.subTest(kappa=repr(kappa), beta=repr(beta)):
                self.assertTrue(mp.almosteq(integ, 1, 1e-9))

    def test_moment_identities(self):
        # exponential-family identity E_f[T(x)] == grad log c -- jointly validates the normalizer AND sampler
        d = Kent(np.eye(3), 8.0, 2.0)
        xs = np.array(d.sampler(seed=1).sample(12000))
        dk = (_log_kent_norm(8.001, 2.0) - _log_kent_norm(7.999, 2.0)) / 0.002
        db = (_log_kent_norm(8.0, 2.001) - _log_kent_norm(8.0, 1.999)) / 0.002
        self.assertAlmostEqual(xs[:, 0].mean(), dk, delta=0.01)
        self.assertAlmostEqual((xs[:, 1] ** 2 - xs[:, 2] ** 2).mean(), db, delta=0.01)

    def test_sampler_on_sphere(self):
        xs = np.array(Kent(np.eye(3), 10.0, 3.0).sampler(seed=0).sample(2000))
        self.assertTrue(np.allclose(np.linalg.norm(xs, axis=1), 1.0))

    def test_seq_matches_scalar(self):
        d = Kent(np.eye(3), 8.0, 2.0)
        xs = d.sampler(seed=0).sample(6)
        scalar = np.array([d.log_density(x) for x in xs])
        self.assertTrue(np.allclose(scalar, d.seq_log_density(d.dist_to_encoder().seq_encode(xs))))

    def test_estimator_recovers_params(self):
        rng = np.random.RandomState(3)
        g, _ = np.linalg.qr(rng.randn(3, 3))
        for kappa, beta in [(15.0, 4.0), (8.0, 1.0)]:
            true = Kent(g, kappa, beta)
            data = np.array(true.sampler(seed=1).sample(12000))
            est = true.estimator()
            acc = est.accumulator_factory().make()
            acc.seq_update(true.dist_to_encoder().seq_encode(data), np.ones(len(data)), None)
            m = est.estimate(len(data), acc.value())
            with self.subTest(kappa=repr(kappa), beta=repr(beta)):
                self.assertAlmostEqual(m.kappa, kappa, delta=1.0)
                self.assertAlmostEqual(m.beta, beta, delta=0.6)
                self.assertAlmostEqual(abs(m.gamma[:, 0] @ g[:, 0]), 1.0, delta=1e-2)

    def test_vmf_limit_and_constraints(self):
        self.assertAlmostEqual(
            Kent(np.eye(3), 5.0, 0.0).density(np.array([1.0, 0.0, 0.0])),
            np.exp(-Kent(np.eye(3), 5.0, 0.0)._log_c + 5.0),
            places=6,
        )
        with self.assertRaises(ValueError):
            Kent(np.eye(3), 4.0, 2.0)  # 2*beta >= kappa

    def test_non_finite_beta_raises(self):
        # `nan < 0.0` and `2.0 * nan >= kappa` are both False (every NaN comparison is False), so a NaN
        # beta used to pass construction validation silently and then made KentSampler._batch's
        # rejection loop spin forever: an all-NaN, therefore always-False, accept mask every iteration,
        # with nothing that would ever change between iterations.
        for bad_beta in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(beta=repr(bad_beta)):
                with self.assertRaises(ValueError):
                    Kent(np.eye(3), 5.0, bad_beta)

    def test_orientation_is_right_handed_copied_and_read_only(self):
        orientation = np.eye(3)
        d = Kent(orientation, 5.0, 1.0)
        orientation[0, 0] = 0.0
        np.testing.assert_array_equal(d.gamma, np.eye(3))
        with self.assertRaises(ValueError):
            d.gamma[0, 0] = 0.0
        reflection = np.eye(3)
        reflection[:, 2] *= -1.0
        with self.assertRaises(ValueError):
            Kent(reflection, 5.0, 1.0)
        with self.assertRaises(ValueError):
            Kent(np.ones((3, 3)), 5.0, 1.0)

    def test_scoring_paths_reject_off_sphere_observations(self):
        from mixle.engines import NUMPY_ENGINE

        d = Kent(np.eye(3), 5.0, 1.0)
        observations = np.asarray([[2.0, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            d.log_density(observations[0])
        with self.assertRaises(ValueError):
            d.seq_log_density(observations)
        with self.assertRaises(ValueError):
            d.dist_to_encoder().seq_encode(observations)
        with self.assertRaises(ValueError):
            d.backend_seq_log_density(observations, NUMPY_ENGINE)

    def test_accumulator_and_estimator_reject_invalid_statistics(self):
        estimator = Kent(np.eye(3), 5.0, 1.0).estimator()
        accumulator = estimator.accumulator_factory().make()
        accumulator.update([1.0, 0.0, 0.0], 1.0, None)
        before = accumulator.value()
        for weight in (-1.0, np.nan, np.inf):
            with self.subTest(weight=repr(weight)), self.assertRaises(ValueError):
                accumulator.update([0.0, 1.0, 0.0], weight, None)
            actual = accumulator.value()
            self.assertEqual(actual[0], before[0])
            np.testing.assert_array_equal(actual[1], before[1])
            np.testing.assert_array_equal(actual[2], before[2])
        with self.assertRaises(KentFitError):
            estimator.estimate(None, (0.0, np.zeros(3), np.zeros((3, 3))))
        for statistics in (
            (1.0, np.asarray([2.0, 0.0, 0.0]), np.eye(3) / 3.0),
            (1.0, np.zeros(3), np.eye(3)),
            (1.0, np.zeros(3), np.diag([2.0, -1.0, 0.0])),
        ):
            with self.subTest(statistics=repr(statistics)), self.assertRaises(ValueError):
                estimator.estimate(None, statistics)

    def test_capabilities(self):
        self.assertTrue(mixle.supports(Kent(np.eye(3), 5.0, 1.0), Fittable))


if __name__ == "__main__":
    unittest.main()
