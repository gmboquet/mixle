"""Tests for the 1-D von Mises distribution (circular density, framework parity, circular MLE)."""

import math
import unittest

import numpy as np
import scipy.stats

from mixle.stats import VonMisesDistribution, VonMisesEstimator, VonMisesFitError


def _dist():
    return VonMisesDistribution(0.7, 2.5)


class VonMisesTestCase(unittest.TestCase):
    def test_log_density_matches_scipy(self):
        dist = _dist()
        ref = scipy.stats.vonmises(dist.kappa, loc=dist.mu)
        x = dist.sampler(seed=1).sample(50)
        scalar = np.asarray([dist.log_density(v) for v in x])
        np.testing.assert_allclose(scalar, ref.logpdf(x), rtol=1.0e-10, atol=1.0e-10)
        np.testing.assert_allclose(dist.seq_log_density(dist.dist_to_encoder().seq_encode(x)), scalar)

    def test_normalizes_on_the_circle(self):
        dist = VonMisesDistribution(0.3, 1.8)
        grid = np.linspace(-np.pi, np.pi, 200001)
        integral = np.trapezoid(np.exp(dist.seq_log_density(dist.dist_to_encoder().seq_encode(grid))), grid)
        self.assertAlmostEqual(integral, 1.0, places=5)

    def test_kappa_zero_is_uniform(self):
        dist = VonMisesDistribution(0.0, 0.0)
        for x in [-3.0, -0.5, 1.0, 3.0]:
            self.assertAlmostEqual(dist.log_density(x), -math.log(2.0 * math.pi), places=12)

    def test_generated_numba_and_torch_match(self):
        import mixle.stats as s

        dist = _dist()
        x = dist.sampler(seed=2).sample(300)
        enc = dist.dist_to_encoder().seq_encode(x)
        ref = np.asarray([dist.log_density(v) for v in x])
        self.assertTrue(s.generated_numba_log_density_available(dist))
        np.testing.assert_allclose(s.generated_numba_log_density(dist, enc), ref, atol=1.0e-9)
        try:
            import torch

            from mixle.engines import TorchEngine
        except Exception as exc:  # pragma: no cover - torch optional  # noqa: BLE001
            self.skipTest("torch unavailable: %s" % exc)
        engine = TorchEngine(dtype=torch.float64)
        backend = np.asarray(engine.to_numpy(dist.backend_seq_log_density(enc, engine)))
        np.testing.assert_allclose(backend, ref, atol=1.0e-9)

    def test_string_round_trip(self):
        dist = VonMisesDistribution(0.7, 2.5, name="vm", keys="k")
        self.assertEqual(str(eval(str(dist))), str(dist))

    def test_estimator_recovers_direction_and_concentration(self):
        true = VonMisesDistribution(1.2, 4.0)
        data = true.sampler(seed=3).sample(20000)
        est = true.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(data), np.ones(len(data)), None)
        fitted = est.estimate(None, acc.value())
        self.assertAlmostEqual(fitted.mu, 1.2, delta=0.05)
        self.assertAlmostEqual(fitted.kappa, 4.0, delta=0.25)

    def test_mu_is_wrapped(self):
        dist = VonMisesDistribution(0.7 + 2.0 * math.pi, 2.5)
        self.assertAlmostEqual(dist.mu, 0.7, places=10)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            VonMisesDistribution(0.0, -1.0)
        for parameters in ((np.nan, 1.0), (0.0, np.nan), (0.0, np.inf)):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                VonMisesDistribution(*parameters)

    def test_forged_encoded_observations_are_rejected(self):
        from mixle.engines import NUMPY_ENGINE

        dist = _dist()
        for encoded in (
            (np.asarray([1.0]), np.asarray([1.0])),
            (np.asarray([1.0, 0.0]), np.asarray([0.0])),
            (np.asarray([np.nan]), np.asarray([0.0])),
        ):
            with self.subTest(encoded=encoded), self.assertRaises(ValueError):
                dist.seq_log_density(encoded)
            with self.assertRaises(ValueError):
                dist.backend_seq_log_density(encoded, NUMPY_ENGINE)

    def test_accumulator_and_estimator_reject_invalid_moments(self):
        estimator = VonMisesEstimator(name="vm")
        accumulator = estimator.accumulator_factory().make()
        self.assertEqual(accumulator.name, "vm")
        accumulator.update(0.0, 1.0, None)
        before = accumulator.value()
        for weight in (-1.0, np.nan, np.inf):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                accumulator.update(1.0, weight, None)
            self.assertEqual(accumulator.value(), before)
        with self.assertRaises(VonMisesFitError):
            estimator.estimate(None, (0.0, 0.0, 0.0))
        with self.assertRaises(VonMisesFitError):
            estimator.estimate(None, (1.0, 1.0, 0.0))
        for statistics in (
            (1.0, 2.0, 0.0),
            (1.0, np.nan, 0.0),
            (-1.0, 0.0, 0.0),
        ):
            with self.subTest(statistics=statistics), self.assertRaises(ValueError):
                estimator.estimate(None, statistics)

    def test_uniform_fit_records_non_identifiable_direction(self):
        fitted = VonMisesEstimator().estimate(None, (2.0, 0.0, 0.0))
        self.assertEqual(fitted.kappa, 0.0)
        self.assertFalse(fitted.fit_metadata["identifiable_direction"])

    def test_invalid_prior_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            VonMisesEstimator(pseudo_count=1.0)
        with self.assertRaises(ValueError):
            VonMisesEstimator(pseudo_count=-1.0, suff_stat=(0.0, 0.0))
        with self.assertRaises(ValueError):
            VonMisesEstimator(pseudo_count=1.0, suff_stat=(2.0, 0.0))


if __name__ == "__main__":
    unittest.main()
