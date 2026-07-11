"""Tests for the multivariate Student's t distribution (density, sampling, EM, engine parity)."""

import unittest

import numpy as np
import scipy.stats

from mixle.inference.estimation import fit
from mixle.stats import MultivariateStudentTDistribution


def _dist():
    return MultivariateStudentTDistribution(6.0, [1.0, -2.0], [[2.0, 0.5], [0.5, 1.0]])


class MultivariateStudentTTestCase(unittest.TestCase):
    def test_log_density_matches_scipy(self):
        dist = _dist()
        ref = scipy.stats.multivariate_t(loc=dist.mu, shape=dist.shape, df=dist.dof)
        x = dist.sampler(seed=1).sample(40)
        scalar = np.asarray([dist.log_density(v) for v in x])
        np.testing.assert_allclose(scalar, ref.logpdf(x), rtol=1.0e-10, atol=1.0e-10)
        np.testing.assert_allclose(dist.seq_log_density(dist.dist_to_encoder().seq_encode(x)), scalar)

    def test_string_round_trip(self):
        dist = MultivariateStudentTDistribution(6.0, [1.0, -2.0], [[2.0, 0.5], [0.5, 1.0]], name="t", keys="k")
        self.assertEqual(str(eval(str(dist))), str(dist))

    def test_torch_backend_matches_scalar(self):
        try:
            import torch

            from mixle.engines import TorchEngine
        except Exception as exc:  # pragma: no cover - torch optional  # noqa: BLE001
            self.skipTest("torch unavailable: %s" % exc)
        dist = _dist()
        x = dist.sampler(seed=2).sample(50)
        enc = dist.dist_to_encoder().seq_encode(x)
        engine = TorchEngine(dtype=torch.float64)
        backend = np.asarray(engine.to_numpy(dist.backend_seq_log_density(enc, engine)))
        np.testing.assert_allclose(backend, dist.seq_log_density(enc), rtol=1.0e-9, atol=1.0e-9)

    def test_stacked_log_density_matches_components(self):
        from mixle.engines import NUMPY_ENGINE

        d0 = _dist()
        d1 = MultivariateStudentTDistribution(10.0, [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        x = d0.sampler(seed=3).sample(30)
        enc = d0.dist_to_encoder().seq_encode(x)
        params = MultivariateStudentTDistribution.backend_stacked_params([d0, d1], NUMPY_ENGINE)
        stacked = np.asarray(MultivariateStudentTDistribution.backend_stacked_log_density(enc, params, NUMPY_ENGINE))
        expected = np.stack([d0.seq_log_density(enc), d1.seq_log_density(enc)], axis=1)
        np.testing.assert_allclose(stacked, expected, rtol=1.0e-10, atol=1.0e-10)

    def test_em_recovers_location_and_shape(self):
        true = MultivariateStudentTDistribution(5.0, [3.0, -1.0], [[1.5, -0.4], [-0.4, 0.8]])
        data = true.sampler(seed=4).sample(8000)
        fitted = fit(data, true.estimator(), max_its=80, rng=np.random.RandomState(0), print_iter=0)
        np.testing.assert_allclose(fitted.mu, true.mu, atol=0.1)
        np.testing.assert_allclose(fitted.shape, true.shape, atol=0.15)
        self.assertEqual(fitted.dof, 5.0)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            MultivariateStudentTDistribution(0.0, [0.0], [[1.0]])
        with self.assertRaises(ValueError):
            MultivariateStudentTDistribution(3.0, [0.0, 0.0], [[1.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
