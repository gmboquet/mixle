"""Tests for the Probabilistic PCA latent-factor model (Woodbury scoring, transform, closed-form MLE)."""

import unittest

import numpy as np
import scipy.stats

from mixle.inference.estimation import fit
from mixle.stats import ProbabilisticPCADistribution


def _dist(seed=0):
    rng = np.random.RandomState(seed)
    w = rng.randn(5, 2) * 0.8
    mu = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
    return ProbabilisticPCADistribution(w, mu, 0.4)


class ProbabilisticPCATestCase(unittest.TestCase):
    def test_matches_explicit_mvn(self):
        dist = _dist()
        cov = dist.w @ dist.w.T + dist.sigma2 * np.eye(dist.dim)
        ref = scipy.stats.multivariate_normal(mean=dist.mu, cov=cov)
        x = dist.sampler(seed=1).sample(8)
        enc = dist.dist_to_encoder().seq_encode(x)
        np.testing.assert_allclose([dist.log_density(v) for v in x], ref.logpdf(x), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(dist.seq_log_density(enc), ref.logpdf(x), rtol=1e-10, atol=1e-10)

    def test_woodbury_inverse_and_logdet_are_exact(self):
        dist = _dist()
        cov = dist.w @ dist.w.T + dist.sigma2 * np.eye(dist.dim)
        np.testing.assert_allclose(dist.inv_covar, np.linalg.inv(cov), atol=1e-10)
        self.assertAlmostEqual(dist.log_det, float(np.linalg.slogdet(cov)[1]), places=9)

    def test_torch_backend_matches(self):
        try:
            import torch

            from mixle.engines import TorchEngine
        except Exception as exc:  # pragma: no cover - torch optional  # noqa: BLE001
            self.skipTest("torch unavailable: %s" % exc)
        dist = _dist()
        enc = dist.dist_to_encoder().seq_encode(dist.sampler(seed=2).sample(8))
        engine = TorchEngine(dtype=torch.float64)
        backend = np.asarray(engine.to_numpy(dist.backend_seq_log_density(enc, engine)))
        np.testing.assert_allclose(backend, dist.seq_log_density(enc), rtol=1e-9, atol=1e-9)

    def test_transform_returns_latent_embedding(self):
        dist = _dist()
        x = dist.sampler(seed=3).sample(7)
        z = dist.transform(x)
        self.assertEqual(z.shape, (dist.latent_dim, 7))

    def test_string_round_trip(self):
        dist = ProbabilisticPCADistribution(
            [[1.0, 0.2], [0.3, 0.8], [0.5, 0.1]], [0.0, 1.0, -1.0], 0.5, name="p", keys="k"
        )
        self.assertEqual(str(eval(str(dist))), str(dist))

    def test_closed_form_mle_recovers_covariance(self):
        true = _dist()
        cov = true.w @ true.w.T + true.sigma2 * np.eye(true.dim)
        data = true.sampler(seed=4).sample(20000)
        fitted = fit(data, true.estimator(), max_its=1, rng=np.random.RandomState(0), print_iter=0)
        cov_fit = fitted.w @ fitted.w.T + fitted.sigma2 * np.eye(fitted.dim)
        np.testing.assert_allclose(cov_fit, cov, atol=0.1)
        self.assertAlmostEqual(fitted.sigma2, true.sigma2, delta=0.05)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            ProbabilisticPCADistribution([[1.0, 0.0]], [0.0, 0.0], 0.5)  # W rows != len(mu)
        with self.assertRaises(ValueError):
            ProbabilisticPCADistribution([[1.0], [0.5]], [0.0, 0.0], 0.0)  # sigma2 must be > 0


if __name__ == "__main__":
    unittest.main()
