"""Tests for the inverse-gamma distribution (density, framework parity, MLE, and conjugate-prior use)."""

import unittest

import numpy as np
import scipy.stats

from mixle.stats import InverseGammaDistribution


class InverseGammaTestCase(unittest.TestCase):
    def test_log_density_matches_scipy(self):
        dist = InverseGammaDistribution(3.0, 2.0)
        xs = [0.3, 0.7, 1.5, 4.0, 9.0]
        ref = scipy.stats.invgamma(3.0, scale=2.0).logpdf(xs)
        enc = dist.dist_to_encoder().seq_encode(xs)
        np.testing.assert_allclose([dist.log_density(x) for x in xs], ref, atol=1e-12)
        np.testing.assert_allclose(dist.seq_log_density(enc), ref, atol=1e-12)

    def test_generated_numba_and_torch_match(self):
        import mixle.stats as s

        dist = InverseGammaDistribution(4.0, 3.0)
        xs = list(1.0 / np.random.RandomState(3).gamma(4.0, 1.0 / 3.0, size=300))
        enc = dist.dist_to_encoder().seq_encode(xs)
        ref = np.asarray([dist.log_density(x) for x in xs])
        self.assertTrue(s.generated_numba_log_density_available(dist))
        np.testing.assert_allclose(s.generated_numba_log_density(dist, enc), ref, atol=1e-9)
        try:
            import torch

            from mixle.engines import TorchEngine
        except Exception as exc:  # pragma: no cover - torch optional  # noqa: BLE001
            self.skipTest("torch unavailable: %s" % exc)
        engine = TorchEngine(dtype=torch.float64)
        backend = np.asarray(engine.to_numpy(dist.backend_seq_log_density(enc, engine)))
        np.testing.assert_allclose(backend, ref, atol=1e-9)

    def test_off_support_is_neg_inf(self):
        dist = InverseGammaDistribution(3.0, 2.0)
        self.assertEqual(dist.log_density(0.0), -np.inf)
        self.assertEqual(dist.log_density(-1.0), -np.inf)

    def test_string_round_trip(self):
        dist = InverseGammaDistribution(3.0, 2.0, name="ig", keys="k")
        self.assertEqual(str(eval(str(dist))), str(dist))

    def test_estimator_recovers_parameters(self):
        true = InverseGammaDistribution(4.0, 3.0)
        data = true.sampler(seed=1).sample(40000)
        est = true.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(data), np.ones(len(data)), None)
        fitted = est.estimate(None, acc.value())
        self.assertAlmostEqual(fitted.alpha, 4.0, delta=0.2)
        self.assertAlmostEqual(fitted.beta, 3.0, delta=0.2)

    def test_prior_use_entropy_and_cross_entropy(self):
        dist = InverseGammaDistribution(3.0, 2.0)
        self.assertEqual(dist.get_parameters(), (3.0, 2.0))
        # entropy matches scipy and equals cross_entropy with itself.
        self.assertAlmostEqual(dist.entropy(), float(scipy.stats.invgamma(3.0, scale=2.0).entropy()), places=8)
        self.assertAlmostEqual(dist.cross_entropy(dist), dist.entropy(), places=10)
        # cross entropy with a different inverse-gamma exceeds its own entropy (Gibbs' inequality).
        other = InverseGammaDistribution(5.0, 4.0)
        self.assertGreater(dist.cross_entropy(other), dist.entropy() - 1e-9)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            InverseGammaDistribution(0.0, 2.0)
        with self.assertRaises(ValueError):
            InverseGammaDistribution(2.0, -1.0)


if __name__ == "__main__":
    unittest.main()
