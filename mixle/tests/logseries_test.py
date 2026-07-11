"""Tests for the logarithmic (log-series) distribution (mass, framework parity, mean-inversion MLE)."""

import unittest

import numpy as np
import scipy.stats

from mixle.stats import LogSeriesDistribution


class LogSeriesTestCase(unittest.TestCase):
    def test_log_mass_matches_scipy(self):
        dist = LogSeriesDistribution(0.6)
        ks = [1, 2, 3, 5, 10, 25]
        ref = scipy.stats.logser(0.6).logpmf(ks)
        enc = dist.dist_to_encoder().seq_encode(ks)
        np.testing.assert_allclose([dist.log_density(k) for k in ks], ref, atol=1.0e-12)
        np.testing.assert_allclose(dist.seq_log_density(enc), ref, atol=1.0e-12)

    def test_normalizes_over_positive_integers(self):
        dist = LogSeriesDistribution(0.7)
        ks = np.arange(1, 4000)
        total = float(np.sum(np.exp(dist.seq_log_density(dist.dist_to_encoder().seq_encode(ks)))))
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_generated_numba_and_torch_match(self):
        import mixle.stats as s

        dist = LogSeriesDistribution(0.55)
        ks = [int(v) for v in np.random.RandomState(3).logseries(0.55, size=300)]
        enc = dist.dist_to_encoder().seq_encode(ks)
        ref = np.asarray([dist.log_density(k) for k in ks])
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

    def test_off_support_is_neg_inf(self):
        dist = LogSeriesDistribution(0.6)
        self.assertEqual(dist.log_density(0), -np.inf)
        self.assertEqual(dist.log_density(1.5), -np.inf)
        self.assertEqual(dist.log_density(-3), -np.inf)

    def test_string_round_trip(self):
        dist = LogSeriesDistribution(0.6, name="ls", keys="k")
        self.assertEqual(str(eval(str(dist))), str(dist))

    def test_estimator_recovers_p(self):
        true = LogSeriesDistribution(0.7)
        data = true.sampler(seed=1).sample(50000)
        est = true.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(data), np.ones(len(data)), None)
        fitted = est.estimate(None, acc.value())
        self.assertAlmostEqual(fitted.p, 0.7, delta=0.02)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            LogSeriesDistribution(0.0)
        with self.assertRaises(ValueError):
            LogSeriesDistribution(1.0)


if __name__ == "__main__":
    unittest.main()
