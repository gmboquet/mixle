"""Tests for the Gumbel (extreme value type I) distribution (density, framework parity, moment MLE)."""

import unittest

import numpy as np
import scipy.stats

from mixle.stats import GumbelDistribution


class GumbelTestCase(unittest.TestCase):
    def test_log_density_matches_scipy(self):
        dist = GumbelDistribution(2.0, 1.5)
        xs = [-3.0, -0.5, 2.0, 4.5, 9.0]
        ref = scipy.stats.gumbel_r(loc=2.0, scale=1.5).logpdf(xs)
        enc = dist.dist_to_encoder().seq_encode(xs)
        np.testing.assert_allclose([dist.log_density(x) for x in xs], ref, atol=1e-12)
        np.testing.assert_allclose(dist.seq_log_density(enc), ref, atol=1e-12)

    def test_generated_numba_and_torch_match(self):
        import mixle.stats as s

        dist = GumbelDistribution(-1.0, 2.0)
        xs = list(np.random.RandomState(7).gumbel(-1.0, 2.0, size=300))
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

    def test_string_round_trip(self):
        dist = GumbelDistribution(loc=2.0, scale=1.5, name="gum", keys="k")
        self.assertEqual(str(eval(str(dist))), str(dist))

    def test_estimator_recovers_parameters(self):
        true = GumbelDistribution(1.5, 2.0)
        data = true.sampler(seed=1).sample(40000)
        est = true.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(data), np.ones(len(data)), None)
        fitted = est.estimate(None, acc.value())
        self.assertAlmostEqual(fitted.loc, 1.5, delta=0.05)
        self.assertAlmostEqual(fitted.scale, 2.0, delta=0.05)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            GumbelDistribution(0.0, 0.0)
        with self.assertRaises(ValueError):
            GumbelDistribution(0.0, -1.0)


if __name__ == "__main__":
    unittest.main()
