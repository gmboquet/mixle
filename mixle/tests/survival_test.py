"""Survival model: censored likelihood (f / S) and the right-censored MLE via imputation EM."""

import math
import unittest

import numpy as np

from mixle.inference import estimate
from mixle.stats import SurvivalDistribution, WeibullDistribution


class SurvivalDistributionTest(unittest.TestCase):
    def setUp(self):
        self.d = SurvivalDistribution(WeibullDistribution(1.5, 2.0))

    def test_event_uses_density_censored_uses_survival(self):
        self.assertAlmostEqual(self.d.log_density((1.3, 1)), self.d.base.log_density(1.3))
        self.assertAlmostEqual(self.d.log_density((1.3, 0)), math.log(1.0 - self.d.base.cdf(1.3)))

    def test_seq_matches_scalar(self):
        data = [(0.5, 1), (1.2, 0), (2.0, 1), (0.8, 0)]
        enc = self.d.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(self.d.seq_log_density(enc), [self.d.log_density(x) for x in data], atol=1e-12)

    def test_requires_cdf_and_quantile(self):
        from mixle.stats import VonMisesDistribution  # circular -- no scalar cdf/quantile

        with self.assertRaises(ValueError):
            SurvivalDistribution(VonMisesDistribution(0.0, 2.0))

    def test_censored_mle_recovers_true_params(self):
        rng = np.random.RandomState(0)
        t = 2.0 * rng.weibull(1.5, size=3000)
        c = 2.5 * rng.weibull(1.2, size=3000)
        obs, event = np.minimum(t, c), (t <= c).astype(int)
        data = list(zip(obs.tolist(), event.tolist()))
        self.assertGreater(1.0 - event.mean(), 0.3)  # substantial censoring

        prev = None
        # 7 rounds is plenty: the imputation EM (driven by the standard estimate() driver) is
        # dominated by per-censored-point quantile calls, and empirically plateaus by ~iteration 5-6
        # (verified via convergence traces across several seeds) -- extra rounds beyond that just
        # burn time without moving the fitted params.
        for _ in range(7):
            prev = estimate(data, self.d.estimator(), prev)
        self.assertAlmostEqual(prev.base.shape, 1.5, delta=0.12)
        self.assertAlmostEqual(prev.base.scale, 2.0, delta=0.12)

        # ignoring censoring (fitting the observed times as if all were events) is badly biased low
        naive = estimate(list(obs), WeibullDistribution(1, 1).estimator())
        self.assertLess(naive.scale, 1.6)


if __name__ == "__main__":
    unittest.main()
