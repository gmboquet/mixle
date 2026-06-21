"""Regression tests for negative-binomial dispersion (r) recovery.

The estimator must solve for the dispersion ``r`` (no closed form: a 1-D solve of the
digamma score equation), not collapse to the mean-matching geometric ``r=1``. These
tests fit NegBin to NegBin-sampled data and check the dispersion / variance, plus the
zero-inflated (ZINB) delegation through the ZeroInflated combinator.
"""

import unittest

import numpy as np

from pysp.stats import NegativeBinomialDistribution, NegativeBinomialEstimator
from pysp.stats.combinator.zero_inflated import ZeroInflatedDistribution, ZeroInflatedEstimator
from pysp.utils.estimation import optimize


class NegativeBinomialEstimatorTestCase(unittest.TestCase):
    def test_recovers_dispersion_and_variance(self):
        # NegBin(r=5, p=0.5): mean 5, variance 10. A mean-matching geometric (r=1) would
        # nail the mean but report variance ~30, so checking the variance/r is the real test.
        true_r, true_p = 5.0, 0.5
        data = NegativeBinomialDistribution(true_r, true_p).sampler(seed=2).sample(8000)

        fit = optimize(
            list(data),
            NegativeBinomialEstimator(),
            max_its=80,
            rng=np.random.RandomState(0),
            print_iter=0,
        )

        self.assertAlmostEqual(fit.r, true_r, delta=0.1 * true_r)  # within ~10%
        self.assertAlmostEqual(fit.p, true_p, delta=0.05)

        fit_mean = fit.r * (1.0 - fit.p) / fit.p
        fit_var = fit.r * (1.0 - fit.p) / (fit.p * fit.p)
        self.assertAlmostEqual(fit_mean, float(np.mean(data)), delta=0.05)
        # The mean is easy; the point of the fix is that the variance is right too.
        self.assertAlmostEqual(fit_var, float(np.var(data)), delta=0.1 * fit_var)

    def test_single_estimate_call_recovers_r(self):
        data = NegativeBinomialDistribution(4.0, 0.6).sampler(seed=7).sample(6000)
        acc = NegativeBinomialEstimator().accumulator_factory().make()
        enc = NegativeBinomialDistribution(1.0, 0.5).dist_to_encoder().seq_encode(list(data))
        acc.seq_update(enc, np.ones(len(data)), None)
        fit = NegativeBinomialEstimator().estimate(None, acc.value())
        self.assertAlmostEqual(fit.r, 4.0, delta=0.4)

    def test_estimate_r_false_holds_r_fixed(self):
        data = NegativeBinomialDistribution(5.0, 0.5).sampler(seed=2).sample(4000)
        fit = optimize(
            list(data),
            NegativeBinomialEstimator(r=2.0, estimate_r=False),
            max_its=20,
            rng=np.random.RandomState(0),
            print_iter=0,
        )
        self.assertEqual(fit.r, 2.0)
        # p still matches the mean for the fixed r.
        self.assertAlmostEqual(fit.r * (1.0 - fit.p) / fit.p, float(np.mean(data)), delta=0.05)

    def test_under_dispersed_data_does_not_crash(self):
        # Poisson data is equi-dispersed; the MLE for r runs to the Poisson limit. The
        # solver must cap r (and not raise) rather than oscillate.
        pois = np.random.RandomState(1).poisson(3.0, 4000)
        fit = optimize(
            list(pois),
            NegativeBinomialEstimator(),
            max_its=20,
            rng=np.random.RandomState(0),
            print_iter=0,
        )
        self.assertTrue(np.isfinite(fit.r))
        self.assertGreater(fit.r, 1.0)
        self.assertAlmostEqual(fit.r * (1.0 - fit.p) / fit.p, float(np.mean(pois)), delta=0.1)

    def test_all_zero_data_is_stable(self):
        fit = NegativeBinomialEstimator().estimate(None, (10.0, 0.0, {0: 10.0}))
        self.assertTrue(np.isfinite(fit.r))
        self.assertGreater(fit.p, 0.0)
        self.assertLess(fit.p, 1.0)

    def test_zinb_delegation_recovers_r(self):
        # The ZeroInflated combinator delegates the base M-step to NegativeBinomialEstimator,
        # so ZINB fits must recover the base dispersion too.
        base = NegativeBinomialDistribution(5.0, 0.5)
        zd = ZeroInflatedDistribution(base, pi=0.3)
        data = zd.sampler(seed=3).sample(8000)

        zfit = optimize(
            list(data),
            ZeroInflatedEstimator(NegativeBinomialEstimator()),
            max_its=80,
            rng=np.random.RandomState(0),
            print_iter=0,
        )

        self.assertAlmostEqual(zfit.pi, 0.3, delta=0.05)
        self.assertAlmostEqual(zfit.base.r, 5.0, delta=0.1 * 5.0)
        self.assertAlmostEqual(zfit.base.p, 0.5, delta=0.05)


if __name__ == "__main__":
    unittest.main()
