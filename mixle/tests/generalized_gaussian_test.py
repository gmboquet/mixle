"""WS-2: GeneralizedGaussianDistribution (exponential-power family), cross-checked vs scipy."""

import unittest

import numpy as np
import scipy.stats as ss

import mixle
from mixle.capability import HasCDF, HasEntropy, HasMoments
from mixle.stats import GeneralizedGaussianDistribution as GG


class GeneralizedGaussianTest(unittest.TestCase):
    def test_density_cdf_moments_match_scipy(self):
        for mu, alpha, beta in [(1.0, 2.0, 1.5), (0.0, 1.0, 1.0), (-1.0, 0.5, 3.0)]:
            d, fr = GG(mu, alpha, beta), ss.gennorm(beta, loc=mu, scale=alpha)
            xs = np.array([mu - 2.0, mu - 0.3, mu + 0.5, mu + 2.0])
            with self.subTest(beta=beta):
                self.assertTrue(np.allclose([d.log_density(x) for x in xs], fr.logpdf(xs)))
                self.assertTrue(np.allclose([d.cdf(x) for x in xs], fr.cdf(xs)))
                self.assertTrue(np.allclose(d.seq_log_density(xs), fr.logpdf(xs)))
                self.assertAlmostEqual(d.variance(), float(fr.var()), places=7)
                self.assertAlmostEqual(d.entropy(), float(fr.entropy()), places=7)
                self.assertAlmostEqual(d.kurtosis(), float(fr.stats(moments="k")), places=6)

    def test_quantile_inverts_cdf(self):
        d = GG(1.0, 2.0, 1.5)
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            self.assertAlmostEqual(d.cdf(d.quantile(q)), q, places=7)

    def test_gaussian_and_laplace_special_cases(self):
        xs = np.array([-1.5, 0.0, 0.7, 2.3])
        self.assertTrue(np.allclose([GG(0.0, np.sqrt(2.0), 2.0).log_density(x) for x in xs], ss.norm(0, 1).logpdf(xs)))
        self.assertTrue(np.allclose([GG(0.0, 1.0, 1.0).log_density(x) for x in xs], ss.laplace(scale=1.0).logpdf(xs)))

    def test_mom_estimator_recovers_params(self):
        true = GG(1.5, 2.0, 1.2)
        data = np.asarray(true.sampler(seed=0).sample(200_000))
        est = true.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(data), np.ones(len(data)), None)
        m = est.estimate(len(data), acc.value())
        self.assertAlmostEqual(m.mu, 1.5, delta=0.05)
        self.assertAlmostEqual(m.alpha, 2.0, delta=0.1)
        self.assertAlmostEqual(m.beta, 1.2, delta=0.15)

    def test_capabilities(self):
        d = GG(1.0, 2.0, 1.5)
        for cap in (HasCDF, HasMoments, HasEntropy):
            self.assertTrue(mixle.supports(d, cap))


if __name__ == "__main__":
    unittest.main()
