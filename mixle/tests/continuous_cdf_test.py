"""CDF / quantile (inverse-CDF) coverage for univariate continuous leaf distributions.

For a continuous family the four enumeration-suite capabilities are realized through the CDF and its
inverse: ``cdf(x)`` is the cumulative probability (the continuous 'index of' a value), ``quantile(q)``
returns the value at cumulative-probability index ``q`` (the continuous 'arbitrary-index' / unranking),
and a quantile grid enumerates the support in order. Each family is checked for: range, monotonicity,
quantile/cdf round-trip, and -- the key correctness tie-in -- that d/dx CDF matches the family's own
``exp(log_density)``.
"""

import math
import unittest

from mixle.stats.univariate.continuous.beta import BetaDistribution
from mixle.stats.univariate.continuous.exponential import ExponentialDistribution
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.continuous.laplace import LaplaceDistribution
from mixle.stats.univariate.continuous.log_gaussian import LogGaussianDistribution
from mixle.stats.univariate.continuous.logistic import LogisticDistribution
from mixle.stats.univariate.continuous.pareto import ParetoDistribution
from mixle.stats.univariate.continuous.rayleigh import RayleighDistribution
from mixle.stats.univariate.continuous.student_t import StudentTDistribution
from mixle.stats.univariate.continuous.uniform import UniformDistribution
from mixle.stats.univariate.continuous.weibull import WeibullDistribution

CASES = [
    ("gaussian", GaussianDistribution(0.7, 2.0), [-1.0, 0.5, 2.0]),
    ("gamma", GammaDistribution(2.5, 1.7), [0.5, 2.0, 5.0]),
    ("exponential", ExponentialDistribution(1.7), [0.2, 1.0, 3.0]),
    ("beta", BetaDistribution(2.0, 3.0), [0.1, 0.5, 0.9]),
    ("laplace", LaplaceDistribution(0.3, 1.2), [-1.0, 0.3, 2.0]),
    ("log_gaussian", LogGaussianDistribution(0.2, 0.5), [0.5, 1.0, 3.0]),
    ("logistic", LogisticDistribution(0.4, 1.3), [-1.0, 0.4, 2.0]),
    ("pareto", ParetoDistribution(1.5, 3.0), [1.6, 2.5, 5.0]),
    ("rayleigh", RayleighDistribution(1.4), [0.3, 1.4, 3.0]),
    ("student_t", StudentTDistribution(5.0, 0.3, 1.2), [-1.0, 0.3, 2.0]),
    ("uniform", UniformDistribution(-1.0, 2.0), [-0.5, 0.0, 1.5]),
    ("weibull", WeibullDistribution(1.8, 2.2), [0.5, 2.0, 4.0]),
]


class ContinuousCDFTestCase(unittest.TestCase):
    def test_cdf_in_unit_interval_and_monotone(self):
        for name, dist, xs in CASES:
            vals = [dist.cdf(x) for x in xs]
            for v in vals:
                self.assertTrue(0.0 <= v <= 1.0, "%s: cdf out of [0,1]" % name)
            for i in range(len(vals) - 1):
                self.assertLessEqual(vals[i], vals[i + 1] + 1e-12, "%s: cdf not monotone" % name)

    def test_quantile_inverts_cdf(self):
        for name, dist, xs in CASES:
            for x in xs:
                self.assertAlmostEqual(dist.quantile(dist.cdf(x)), x, delta=1e-5, msg="%s: round-trip" % name)
            for q in (0.05, 0.5, 0.95):
                self.assertAlmostEqual(dist.cdf(dist.quantile(q)), q, delta=1e-6, msg="%s: cdf(quantile)" % name)

    def test_cdf_derivative_matches_density(self):
        # The strongest check: the CDF's slope is the family's own density, tying cdf/quantile to
        # log_density (so the scipy parameterization provably matches each distribution).
        h = 1e-5
        for name, dist, xs in CASES:
            for x in xs:
                fd = (dist.cdf(x + h) - dist.cdf(x - h)) / (2.0 * h)
                self.assertAlmostEqual(fd, math.exp(dist.log_density(x)), delta=1e-3, msg="%s at %s" % (name, x))


class MultivariateCumulativeTestCase(unittest.TestCase):
    """Multivariate Gaussians have no coordinate-wise CDF (no total order on R^d), but the
    probability-ordered cumulative G(x)=P(p(Y)>=p(x)) -- the highest-density-region mass through x --
    is exact via the chi-square of the Mahalanobis distance, and density_rank uses it."""

    def test_mvn_exact_cumulative_matches_sampling(self):
        import numpy as np

        from mixle.enumeration.density_rank import density_rank
        from mixle.stats.multivariate.diagonal_gaussian import DiagonalGaussianDistribution
        from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution

        mvn = MultivariateGaussianDistribution(np.array([0.5, -1.0]), np.array([[2.0, 0.3], [0.3, 1.0]]))
        dmvn = DiagonalGaussianDistribution(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 0.5]))
        for dist, x in [(mvn, np.array([1.0, 0.0])), (dmvn, np.array([0.5, 1.5, 2.0]))]:
            r = density_rank(dist, x, n_samples=1)
            self.assertEqual(r.method, "exact-analytic")
            # large-sample Monte-Carlo reference
            t = float(dist.log_density(x))
            ys = dist.sampler(0).sample(100000)
            samp = sum(1 for y in ys if float(dist.log_density(y)) >= t - 1e-9) / 100000
            self.assertAlmostEqual(r.cumulative_probability, samp, delta=0.02)
            self.assertTrue(0.0 <= r.cumulative_probability <= 1.0)
        # the mode has nothing strictly more probable -> G == 0
        self.assertAlmostEqual(density_rank(mvn, mvn.mu).cumulative_probability, 0.0, delta=1e-9)

    def test_vmf_exact_cumulative(self):
        import numpy as np

        from mixle.enumeration.density_rank import density_rank
        from mixle.stats.directional.von_mises_fisher import VonMisesFisherDistribution

        # d=3: closed form G = (e^k - e^{k t}) / (e^k - e^{-k}), t = mu . x
        mu = np.array([1.0, 0.0, 0.0])
        k = 4.0
        d3 = VonMisesFisherDistribution(mu, k)
        for x in (mu, np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0])):
            xx = x / np.linalg.norm(x)
            t = float(mu @ xx)
            closed = (math.exp(k) - math.exp(k * t)) / (math.exp(k) - math.exp(-k))
            self.assertAlmostEqual(d3.density_cumulative(xx), closed, places=6)
        # mode -> 0; density_rank routes through the exact cumulative
        self.assertAlmostEqual(d3.density_cumulative(mu), 0.0, delta=1e-9)
        self.assertEqual(density_rank(d3, mu, n_samples=1).method, "exact-analytic")

        # higher dimension (no elementary closed form): quadrature matches Monte-Carlo
        d5 = VonMisesFisherDistribution(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 3.0)
        x5 = d5.sampler(0).sample()
        t = float(d5.log_density(x5))
        ys = d5.sampler(1).sample(60000)
        samp = sum(1 for y in ys if float(d5.log_density(y)) >= t - 1e-9) / 60000
        self.assertAlmostEqual(d5.density_cumulative(x5), samp, delta=0.02)

    def test_density_quantile_inverts_cumulative(self):
        # density_quantile is the multivariate arbitrary-index (inverse-CDF in the density ordering):
        # density_cumulative(density_quantile(q)) == q, and density falls monotonically as q grows
        # (sweeping q enumerates the support in descending density).
        import numpy as np

        from mixle.stats.directional.von_mises_fisher import VonMisesFisherDistribution
        from mixle.stats.multivariate.diagonal_gaussian import DiagonalGaussianDistribution
        from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution

        dists = [
            MultivariateGaussianDistribution(np.array([0.5, -1.0]), np.array([[2.0, 0.3], [0.3, 1.0]])),
            DiagonalGaussianDistribution(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 0.5])),
            VonMisesFisherDistribution(np.array([1.0, 0.0, 0.0]), 3.0),
        ]
        for d in dists:
            for q in (0.1, 0.3, 0.5, 0.7, 0.9):
                self.assertAlmostEqual(d.density_cumulative(d.density_quantile(q)), q, delta=1e-6)
            lps = [d.log_density(d.density_quantile(q)) for q in (0.1, 0.5, 0.9)]
            self.assertTrue(all(lps[i] >= lps[i + 1] - 1e-9 for i in range(len(lps) - 1)))
            with self.assertRaises(ValueError):
                d.density_quantile(1.5)


if __name__ == "__main__":
    unittest.main()
