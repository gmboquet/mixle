"""WS-4/WS-10: mixle.summarize -- capability-driven closed-form statistic summary."""

import unittest

import mixle
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.continuous.pareto import ParetoDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution


class SummarizeTest(unittest.TestCase):
    def test_gaussian_full_summary(self):
        s = mixle.summarize(GaussianDistribution(1.0, 4.0))
        self.assertAlmostEqual(s["mean"], 1.0)
        self.assertAlmostEqual(s["variance"], 4.0)
        self.assertAlmostEqual(s["std"], 2.0)
        self.assertAlmostEqual(s["skewness"], 0.0)
        self.assertAlmostEqual(s["kurtosis"], 0.0)
        self.assertAlmostEqual(
            s["entropy"], 0.5 * __import__("math").log(2 * __import__("math").pi * __import__("math").e * 4.0)
        )
        self.assertAlmostEqual(s["median"], 1.0)

    def test_discrete_summary_has_median_via_quantile(self):
        s = mixle.summarize(PoissonDistribution(4.0))
        self.assertAlmostEqual(s["mean"], 4.0)
        self.assertAlmostEqual(s["variance"], 4.0)
        self.assertIn("median", s)  # from cdf/quantile

    def test_never_raises_on_undefined_moments(self):
        # Pareto(alpha=0.5): mean/variance are infinite, not an error
        s = mixle.summarize(ParetoDistribution(1.0, 0.5))
        self.assertEqual(s["mean"], float("inf"))
        self.assertIn("median", s)


if __name__ == "__main__":
    unittest.main()
