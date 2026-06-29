"""WS-13: mean/variance/cdf for Skellam and LogSeries, cross-checked against scipy."""

import unittest

import scipy.stats as ss

import mixle
from mixle.capability import HasCDF, HasMoments
from mixle.stats.univariate.discrete.logseries import LogSeriesDistribution
from mixle.stats.univariate.discrete.skellam import SkellamDistribution


class SkellamLogSeriesMethodsTest(unittest.TestCase):
    def test_skellam_matches_scipy(self):
        d, fr = SkellamDistribution(3.0, 1.5), ss.skellam(3.0, 1.5)
        self.assertAlmostEqual(d.mean(), float(fr.mean()), places=9)
        self.assertAlmostEqual(d.variance(), float(fr.var()), places=9)
        for k in (-3, 0, 1, 4):
            self.assertAlmostEqual(d.cdf(k), float(fr.cdf(k)), places=9)

    def test_logseries_matches_scipy(self):
        d, fr = LogSeriesDistribution(0.6), ss.logser(0.6)
        self.assertAlmostEqual(d.mean(), float(fr.mean()), places=9)
        self.assertAlmostEqual(d.variance(), float(fr.var()), places=9)
        for k in (1, 2, 5, 10):
            self.assertAlmostEqual(d.cdf(k), float(fr.cdf(k)), places=9)
        self.assertEqual(d.cdf(0), 0.0)

    def test_capabilities(self):
        for d in (SkellamDistribution(3.0, 1.5), LogSeriesDistribution(0.6)):
            self.assertTrue(mixle.supports(d, HasMoments))
            self.assertTrue(mixle.supports(d, HasCDF))


if __name__ == "__main__":
    unittest.main()
