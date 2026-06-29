"""WS-13: mean/variance for GEV and GPD (range-of-existence guards), checked vs scipy."""

import unittest

import scipy.stats as ss

import mixle
from mixle.capability import HasMoments
from mixle.stats.univariate.continuous.generalized_extreme_value import GeneralizedExtremeValueDistribution as GEV
from mixle.stats.univariate.continuous.generalized_pareto import GeneralizedParetoDistribution as GPD


class GEVGPDMomentsTest(unittest.TestCase):
    def test_gev_matches_scipy(self):
        # mixle shape xi corresponds to scipy genextreme c = -xi
        for loc, scale, xi in [(0.0, 1.0, 0.2), (1.0, 2.0, 0.0), (-1.0, 1.5, -0.3)]:
            d, fr = GEV(loc, scale, xi), ss.genextreme(c=-xi, loc=loc, scale=scale)
            with self.subTest(xi=xi):
                self.assertAlmostEqual(d.mean(), float(fr.mean()), places=7)
                self.assertAlmostEqual(d.variance(), float(fr.var()), places=7)

    def test_gpd_matches_scipy(self):
        # mixle shape xi corresponds to scipy genpareto c = xi
        for scale, xi, loc in [(1.0, 0.3, 0.0), (2.0, 0.0, 1.0), (1.5, -0.5, 0.0)]:
            d, fr = GPD(scale, xi, loc), ss.genpareto(c=xi, loc=loc, scale=scale)
            with self.subTest(xi=xi):
                self.assertAlmostEqual(d.mean(), float(fr.mean()), places=7)
                self.assertAlmostEqual(d.variance(), float(fr.var()), places=7)

    def test_range_of_existence_guards(self):
        self.assertEqual(GEV(0.0, 1.0, 1.2).mean(), float("inf"))  # xi >= 1
        self.assertEqual(GEV(0.0, 1.0, 0.7).variance(), float("inf"))  # xi >= 1/2
        self.assertEqual(GPD(1.0, 1.5, 0.0).mean(), float("inf"))  # xi >= 1
        self.assertEqual(GPD(1.0, 0.7, 0.0).variance(), float("inf"))  # xi >= 1/2
        for d in (GEV(0.0, 1.0, 0.2), GPD(1.0, 0.3, 0.0)):
            self.assertTrue(mixle.supports(d, HasMoments))


if __name__ == "__main__":
    unittest.main()
