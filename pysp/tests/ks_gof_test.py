"""WS-13: Kolmogorov-Smirnov goodness-of-fit test (utils.evaluation.ks_test), checked vs scipy."""

import unittest

import numpy as np
import scipy.stats as ss

from pysp.stats.univariate.continuous.gamma import GammaDistribution
from pysp.stats.univariate.continuous.gaussian import GaussianDistribution
from pysp.utils.evaluation import ks_test


class KSGoodnessOfFitTest(unittest.TestCase):
    def test_matches_scipy(self):
        for dist, frozen in [
            (GaussianDistribution(0.5, 4.0), None),
            (GammaDistribution(3.0, 0.5), None),
        ]:
            data = np.asarray(dist.sampler(seed=0).sample(2000), dtype=float)
            d, p = ks_test(data, dist)
            d_sp, p_sp = ss.kstest(data, lambda v, dd=dist: np.array([dd.cdf(vi) for vi in np.atleast_1d(v)]))
            with self.subTest(dist=type(dist).__name__):
                self.assertAlmostEqual(d, float(d_sp), places=9)
                self.assertAlmostEqual(p, float(p_sp), places=9)

    def test_well_specified_not_rejected(self):
        d = GaussianDistribution(0.5, 4.0)
        data = np.asarray(d.sampler(seed=1).sample(3000), dtype=float)
        _, p = ks_test(data, d)
        self.assertGreater(p, 0.01)  # correct model: not rejected at 1%

    def test_misspecified_rejected(self):
        data = np.asarray(GaussianDistribution(2.0, 4.0).sampler(seed=2).sample(3000), dtype=float)
        _, p = ks_test(data, GaussianDistribution(0.5, 4.0))  # wrong mean
        self.assertLess(p, 1e-6)


if __name__ == "__main__":
    unittest.main()
