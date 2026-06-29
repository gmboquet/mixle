"""WS-13: Pearson chi-square goodness-of-fit for discrete distributions (utils.evaluation)."""

import unittest

import numpy as np
import scipy.stats as ss

from mixle.stats.univariate.discrete.binomial import BinomialDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution
from mixle.utils.evaluation import chi_square_test


class ChiSquareGOFTest(unittest.TestCase):
    def test_statistic_matches_scipy(self):
        d = PoissonDistribution(4.0)
        data = np.asarray(d.sampler(seed=0).sample(5000), dtype=int)
        chi2, dof, _ = chi_square_test(data, d, lo=0, hi=12)
        # rebuild the same cells and compare the statistic to scipy.chisquare
        ks = list(range(0, 13))
        O = np.array([np.sum(data == k) for k in ks] + [np.sum(data > 12)], dtype=float)
        p = np.array([np.exp(d.log_density(k)) for k in ks] + [1.0 - d.cdf(12)], dtype=float)
        c_sp, _ = ss.chisquare(O, O.sum() * p)
        self.assertAlmostEqual(chi2, float(c_sp), places=8)
        self.assertEqual(dof, len(ks))  # (13 + tail) - 1

    def test_well_specified_not_rejected(self):
        d = PoissonDistribution(4.0)
        data = np.asarray(d.sampler(seed=1).sample(8000), dtype=int)
        _, _, p = chi_square_test(data, d)
        self.assertGreater(p, 0.01)

    def test_misspecified_rejected(self):
        data = np.asarray(BinomialDistribution(0.6, 10).sampler(seed=2).sample(8000), dtype=int)
        _, _, p = chi_square_test(data, PoissonDistribution(6.0))  # wrong family
        self.assertLess(p, 1e-6)


if __name__ == "__main__":
    unittest.main()
