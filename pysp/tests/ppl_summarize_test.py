"""Highest-density intervals and the posterior-summary table (pysp.ppl.summarize)."""

import unittest
import warnings

import numpy as np

import pysp.ppl as P
from pysp.ppl.summarize import hdi, posterior_summary
from pysp.stats.univariate.continuous.gaussian import GaussianDistribution


class HdiTest(unittest.TestCase):
    def test_symmetric_for_normal(self):
        z = np.random.RandomState(0).standard_normal(50000)
        lo, hi = hdi(z, 0.94)
        self.assertLess(abs(lo + hi), 0.1)  # symmetric about 0
        self.assertAlmostEqual(lo, -1.88, delta=0.1)

    def test_narrower_than_equal_tailed_for_skewed(self):
        e = np.random.RandomState(0).exponential(1.0, 50000)
        lo, hi = hdi(e, 0.9)
        ql, qh = np.quantile(e, [0.05, 0.95])
        self.assertLess(lo, 0.1)  # HDI starts near the mode at 0
        self.assertLess(hi - lo, qh - ql)  # and is narrower than the equal-tailed interval

    def test_rejects_bad_prob(self):
        with self.assertRaises(ValueError):
            hdi([1.0, 2.0, 3.0], prob=1.5)


class PosteriorSummaryTest(unittest.TestCase):
    def test_table_has_mean_sd_hdi(self):
        data = GaussianDistribution(1.0, 4.0).sampler(seed=0).sample(300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = P.Normal(P.Normal(0, 10), P.HalfNormal(5)).fit(data, how="mcmc", draws=400)
        ps = posterior_summary(m)
        self.assertGreaterEqual(len(ps), 2)
        for row in ps.values():
            for key in ("mean", "sd", "hdi_low", "hdi_high"):
                self.assertIn(key, row)
            self.assertLessEqual(row["hdi_low"], row["mean"])
            self.assertLessEqual(row["mean"], row["hdi_high"])


if __name__ == "__main__":
    unittest.main()
