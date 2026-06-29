"""automatic: negative-binomial detected on overdispersed counts; Poisson not stolen on equidispersed."""

import unittest

import numpy as np

from mixle.inference.estimation import fit
from mixle.utils.automatic import get_estimator


class AutomaticNegativeBinomialTest(unittest.TestCase):
    def test_overdispersed_counts_recover_negative_binomial(self):
        rng = np.random.RandomState(0)
        for r, p in [(3.0, 0.3), (5.0, 0.5)]:  # var = mean/p > mean
            data = list(rng.negative_binomial(r, p, size=6000))
            m = fit(data, get_estimator(data), max_its=30, out=None)
            with self.subTest(r=r, p=p):
                self.assertEqual(type(m).__name__, "NegativeBinomialDistribution")

    def test_poisson_counts_not_stolen(self):
        rng = np.random.RandomState(1)
        data = list(rng.poisson(4.0, size=6000))  # var == mean: not overdispersed
        m = fit(data, get_estimator(data), max_its=30, out=None)
        self.assertNotEqual(type(m).__name__, "NegativeBinomialDistribution")


if __name__ == "__main__":
    unittest.main()
