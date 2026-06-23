"""Tests for the logistic candidate in the automatic model selector.

Real-valued, symmetric, heavier-tailed-than-Gaussian data should be recommended (and built) as a
LogisticDistribution. The selector must NOT steal clearly-Gaussian or clearly-exponential data.
"""

import unittest

import numpy as np

from pysp.inference.estimation import fit
from pysp.stats import LogisticDistribution
from pysp.utils.automatic import get_estimator


class AutomaticLogisticTest(unittest.TestCase):
    def test_recovery_logistic_data_builds_logistic(self):
        for loc, scale in [(0.0, 1.0), (5.0, 2.5)]:
            rng = np.random.RandomState(0)
            data = list(rng.logistic(loc=loc, scale=scale, size=5000))
            model = fit(data, get_estimator(data), max_its=25, out=None)
            self.assertIsInstance(model, LogisticDistribution, msg=f"logistic(loc={loc}, scale={scale}) not recovered")

    def test_no_steal_gaussian(self):
        rng = np.random.RandomState(1)
        data = list(rng.normal(0.0, 1.0, size=5000))
        model = fit(data, get_estimator(data), max_its=25, out=None)
        self.assertNotIsInstance(model, LogisticDistribution)

    def test_no_steal_exponential(self):
        rng = np.random.RandomState(2)
        data = list(rng.exponential(scale=1.0, size=5000))
        model = fit(data, get_estimator(data), max_its=25, out=None)
        self.assertNotIsInstance(model, LogisticDistribution)


if __name__ == "__main__":
    unittest.main()
