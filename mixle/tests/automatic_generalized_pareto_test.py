"""Acceptance tests for the generalized-Pareto candidate in the automatic model selector.

RECOVERY: heavy-tailed exceedance data drawn from a true GPD (clearly positive shape) is
recommended and built as a GeneralizedParetoDistribution. NO-STEAL: Gaussian(0,1) (signed, so
outside the positive support) and Exponential(scale=1) (the xi=0 limit, owned by gamma) are NOT
recommended as generalized_pareto -- the detector only fires on an unmistakably heavy tail.
"""

import unittest

import numpy as np
from scipy import stats

from mixle.inference.estimation import fit
from mixle.stats import GeneralizedParetoDistribution
from mixle.utils.automatic import get_estimator


class AutomaticGeneralizedParetoTest(unittest.TestCase):
    def test_recovery_heavy_tail_xi_03(self):
        rng = np.random.RandomState(0)
        data = list(stats.genpareto.rvs(0.3, loc=0.0, scale=1.0, size=5000, random_state=rng))
        model = fit(data, get_estimator(data), max_its=25, out=None)
        self.assertIsInstance(model, GeneralizedParetoDistribution)

    def test_recovery_heavy_tail_xi_05(self):
        rng = np.random.RandomState(1)
        data = list(stats.genpareto.rvs(0.5, loc=0.0, scale=2.0, size=5000, random_state=rng))
        model = fit(data, get_estimator(data), max_its=25, out=None)
        self.assertIsInstance(model, GeneralizedParetoDistribution)

    def test_no_steal_gaussian(self):
        rng = np.random.RandomState(2)
        data = list(rng.normal(0.0, 1.0, size=5000))
        model = fit(data, get_estimator(data), max_its=25, out=None)
        self.assertNotIsInstance(model, GeneralizedParetoDistribution)

    def test_no_steal_exponential(self):
        rng = np.random.RandomState(3)
        data = list(rng.exponential(1.0, size=5000))
        model = fit(data, get_estimator(data), max_its=25, out=None)
        self.assertNotIsInstance(model, GeneralizedParetoDistribution)


if __name__ == "__main__":
    unittest.main()
