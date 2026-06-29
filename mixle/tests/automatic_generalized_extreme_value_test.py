"""Acceptance tests for the generalized_extreme_value auto-selection candidate.

RECOVERY: GEV-distributed block-maxima data (two clearly-in-family shapes) is recommended as
generalized_extreme_value and ``fit`` builds a GeneralizedExtremeValueDistribution. NO-STEAL: Gaussian and
Exponential data are NOT recommended as GEV -- the 3-parameter BIC keeps it from beating the honest fit.
"""

import unittest

import numpy as np
from scipy import stats

from mixle.inference.estimation import fit
from mixle.stats import GeneralizedExtremeValueDistribution
from mixle.utils.automatic import get_estimator


class AutomaticGeneralizedExtremeValueTest(unittest.TestCase):
    def test_recovery(self):
        # scipy genextreme uses c = -xi; draw two clearly-in-family GEV settings.
        # scipy c = -xi, so c < 0 is the heavy-tailed Frechet type -- strongly skewed, unmistakably GEV.
        for seed, c, loc, scale in [(0, -0.2, 5.0, 2.0), (1, -0.3, -1.0, 1.5)]:
            rng = np.random.RandomState(seed)
            data = list(stats.genextreme.rvs(c, loc=loc, scale=scale, size=5000, random_state=rng))
            model = fit(data, get_estimator(data), max_its=25, out=None)
            self.assertIsInstance(
                model,
                GeneralizedExtremeValueDistribution,
                msg=f"GEV(c={c}) not recovered; got {type(model).__name__}",
            )

    def test_no_steal_gaussian(self):
        rng = np.random.RandomState(2)
        data = list(rng.normal(0.0, 1.0, size=5000))
        model = fit(data, get_estimator(data), max_its=25, out=None)
        self.assertNotIsInstance(model, GeneralizedExtremeValueDistribution)

    def test_no_steal_exponential(self):
        rng = np.random.RandomState(3)
        data = list(rng.exponential(scale=1.0, size=5000))
        model = fit(data, get_estimator(data), max_its=25, out=None)
        self.assertNotIsInstance(model, GeneralizedExtremeValueDistribution)


if __name__ == "__main__":
    unittest.main()
