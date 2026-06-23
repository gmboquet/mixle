"""Tests for the generalized-Gaussian (exponential-power) candidate in the automatic selector.

Symmetric real-valued data whose shape clearly departs from Gaussian (``beta != 2``) should be
recommended (and built) as a GeneralizedGaussianDistribution. The selector must NOT steal
clearly-Gaussian or clearly-exponential data.
"""

import unittest

import numpy as np
from scipy import stats

from pysp.inference.estimation import fit
from pysp.stats import GeneralizedGaussianDistribution
from pysp.utils.automatic import get_estimator


class AutomaticGeneralizedGaussianTest(unittest.TestCase):
    def test_recovery_generalized_gaussian_data_builds_generalized_gaussian(self):
        # beta=1.2 (peaky, near-Laplace) and beta=4.0 (flat-topped, sub-Gaussian) -- both clearly off 2.
        for beta, loc, scale in [(1.2, 0.0, 2.0), (4.0, 3.0, 1.5)]:
            data = list(stats.gennorm.rvs(beta, loc=loc, scale=scale, size=5000, random_state=1))
            model = fit(data, get_estimator(data), max_its=25, out=None)
            self.assertIsInstance(
                model,
                GeneralizedGaussianDistribution,
                msg=f"gennorm(beta={beta}, loc={loc}, scale={scale}) not recovered",
            )

    def test_no_steal_gaussian(self):
        rng = np.random.RandomState(1)
        data = list(rng.normal(0.0, 1.0, size=5000))
        model = fit(data, get_estimator(data), max_its=25, out=None)
        self.assertNotIsInstance(model, GeneralizedGaussianDistribution)

    def test_no_steal_exponential(self):
        rng = np.random.RandomState(2)
        data = list(rng.exponential(scale=1.0, size=5000))
        model = fit(data, get_estimator(data), max_its=25, out=None)
        self.assertNotIsInstance(model, GeneralizedGaussianDistribution)


if __name__ == "__main__":
    unittest.main()
