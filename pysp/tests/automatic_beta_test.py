"""Tests for the Beta candidate in the automatic model selector.

The Beta family lives on the open unit interval (0, 1); the detector only fires when every value is
strictly inside it. The acceptance bar is recovery on genuine Beta data and no-steal on Gaussian /
Exponential data (whose support spills outside (0, 1), so the gate keeps Beta out of the running).
"""

import unittest

import numpy as np

from pysp.inference.estimation import fit
from pysp.stats import BetaDistribution
from pysp.utils.automatic import get_estimator


class AutomaticBetaTest(unittest.TestCase):
    def test_recovers_beta(self):
        rng = np.random.RandomState(0)
        for a, b in ((2.0, 5.0), (0.7, 0.7)):
            data = list(rng.beta(a, b, size=5000))
            model = fit(data, get_estimator(data), max_its=25, out=None)
            self.assertIsInstance(
                model, BetaDistribution, msg="failed to recover Beta for (a=%s, b=%s)" % (a, b)
            )

    def test_does_not_steal_gaussian(self):
        rng = np.random.RandomState(1)
        data = list(rng.normal(0.0, 1.0, size=5000))
        model = fit(data, get_estimator(data), max_its=25, out=None)
        self.assertNotIsInstance(model, BetaDistribution)

    def test_does_not_steal_exponential(self):
        rng = np.random.RandomState(2)
        data = list(rng.exponential(1.0, size=5000))
        model = fit(data, get_estimator(data), max_its=25, out=None)
        self.assertNotIsInstance(model, BetaDistribution)


if __name__ == "__main__":
    unittest.main()
