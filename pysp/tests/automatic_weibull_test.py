"""Automatic selection of the Weibull family: recovered from true Weibull data, steals nothing."""

import unittest

import numpy as np

from pysp.inference.estimation import fit
from pysp.utils.automatic import get_estimator


class AutomaticWeibullTest(unittest.TestCase):
    def test_recovers_weibull(self):
        rng = np.random.RandomState(0)
        for shape, scale in ((2.5, 3.0), (0.6, 1.5)):
            data = list(scale * rng.weibull(shape, size=5000))
            m = fit(data, get_estimator(data), max_its=25, out=None)
            self.assertEqual(
                type(m).__name__,
                "WeibullDistribution",
                msg=f"failed to recover Weibull(shape={shape}, scale={scale})",
            )

    def test_does_not_steal_gaussian(self):
        rng = np.random.RandomState(1)
        g = list(rng.normal(0.0, 1.0, size=5000))
        m = fit(g, get_estimator(g), max_its=25, out=None)
        self.assertNotEqual(type(m).__name__, "WeibullDistribution")

    def test_does_not_steal_exponential(self):
        rng = np.random.RandomState(2)
        e = list(rng.exponential(scale=1.0, size=5000))
        m = fit(e, get_estimator(e), max_its=25, out=None)
        self.assertNotEqual(type(m).__name__, "WeibullDistribution")


if __name__ == "__main__":
    unittest.main()
