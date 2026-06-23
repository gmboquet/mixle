"""Automatic selection of the Pareto family: recovered from true Pareto data, steals nothing."""

import unittest

import numpy as np

from pysp.inference.estimation import fit
from pysp.utils.automatic import get_estimator


class AutomaticParetoTest(unittest.TestCase):
    def test_recovers_pareto(self):
        rng = np.random.RandomState(0)
        # numpy's pareto draws X >= 0 from a Lomax; xm + xm*X is Pareto type-I on [xm, inf).
        for xm, alpha in ((1.0, 1.5), (2.0, 3.0)):
            data = list(xm * (rng.pareto(alpha, size=5000) + 1.0))
            m = fit(data, get_estimator(data), max_its=25, out=None)
            self.assertEqual(
                type(m).__name__, "ParetoDistribution",
                msg=f"failed to recover Pareto(xm={xm}, alpha={alpha})",
            )

    def test_does_not_steal_gaussian(self):
        rng = np.random.RandomState(1)
        g = list(rng.normal(0.0, 1.0, size=5000))
        m = fit(g, get_estimator(g), max_its=25, out=None)
        self.assertNotEqual(type(m).__name__, "ParetoDistribution")

    def test_does_not_steal_exponential(self):
        rng = np.random.RandomState(2)
        e = list(rng.exponential(scale=1.0, size=5000))
        m = fit(e, get_estimator(e), max_its=25, out=None)
        self.assertNotEqual(type(m).__name__, "ParetoDistribution")


if __name__ == "__main__":
    unittest.main()
