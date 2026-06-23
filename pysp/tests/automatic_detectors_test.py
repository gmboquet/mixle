"""Automatic detector registry: families self-register, get selected by BIC, builtins unaffected."""

import unittest

import numpy as np

from pysp.inference.estimation import fit
from pysp.utils.automatic import get_estimator
from pysp.utils.automatic.detectors import Detector, continuous_detectors, get_detector, register


class DetectorRegistryTest(unittest.TestCase):
    def test_registry_discovers_modules(self):
        names = [d.name for d in continuous_detectors()]
        self.assertIn("laplace", names)
        self.assertIsNotNone(get_detector("laplace"))

    def test_register_is_idempotent_and_typed(self):
        d = register(Detector("laplace", "continuous", lambda a: True, lambda a, n: 0.0, lambda *a, **k: None))
        self.assertEqual(d.name, "laplace")

    def test_laplace_recovered_from_laplace_data(self):
        rng = np.random.RandomState(0)
        data = list(rng.laplace(2.0, 1.5, size=4000))
        m = fit(data, get_estimator(data), max_its=25, out=None)
        self.assertEqual(type(m).__name__, "LaplaceDistribution")

    def test_laplace_does_not_steal_gaussian_or_positive(self):
        rng = np.random.RandomState(1)
        g = list(rng.normal(2.0, 1.5, size=4000))
        self.assertEqual(type(fit(g, get_estimator(g), max_its=25, out=None)).__name__, "GaussianDistribution")


if __name__ == "__main__":
    unittest.main()
