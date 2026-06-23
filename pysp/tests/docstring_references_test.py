"""WS-6: the standard base-distribution modules carry a literature reference in their docstring."""

import importlib
import unittest

STANDARD = [
    "gaussian", "gamma", "beta", "exponential", "uniform", "laplace", "logistic", "rayleigh",
    "weibull", "pareto", "gumbel", "student_t", "poisson", "binomial", "bernoulli", "geometric",
    "negative_binomial",
]


class DocstringReferencesTest(unittest.TestCase):
    def test_standard_families_cite_a_reference(self):
        for name in STANDARD:
            mod = importlib.import_module(f"pysp.stats.base.{name}")
            with self.subTest(module=name):
                self.assertIsNotNone(mod.__doc__)
                self.assertIn("Reference", mod.__doc__)


if __name__ == "__main__":
    unittest.main()
