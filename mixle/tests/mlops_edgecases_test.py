"""Regression test for the MLOps edge case the maturity audit found: the registry's error surface on an
unknown name/version (a clear KeyError, not a bare IndexError or a path-leaking FileNotFoundError)."""

import tempfile
import unittest

import numpy as np

from mixle.inference.production import Registry, fit_with_provenance
from mixle.stats import GaussianDistribution


def _fit(mu=0.0, seed=0):
    data = np.random.RandomState(seed).normal(mu, 1.0, 200).tolist()
    model, _ = fit_with_provenance(data, GaussianDistribution(0, 1).estimator(), max_its=10)
    return model


class RegistryErrorSurfaceTest(unittest.TestCase):
    def test_unknown_name_raises_keyerror(self):
        # header()/metadata() previously raised a bare IndexError on an unregistered name; now a clear KeyError.
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            for fn in (reg.get, reg.header, reg.metadata):
                with self.assertRaises(KeyError):
                    fn("never_registered")

    def test_unknown_version_raises_keyerror(self):
        # get(name, "v99") previously leaked a raw FileNotFoundError with the store path; now a clear KeyError.
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            reg.register(_fit(), "g")
            for fn in (reg.get, reg.header, reg.metadata):
                with self.assertRaises(KeyError):
                    fn("g", "v99")


if __name__ == "__main__":
    unittest.main()
