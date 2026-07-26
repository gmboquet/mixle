"""Automatic detector registry: families self-register, get selected by BIC, builtins unaffected."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import mixle.utils.automatic.detectors as detector_registry
from mixle.inference.estimation import fit
from mixle.utils.automatic import get_estimator
from mixle.utils.automatic.detectors import Detector, continuous_detectors, get_detector, register


class DetectorRegistryTest(unittest.TestCase):
    def test_registry_discovers_modules(self):
        names = [d.name for d in continuous_detectors()]
        self.assertIn("laplace", names)
        self.assertIsNotNone(get_detector("laplace"))

    def test_detector_constructs_and_register_returns_it(self):
        # NB: never register a real family name (or an always-applying detector) from a test -- the
        # registry is a process-global singleton, so that would poison every other automatic test.
        d = Detector("__never_applies__", "continuous", lambda a: False, lambda a, n: 0.0, lambda *a, **k: None)
        self.assertEqual(d.name, "__never_applies__")
        self.assertIs(register(d), d)  # registering an inert (applies->False) detector cannot affect selection

    def test_registry_rejects_invalid_and_duplicate_entries(self):
        d = Detector("__unique_probe__", "continuous", lambda a: False, lambda a, n: 0.0, lambda: None)
        self.assertIs(register(d), d)
        self.assertIs(register(d), d)
        duplicate = Detector("__unique_probe__", "continuous", lambda a: False, lambda a, n: 0.0, lambda: None)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register(duplicate)
        other_kind = Detector("__unique_probe__", "discrete", lambda a: False, lambda a, n: 0.0, lambda: None)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register(other_kind)
        with self.assertRaisesRegex(ValueError, "kind"):
            register(Detector("__bad_kind__", "other", lambda a: False, lambda a, n: 0.0, lambda: None))

    def test_failed_discovery_is_atomic_and_retryable(self):
        original_registry = detector_registry._REGISTRY
        original_discovered = detector_registry._DISCOVERED
        original_staging = detector_registry._DISCOVERY_STAGING
        calls = 0
        probe = Detector("__transaction_probe__", "continuous", lambda a: False, lambda a, n: 0.0, lambda: None)

        def import_probe(_name):
            nonlocal calls
            calls += 1
            register(probe)
            if calls == 1:
                raise RuntimeError("transient detector import")
            return object()

        detector_registry._publish({"continuous": {}, "discrete": {}})
        detector_registry._DISCOVERED = False
        detector_registry._DISCOVERY_STAGING = None
        try:
            with (
                mock.patch.object(
                    detector_registry.pkgutil,
                    "iter_modules",
                    return_value=[SimpleNamespace(name="transaction_probe")],
                ),
                mock.patch.object(detector_registry.importlib, "import_module", side_effect=import_probe),
            ):
                with self.assertRaisesRegex(RuntimeError, "transient detector import"):
                    detector_registry._discover()
                self.assertFalse(detector_registry._DISCOVERED)
                self.assertNotIn("__transaction_probe__", detector_registry._REGISTRY["continuous"])

                detector_registry._discover()
                self.assertTrue(detector_registry._DISCOVERED)
                self.assertIs(detector_registry.get_detector("__transaction_probe__"), probe)
                self.assertEqual(calls, 2)
        finally:
            detector_registry._REGISTRY = original_registry
            detector_registry._DISCOVERED = original_discovered
            detector_registry._DISCOVERY_STAGING = original_staging

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
