"""Production layer: versioned Registry (register/promote/swap) and Service (scoring + logging)."""

import tempfile
import unittest

import numpy as np

from pysp.inference.production import Registry, Service, fit_with_provenance
from pysp.stats import GaussianDistribution


class ModelRegistryTest(unittest.TestCase):
    def _fit(self, mu, seed):
        data = np.random.RandomState(seed).normal(mu, 1.0, 300).tolist()
        model, _ = fit_with_provenance(data, GaussianDistribution(0, 1).estimator(), max_its=20)
        return model

    def test_register_versions_get_and_header(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            v1 = reg.register(self._fit(0.0, 0), "g")
            v2 = reg.register(self._fit(5.0, 1), "g")
            self.assertEqual([v1, v2], ["v1", "v2"])
            self.assertEqual(reg.versions("g"), ["v1", "v2"])
            self.assertEqual(reg.names(), ["g"])
            model, header = reg.get("g", "latest")
            self.assertAlmostEqual(model.mu, 5.0, delta=0.4)  # latest is the mu=5 model, deserialized
            self.assertIsNotNone(header)
            self.assertEqual(reg.header("g", "v1")["model_type"], "GaussianDistribution")

    def test_promote_and_current_swap(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            reg.register(self._fit(0.0, 2), "g")
            reg.register(self._fit(5.0, 3), "g")
            reg.promote("g", "v1", alias="production")
            prod, _ = reg.current("g", "production")
            self.assertAlmostEqual(prod.mu, 0.0, delta=0.4)  # production pinned to v1
            reg.promote("g", "v2", alias="production")  # swap
            prod2, _ = reg.current("g", "production")
            self.assertAlmostEqual(prod2.mu, 5.0, delta=0.4)

    def test_current_falls_back_to_latest(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            reg.register(self._fit(0.0, 4), "g")
            model, _ = reg.current("g")  # no alias set -> latest
            self.assertIsInstance(model, GaussianDistribution)


class ModelServiceTest(unittest.TestCase):
    def test_score_logs_activity_and_health(self):
        rng = np.random.RandomState(5)
        ref = rng.normal(0, 1, 1000).tolist()
        model, _ = fit_with_provenance(ref, GaussianDistribution(0, 1).estimator(), max_its=20)
        svc = Service(model, name="g", reference=ref)
        lp = svc.score(rng.normal(0, 1, 200).tolist())
        self.assertEqual(lp.shape, (200,))
        self.assertEqual(len(svc.activity), 1)
        h = svc.health()
        self.assertEqual(h["records"], 200)
        self.assertEqual(h["unscorable_rate"], 0.0)
        self.assertIsNotNone(h["mean_loglik"])

    def test_unscorable_records_surface_as_problem(self):
        model = GaussianDistribution(0.0, 1.0)
        svc = Service(model, name="g")
        svc.score([1.0, 2.0, float("inf"), float("nan")])  # inf/nan are outside support -> unscorable
        self.assertGreater(svc.health()["unscorable_rate"], 0.0)

    def test_from_registry_and_drift(self):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.RandomState(6)
            ref = rng.normal(0, 1, 1000).tolist()
            model, _ = fit_with_provenance(ref, GaussianDistribution(0, 1).estimator(), max_its=20)
            reg = Registry(d)
            reg.register(model, "g")
            reg.promote("g", "v1")
            svc = Service.from_registry(reg, "g", reference=ref)
            # the service carries the model's provenance header loaded from the registry
            self.assertIsNotNone(svc.header)
            self.assertEqual(svc.header["model_type"], "GaussianDistribution")
            self.assertTrue(svc.check_drift(rng.normal(4, 1, 500).tolist()).drift)
            self.assertEqual(svc.health()["drift_events"], 1)


if __name__ == "__main__":
    unittest.main()
