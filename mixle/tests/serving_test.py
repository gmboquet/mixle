"""Production layer: versioned Registry (register/promote/swap) and Service (scoring + logging)."""

import tempfile
import unittest

import numpy as np

from mixle.inference.production import Registry, Service, fit_with_provenance
from mixle.stats import GaussianDistribution


class _IdentityEncoder:
    """Minimal stand-in for a real ``DataSequenceEncoder``: ``seq_encode`` is the identity."""

    def seq_encode(self, recs):
        return recs


class _WrongCardinalityModel:
    """A model whose batch scoring path always returns one fewer log-density than requested."""

    def dist_to_encoder(self):
        return _IdentityEncoder()

    def seq_log_density(self, enc):
        return np.zeros(max(len(enc) - 1, 0), dtype=float)

    def log_density(self, r):
        return 0.0


class _OutageModel:
    """A model whose backend is entirely down: every call -- batch or per-record -- raises."""

    def dist_to_encoder(self):
        return _IdentityEncoder()

    def seq_log_density(self, enc):
        raise ConnectionError("simulated backend outage")

    def log_density(self, r):
        raise ConnectionError("simulated backend outage")


class _FlakyModel:
    """A model whose batch path always fails, and whose per-record path fails for one sentinel value
    only (a network blip on a single record, not a full outage)."""

    def dist_to_encoder(self):
        return _IdentityEncoder()

    def seq_log_density(self, enc):
        raise TimeoutError("simulated backend timeout")

    def log_density(self, r):
        if r == 2.0:
            raise TimeoutError("simulated backend timeout")
        return -0.5


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


class ModelServiceCardinalityTest(unittest.TestCase):
    """A model's batch output must have exactly one log-density per input record, or scoring rejects it."""

    def test_wrong_cardinality_output_is_rejected(self):
        svc = Service(_WrongCardinalityModel(), name="bad-cardinality")
        with self.assertRaises(ValueError):
            svc.score([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_wrong_cardinality_single_record_is_also_rejected(self):
        # a len-1 batch against a model that always returns one fewer output -> 0 outputs for 1 record;
        # still a mismatch, not an edge case that should slip through.
        svc = Service(_WrongCardinalityModel(), name="bad-cardinality")
        with self.assertRaises(ValueError):
            svc.score([1.0])

    def test_correct_cardinality_model_is_unaffected(self):
        model = GaussianDistribution(0.0, 1.0)
        svc = Service(model, name="g")
        lp = svc.score([0.0, 0.5, 1.0])
        self.assertEqual(lp.shape, (3,))


class ModelServiceOutageTest(unittest.TestCase):
    """A model call that raises (network failure, timeout, backend error) is an outage, not a score --
    it must not be silently folded into an ordinary, indistinguishable ``-inf`` observation."""

    def test_outage_still_returns_the_contracted_array_shape(self):
        svc = Service(_OutageModel(), name="outage")
        lp = svc.score([1.0, 2.0, 3.0])
        self.assertEqual(lp.shape, (3,))
        self.assertTrue(np.all(~np.isfinite(lp)))

    def test_outage_is_counted_separately_from_ordinary_unscorable_records(self):
        svc = Service(_OutageModel(), name="outage")
        svc.score([1.0, 2.0, 3.0])
        self.assertEqual(svc.activity[-1]["n_unscorable"], 3)
        self.assertEqual(svc.activity[-1]["n_unavailable"], 3)
        self.assertEqual(svc.health()["unavailable_rate"], 1.0)

    def test_outage_is_distinguishable_from_a_genuine_low_probability_score(self):
        # a model that legitimately, successfully scores every record as impossible -- no exception,
        # the model computed and returned -inf itself -- must NOT be flagged as an outage, even though
        # it is equally "unscorable" by the finite-value check.
        model = GaussianDistribution(0.0, 1.0)
        svc_genuine = Service(model, name="genuine")
        lp_genuine = svc_genuine.score([float("inf"), float("inf"), float("inf")])
        self.assertTrue(np.all(~np.isfinite(lp_genuine)))
        self.assertEqual(svc_genuine.activity[-1]["n_unscorable"], 3)
        self.assertEqual(svc_genuine.activity[-1]["n_unavailable"], 0)
        self.assertEqual(svc_genuine.health()["unavailable_rate"], 0.0)

        svc_outage = Service(_OutageModel(), name="outage")
        lp_outage = svc_outage.score([1.0, 2.0, 3.0])

        # both report the same array values and the same unscorable_rate ...
        self.assertTrue(np.array_equal(lp_outage, lp_genuine))
        self.assertEqual(svc_outage.health()["unscorable_rate"], svc_genuine.health()["unscorable_rate"])
        # ... but only the outage is flagged unavailable: the availability signal is not lost.
        self.assertNotEqual(svc_outage.health()["unavailable_rate"], svc_genuine.health()["unavailable_rate"])

    def test_partial_outage_flags_only_the_failing_records(self):
        svc = Service(_FlakyModel(), name="flaky")
        lp = svc.score([1.0, 2.0, 3.0])
        self.assertTrue(np.isfinite(lp[0]))
        self.assertFalse(np.isfinite(lp[1]))  # the one sentinel record that failed
        self.assertTrue(np.isfinite(lp[2]))
        self.assertEqual(svc.activity[-1]["n_unavailable"], 1)
        self.assertEqual(svc.activity[-1]["n_unscorable"], 1)

    def test_health_omits_unavailable_rate_when_no_scoring_events_yet(self):
        svc = Service(GaussianDistribution(0.0, 1.0), name="g")
        h = svc.health()
        self.assertEqual(h, {"events": 0, "drift_events": 0})


if __name__ == "__main__":
    unittest.main()
