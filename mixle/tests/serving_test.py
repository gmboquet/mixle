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


class _BatchDownModel:
    """A model whose batch endpoint always fails, but whose per-record endpoint always succeeds: the
    batch path specifically is degraded, but the fallback recovers every record and no data is lost."""

    def dist_to_encoder(self):
        return _IdentityEncoder()

    def seq_log_density(self, enc):
        raise ConnectionError("simulated batch endpoint outage")

    def log_density(self, r):
        return -0.5


class _BatchOKModel:
    """Control for _BatchDownModel: the batch endpoint succeeds on the first try, returning the exact
    same values _BatchDownModel's per-record fallback recovers -- so the two Services' returned arrays
    and unscorable/unavailable signals are identical, and only batch_unavailable_rate tells them apart."""

    def dist_to_encoder(self):
        return _IdentityEncoder()

    def seq_log_density(self, enc):
        return np.full(len(enc), -0.5, dtype=float)

    def log_density(self, r):
        return -0.5


class _InternalTypeErrorModel:
    """A user/model implementation bug must propagate, not be retried as an outage."""

    def __init__(self):
        self.batch_calls = 0
        self.scalar_calls = 0

    def dist_to_encoder(self):
        return _IdentityEncoder()

    def seq_log_density(self, enc):
        self.batch_calls += 1
        raise TypeError("internal scoring bug")

    def log_density(self, r):
        self.scalar_calls += 1
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

    def test_named_but_absent_alias_refuses_instead_of_serving_an_unpromoted_version(self):
        """SYS-05: a misspelled/absent NAMED alias must not silently serve the latest registration.

        The fail-open this pins against is worse than it first looks: the caller asked for the
        version promoted to an alias and would have received an unpromoted one, and because the
        substitution follows ``latest`` rather than the alias, rolling the alias back to a
        known-good version does not change what a misspelled caller is served.
        """
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            reg.register(self._fit(0.0, 6), "g")  # v1, promoted below
            reg.register(self._fit(5.0, 7), "g")  # v2, never promoted
            reg.promote("g", "v1", alias="production")

            promoted, _ = reg.current("g", "production")
            self.assertAlmostEqual(promoted.mu, 0.0, delta=0.4)

            # Case variants are deliberately NOT asserted here: alias lookup is a filename open, so
            # "PRODUCTION" refuses on a case-sensitive filesystem and resolves to the production
            # alias on a case-insensitive one (macOS APFS). That difference is real but benign for
            # this finding -- where it resolves, it resolves to the PROMOTED version, never to an
            # unpromoted one -- so pinning either outcome would pin the filesystem, not the contract.
            for absent in ("prodcution", "production ", "staging", "prod"):
                with self.assertRaises(KeyError, msg=f"alias {absent!r} must refuse"):
                    reg.current("g", absent)

            # the same refusal reaches the serving entry point, which is where an unpromoted
            # model would actually have been served to traffic.
            with self.assertRaises(KeyError):
                Service.from_registry(reg, "g", alias="staging")

            # and the bootstrap path (no alias NAMED) is deliberately still permitted.
            model, _ = reg.current("g")
            self.assertAlmostEqual(model.mu, 0.0, delta=0.4)  # resolves the production alias

    def test_rollback_is_observable_through_a_named_alias(self):
        """A rollback must change what a named-alias caller is served (the SYS-05 consequence)."""
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            reg.register(self._fit(0.0, 8), "g")
            reg.register(self._fit(5.0, 9), "g")
            reg.promote("g", "v2", alias="production")
            self.assertAlmostEqual(reg.current("g", "production")[0].mu, 5.0, delta=0.4)
            reg.promote("g", "v1", alias="production")  # roll back
            self.assertAlmostEqual(reg.current("g", "production")[0].mu, 0.0, delta=0.4)


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


class ModelServiceBatchFallbackTest(unittest.TestCase):
    """The batch call itself failing (``seq_log_density`` raising) must be visible even when every
    per-record retry recovers a real score -- otherwise a batch endpoint that fails over to per-record
    scoring on every call, but never actually loses a record, is completely invisible to health()."""

    def test_batch_fallback_is_counted_even_when_every_retry_recovers(self):
        svc = Service(_BatchDownModel(), name="batch-down")
        lp = svc.score([1.0, 2.0, 3.0])
        # every record recovered a real score through the per-record retry -- nothing was lost ...
        self.assertTrue(np.all(np.isfinite(lp)))
        self.assertEqual(svc.activity[-1]["n_unscorable"], 0)
        self.assertEqual(svc.activity[-1]["n_unavailable"], 0)
        # ... but the batch call itself failed for every record, and that must not be silently dropped.
        self.assertEqual(svc.activity[-1]["n_batch_unavailable"], 3)
        self.assertEqual(svc.health()["batch_unavailable_rate"], 1.0)

    def test_batch_fallback_is_indistinguishable_from_first_try_success_without_the_new_signal(self):
        # a batch call that succeeds on the first try and one that fails-then-fully-recovers return the
        # identical array and the identical unscorable/unavailable signals; only batch_unavailable_rate
        # tells the two apart -- the same "equal everywhere except the one signal that matters" shape as
        # ModelServiceOutageTest.test_outage_is_distinguishable_from_a_genuine_low_probability_score.
        svc_ok = Service(_BatchOKModel(), name="batch-ok")
        lp_ok = svc_ok.score([1.0, 2.0, 3.0])

        svc_fallback = Service(_BatchDownModel(), name="batch-down")
        lp_fallback = svc_fallback.score([1.0, 2.0, 3.0])

        self.assertTrue(np.array_equal(lp_ok, lp_fallback))
        self.assertEqual(svc_ok.health()["unscorable_rate"], svc_fallback.health()["unscorable_rate"])
        self.assertEqual(svc_ok.health()["unavailable_rate"], svc_fallback.health()["unavailable_rate"])
        self.assertEqual(svc_ok.health()["batch_unavailable_rate"], 0.0)
        self.assertNotEqual(svc_ok.health()["batch_unavailable_rate"], svc_fallback.health()["batch_unavailable_rate"])

    def test_batch_unavailable_still_counts_records_that_also_fail_their_own_retry(self):
        # _FlakyModel: batch always fails; the per-record retry fails only for the r == 2.0 sentinel.
        # Every record in the batch went through the degraded batch-fallback path (n_batch_unavailable
        # == 3), even though only one of them additionally failed its own retry (n_unavailable == 1) --
        # the two counters answer different questions and neither subsumes the other.
        svc = Service(_FlakyModel(), name="flaky")
        svc.score([1.0, 2.0, 3.0])
        self.assertEqual(svc.activity[-1]["n_batch_unavailable"], 3)
        self.assertEqual(svc.activity[-1]["n_unavailable"], 1)

    def test_cardinality_mismatch_is_not_treated_as_batch_unavailable(self):
        # a batch call that *succeeds* but returns the wrong shape is a contract violation to raise on,
        # not an exception to catch and silently retry per-record.
        svc = Service(_WrongCardinalityModel(), name="bad-cardinality")
        with self.assertRaises(ValueError):
            svc.score([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(len(svc.activity), 0)  # score() raised before logging anything

    def test_healthy_batch_call_reports_zero_batch_unavailable_rate(self):
        model = GaussianDistribution(0.0, 1.0)
        svc = Service(model, name="g")
        svc.score([0.0, 0.5, 1.0])
        self.assertEqual(svc.activity[-1]["n_batch_unavailable"], 0)
        self.assertEqual(svc.health()["batch_unavailable_rate"], 0.0)

    def test_internal_type_error_is_not_masked_as_a_compatibility_retry(self):
        model = _InternalTypeErrorModel()
        svc = Service(model, name="buggy")

        with self.assertRaisesRegex(TypeError, "internal scoring bug"):
            svc.score([1.0, 2.0])

        self.assertEqual(model.batch_calls, 1)
        self.assertEqual(model.scalar_calls, 0)
        self.assertEqual(svc.activity, [])

    def test_custom_availability_error_is_explicitly_opted_in(self):
        class CustomUnavailable(Exception):
            pass

        class CustomOutage(_BatchOKModel):
            def seq_log_density(self, enc):
                raise CustomUnavailable("custom transport is down")

        svc = Service(CustomOutage(), availability_errors=(CustomUnavailable,))
        values = svc.score([1.0, 2.0])
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertEqual(svc.activity[-1]["n_batch_unavailable"], 2)


class _ConstantScoreModel:
    """Scores every record in a batch with one fixed log-density (whatever the caller set last)."""

    def __init__(self):
        self.value = 0.0

    def dist_to_encoder(self):
        return _IdentityEncoder()

    def seq_log_density(self, enc):
        return np.full(len(enc), self.value, dtype=float)

    def log_density(self, r):
        return float(self.value)


class _OutOfSupportModel:
    """Scores a sentinel record as a genuine ``-inf`` (outside support) and everything else at -0.5."""

    SENTINEL = -12345.0

    def dist_to_encoder(self):
        return _IdentityEncoder()

    def seq_log_density(self, enc):
        return np.asarray([self.log_density(r) for r in enc], dtype=float)

    def log_density(self, r):
        return float("-inf") if r == self.SENTINEL else -0.5


class ServiceHealthWeightingTest(unittest.TestCase):
    """MXR-080-1654: health() is a record-weighted signal and validates its event window."""

    def test_mean_loglik_weights_records_not_batches(self):
        model = _ConstantScoreModel()
        svc = Service(model, name="g")
        model.value = 0.0
        svc.score([1.0])  # one record at 0.0
        model.value = -10.0
        svc.score([1.0] * 100)  # a hundred records at -10.0
        h = svc.health()
        self.assertEqual(h["records"], 101)
        self.assertEqual(h["scored_records"], 101)
        # the true 101-record mean is -1000/101 = -9.9009...; the old mean-of-means reported -5.0
        self.assertAlmostEqual(h["mean_loglik"], -1000.0 / 101.0, places=9)

    def test_rebatching_the_same_scores_does_not_move_health(self):
        model = _ConstantScoreModel()
        one_call = Service(model, name="a")
        model.value = -2.0
        one_call.score([1.0] * 8)
        many_calls = Service(model, name="b")
        for _ in range(8):
            many_calls.score([1.0])
        self.assertAlmostEqual(one_call.health()["mean_loglik"], many_calls.health()["mean_loglik"], places=12)

    def test_unscorable_records_are_excluded_from_the_record_mean(self):
        svc = Service(_OutOfSupportModel(), name="g")
        svc.score([1.0, _OutOfSupportModel.SENTINEL])
        h = svc.health()
        self.assertEqual(h["records"], 2)
        self.assertEqual(h["scored_records"], 1)
        self.assertAlmostEqual(h["mean_loglik"], -0.5, places=12)
        self.assertEqual(h["unscorable_rate"], 0.5)

    def test_all_unscorable_window_reports_no_mean(self):
        svc = Service(_OutOfSupportModel(), name="g")
        svc.score([_OutOfSupportModel.SENTINEL] * 3)
        h = svc.health()
        self.assertEqual(h["scored_records"], 0)
        self.assertIsNone(h["mean_loglik"])

    def test_window_must_be_an_exact_positive_integer(self):
        svc = Service(_ConstantScoreModel(), name="g")
        svc.score([1.0])
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                svc.health(window=bad)
        for bad in (True, 1.5, "5"):
            with self.assertRaises(TypeError):
                svc.health(window=bad)

    def test_window_selects_the_most_recent_events(self):
        model = _ConstantScoreModel()
        svc = Service(model, name="g")
        model.value = -1.0
        svc.score([1.0])
        model.value = -3.0
        svc.score([1.0])
        self.assertEqual(svc.health(window=1)["events"], 1)
        self.assertAlmostEqual(svc.health(window=1)["mean_loglik"], -3.0, places=12)
        self.assertAlmostEqual(svc.health(window=2)["mean_loglik"], -2.0, places=12)


class RecordDigestRoundTripTest(unittest.TestCase):
    """A version envelope must verify against the digest that was stored with it.

    ``_record_digest`` ran over the *in-memory* payload while ``_load_payload`` reran it over the
    JSON-parsed one. ``_canonical`` deliberately tags lists and tuples differently (MXR-080-1601:
    otherwise ``[[1, 2]]`` and ``[(1, 2)]`` collide), but JSON has only arrays -- and a provenance
    header carries ``schema`` as ``("value", "Real")`` pairs. Every version file written for a
    fitted model therefore failed its own integrity check the moment it was read back, so a healthy
    registry was indistinguishable from a tampered one.
    """

    def _fitted(self, seed=0):
        data = np.random.RandomState(seed).normal(0.0, 1.0, 200).tolist()
        model, _ = fit_with_provenance(data, GaussianDistribution(0, 1).estimator(), max_its=10)
        return model

    def test_a_freshly_registered_version_verifies_on_load(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            reg.register(self._fitted(), "g")
            # every reader path re-verifies the digest; none of them may reject an untampered file
            self.assertIsNotNone(reg.get("g", "v1")[0])
            self.assertEqual(reg.header("g", "v1")["model_type"], "GaussianDistribution")
            self.assertIsInstance(reg.record_digest("g", "v1"), str)
            self.assertIsInstance(reg.metadata("g", "v1"), dict)

    def test_a_tampered_version_still_fails(self):
        import json
        import os

        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            reg.register(self._fitted(), "g")
            path = os.path.join(d, "g", "v1.json")
            with open(path) as f:
                payload = json.load(f)
            payload["header"]["model_type"] = "SomethingElse"
            with open(path, "w") as f:
                json.dump(payload, f)
            with self.assertRaisesRegex(ValueError, "integrity failure"):
                reg.get("g", "v1")


class UnscorableRecordIsNotAnOutageTest(unittest.TestCase):
    """A record the model refuses is unscorable data, not an unavailable model.

    ``_safe_logd`` caught only the configured availability errors, so when the scalar observation
    contract began refusing NaN/Inf the rejection escaped ``score()`` entirely -- a single malformed
    record in a production batch took down the whole call instead of being counted, which is what
    ``unscorable_rate`` exists to report.
    """

    def test_nonfinite_records_are_counted_not_raised(self):
        svc = Service(GaussianDistribution(0.0, 1.0), name="g")
        lp = svc.score([1.0, 2.0, float("inf"), float("nan")])
        self.assertEqual(lp.shape, (4,))
        self.assertTrue(np.isfinite(lp[:2]).all())
        self.assertTrue((lp[2:] == float("-inf")).all())
        health = svc.health()
        self.assertAlmostEqual(health["unscorable_rate"], 0.5)
        # unscorable is not an outage: the model answered, so availability must stay clean
        self.assertEqual(health["unavailable_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
