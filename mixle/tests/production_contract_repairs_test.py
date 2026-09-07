"""Production surfaces: records that are what they say, receipts that repeat, inputs that are checked.

Pass 05 of the ten adversarial reviews of the 0.8.1 candidate (P05-F01..F17) found a registry that
served a swapped version file and a substituted model family without noticing, a provenance header
that made its own model unserializable and whose request digest hashed a memory address, a
verification function that turned a caller's flag error into a false verdict, a drift detector whose
answer depended on whether the batch arrived as a list or an array and which died on the first NaN,
and several constructor arguments taken entirely on trust.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
import unittest
import warnings

import numpy as np

import mixle.stats as S
from mixle.inference import optimize
from mixle.inference.production import Registry, Service, detect_drift, fit_with_provenance, verify_lineage
from mixle.inference.production.registry import _record_digest
from mixle.stats import dump_models, load_models
from mixle.utils.serialization import to_serializable


def _quiet(callable_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return callable_()


class _TemporaryRegistry(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.registry = Registry(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


class RegistryIntegrityTest(_TemporaryRegistry):
    """P05-F01, F02, F11: a swapped file, a substituted family, and raw file errors all went unnoticed."""

    def _register_two(self):
        self.registry.register(S.GaussianDistribution(1.0, 1.0), "m")
        self.registry.register(S.GaussianDistribution(9.0, 1.0), "m")
        self.registry.promote("m", "v1", "production")

    def test_swapping_two_version_files_is_refused_rather_than_served(self):
        self._register_two()
        self.assertEqual(self.registry.get("m", "v1")[0].mu, 1.0)
        first = os.path.join(self.root, "m", "v1.json")
        second = os.path.join(self.root, "m", "v2.json")
        a, b = pathlib.Path(first).read_text(), pathlib.Path(second).read_text()
        pathlib.Path(first).write_text(b)
        pathlib.Path(second).write_text(a)
        for label, produce in (
            ("get", lambda: self.registry.get("m", "v1")),
            ("current", lambda: self.registry.current("m", "production")),
        ):
            with self.subTest(entry=label):
                with self.assertRaises(ValueError) as caught:
                    produce()
                self.assertIn("declares version", str(caught.exception))

    def test_a_substituted_model_family_is_refused_against_its_own_header(self):
        data = [float(value) for value in np.random.RandomState(0).normal(0.0, 1.0, 200)]
        model, header = _quiet(lambda: fit_with_provenance(data, S.GaussianEstimator()))
        self.registry.register(model, "g", header=header)
        self.assertIsInstance(self.registry.get("g")[0], S.GaussianDistribution)
        path = os.path.join(self.root, "g", "v1.json")
        payload = json.loads(pathlib.Path(path).read_text())
        payload["model"] = to_serializable(S.PoissonDistribution(3.0))
        payload.pop("record_digest", None)
        payload["record_digest"] = _record_digest(payload)
        pathlib.Path(path).write_text(json.dumps(payload))
        with self.assertRaises(ValueError) as caught:
            self.registry.get("g")
        self.assertIn("describes a GaussianDistribution", str(caught.exception))

    def test_a_damaged_file_and_a_stray_name_are_reported_as_registry_problems(self):
        self.registry.register(S.GaussianDistribution(1.0, 1.0), "m")
        pathlib.Path(self.root, "m", "vX.json").write_text("{}")
        self.assertEqual(self.registry.versions("m"), ["v1"])
        pathlib.Path(self.root, "m", "v1.json").write_text('{"version": ')
        with self.assertRaises(ValueError) as caught:
            self.registry.get("m", "v1")
        self.assertIn("not valid JSON", str(caught.exception))

    def test_a_non_finite_header_value_is_refused_at_register_time(self):
        with self.assertRaises(ValueError) as caught:
            self.registry.register(
                S.GaussianDistribution(1.0, 1.0), "n", header={"training": {"final_loglik": float("nan")}}
            )
        self.assertIn("strict JSON", str(caught.exception))


class VerifyChainTrustCodeTest(_TemporaryRegistry):
    """P05-F05: an invalid trust_code was swallowed into a false 'does not verify'."""

    def test_an_invalid_flag_raises_instead_of_returning_a_verdict(self):
        self.registry.register(S.GaussianDistribution(1.0, 1.0), "ck")
        self.assertIsInstance(self.registry.verify_chain("ck"), bool)
        for value in ("yes", 1, None):
            with self.subTest(trust_code=repr(value)):
                with self.assertRaises(ValueError) as caught:
                    self.registry.verify_chain("ck", trust_code=value)
                self.assertIn("must be exactly True", str(caught.exception))


class CheckpointerAndServiceValidationTest(_TemporaryRegistry):
    """P05-F09, F10, F16: every/resume/keep were taken on trust; trust_code could not be forwarded."""

    def test_checkpointer_every_must_be_a_positive_integer(self):
        for value in (0, -1, True, 1.5, "2", None):
            with self.subTest(every=repr(value)):
                with self.assertRaises(ValueError):
                    self.registry.checkpointer("c", every=value)
        self.assertTrue(callable(self.registry.checkpointer("c", every=5)))

    def test_checkpointer_resume_must_be_an_exact_bool(self):
        with self.assertRaises(TypeError) as caught:
            self.registry.checkpointer("c", resume="no")
        self.assertIn("resume must be True or False", str(caught.exception))

    def test_service_keep_must_be_a_positive_integer(self):
        model = S.GaussianDistribution(0.0, 1.0)
        for value in (0, -1, True, 1.5):
            with self.subTest(keep=repr(value)):
                with self.assertRaises(ValueError):
                    Service(model, keep=value)
        self.assertEqual(Service(model, keep=5).keep, 5)

    def test_from_registry_forwards_trust_code_to_the_load(self):
        self.registry.register(S.GaussianDistribution(0.0, 1.0), "g")
        self.assertIsInstance(Service.from_registry(self.registry, "g", trust_code=True), Service)
        self.assertIsInstance(Service.from_registry(self.registry, "g"), Service)


class ProvenanceTest(unittest.TestCase):
    """P05-F03, F04: the attached header broke serialization; the request digest hashed an address."""

    DATA = [float(value) for value in np.random.RandomState(0).normal(0.0, 1.0, 200)]

    def test_a_model_carrying_a_header_still_serializes_as_json(self):
        model, _ = _quiet(lambda: fit_with_provenance(self.DATA, S.GaussianEstimator()))
        self.assertTrue(hasattr(model, "header"))
        restored = load_models(dump_models(model))
        self.assertAlmostEqual(model.log_density(0.3), restored.log_density(0.3), places=12)
        self.assertGreater(len(model.to_json()), 0)

    def test_two_identical_requests_have_the_same_digest(self):
        _, first = _quiet(lambda: fit_with_provenance(self.DATA, S.GaussianEstimator()))
        _, second = _quiet(lambda: fit_with_provenance(self.DATA, S.GaussianEstimator()))
        self.assertEqual(first.training["fit_request_digest"], second.training["fit_request_digest"])
        self.assertNotIn("0x", first.training["fit_request"]["estimator_repr"])

    def test_a_different_configuration_still_has_a_different_digest(self):
        _, plain = _quiet(lambda: fit_with_provenance(self.DATA, S.GaussianEstimator()))
        _, primed = _quiet(
            lambda: fit_with_provenance(self.DATA, S.GaussianEstimator(pseudo_count=(1.0, 1.0), suff_stat=(0.0, 1.0)))
        )
        self.assertNotEqual(plain.training["fit_request_digest"], primed.training["fit_request_digest"])

    def test_verify_lineage_reports_a_non_header_as_unverified(self):
        for value in ("x", None, 3, {}):
            with self.subTest(header=repr(value)):
                self.assertFalse(verify_lineage(value))


class DriftContractTest(unittest.TestCase):
    """P05-F06, F12, F16: the verdict depended on the container, a NaN killed the loop, thresholds were untyped."""

    @staticmethod
    def _rows(count: int, seed: int):
        rng = np.random.RandomState(seed)
        return [(float(a), float(b)) for a, b in zip(rng.normal(0.0, 1.0, count), rng.normal(0.0, 1.0, count))]

    def test_a_list_batch_and_an_array_batch_give_the_same_answer(self):
        reference, current = self._rows(200, 1), self._rows(100, 2)
        model = _quiet(lambda: optimize(reference, max_its=3, out=None))
        as_list = detect_drift(model, reference, current)
        as_array = detect_drift(model, reference, np.array(current))
        self.assertEqual(as_list.drift, as_array.drift)
        self.assertEqual(sorted(as_list.per_feature), sorted(as_array.per_feature))
        for field in as_list.per_feature:
            self.assertAlmostEqual(as_list.per_feature[field]["psi"], as_array.per_feature[field]["psi"], places=9)

    def test_a_nan_record_is_counted_rather_than_raised(self):
        reference = [float(value) for value in np.random.RandomState(1).normal(0.0, 1.0, 300)]
        current = [float(value) for value in np.random.RandomState(2).normal(0.0, 1.0, 100)]
        current[3] = float("nan")
        model = _quiet(lambda: optimize(reference, S.GaussianEstimator(), max_its=5, out=None))
        report = detect_drift(model, reference, current)
        self.assertAlmostEqual(report.score["fraction_unscorable_current"], 0.01, places=6)
        service = Service(model, reference=reference)
        self.assertIsNotNone(service.check_drift(current))

    def test_thresholds_are_checked_for_type_and_for_being_meetable(self):
        reference, current = self._rows(200, 1), self._rows(100, 2)
        model = _quiet(lambda: optimize(reference, max_its=3, out=None))
        with self.assertRaises(TypeError) as caught:
            detect_drift(model, reference, current, psi_threshold="0.5")
        self.assertIn("must be a real number", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            detect_drift(model, reference, current, loglik_shift_threshold=0.5)
        self.assertIn("must be <= 0", str(caught.exception))
        self.assertIsNotNone(detect_drift(model, reference, current, loglik_shift_threshold=-0.5))


class WriteOnlySerializationTest(unittest.TestCase):
    """P05-F17, P09-F05, P10-F05: eight mainstream families produced JSON nothing could read back."""

    def _round_trip(self, distribution, probe):
        restored = load_models(dump_models(distribution))
        self.assertAlmostEqual(float(distribution.log_density(probe)), float(restored.log_density(probe)), places=9)

    def test_the_families_the_changelog_did_not_disclose_now_round_trip(self):
        topics = [
            S.CategoricalDistribution({"a": 0.5, "b": 0.25, "c": 0.25}),
            S.CategoricalDistribution({"a": 0.25, "b": 0.5, "c": 0.25}),
            S.CategoricalDistribution({"a": 0.25, "b": 0.25, "c": 0.5}),
        ]
        composite = [
            S.CompositeDistribution([S.CategoricalDistribution({"a": 1.0, "b": 0.0}), S.GaussianDistribution(-6, 1)]),
            S.CompositeDistribution([S.CategoricalDistribution({"a": 0.0, "b": 1.0}), S.GaussianDistribution(0, 1)]),
        ]
        sequences = [
            S.SequenceDistribution(
                S.CompositeDistribution([S.GaussianDistribution(-6, 1), S.GammaDistribution(1, 3)]),
                S.PoissonDistribution(3),
            ),
            S.SequenceDistribution(
                S.CompositeDistribution([S.GaussianDistribution(0, 1), S.GammaDistribution(3, 3)]),
                S.PoissonDistribution(3),
            ),
        ]
        cases = (
            ("Dirichlet", S.DirichletDistribution([1.0, 2.0, 3.0]), [0.2, 0.3, 0.5]),
            (
                "HierarchicalMixture",
                S.HierarchicalMixtureDistribution(
                    topics,
                    [0.25] * 4,
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.3, 0.4, 0.3]],
                    len_dist=S.CategoricalDistribution({8: 1.0}),
                ),
                ["a", "b", "a", "c", "a", "b", "a", "c"],
            ),
            (
                "JointMixture",
                S.JointMixtureDistribution(composite, sequences, joint_weights=[[0.5, 0.1], [0.1, 0.3]]),
                (("a", -6.0), [(-6.0, 1.0)]),
            ),
            (
                "MultivariateStudentT",
                S.MultivariateStudentTDistribution(5.0, [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
                [0.1, 0.2],
            ),
            (
                "ProbabilisticPCA",
                S.ProbabilisticPCADistribution(np.eye(3)[:, :2], np.zeros(3), 0.5),
                [0.1, 0.2, 0.3],
            ),
            ("GaussianCopula", S.GaussianCopulaDistribution(np.eye(3)), [0.3, 0.4, 0.5]),
            ("RVineCopula", S.RVineCopulaDistribution.independence(3), [0.3, 0.4, 0.5]),
        )
        for label, distribution, probe in cases:
            with self.subTest(family=label):
                self._round_trip(distribution, probe)

    def test_a_fitted_vine_and_a_fitted_gaussian_copula_round_trip(self):
        from scipy.stats import norm

        rng = np.random.RandomState(0)
        z = rng.normal(size=(1500, 3))
        z[:, 1] += 0.8 * z[:, 0]
        z[:, 2] += 0.6 * z[:, 1]
        u = norm.cdf(z).tolist()
        prototype = S.RVineCopulaDistribution.independence(3)
        vine = _quiet(lambda: optimize(u, prototype.estimator(), prev_estimate=prototype, max_its=3, out=None))
        self._round_trip(vine, u[0])
        base = S.GaussianCopulaDistribution(np.eye(3))
        gaussian = _quiet(lambda: optimize(u, base.estimator(), prev_estimate=base, max_its=3, out=None))
        self._round_trip(gaussian, u[0])

    def test_a_tampered_state_is_refused_by_the_constructor(self):
        from mixle.utils.serialization import SerializationError, rebuild_through_init

        law = S.DirichletDistribution([1.0, 2.0, 3.0])
        with self.assertRaises(SerializationError):
            rebuild_through_init(law, {"alpha": [1.0]}, parameters=("alpha", "name", "keys"), label="Dirichlet")
        with self.assertRaises(SerializationError):
            rebuild_through_init(
                law, {"alpha": [-1.0], "name": None, "keys": None}, parameters=("alpha", "name", "keys"), label="D"
            )


if __name__ == "__main__":
    unittest.main()
