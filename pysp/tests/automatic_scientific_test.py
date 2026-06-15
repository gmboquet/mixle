"""Synthetic coverage tests for scientific automatic estimator profiling."""

import json
import unittest

import numpy as np

from pysp.stats import (
    BernoulliSetEstimator,
    CompositeEstimator,
    GaussianEstimator,
    IgnoredEstimator,
    IntegerCategoricalEstimator,
    PoissonEstimator,
    SequenceEstimator,
    estimate,
    initialize,
    seq_encode,
    seq_estimate,
    seq_log_density_sum,
)
from pysp.utils.automatic import DictRecordDistribution, DictRecordEstimator, analyze_structure, get_estimator


class AutomaticScientificProfilingTestCase(unittest.TestCase):
    def assert_vectorized_round_trip(self, data):
        profile = analyze_structure(data, pairwise=False)
        est = profile.recommend()
        init = initialize(data, est, rng=np.random.RandomState(7), p=1.0)
        model = estimate(data, est, init)
        enc = seq_encode(data, model=model, num_chunks=3)

        total, ll_sum = seq_log_density_sum(enc, model)
        self.assertEqual(total, len(data))
        self.assertTrue(np.isfinite(ll_sum) or ll_sum == -np.inf)

        seq_model = seq_estimate(enc, est, model)
        seq_total, seq_ll_sum = seq_log_density_sum(enc, seq_model)
        self.assertEqual(seq_total, len(data))
        self.assertTrue(np.isfinite(seq_ll_sum) or seq_ll_sum == -np.inf)

        summary = profile.summary()
        self.assertEqual(summary["total_rows"], len(data))
        json.dumps(summary, sort_keys=True)
        return profile, model, seq_model

    def test_marginal_model_selection_synthetic_grid(self):
        rng = np.random.RandomState(11)
        cases = [
            ("dense_integer", [0, 1, 2, 1] * 100, IntegerCategoricalEstimator, "integer_categorical"),
            ("poisson_count", list(rng.poisson(30, 500)), PoissonEstimator, "poisson"),
            ("wide_signed_integer", list(rng.randint(-1000, 1000, 500)), GaussianEstimator, "gaussian"),
            ("sparse_integer_identifier", [10_000_000 + 1000 * i for i in range(200)], IgnoredEstimator, "ignored"),
        ]

        for name, data, estimator_type, recommendation in cases:
            with self.subTest(name=name):
                self.assertIsInstance(get_estimator(data), estimator_type)
                profile = analyze_structure(data, pairwise=False)
                field = profile.fields[0]
                self.assertEqual(field.recommendation, recommendation)
                if recommendation != "ignored":
                    self.assertIn(recommendation, field.model_scores_bits)
                if len(field.model_scores_bits) > 1:
                    self.assertIsNotNone(field.model_score_gap_bits)
                    self.assertGreaterEqual(field.model_score_gap_bits, 0.0)

    def test_auto_estimators_round_trip_through_vectorized_estimation_matrix(self):
        rng = np.random.RandomState(31)
        cases = [
            ("float_gaussian", [1.2, 3.4, -0.5, 2.2] * 30),
            ("poisson", list(rng.poisson(20, 200))),
            ("categorical", ["a", "b", "a", "c"] * 50),
            ("optional", [1.5, None, 2.5, 3.5] * 30),
            ("tuple", [("a", 1.5, 3), ("b", 2.5, 1), ("a", 0.5, 3)] * 30),
            ("variable_length_list", [["a", "b"], ["a"], ["b", "c", "a"]] * 30),
            ("fixed_word_list", [["a", "b", "c"], ["b", "a", "a"]] * 40),
            ("fixed_numeric_vector", np.random.RandomState(32).normal(size=(80, 3)).tolist()),
            ("set", [{"x", "y"}, {"y"}, {"x", "z"}] * 40),
            ("dict", [{"kind": "a", "score": 1.0}, {"kind": "b"}, {"kind": "a", "score": 2.0}] * 40),
            (
                "nested_dict",
                [{"meta": {"kind": "a"}, "value": 0}, {"meta": {"kind": "b"}, "value": 1}, {"value": 2}] * 40,
            ),
            ("fixed_mixed_dict_set", [[{"a": 1}, {"x"}], [{"a": 2}, {"y"}]] * 20),
            ("fixed_mixed_tuple_list", [[("a", 1), ["x", "y"]], [("b", 2), ["z"]]] * 20),
            ("fixed_dict_sequence", [[{"a": 1}, {"a": 2}], [{"a": 3}, {"a": 4}]] * 20),
            ("fixed_set_sequence", [[{"x"}, {"y"}], [{"z"}, {"x"}]] * 20),
            ("empty_dict", [{} for _ in range(20)]),
            ("ignored_mixed", [1.0, [1.0, 2.0], "a"] * 30),
        ]

        for name, data in cases:
            with self.subTest(name=name):
                self.assert_vectorized_round_trip(data)

    def test_fixed_list_container_topology_is_not_over_merged(self):
        dict_set = [[{"a": 1}, {"x"}], [{"a": 2}, {"y"}]] * 20
        dict_set_profile, _, _ = self.assert_vectorized_round_trip(dict_set)
        self.assertIsInstance(dict_set_profile.recommend(), CompositeEstimator)
        self.assertIsInstance(dict_set_profile.recommend().estimators[0], DictRecordEstimator)
        self.assertIsInstance(dict_set_profile.recommend().estimators[1], BernoulliSetEstimator)

        tuple_list = [[("a", 1), ["x", "y"]], [("b", 2), ["z"]]] * 20
        tuple_list_profile, _, _ = self.assert_vectorized_round_trip(tuple_list)
        self.assertIsInstance(tuple_list_profile.recommend(), CompositeEstimator)
        self.assertIsInstance(tuple_list_profile.recommend().estimators[0], CompositeEstimator)
        self.assertIsInstance(tuple_list_profile.recommend().estimators[1], SequenceEstimator)

        dict_dict = [[{"a": 1}, {"a": 2}], [{"a": 3}, {"a": 4}]] * 20
        dict_dict_profile, _, _ = self.assert_vectorized_round_trip(dict_dict)
        self.assertIsInstance(dict_dict_profile.recommend(), SequenceEstimator)
        self.assertIsInstance(dict_dict_profile.recommend().estimator, DictRecordEstimator)

        set_set = [[{"x"}, {"y"}], [{"z"}, {"x"}]] * 20
        set_set_profile, _, _ = self.assert_vectorized_round_trip(set_set)
        self.assertIsInstance(set_set_profile.recommend(), SequenceEstimator)
        self.assertIsInstance(set_set_profile.recommend().estimator, BernoulliSetEstimator)

    def test_marginal_validation_confirms_predictive_recommendations(self):
        rng = np.random.RandomState(21)
        cases = [
            ("dense_integer", [0, 1, 2, 1] * 200, "integer_categorical"),
            ("poisson_count", list(rng.poisson(30, 1000)), "poisson"),
            ("categorical_string", ["a", "b", "a", "c"] * 200, "categorical"),
        ]

        for name, data, recommendation in cases:
            with self.subTest(name=name):
                profile = analyze_structure(data, pairwise=False)
                field = profile.fields[0]
                self.assertEqual(field.recommendation, recommendation)
                self.assertEqual(field.validation_recommendation, recommendation)
                self.assertGreater(field.validation_count, 0)
                self.assertLessEqual(field.validation_count, 1000)
                self.assertIn(recommendation, field.validation_scores_bits)
                self.assertTrue(any("validation:" in line for line in profile.explain()))

    def test_marginal_validation_is_bounded_and_optional(self):
        data = list(np.random.RandomState(22).poisson(10, 1000))
        profile = analyze_structure(data, pairwise=False, validation_fraction=0.50, max_validation_rows=12)
        field = profile.fields[0]
        self.assertEqual(field.validation_count, 12)
        self.assertIn(field.validation_recommendation, field.validation_scores_bits)
        self.assertEqual(profile.summary()["fields"][0]["validation_count"], 12)

        disabled = analyze_structure(data, pairwise=False, validate_marginals=False)
        self.assertEqual(disabled.fields[0].validation_scores_bits, {})
        self.assertIsNone(disabled.fields[0].validation_recommendation)
        self.assertEqual(disabled.fields[0].validation_count, 0)

    def test_integer_categorical_validation_handles_wide_dense_support(self):
        data = list(range(1500)) * 2
        profile = analyze_structure(data, pairwise=False, validation_fraction=0.5, max_validation_rows=25)
        field = profile.fields[0]

        self.assertIn("integer_categorical", field.model_scores_bits)
        self.assertIn("integer_categorical", field.validation_scores_bits)
        self.assertEqual(field.validation_count, 25)
        self.assertTrue(np.isfinite(field.validation_scores_bits["integer_categorical"]))

    def test_marginal_validation_is_deterministic_and_summary_serializable(self):
        data = list(np.random.RandomState(23).poisson(12, 400))
        p1 = analyze_structure(
            data, pairwise=True, max_pairwise_pairs=4, validation_seed=123, rng=np.random.RandomState(5)
        )
        p2 = analyze_structure(
            data, pairwise=True, max_pairwise_pairs=4, validation_seed=123, rng=np.random.RandomState(5)
        )

        self.assertEqual(p1.summary(), p2.summary())
        encoded = json.dumps(p1.summary(), sort_keys=True)
        self.assertIn("validation_scores_bits", encoded)

    def test_marginal_validation_disagreement_is_explained(self):
        data = [0] * 30 + [1] * 10 + [10] * 2
        profile = analyze_structure(
            data, pairwise=False, validation_seed=17, validation_fraction=0.25, validation_min_count=10
        )
        field = profile.fields[0]

        self.assertEqual(field.recommendation, "integer_categorical")
        self.assertEqual(field.validation_recommendation, "poisson")
        self.assertIn(
            "validation prefers poisson over marginal recommendation integer_categorical", field.validation_notes
        )
        self.assertTrue(any("validation disagrees" in warning for warning in profile.warnings))
        self.assertTrue(any("validation prefers poisson" in line for line in profile.explain()))

    def test_marginal_validation_skips_ignored_identifiers(self):
        profile = analyze_structure(["id_%d" % i for i in range(200)], pairwise=False)
        field = profile.fields[0]
        self.assertEqual(field.recommendation, "ignored")
        self.assertEqual(field.validation_scores_bits, {})
        self.assertEqual(field.validation_count, 0)
        self.assertIn("predictive validation skipped for ignored field", field.validation_notes)

    def test_missing_values_are_accounted_for_in_profile(self):
        data = [1.0, None, np.nan, 2.5] * 40
        profile = analyze_structure(data, pairwise=False)
        field = profile.fields[0]
        self.assertEqual(field.count, 160)
        self.assertEqual(field.missing_count, 80)
        self.assertEqual(field.missing_fraction, 0.5)
        self.assertEqual(field.observed_count, 80)
        self.assertEqual(field.recommendation, "gaussian")
        self.assertAlmostEqual(field.unique_fraction, 2.0 / 80.0)
        self.assertGreater(field.effective_cardinality, 1.0)
        self.assertEqual(field.validation_count, 20)
        self.assertEqual(field.validation_recommendation, "gaussian")

    def test_constant_field_diagnostics_are_explicit(self):
        profile = analyze_structure([3.5] * 100, pairwise=False)
        field = profile.fields[0]
        self.assertTrue(field.is_constant)
        self.assertEqual(field.cardinality, 1)
        self.assertEqual(field.top_mass, 1.0)
        self.assertIn("observed values are constant", field.notes)
        self.assertEqual(field.effective_cardinality, 1.0)

    def test_independent_product_has_no_dependency_tree(self):
        rng = np.random.RandomState(12)
        data = [(int(rng.randint(0, 2)), int(rng.randint(0, 3)), int(rng.randint(0, 4))) for _ in range(1000)]
        profile = analyze_structure(data, pairwise=True, mi_threshold_bits=0.02)
        self.assertEqual(profile.pairwise_hints, [])
        self.assertEqual(profile.dependency_tree_edges, [])
        self.assertEqual(profile.pairwise_pair_strategy, "exhaustive")

    def test_dependency_forest_recovers_noisy_binary_chain(self):
        rng = np.random.RandomState(13)
        data = []
        for _ in range(1500):
            a = int(rng.randint(0, 2))
            b = a ^ int(rng.rand() < 0.10)
            c = b ^ int(rng.rand() < 0.10)
            d = int(rng.randint(0, 2))
            data.append((a, b, c, d))

        profile = analyze_structure(data, pairwise=True, mi_threshold_bits=0.01)
        tree_pairs = {frozenset((edge.left, edge.right)) for edge in profile.dependency_tree_edges}
        self.assertIn(frozenset(((0,), (1,))), tree_pairs)
        self.assertIn(frozenset(((1,), (2,))), tree_pairs)
        self.assertNotIn(frozenset(((0,), (2,))), tree_pairs)
        self.assertTrue(all((3,) not in (edge.left, edge.right) for edge in profile.dependency_tree_edges))

    def test_latent_common_cause_leaves_residual_dependency_edges(self):
        rng = np.random.RandomState(16)
        data = []
        for _ in range(1600):
            latent = int(rng.randint(0, 2))
            data.append(tuple(latent ^ int(rng.rand() < 0.08) for _ in range(3)))

        profile = analyze_structure(data, pairwise=True, mi_threshold_bits=0.02)
        self.assertEqual(len(profile.dependency_tree_edges), 2)
        self.assertEqual(len(profile.dependency_residual_edges), 1)
        self.assertAlmostEqual(profile.dependency_redundancy_ratio, 1.0 / 3.0)
        self.assertTrue(any("latent/common-cause" in warning for warning in profile.warnings))
        self.assertTrue(any("dependency residuals" in line for line in profile.explain()))

    def test_dict_records_are_profiled_and_estimable_by_key(self):
        data = [
            {"kind": "a", "count": 1, "score": 1.0},
            {"kind": "b", "count": 2},
            {"kind": "a", "count": 1, "score": 2.0},
        ] * 40
        est = get_estimator(data)
        self.assertIsInstance(est, DictRecordEstimator)
        acc = est.accumulator_factory().make()
        for row in data:
            acc.update(row, 1.0, None)
        model = est.estimate(len(data), acc.value())
        self.assertIsInstance(model, DictRecordDistribution)
        self.assertTrue(np.isfinite(model.log_density({"kind": "a", "count": 1, "score": 1.5})))
        self.assertTrue(np.isfinite(model.log_density({"kind": "b", "count": 2})))
        enc = model.dist_to_encoder().seq_encode(data)
        seq_model = seq_estimate([(len(data), enc)], est, model)
        self.assertTrue(np.isfinite(seq_model.log_density({"kind": "a", "count": 1, "score": 1.5})))

        profile = analyze_structure(data, pairwise=False)
        by_path = {field.path: field for field in profile.fields}
        self.assertEqual(by_path[("key", "kind")].recommendation, "categorical")
        self.assertEqual(by_path[("key", "count")].recommendation, "integer_categorical")
        self.assertEqual(by_path[("key", "score")].missing_count, 40)
        self.assertFalse(any("dict records are profiled by key" in warning for warning in profile.warnings))

    def test_empty_dict_records_have_vectorized_batch_shape(self):
        data = [{} for _ in range(17)]
        est = get_estimator(data)
        self.assertIsInstance(est, DictRecordEstimator)
        self.assertEqual(est.keys, ())

        acc = est.accumulator_factory().make()
        for row in data:
            acc.update(row, 1.0, None)
        model = est.estimate(len(data), acc.value())
        enc = model.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(model.seq_log_density(enc), np.zeros(len(data)))
        seq_model = seq_estimate([(len(data), enc)], est, model)
        np.testing.assert_allclose(seq_model.seq_log_density(enc), np.zeros(len(data)))

    def test_nested_dict_records_are_profiled_recursively(self):
        data = [
            {"meta": {"kind": "a", "flag": True}, "value": 0},
            {"meta": {"kind": "b", "flag": False}, "value": 1},
            {"value": 2},
        ] * 50

        est = get_estimator(data)
        self.assertIsInstance(est, DictRecordEstimator)
        acc = est.accumulator_factory().make()
        for row in data:
            acc.update(row, 1.0, None)
        model = est.estimate(len(data), acc.value())
        self.assertTrue(np.isfinite(model.log_density({"meta": {"kind": "a", "flag": True}, "value": 0})))
        self.assertTrue(np.isfinite(model.log_density({"value": 2})))

        profile = analyze_structure(data, pairwise=True, mi_threshold_bits=0.01)
        by_path = {field.path: field for field in profile.fields}
        kind_path = ("key", "meta", "key", "kind")
        flag_path = ("key", "meta", "key", "flag")
        value_path = ("key", "value")

        self.assertEqual(by_path[kind_path].recommendation, "categorical")
        self.assertEqual(by_path[kind_path].missing_count, 50)
        self.assertEqual(by_path[flag_path].recommendation, "categorical")
        self.assertEqual(by_path[flag_path].missing_count, 50)
        self.assertEqual(by_path[value_path].recommendation, "integer_categorical")

        hint_pairs = {frozenset((hint.left, hint.right)) for hint in profile.pairwise_hints}
        self.assertIn(frozenset((kind_path, value_path)), hint_pairs)

    def test_nested_record_fields_participate_in_pairwise_profile(self):
        rng = np.random.RandomState(17)
        data = []
        for _ in range(500):
            signal = int(rng.randint(0, 2))
            data.append((signal, {"mirror": signal, "noise": int(rng.randint(0, 3))}))

        profile = analyze_structure(data, pairwise=True, mi_threshold_bits=0.05)
        by_path = {field.path: field for field in profile.fields}
        signal_path = (0,)
        mirror_path = (1, "key", "mirror")
        noise_path = (1, "key", "noise")

        self.assertEqual(by_path[signal_path].recommendation, "integer_categorical")
        self.assertEqual(by_path[mirror_path].recommendation, "integer_categorical")
        self.assertEqual(by_path[noise_path].recommendation, "integer_categorical")

        hint_pairs = {frozenset((hint.left, hint.right)) for hint in profile.pairwise_hints}
        self.assertIn(frozenset((signal_path, mirror_path)), hint_pairs)
        self.assertNotIn(frozenset((signal_path, noise_path)), hint_pairs)

    def test_bayesian_dict_records_warn_about_estimator_gap(self):
        data = [{"kind": "a", "count": 1}, {"kind": "b"}] * 40
        profile = analyze_structure(data, pairwise=False, use_bstats=True)
        self.assertTrue(any("Bayesian (conjugate-prior) automatic" in warning for warning in profile.warnings))

    def test_bayesian_gaussian_provider_carries_conjugate_default_prior(self):
        from pysp.stats import GaussianEstimator
        from pysp.stats.bayes.normgamma import NormalGammaDistribution

        est = get_estimator([1.0, 2.0, 3.0] * 20, use_bstats=True)

        # The Bayesian path now builds a pysp.stats estimator carrying the
        # conjugate default NormalGamma prior, so estimation runs the
        # closed-form conjugate / MAP update.
        self.assertIsInstance(est, GaussianEstimator)
        self.assertIsInstance(est.prior, NormalGammaDistribution)
        self.assertTrue(est.has_conj_prior)
        self.assertEqual((est.prior.mu, est.prior.lam, est.prior.a, est.prior.b), (0.0, 1.0e-8, 0.500001, 1.0))

    def test_stratified_pair_budget_can_find_late_field_dependency(self):
        rng = np.random.RandomState(15)
        data = []
        for _ in range(800):
            row = [int(rng.randint(0, 2)) for _ in range(18)]
            signal = int(rng.randint(0, 2))
            row.extend([signal, signal])
            data.append(tuple(row))

        profile = analyze_structure(
            data, pairwise=True, max_pairwise_fields=20, max_pairwise_pairs=10, mi_threshold_bits=0.1
        )
        hint_pairs = {frozenset((hint.left, hint.right)) for hint in profile.pairwise_hints}
        self.assertEqual(profile.pairwise_pair_strategy, "stratified")
        self.assertEqual(profile.pairwise_pairs_available, 190)
        self.assertEqual(profile.pairwise_pairs_checked, 10)
        self.assertIn(frozenset(((18,), (19,))), hint_pairs)
        self.assertTrue(any("checked 10 of 190" in warning for warning in profile.warnings))

    def test_sampling_and_pairwise_budget_are_reported(self):
        data = [(i % 2, i % 2, i % 2, i % 2, i % 2, i % 2) for i in range(1000)]
        profile = analyze_structure(
            data,
            pairwise=True,
            sample_size=200,
            max_pairwise_fields=4,
            max_pairwise_pairs=3,
            mi_threshold_bits=0.0,
            rng=np.random.RandomState(14),
        )
        self.assertEqual(profile.sampled_rows, 200)
        self.assertEqual(profile.pairwise_fields_available, 6)
        self.assertEqual(profile.encoded_pairwise_fields, 4)
        self.assertEqual(profile.pairwise_pairs_available, 6)
        self.assertEqual(profile.pairwise_pairs_checked, 3)
        self.assertEqual(profile.pairwise_pair_strategy, "stratified")
        self.assertLessEqual(len(profile.pairwise_hints), 3)
        self.assertLessEqual(len(profile.dependency_tree_edges), 3)
        self.assertTrue(any("sampled 200 of 1000" in warning for warning in profile.warnings))
        self.assertTrue(any("encoded 4 of 6" in warning for warning in profile.warnings))
        self.assertTrue(any("checked 3 of 6" in warning for warning in profile.warnings))
        summary = profile.summary()
        self.assertEqual(summary["pairwise_fields_available"], 6)


if __name__ == "__main__":
    unittest.main()
