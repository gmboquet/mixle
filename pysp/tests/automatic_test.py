"""Tests for automatic estimator detection (pysp.utils.automatic)."""

import unittest

import numpy as np

from pysp.stats import (
    BernoulliSetEstimator,
    CategoricalEstimator,
    CompositeEstimator,
    GaussianEstimator,
    IgnoredEstimator,
    IntegerCategoricalEstimator,
    MultivariateGaussianEstimator,
    OptionalEstimator,
    PoissonEstimator,
    SequenceEstimator,
    estimate,
    initialize,
)
from pysp.utils.automatic import DictRecordEstimator, analyze_structure, get_estimator


class AutomaticDetectionTestCase(unittest.TestCase):
    def test_floats_gaussian(self):
        est = get_estimator([1.2, 3.4, -0.5, 2.2] * 30)
        self.assertIsInstance(est, GaussianEstimator)

    def test_integral_valued_floats_are_gaussian(self):
        # Float-valued measurements can land exactly on integers after
        # preprocessing, for example standardized constant columns at 0.0.
        est = get_estimator([0.0, 1.0, 2.0, 3.0] * 30)
        self.assertIsInstance(est, GaussianEstimator)

    def test_bools_categorical(self):
        # bool is a subclass of int in Python; must not fall into the int path
        est = get_estimator([True, False, True] * 40)
        self.assertIsInstance(est, CategoricalEstimator)

    def test_small_cardinality_ints_categorical(self):
        est = get_estimator([1, 2, 3, 2, 1] * 30)
        self.assertIsInstance(est, IntegerCategoricalEstimator)

    def test_counts_poisson(self):
        data = list(np.random.RandomState(0).poisson(50, 300))
        est = get_estimator(data)
        self.assertIsInstance(est, PoissonEstimator)

    def test_signed_ints_gaussian(self):
        data = list(np.random.RandomState(0).randint(-500, 500, 300))
        est = get_estimator(data)
        self.assertIsInstance(est, GaussianEstimator)

    def test_sparse_integer_ids_ignored(self):
        data = [10_000_000 + 1000 * i for i in range(200)]
        est = get_estimator(data)
        self.assertIsInstance(est, IgnoredEstimator)

    def test_overdispersed_nonnegative_ints_gaussian(self):
        data = list(range(300))
        est = get_estimator(data)
        self.assertIsInstance(est, GaussianEstimator)

    def test_id_strings_ignored(self):
        est = get_estimator(["id_%d" % i for i in range(200)])
        self.assertIsInstance(est, IgnoredEstimator)

    def test_cat_strings_categorical(self):
        est = get_estimator(["a", "b", "c"] * 50)
        self.assertIsInstance(est, CategoricalEstimator)

    def test_tuples_composite(self):
        est = get_estimator([("a", 1.5, 3), ("b", 2.5, 1)] * 50)
        self.assertIsInstance(est, CompositeEstimator)
        self.assertEqual(len(est.estimators), 3)

    def test_variable_length_lists_sequence_with_length_model(self):
        est = get_estimator([["a", "b"], ["a"], ["b", "c", "a"]] * 40)
        self.assertIsInstance(est, SequenceEstimator)
        self.assertIsInstance(est.estimator, CategoricalEstimator)
        self.assertIsInstance(est.len_estimator, IntegerCategoricalEstimator)

    def test_fixed_length_word_lists_sequence(self):
        # homogeneous fixed-length lists are sequences, not 3-slot records
        est = get_estimator([["a", "b", "c"], ["b", "a", "a"]] * 50)
        self.assertIsInstance(est, SequenceEstimator)
        self.assertIsInstance(est.len_estimator, IntegerCategoricalEstimator)

    def test_fixed_length_float_lists_mvn(self):
        # numeric fixed-length lists are vectors with covariance, not independent records
        data = np.random.RandomState(2).normal(size=(80, 3)).tolist()
        est = get_estimator(data)
        self.assertIsInstance(est, MultivariateGaussianEstimator)
        self.assertEqual(est.dim, 3)

    def test_sets_bernoulli_set(self):
        est = get_estimator([{"x", "y"}, {"y"}, {"x", "z"}] * 40)
        self.assertIsInstance(est, BernoulliSetEstimator)

    def test_none_optional(self):
        est = get_estimator([1.5, None, 2.5, 3.5] * 30)
        self.assertIsInstance(est, OptionalEstimator)

    def test_dicts_keyed_record(self):
        est = get_estimator([{"k": 1}] * 20)
        self.assertIsInstance(est, DictRecordEstimator)
        self.assertEqual(est.keys, ("k",))

    def test_mixed_scalars_and_containers_ignored(self):
        est = get_estimator([1.0, [1.0, 2.0], "a"] * 20)
        self.assertIsInstance(est, IgnoredEstimator)

    def test_detected_estimators_are_fittable(self):
        # estimators built from detection must round-trip through estimation
        cases = [
            [("a", 1.5, 3), ("b", 2.5, 1), ("a", 0.5, 3)] * 20,
            [["a", "b"], ["a"], ["b", "c", "a"]] * 20,
            np.random.RandomState(3).normal(size=(80, 3)).tolist(),
            [{"x", "y"}, {"y"}, {"x", "z"}] * 20,
            [{"kind": "a", "score": 1.0}, {"kind": "b"}, {"kind": "a", "score": 2.0}] * 20,
            [1.5, None, 2.5, 3.5] * 20,
        ]
        for data in cases:
            est = get_estimator(data)
            init = initialize(data, est, rng=np.random.RandomState(1), p=1.0)
            model = estimate(data, est, init)
            ll = model.log_density(data[0])
            self.assertTrue(np.isfinite(ll) or ll == -np.inf)

    def test_bayesian_estimators_constructible(self):
        for data in (
            [("a", 1.5, [1.0, 2.0]), ("b", 2.5, [0.5, 1.5])] * 30,
            [["a", "b"], ["a"], ["b", "c", "a"]] * 20,
            list(np.random.RandomState(0).poisson(50, 200)),
        ):
            est = get_estimator(data, use_bstats=True)
            self.assertIsNotNone(
                est.accumulator_factory() if hasattr(est, "accumulator_factory") else est.accumulatorFactory()
            )

    def test_structure_profile_reports_marginals(self):
        data = [["a", "b"], ["a"], ["b", "c", "a"]] * 20
        profile = analyze_structure(data, pairwise=False)
        self.assertIsInstance(profile.recommend(), SequenceEstimator)
        by_path = {u.path: u for u in profile.fields}
        self.assertEqual(by_path[("length",)].recommendation, "integer_categorical")
        self.assertEqual(by_path[("element",)].recommendation, "categorical")
        self.assertIn("integer_categorical", by_path[("length",)].model_scores_bits)
        self.assertTrue(any("length" in line for line in profile.explain()))

    def test_structure_profile_scores_count_models(self):
        data = list(np.random.RandomState(0).poisson(50, 300))
        profile = analyze_structure(data, pairwise=False)
        field = profile.fields[0]
        self.assertEqual(field.recommendation, "poisson")
        self.assertIn("poisson", field.model_scores_bits)
        self.assertIn("gaussian", field.model_scores_bits)
        self.assertLess(field.model_scores_bits["poisson"], field.model_scores_bits["gaussian"])

    def test_structure_profile_pairwise_hint(self):
        data = [("left", 0), ("left", 0), ("right", 1), ("right", 1)] * 50
        profile = analyze_structure(data, pairwise=True, mi_threshold_bits=0.01)
        self.assertGreaterEqual(len(profile.pairwise_hints), 1)
        hint = profile.pairwise_hints[0]
        self.assertEqual({hint.left, hint.right}, {(0,), (1,)})
        self.assertGreater(hint.mi_bits, 0.5)
        self.assertGreater(hint.adjusted_mi_bits, 0.5)
        self.assertGreater(hint.bic_gain_bits, 0.0)

    def test_structure_profile_suppresses_high_cardinality_coincidence(self):
        data = [("id_%d" % i, i) for i in range(30)]
        profile = analyze_structure(data, pairwise=True, mi_threshold_bits=0.0)
        self.assertEqual(profile.pairwise_hints, [])

    def test_structure_profile_pairwise_permutation_p_value(self):
        data = [("left", 0), ("left", 0), ("right", 1), ("right", 1)] * 50
        profile = analyze_structure(
            data,
            pairwise=True,
            mi_threshold_bits=0.01,
            pairwise_permutations=25,
            permutation_alpha=0.10,
            rng=np.random.RandomState(1),
        )
        self.assertGreaterEqual(len(profile.pairwise_hints), 1)
        self.assertIsNotNone(profile.pairwise_hints[0].p_value)
        self.assertLessEqual(profile.pairwise_hints[0].p_value, 0.10)

    def test_structure_profile_recommends_mvn_for_numeric_vectors(self):
        data = np.random.RandomState(4).normal(size=(120, 3)).tolist()
        profile = analyze_structure(data, pairwise=True)
        self.assertIsInstance(profile.recommend(), MultivariateGaussianEstimator)
        self.assertEqual(profile.recommend().dim, 3)
        self.assertEqual(len(profile.fields), 3)

    def test_bayesian_integral_float_record_fields_are_fittable(self):
        from pysp.stats import initialize
        from pysp.stats.bayes.normgamma import NormalGammaDistribution
        from pysp.utils.estimation import fit

        data = [(0.0, float(i % 4) + 0.25) for i in range(80)]
        est = get_estimator(data, use_bstats=True)
        self.assertEqual(type(est.estimators[0]).__name__, "GaussianEstimator")
        # Bayesian path: each Gaussian leaf carries the conjugate default prior.
        self.assertIsInstance(est.estimators[0].prior, NormalGammaDistribution)
        init = initialize(data, est, rng=np.random.RandomState(1), p=1.0)
        model = fit(data, est, prev_estimate=init, rng=np.random.RandomState(1))
        self.assertTrue(np.isfinite(model.log_density(data[0])) or model.log_density(data[0]) == -np.inf)


if __name__ == "__main__":
    unittest.main()
