"""Focused contracts for exact count encoding and fixed-support fitting."""

import unittest

import numpy as np

from mixle.stats import (
    BetaBinomialDistribution,
    BetaDistribution,
    BinomialDataEncoder,
    BinomialDistribution,
    CategoricalDistribution,
    CategoricalEstimator,
    DirichletDistribution,
    GammaDistribution,
    GeometricDistribution,
    IntegerCategoricalDistribution,
    LogSeriesDistribution,
    NegativeBinomialDistribution,
    PoissonDistribution,
)
from mixle.stats.univariate.discrete.categorical import CategoricalAccumulator
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalAccumulator
from mixle.stats.univariate.discrete.integer_uniform_spike import IntegerUniformSpikeAccumulator


class BetaBinomialEvidenceContractTest(unittest.TestCase):
    def test_encoder_and_accumulator_reject_counts_outside_fixed_support(self):
        distribution = BetaBinomialDistribution(2, 1.5, 2.5)
        encoder = distribution.dist_to_encoder()
        accumulator = distribution.estimator().accumulator_factory().make()
        np.testing.assert_array_equal(encoder.seq_encode([0, 1, 2]), np.asarray([0, 1, 2]))

        for invalid in (-1, 1.5, 3, np.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                encoder.seq_encode([invalid])
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                accumulator.update(invalid, 1.0, distribution)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                accumulator.seq_update(np.asarray([invalid]), np.ones(1), distribution)

    def test_encoder_identity_includes_trial_count(self):
        self.assertNotEqual(
            BetaBinomialDistribution(2, 1.0, 1.0).dist_to_encoder(),
            BetaBinomialDistribution(3, 1.0, 1.0).dist_to_encoder(),
        )


class BinomialEvidenceContractTest(unittest.TestCase):
    def test_model_owned_estimators_preserve_shifted_support_and_identity(self):
        distribution = BinomialDistribution(0.4, 10, min_val=5, name="counts", keys="shared")
        statistics = (3.0, 21.0, 7, 7)
        for estimator in (distribution.estimator(), distribution.estimator(pseudo_count=2.0)):
            with self.subTest(pseudo_count=estimator.pseudo_count):
                fitted = estimator.estimate(None, statistics)
                self.assertEqual((fitted.n, fitted.min_val), (10, 5))
                self.assertEqual((fitted.name, fitted.keys), ("counts", "shared"))

    def test_shifted_encoder_accepts_negative_support_and_rejects_outside_it(self):
        encoder = BinomialDistribution(0.5, 2, min_val=-2).dist_to_encoder()
        _, _, values, minimum, maximum = encoder.seq_encode([-2, -1, 0])
        np.testing.assert_array_equal(values, np.asarray([-2, -1, 0], dtype=np.int64))
        self.assertEqual((minimum, maximum), (-2, 0))
        for invalid in (-3, 1, np.inf, 0.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                encoder.seq_encode([invalid])

    def test_generic_encoder_uses_checked_int64_without_value_corruption(self):
        value = 2**40
        encoded = BinomialDataEncoder().seq_encode([value])
        self.assertEqual(encoded[2].dtype, np.dtype(np.int64))
        self.assertEqual(int(encoded[2][0]), value)
        for invalid in (2**63, -(2**63) - 1, np.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                BinomialDataEncoder().seq_encode([invalid])

    def test_zero_weight_does_not_expand_inferred_support(self):
        accumulator = BinomialDataEncoder().seq_encode([2, 100])
        estimator = BinomialDistribution(0.5, 2).estimator()
        learned = estimator.accumulator_factory().make()
        learned.seq_update(accumulator, np.asarray([1.0, 0.0]), None)
        self.assertEqual(learned.value(), (1.0, 2.0, 0, 2))


class DiscreteVariationalAndCountContractTest(unittest.TestCase):
    def test_geometric_boundary_map_falls_back_to_valid_posterior_mean(self):
        fitted = (
            GeometricDistribution(0.5, prior=BetaDistribution(0.5, 2.0))
            .estimator()
            .estimate(
                None,
                (0.0, 0.0),
            )
        )
        self.assertAlmostEqual(fitted.p, 0.2)
        self.assertGreater(fitted.p, 0.0)

    def test_variational_scorers_reject_fractional_and_nonfinite_counts(self):
        distributions_and_encoded = (
            (
                GeometricDistribution(0.5, prior=BetaDistribution(2.0, 3.0)),
                np.asarray([1.0, 1.5, np.inf]),
            ),
            (
                PoissonDistribution(2.0, prior=GammaDistribution(2.0, 1.0)),
                (
                    np.asarray([0.0, 0.5, np.inf]),
                    np.asarray([0.0, 0.0, np.inf]),
                ),
            ),
            (
                IntegerCategoricalDistribution(
                    0,
                    np.asarray([0.5, 0.5]),
                    prior=DirichletDistribution(np.asarray([2.0, 3.0])),
                ),
                np.asarray([0.0, 0.5, np.inf]),
            ),
        )
        for distribution, encoded in distributions_and_encoded:
            with self.subTest(distribution=distribution):
                self.assertEqual(distribution.expected_log_density(0.5), -np.inf)
                scored = distribution.seq_expected_log_density(encoded)
                self.assertEqual(scored[1], -np.inf)
                self.assertEqual(scored[2], -np.inf)

    def test_count_encoders_reject_positive_infinity(self):
        distributions = (
            GeometricDistribution(0.5),
            NegativeBinomialDistribution(2.0, 0.5),
            PoissonDistribution(2.0),
        )
        for distribution in distributions:
            with self.subTest(distribution=distribution), self.assertRaises(ValueError):
                distribution.dist_to_encoder().seq_encode([np.inf])

    def test_negative_binomial_rejects_nan_probability(self):
        with self.assertRaises(ValueError):
            NegativeBinomialDistribution(2.0, np.nan)

    def test_log_series_infinity_is_off_support_not_an_exception(self):
        self.assertEqual(LogSeriesDistribution(0.5).log_density(np.inf), -np.inf)


class CategoricalEvidenceContractTest(unittest.TestCase):
    def test_mixed_type_labels_remain_distinct(self):
        distribution = CategoricalDistribution({1: 0.9, "1": 0.1})
        encoded = distribution.dist_to_encoder().seq_encode([1, "1"])
        np.testing.assert_array_equal(encoded[0], np.asarray([0, 1]))
        self.assertEqual(encoded[1].tolist(), [1, "1"])
        np.testing.assert_allclose(
            distribution.seq_log_density(encoded),
            np.log(np.asarray([0.9, 0.1])),
        )

    def test_heterogeneous_labels_have_deterministic_stringification(self):
        rendered = str(CategoricalDistribution({1: 0.5, "a": 0.5}))
        self.assertIn("1: 0.5", rendered)
        self.assertIn("'a': 0.5", rendered)

    def test_prior_statistics_without_multiplier_have_defined_unit_weight(self):
        estimator = CategoricalEstimator(suff_stat={"a": 1.0})
        self.assertEqual(estimator.pseudo_count, 1.0)
        fitted = estimator.estimate(None, {"b": 1.0})
        self.assertEqual(fitted.pmap, {"a": 0.5, "b": 0.5})

    def test_zero_weight_does_not_create_or_expand_categories(self):
        categorical = CategoricalAccumulator()
        categorical.update("ignored", 0.0, None)
        categorical.seq_update(
            CategoricalDistribution({"used": 0.5, "ignored": 0.5}).dist_to_encoder().seq_encode(["used", "ignored"]),
            np.asarray([1.0, 0.0]),
            None,
        )
        self.assertEqual(categorical.value(), {"used": 1.0})

        for accumulator in (
            IntegerCategoricalAccumulator(),
            IntegerUniformSpikeAccumulator(None, None),
        ):
            with self.subTest(accumulator=accumulator):
                accumulator.update(2, 1.0, None)
                accumulator.update(100, 0.0, None)
                self.assertEqual((accumulator.min_val, accumulator.max_val), (2, 2))
                np.testing.assert_array_equal(accumulator.count_vec, np.asarray([1.0]))


if __name__ == "__main__":
    unittest.main()
