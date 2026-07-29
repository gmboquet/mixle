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
    IntegerCategoricalEstimator,
    IntegerUniformSpikeDistribution,
    IntegerUniformSpikeEstimator,
    LogSeriesDistribution,
    NegativeBinomialDistribution,
    PointMassDistribution,
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
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                encoder.seq_encode([invalid])
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                accumulator.update(invalid, 1.0, distribution)
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
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
            with self.subTest(pseudo_count=repr(estimator.pseudo_count)):
                fitted = estimator.estimate(None, statistics)
                self.assertEqual((fitted.n, fitted.min_val), (10, 5))
                self.assertEqual((fitted.name, fitted.keys), ("counts", "shared"))

    def test_shifted_encoder_keeps_its_support_and_scores_outside_it_as_impossible(self):
        """A shifted support is preserved through encoding, and a count outside it scores -inf.

        Being an exact integer is a type contract -- a fractional or non-finite value is not an
        observation of a count family at all -- but being inside the support is a probability
        question. log_density has always answered it with -inf, so rejecting those counts at
        encode time left seq_log_density unable to score what the scalar path scores fine.
        """
        dist = BinomialDistribution(0.5, 2, min_val=-2)
        encoder = dist.dist_to_encoder()
        _, _, values, minimum, maximum = encoder.seq_encode([-2, -1, 0])
        np.testing.assert_array_equal(values, np.asarray([-2, -1, 0], dtype=np.int64))
        self.assertEqual((minimum, maximum), (-2, 0))

        outside = [-3, 1]
        scored = np.asarray(dist.seq_log_density(encoder.seq_encode(outside)))
        np.testing.assert_array_equal(np.isneginf(scored), [True, True])
        np.testing.assert_allclose(scored, [dist.log_density(v) for v in outside])

        for invalid in (np.inf, 0.5):  # not a count at all, still rejected
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                encoder.seq_encode([invalid])

    def test_generic_encoder_uses_checked_int64_without_value_corruption(self):
        value = 2**40
        encoded = BinomialDataEncoder().seq_encode([value])
        self.assertEqual(encoded[2].dtype, np.dtype(np.int64))
        self.assertEqual(int(encoded[2][0]), value)
        # subTest labels are repr()'d, not passed raw: xdist ships every subtest report through
        # execnet, whose integer wire format is 32-bit, so a raw 2**63 label crashes the report
        # channel (struct.error) even when the assertion itself passes. Same reason objects are
        # labelled by repr elsewhere in this file.
        for invalid in (2**63, -(2**63) - 1, np.inf):
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
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
            with self.subTest(distribution=repr(distribution)):
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
            with self.subTest(distribution=repr(distribution)), self.assertRaises(ValueError):
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
            with self.subTest(accumulator=type(accumulator).__name__):
                accumulator.update(2, 1.0, None)
                accumulator.update(100, 0.0, None)
                self.assertEqual((accumulator.min_val, accumulator.max_val), (2, 2))
                np.testing.assert_array_equal(accumulator.count_vec, np.asarray([1.0]))


class DiscreteProbabilityLawContractTest(unittest.TestCase):
    def test_empty_categorical_fit_widens_a_declared_support_and_otherwise_stays_empty(self):
        """A zero-evidence fit is a state EM legitimately reaches, so it estimates rather than raises.

        A mixture or HMM component can win zero responsibility for an iteration and recover on the
        next one; aborting the whole fit there would lose a model the very next E-step repairs. So
        the contract is: a declared support (prior or ``suff_stat``) is widened into with zero
        counts, and with no declared support the honest estimate is the empty categorical -- which
        reports itself unnormalized, deferring refusal to the point of use.
        """
        fitted = CategoricalEstimator(suff_stat={"known": 1.0}).estimate(None, {})
        self.assertEqual(fitted.pmap, {"known": 1.0})

        empty = CategoricalEstimator().estimate(None, {})
        self.assertEqual(empty.pmap, {})
        self.assertFalse(empty.is_normalized_probability)

    def test_unnormalized_categorical_constructs_but_reports_itself_unnormalized(self):
        # An unnormalized pmap (with or without a positive fallback) is a legal construction by
        # design -- see CategoricalDistribution.__init__ -- so the contract is that it constructs
        # and then tells the truth about itself, not that it is rejected.
        for kwargs in (
            {"pmap": {"a": 0.5}},
            {"pmap": {"a": 0.5}, "default_value": 0.1},
            {"pmap": {}},
        ):
            with self.subTest(kwargs=repr(kwargs)):
                distribution = CategoricalDistribution(**kwargs)
                self.assertFalse(distribution.scoring_only)
                self.assertFalse(distribution.is_normalized_probability)

    def test_explicit_scoring_only_categorical_is_not_sampleable(self):
        scorer = CategoricalDistribution(
            {"a": 0.5},
            default_value=0.1,
            scoring_only=True,
        )
        self.assertTrue(scorer.scoring_only)
        self.assertFalse(scorer.is_normalized_probability)
        with self.assertRaises(ValueError):
            scorer.sampler()

    def test_integer_categorical_requires_exact_origin_and_valid_weights(self):
        for origin, probabilities in (
            (0.5, [0.5, 0.5]),
            (0, [-0.5, 1.5]),
            (0, [float("nan"), 0.5]),
        ):
            with self.subTest(origin=repr(origin), probabilities=repr(probabilities)), self.assertRaises(ValueError):
                IntegerCategoricalDistribution(origin, probabilities)

        # Mirroring CategoricalDistribution.pmap, an unnormalized (or empty, or all-zero) weight
        # vector is deliberately still constructible; only finiteness/non-negativity is enforced.
        for probabilities in ([], [0.0, 0.0], [0.8, 0.8]):
            with self.subTest(probabilities=repr(probabilities)):
                distribution = IntegerCategoricalDistribution(0, probabilities)
                self.assertEqual(distribution.num_vals, len(probabilities))

    def test_integer_categorical_scalar_and_batch_use_exact_integrality(self):
        distribution = IntegerCategoricalDistribution(1, [0.5, 0.5])
        fractional = 1.0 + 5.0e-10
        self.assertEqual(distribution.density(fractional), 0.0)
        self.assertEqual(distribution.log_density(fractional), -np.inf)
        self.assertEqual(distribution.seq_log_density(np.asarray([fractional]))[0], -np.inf)
        with self.assertRaises(ValueError):
            distribution.dist_to_encoder().seq_encode([fractional])

    def test_integer_categorical_empty_batches_and_zero_count_fits_are_defined(self):
        accumulator = IntegerCategoricalEstimator(min_val=2, max_val=3).accumulator_factory().make()
        accumulator.seq_update(np.asarray([], dtype=np.int64), np.asarray([], dtype=float), None)
        fitted = IntegerCategoricalEstimator(min_val=2, max_val=3).estimate(
            None,
            accumulator.value(),
        )
        self.assertEqual(fitted.min_val, 2)
        np.testing.assert_allclose(fitted.p_vec, [0.5, 0.5])
        with self.assertRaises(ValueError):
            IntegerCategoricalEstimator().estimate(None, None)

    def test_integer_count_accumulator_arrays_are_owned(self):
        for accumulator in (
            IntegerCategoricalAccumulator(),
            IntegerUniformSpikeAccumulator(None, None),
        ):
            with self.subTest(accumulator=type(accumulator).__name__):
                donor = np.asarray([1.0, 2.0])
                accumulator.combine((4, donor))
                donor[0] = 99.0
                self.assertEqual(accumulator.value()[1][0], 1.0)

                exposed = accumulator.value()[1]
                exposed[0] = 88.0
                self.assertEqual(accumulator.value()[1][0], 1.0)

                replacement = np.asarray([3.0, 4.0])
                accumulator.from_value((7, replacement))
                replacement[0] = 77.0
                self.assertEqual(accumulator.value()[1][0], 3.0)

    def test_integer_uniform_spike_requires_a_valid_integer_probability_law(self):
        invalid = (
            (0.5, 2, 0.5, 0),
            (0, 1.5, 0.5, 0),
            (0, 0, 0.5, 0),
            (0, 1, 0.2, 0),
        )
        for k, size, probability, minimum in invalid:
            with self.subTest(values=repr((k, size, probability, minimum))), self.assertRaises(ValueError):
                IntegerUniformSpikeDistribution(k, size, probability, minimum)

    def test_integer_uniform_spike_rejects_fractional_states_in_every_path(self):
        distribution = IntegerUniformSpikeDistribution(1, 3, 0.5, min_val=0)
        self.assertEqual(distribution.log_density(1.5), -np.inf)
        self.assertEqual(distribution.seq_log_density(np.asarray([1.5]))[0], -np.inf)
        with self.assertRaises(ValueError):
            distribution.dist_to_encoder().seq_encode([1.5])

    def test_integer_uniform_spike_empty_and_prior_support_contracts(self):
        fitted = IntegerUniformSpikeEstimator(min_val=0, max_val=1).estimate(
            None,
            (0, np.zeros(2)),
        )
        self.assertEqual((fitted.k, fitted.p), (0, 0.5))
        with self.assertRaises(ValueError):
            IntegerUniformSpikeEstimator(
                min_val=0,
                max_val=1,
                pseudo_count=1.0,
                suff_stat=(2, None),
            )
        with self.assertRaises(ValueError):
            IntegerUniformSpikeEstimator(min_val=0, max_val=1).estimate(
                None,
                (0, np.asarray([1.0, -1.0])),
            )

    def test_point_mass_owns_and_copies_mutable_atoms(self):
        source = {"values": [1]}
        distribution = PointMassDistribution(source)
        source["values"].append(2)
        self.assertEqual(distribution.value, {"values": [1]})

        draws = distribution.sampler().sample(2)
        draws[0]["values"].append(3)
        self.assertEqual(draws[1], {"values": [1]})
        self.assertEqual(distribution.value, {"values": [1]})


if __name__ == "__main__":
    unittest.main()
