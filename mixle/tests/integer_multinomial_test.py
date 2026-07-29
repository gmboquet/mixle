"""Tests for IntegerMultinomialDistribution's log_density/seq_log_density agreement, in particular
with a real len_dist (trial-count distribution) set. log_density previously omitted the len_dist term
entirely -- seq_log_density and the sibling MultinomialDistribution.log_density both include it -- so
the two disagreed by exactly the trial-count term whenever len_dist was not the default NullDistribution.
"""

import unittest

import numpy as np
from scipy.special import gammaln

from mixle.engines import NUMPY_ENGINE
from mixle.stats.multivariate.integer_multinomial import (
    IntegerMultinomialAccumulator,
    IntegerMultinomialDistribution,
    IntegerMultinomialEstimator,
)
from mixle.stats.univariate.discrete.poisson import PoissonDistribution

DATA = [
    [(0, 2.0), (2, 1.0)],
    [(1, 3.0)],
    [(0, 1.0), (1, 1.0), (2, 1.0)],
    [(0, 5.0)],
]


class IntegerMultinomialLenDistTestCase(unittest.TestCase):
    def test_log_density_includes_len_dist_term(self):
        d = IntegerMultinomialDistribution(min_val=0, p_vec=[0.2, 0.3, 0.5], len_dist=PoissonDistribution(3.0))
        for x in DATA:
            with self.subTest(x=repr(x)):
                total = sum(cnt for _, cnt in x)
                category_term = sum(d.log_p_vec[v] * cnt for v, cnt in x)
                coefficient = gammaln(total + 1.0) - sum(gammaln(cnt + 1.0) for _, cnt in x)
                expected = coefficient + category_term + PoissonDistribution(3.0).log_density(total)
                self.assertAlmostEqual(d.log_density(x), expected, places=10)

    def test_log_density_matches_seq_log_density_with_len_dist(self):
        d = IntegerMultinomialDistribution(min_val=0, p_vec=[0.2, 0.3, 0.5], len_dist=PoissonDistribution(3.0))
        enc = d.dist_to_encoder().seq_encode(DATA)
        seq = np.asarray(d.seq_log_density(enc))
        scalar = np.array([d.log_density(x) for x in DATA])
        np.testing.assert_allclose(seq, scalar, atol=1e-10)

    def test_log_density_matches_seq_log_density_without_len_dist(self):
        # sanity: the no-len_dist (default NullDistribution) path is unaffected by the fix.
        d = IntegerMultinomialDistribution(min_val=0, p_vec=[0.2, 0.3, 0.5])
        enc = d.dist_to_encoder().seq_encode(DATA)
        seq = np.asarray(d.seq_log_density(enc))
        scalar = np.array([d.log_density(x) for x in DATA])
        np.testing.assert_allclose(seq, scalar, atol=1e-10)

    def test_len_dist_term_is_nonzero_when_set(self):
        # guards against a fix that adds the term but has it silently evaluate to 0 (e.g. wrong count).
        d_null = IntegerMultinomialDistribution(min_val=0, p_vec=[0.2, 0.3, 0.5])
        d_poisson = IntegerMultinomialDistribution(min_val=0, p_vec=[0.2, 0.3, 0.5], len_dist=PoissonDistribution(3.0))
        x = DATA[0]
        self.assertNotAlmostEqual(d_null.log_density(x), d_poisson.log_density(x), places=6)


class IntegerMultinomialZeroProbZeroCountTestCase(unittest.TestCase):
    """seq_log_density previously computed (-inf) * 0 = NaN for an in-support category k with
    p_vec[k] == 0 whenever an observation contained (k, count=0). The scalar log_density path
    already special-cased cnt == 0 to avoid this; seq_log_density now matches it.
    """

    ZERO_PROB_DATA = [
        [(0, 0.0), (1, 2.0), (2, 1.0)],  # zero-prob category present with count=0: must not be NaN
        [(1, 3.0)],  # zero-prob category absent entirely
        [(0, 1.0), (1, 1.0), (2, 1.0)],  # zero-prob category present with count>0: must be -inf
        [(0, 0.0)],  # observation consisting solely of a zero-count, zero-prob entry
    ]

    def _dist(self):
        return IntegerMultinomialDistribution(min_val=0, p_vec=[0.0, 0.3, 0.7])

    def test_seq_log_density_no_nan_for_zero_count_zero_prob_entry(self):
        d = self._dist()
        enc = d.dist_to_encoder().seq_encode(self.ZERO_PROB_DATA)
        seq = np.asarray(d.seq_log_density(enc))
        self.assertFalse(np.any(np.isnan(seq)), msg=f"seq_log_density produced NaN: {seq}")

    def test_seq_log_density_matches_scalar_with_zero_prob_category(self):
        d = self._dist()
        enc = d.dist_to_encoder().seq_encode(self.ZERO_PROB_DATA)
        seq = np.asarray(d.seq_log_density(enc))
        scalar = np.array([d.log_density(x) for x in self.ZERO_PROB_DATA])
        np.testing.assert_allclose(seq, scalar, atol=1e-10)

    def test_seq_log_density_still_neg_inf_for_nonzero_count_zero_prob_entry(self):
        # Guards against an overcorrection that zeroes out every zero-prob-category contribution
        # regardless of count -- it must remain -inf when the count is actually nonzero.
        d = self._dist()
        enc = d.dist_to_encoder().seq_encode(self.ZERO_PROB_DATA)
        seq = np.asarray(d.seq_log_density(enc))
        self.assertEqual(seq[2], -np.inf)


class IntegerMultinomialAuditContractTestCase(unittest.TestCase):
    def test_constructor_requires_owned_probability_simplex(self):
        for invalid in ([], [0.8, 0.8], [-0.1, 1.1], [np.nan, 1.0]):
            with self.assertRaises(ValueError):
                IntegerMultinomialDistribution(0, invalid)
        probabilities = np.array([0.4, 0.6])
        dist = IntegerMultinomialDistribution(0, probabilities)
        before = dist.log_density([(0, 1), (1, 1)])
        probabilities[:] = [1.0, 0.0]
        self.assertEqual(dist.log_density([(0, 1), (1, 1)]), before)
        with self.assertRaises(ValueError):
            dist.p_vec[0] = 1.0

    def test_score_is_normalized_multinomial_mass(self):
        dist = IntegerMultinomialDistribution(0, [0.5, 0.5])
        self.assertAlmostEqual(dist.density([(0, 1), (1, 1)]), 0.5)
        self.assertAlmostEqual(dist.density([(0, 2)]), 0.25)
        self.assertAlmostEqual(dist.density([(1, 2)]), 0.25)

    def test_sparse_bag_is_canonical_and_integral(self):
        dist = IntegerMultinomialDistribution(0, [0.5, 0.5])
        self.assertEqual(
            dist.log_density([(0, 1), (0, 1), (1, 1)]),
            dist.log_density([(0, 2), (1, 1)]),
        )
        for invalid in ([(0.5, 1)], [(0, 1.5)], [(0, -1)], [(0, np.nan)]):
            with self.assertRaises((TypeError, ValueError)):
                dist.log_density(invalid)

    def test_encoder_does_not_truncate_and_uses_int64(self):
        encoder = IntegerMultinomialDistribution(0, [0.5, 0.5]).dist_to_encoder()
        for invalid in (
            [[(1.9, 1)]],
            [[(1, 1.9)]],
            [[(1, -1)]],
            [[(1, np.nan)]],
        ):
            with self.assertRaises((TypeError, ValueError)):
                encoder.seq_encode(invalid)
        encoded = encoder.seq_encode([[(0, 1), (0, 2)], []])
        self.assertEqual(encoded[1].dtype, np.int64)
        self.assertEqual(encoded[2].dtype, np.int64)
        self.assertEqual(encoded[3].dtype, np.int64)
        np.testing.assert_array_equal(encoded[2], [3])

    def test_numpy_and_generated_scores_agree_on_zero_counts(self):
        dist = IntegerMultinomialDistribution(0, [0.0, 1.0])
        encoded = dist.dist_to_encoder().seq_encode([[(0, 0), (1, 2)], [(10, 0)], [(0, 1)]])
        expected = dist.seq_log_density(encoded)
        actual = np.asarray(dist.backend_seq_log_density(encoded, NUMPY_ENGINE))
        np.testing.assert_allclose(actual, expected)
        self.assertTrue(np.isfinite(actual[0]))
        self.assertTrue(np.isfinite(actual[1]))
        self.assertEqual(actual[2], -np.inf)

    def test_stacked_scores_include_the_same_base_measure(self):
        first = IntegerMultinomialDistribution(0, [0.25, 0.75])
        second = IntegerMultinomialDistribution(0, [0.6, 0.4])
        encoded = first.dist_to_encoder().seq_encode([[(0, 1), (1, 1)], [(1, 2)], []])
        params = IntegerMultinomialDistribution.backend_stacked_params(
            [first, second],
            NUMPY_ENGINE,
        )
        actual = np.asarray(
            IntegerMultinomialDistribution.backend_stacked_log_density(
                encoded,
                params,
                NUMPY_ENGINE,
            )
        )
        expected = np.column_stack((first.seq_log_density(encoded), second.seq_log_density(encoded)))
        np.testing.assert_allclose(actual, expected)

    def test_empty_batches_accumulate_with_fixed_support(self):
        estimator = IntegerMultinomialDistribution(3, [0.25, 0.75]).estimator()
        accumulator = estimator.accumulator_factory().make()
        encoded = accumulator.acc_to_encoder().seq_encode([[], []])
        accumulator.seq_update(encoded, np.ones(2), None)
        fitted = estimator.estimate(None, accumulator.value())
        self.assertEqual(fitted.min_val, 3)
        np.testing.assert_allclose(fitted.p_vec, [0.5, 0.5])

    def test_statistics_are_copied_at_every_boundary(self):
        counts = np.array([1.0, 2.0])
        accumulator = IntegerMultinomialAccumulator().from_value((0, counts, None))
        counts[0] = 99.0
        self.assertEqual(accumulator.count_vec[0], 1.0)
        serialized = accumulator.value()
        serialized[1][0] = 88.0
        self.assertEqual(accumulator.count_vec[0], 1.0)
        recipient = IntegerMultinomialAccumulator()
        recipient.combine(accumulator.value())
        accumulator.count_vec[0] = 77.0
        self.assertEqual(recipient.count_vec[0], 1.0)

    def test_fixed_support_is_not_expanded(self):
        estimator = IntegerMultinomialEstimator(min_val=0, max_val=1)
        accumulator = estimator.accumulator_factory().make()
        with self.assertRaises(ValueError):
            accumulator.update([(2, 1)], 1.0, None)
        with self.assertRaises(ValueError):
            estimator.estimate(None, (0, np.ones(3), None))

    def test_estimator_controls_and_empty_learned_support_fail_closed(self):
        for kwargs in (
            {"min_val": 2, "max_val": 1},
            {"max_val": 1},
            {"pseudo_count": -1.0},
            {"pseudo_count": np.nan},
            {"pseudo_count": 1.0, "suff_stat": (0, [0.0, 0.0])},
        ):
            with self.assertRaises((TypeError, ValueError)):
                IntegerMultinomialEstimator(**kwargs)
        estimator = IntegerMultinomialEstimator()
        accumulator = estimator.accumulator_factory().make()
        encoded = accumulator.acc_to_encoder().seq_encode([[]])
        accumulator.seq_update(encoded, np.ones(1), None)
        with self.assertRaises(ValueError, msg="support"):
            estimator.estimate(None, accumulator.value())

    def test_min_val_alone_pins_the_floor_and_learns_the_ceiling(self):
        # min_val without max_val is a supported configuration: the floor is fixed, the ceiling
        # grows with the data. Categories below the floor are still rejected.
        estimator = IntegerMultinomialEstimator(min_val=0, pseudo_count=1.0)
        accumulator = estimator.accumulator_factory().make()
        for datum in ([(3, 2)], [(5, 1)], [(4, 3)]):
            accumulator.update(datum, 1.0, None)
        distribution = estimator.estimate(None, accumulator.value())
        self.assertEqual(distribution.min_val, 0)
        self.assertEqual(distribution.max_val, 5)
        self.assertEqual(len(distribution.p_vec), 6)

        below_floor = estimator.accumulator_factory().make()
        with self.assertRaises(ValueError):
            below_floor.update([(-1, 1)], 1.0, None)

        seq = estimator.accumulator_factory().make()
        encoded = seq.acc_to_encoder().seq_encode([[(3, 2)], [(5, 1)]])
        seq.seq_update(encoded, np.ones(2), None)
        self.assertEqual(seq.value()[0], 0)
        np.testing.assert_allclose(seq.value()[1], [0.0, 0.0, 0.0, 2.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
