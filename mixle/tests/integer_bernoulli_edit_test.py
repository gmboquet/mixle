"""Tests for IntegerBernoulliEditDistribution's edit-transition log-density, including the "forced
transition" case: a value with p_mat(present | missing) == 1.0 exactly (so p_mat(missing | missing) == 0,
log = -inf). log_nsum/log_dvec's baseline-plus-delta vectorization trick relies on ordinary finite-
arithmetic cancellation and breaks silently for such values unless they are tracked and special-cased --
previously producing a wrongly-finite log_density for an impossible observation (a forced value observed
to stay missing) and a wrongly-infinite one for the correct/expected observation (the forced value
actually becoming present), instead of -inf and a proper finite value respectively.
"""

import itertools
import math
import unittest

import numpy as np

from mixle.engines import NumpyEngine
from mixle.stats.combinator.null_dist import NullDistribution
from mixle.stats.sets.integer_bernoulli_edit import (
    IntegerBernoulliEditAccumulator,
    IntegerBernoulliEditDistribution,
    IntegerBernoulliEditEstimator,
    IntegerBernoulliEditFitError,
)
from mixle.stats.sets.integer_bernoulli_set import (
    IntegerBernoulliSetDistribution,
    IntegerBernoulliSetEstimator,
)


def _all_subsets(n):
    for r in range(n + 1):
        yield from (list(c) for c in itertools.combinations(range(n), r))


class IntegerBernoulliEditBruteForceTestCase(unittest.TestCase):
    def setUp(self):
        # No forced values -- every p_mat(present|missing)/p_mat(present|present) strictly inside (0, 1).
        self.num_vals = 3
        self.dist = IntegerBernoulliEditDistribution(
            [
                (math.log(0.3), math.log(0.6)),
                (math.log(0.4), math.log(0.8)),
                (math.log(0.1), math.log(0.5)),
            ]
        )

    def test_conditional_density_sums_to_one_over_all_next_sets(self):
        for x0 in _all_subsets(self.num_vals):
            total = sum(
                np.exp(self.dist.conditional_log_density(x0, x1))
                for x1 in _all_subsets(self.num_vals)
            )
            self.assertAlmostEqual(total, 1.0, places=10)

    def test_default_joint_density_is_normalized(self):
        total = sum(
            self.dist.density((x0, x1))
            for x0 in _all_subsets(self.num_vals)
            for x1 in _all_subsets(self.num_vals)
        )
        self.assertAlmostEqual(total, 1.0, places=10)
        self.assertTrue(
            all(
                self.dist.log_density((x0, [])) == -np.inf
                for x0 in _all_subsets(self.num_vals)
                if x0
            )
        )

    def test_seq_log_density_matches_scalar_log_density(self):
        data = [(x0, x1) for x0 in _all_subsets(self.num_vals) for x1 in _all_subsets(self.num_vals)]
        enc = self.dist.dist_to_encoder().seq_encode(data)
        seq = self.dist.seq_log_density(enc)
        scalar = np.array([self.dist.log_density(d) for d in data])
        np.testing.assert_allclose(seq, scalar, atol=1e-10)

    def test_empty_edit_rows_score_consistently(self):
        data = [([], []), ([], [0])]
        encoded = self.dist.dist_to_encoder().seq_encode(data)
        expected = np.asarray([self.dist.log_density(row) for row in data])
        np.testing.assert_array_equal(self.dist.seq_log_density(encoded), expected)
        np.testing.assert_array_equal(
            NumpyEngine().to_numpy(
                self.dist.backend_seq_log_density(encoded, NumpyEngine())
            ),
            expected,
        )

    def test_default_enumerator_is_a_normalized_joint_law(self):
        items = list(self.dist.enumerator())
        self.assertAlmostEqual(sum(np.exp(score) for _, score in items), 1.0)
        self.assertTrue(all(previous == [] for (previous, _), _ in items))


class IntegerBernoulliEditContractTestCase(unittest.TestCase):
    def test_constructor_validates_stochastic_kernel_geometry(self):
        invalid = (
            np.zeros(2),
            np.zeros((2, 3)),
            np.asarray([[np.nan, np.log(0.5)]]),
            np.asarray([[0.1, np.log(0.5)]]),
            np.zeros((1, 4)),
        )
        for kernel in invalid:
            with self.subTest(kernel=kernel), self.assertRaises(ValueError):
                IntegerBernoulliEditDistribution(kernel)
        with self.assertRaises(ValueError):
            IntegerBernoulliEditDistribution(
                np.log([[0.2, 0.8, 0.2, 0.8]])
            )

    def test_constructor_copies_freezes_and_canonicalizes_kernel(self):
        source = np.log([[0.2, 0.7], [0.8, 0.3]])
        dist = IntegerBernoulliEditDistribution(source)
        source[:] = np.log(0.5)
        np.testing.assert_allclose(
            np.exp(dist.log_edit_pmat[:, 0])
            + np.exp(dist.log_edit_pmat[:, 2]),
            1.0,
        )
        np.testing.assert_allclose(
            np.exp(dist.log_edit_pmat[:, 1])
            + np.exp(dist.log_edit_pmat[:, 3]),
            1.0,
        )
        with self.assertRaises(ValueError):
            dist.log_edit_pmat[0, 0] = 0.0

    def test_neutral_initial_child_is_rejected(self):
        with self.assertRaises(ValueError):
            IntegerBernoulliEditDistribution(
                np.log([[0.2, 0.7]]),
                init_dist=NullDistribution(),
            )

    def test_raw_and_encoded_events_require_unique_supported_integers(self):
        dist = IntegerBernoulliEditDistribution(np.log([[0.2, 0.7], [0.8, 0.3]]))
        encoder = dist.dist_to_encoder()
        for observation in (
            ([0.9], []),
            ([-1], []),
            ([2], []),
            ([0, 0], []),
            ([], [1, 1]),
        ):
            with self.subTest(observation=observation):
                with self.assertRaises((TypeError, ValueError)):
                    dist.log_density(observation)
                with self.assertRaises((TypeError, ValueError)):
                    encoder.seq_encode([observation])

    def test_forced_transitions_are_impossible_in_every_backend(self):
        dists = (
            IntegerBernoulliEditDistribution(np.asarray([[0.0, np.log(0.8)]])),
            IntegerBernoulliEditDistribution(np.asarray([[0.0, np.log(0.3)]])),
        )
        encoded = dists[0].dist_to_encoder().seq_encode([([], [])])
        scalar = dists[0].log_density(([], []))
        self.assertEqual(scalar, -np.inf)
        self.assertEqual(dists[0].seq_log_density(encoded)[0], -np.inf)
        self.assertEqual(
            NumpyEngine().to_numpy(
                dists[0].backend_seq_log_density(encoded, NumpyEngine())
            )[0],
            -np.inf,
        )
        params = IntegerBernoulliEditDistribution.backend_stacked_params(
            dists,
            NumpyEngine(),
        )
        stacked = NumpyEngine().to_numpy(
            IntegerBernoulliEditDistribution.backend_stacked_log_density(
                encoded,
                params,
                NumpyEngine(),
            )
        )
        np.testing.assert_array_equal(stacked, [[-np.inf, -np.inf]])

    def test_distribution_estimator_preserves_nested_model_and_identity(self):
        initial = IntegerBernoulliSetDistribution(np.log([0.3, 0.6]))
        dist = IntegerBernoulliEditDistribution(
            np.log([[0.2, 0.7], [0.8, 0.3]]),
            init_dist=initial,
            name="edits",
            keys="shared",
        )
        estimator = dist.estimator(pseudo_count=2.0)
        self.assertIsInstance(estimator.init_est, IntegerBernoulliSetEstimator)
        self.assertEqual(estimator.keys, "shared")
        np.testing.assert_allclose(
            estimator.suff_stat,
            np.exp(dist.log_edit_pmat),
        )
        accumulator = estimator.accumulator_factory().make()
        accumulator.update(([0], [0, 1]), 1.0, dist)
        fitted = estimator.estimate(None, accumulator.value())
        self.assertIsInstance(
            fitted.init_dist,
            IntegerBernoulliSetDistribution,
        )
        self.assertEqual(fitted.name, "edits")
        self.assertEqual(fitted.keys, "shared")

    def test_accumulator_validates_atomically_and_copies_state(self):
        acc = IntegerBernoulliEditAccumulator(2)
        acc.update(([0], [1]), 2.0, None)
        before = acc.value()
        for observation, weight in (
            (([0, 0], []), 1.0),
            (([-1], []), 1.0),
            (([2], []), 1.0),
            (([], [1]), -1.0),
            (([], [1]), np.nan),
        ):
            with self.subTest(observation=observation, weight=weight):
                with self.assertRaises((TypeError, ValueError)):
                    acc.update(observation, weight, None)
                np.testing.assert_array_equal(acc.value()[0], before[0])
                self.assertEqual(acc.value()[1], before[1])
        restored = acc.value()
        acc.from_value(restored)
        restored[0][0, 0] = 99.0
        self.assertNotEqual(acc.value()[0][0, 0], 99.0)

    def test_estimator_rejects_invalid_configuration_and_statistics(self):
        for min_prob in (-0.1, 0.51, np.nan):
            with self.subTest(min_prob=min_prob), self.assertRaises(ValueError):
                IntegerBernoulliEditEstimator(2, min_prob=min_prob)
        with self.assertRaises(ValueError):
            IntegerBernoulliEditEstimator(2, pseudo_count=-1.0)
        with self.assertRaises(ValueError):
            IntegerBernoulliEditEstimator(
                2,
                pseudo_count=1.0,
                suff_stat=np.full((2, 4), 0.8),
            )
        estimator = IntegerBernoulliEditEstimator(2)
        with self.assertRaises(IntegerBernoulliEditFitError):
            estimator.estimate(
                None,
                (np.zeros((2, 3)), 0.0, (np.zeros(2), 0.0)),
            )
        invalid_counts = np.zeros((2, 3))
        invalid_counts[0] = [0.8, 0.3, 0.4]
        with self.assertRaises(ValueError):
            estimator.estimate(
                None,
                (invalid_counts, 1.0, (np.zeros(2), 1.0)),
            )

    def test_fitted_transition_pairs_remain_normalized_after_flooring(self):
        estimator = IntegerBernoulliEditEstimator(1, min_prob=0.1)
        fitted = estimator.estimate(
            None,
            (
                np.asarray([[0.0, 0.0, 1.0]]),
                1.0,
                (np.asarray([1.0]), 1.0),
            ),
        )
        probabilities = np.exp(fitted.log_edit_pmat)
        np.testing.assert_allclose(probabilities[:, 0] + probabilities[:, 2], 1.0)
        np.testing.assert_allclose(probabilities[:, 1] + probabilities[:, 3], 1.0)


class IntegerBernoulliEditForcedTransitionTestCase(unittest.TestCase):
    def setUp(self):
        # value 0 is forced: p_mat(present | missing) == 1.0 exactly -> p_mat(missing | missing) == 0
        # (log = -inf). value 1 is an ordinary, non-forced value.
        self.p_present_present_0 = 0.9
        self.p_present_missing_1 = 0.4
        self.dist = IntegerBernoulliEditDistribution(
            [(0.0, math.log(self.p_present_present_0)), (math.log(self.p_present_missing_1), math.log(0.8))]
        )

    def test_forced_value_left_missing_in_both_sets_is_impossible(self):
        self.assertEqual(self.dist.conditional_log_density([], []), -np.inf)

    def test_forced_value_present_in_next_set_is_finite_not_positive_infinity(self):
        # value 0: forced transition happened as required (missing -> present). Previously: +inf.
        ld = self.dist.conditional_log_density([], [0])
        self.assertTrue(np.isfinite(ld), msg=f"expected a finite log-density, got {ld}")
        expected = 0.0 + math.log(
            1.0 - self.p_present_missing_1
        )  # log p(present|missing)[0] + log p(missing|missing)[1]
        self.assertAlmostEqual(ld, expected, places=10)

    def test_forced_value_kept_present_is_finite(self):
        ld = self.dist.conditional_log_density([0], [0])
        self.assertTrue(np.isfinite(ld), msg=f"expected a finite log-density, got {ld}")
        expected = math.log(self.p_present_present_0) + math.log(1.0 - self.p_present_missing_1)
        self.assertAlmostEqual(ld, expected, places=10)

    def test_forced_value_removed_is_finite(self):
        ld = self.dist.conditional_log_density([0], [])
        self.assertTrue(np.isfinite(ld), msg=f"expected a finite log-density, got {ld}")
        expected = math.log(1.0 - self.p_present_present_0) + math.log(1.0 - self.p_present_missing_1)
        self.assertAlmostEqual(ld, expected, places=10)

    def test_non_forced_value_alone_is_unaffected(self):
        # sanity: value 1's own ordinary (kept present->present) transition is untouched by value 0
        # being forced -- value 0 must also be touched here (added), or the whole observation would be
        # the impossible "forced value left missing" case from the test above.
        ld = self.dist.conditional_log_density([1], [0, 1])
        expected = 0.0 + math.log(0.8)  # log p(present|missing)[0] (forced, ==1.0) + log p(present|present)[1]
        self.assertAlmostEqual(ld, expected, places=10)

    def test_seq_log_density_matches_scalar_including_impossible_rows(self):
        data = [([], []), ([], [0]), ([0], [0]), ([0], []), ([1], [0, 1]), ([0, 1], [])]
        enc = self.dist.dist_to_encoder().seq_encode(data)
        seq = self.dist.seq_log_density(enc)
        scalar = np.array([self.dist.log_density(d) for d in data])
        np.testing.assert_array_equal(seq, scalar)  # exact match including -inf, not just close

    def test_seq_log_density_has_no_nan_across_a_mixed_batch(self):
        data = [([], []), ([], [0]), ([0], [0]), ([0], []), ([1], [0, 1])]
        enc = self.dist.dist_to_encoder().seq_encode(data)
        seq = self.dist.seq_log_density(enc)
        self.assertFalse(np.any(np.isnan(seq)), msg=f"seq_log_density produced NaN: {seq}")


if __name__ == "__main__":
    unittest.main()
