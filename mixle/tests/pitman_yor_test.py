"""Tests for the Pitman-Yor process distribution over set partitions (EPPF, sampling, estimation)."""

import math
import unittest

import numpy as np

from mixle.inference.estimation import fit
from mixle.stats import PitmanYorProcessDistribution, PitmanYorProcessEstimator
from mixle.stats.bayes.pitman_yor import PitmanYorProcessAccumulator


def _set_partitions(collection):
    collection = list(collection)
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in _set_partitions(collection[1:]):
        for i, subset in enumerate(smaller):
            yield smaller[:i] + [[first] + subset] + smaller[i + 1 :]
        yield [[first]] + smaller


def _labels(part, n):
    lab = [0] * n
    for cid, block in enumerate(part):
        for e in block:
            lab[e] = cid
    return lab


class PitmanYorTestCase(unittest.TestCase):
    def test_eppf_normalizes_over_partitions(self):
        for alpha, discount in [(1.0, 0.0), (2.5, 0.0), (1.0, 0.4), (3.0, 0.6)]:
            dist = PitmanYorProcessDistribution(alpha, discount)
            total = sum(math.exp(dist.log_density(_labels(p, 5))) for p in _set_partitions(range(5)))
            self.assertAlmostEqual(total, 1.0, places=10)

    def test_discount_zero_matches_crp_closed_form(self):
        dist = PitmanYorProcessDistribution(2.0, 0.0)
        labels = [0, 0, 1, 0, 2, 1]  # block sizes 3, 2, 1
        sizes, a, n = [3, 2, 1], 2.0, 6
        crp = 3 * math.log(a) + sum(math.lgamma(s) for s in sizes) - (math.lgamma(a + n) - math.lgamma(a))
        self.assertAlmostEqual(dist.log_density(labels), crp, places=10)

    def test_density_is_label_invariant(self):
        dist = PitmanYorProcessDistribution(1.5, 0.3)
        self.assertAlmostEqual(dist.log_density([0, 0, 1, 2]), dist.log_density([2, 2, 0, 1]), places=12)

    def test_seq_matches_scalar(self):
        dist = PitmanYorProcessDistribution(1.2, 0.25)
        data = [[0, 0, 1], [0, 1, 2, 0], [0, 0, 0, 0]]
        enc = dist.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(dist.seq_log_density(enc), [dist.log_density(x) for x in data])

    def test_string_round_trip(self):
        dist = PitmanYorProcessDistribution(1.5, 0.3, num_elements=8, name="py", keys="k")
        self.assertEqual(str(eval(str(dist))), str(dist))

    def test_sampler_matches_eppf_by_shape(self):
        dist = PitmanYorProcessDistribution(1.5, 0.3, num_elements=4)
        n = 60000
        samples = dist.sampler(seed=0).sample(n)

        def shape(labels):
            _, counts = np.unique(labels, return_counts=True)
            return tuple(sorted(counts, reverse=True))

        from collections import Counter

        empirical = Counter(shape(s) for s in samples)
        expected = {}
        for part in _set_partitions(range(4)):
            sh = shape(_labels(part, 4))
            expected[sh] = expected.get(sh, 0.0) + math.exp(dist.log_density(_labels(part, 4)))
        for sh, p in expected.items():
            self.assertAlmostEqual(empirical[sh] / n, p, delta=0.02)

    def test_estimator_recovers_dp_concentration(self):
        true = PitmanYorProcessDistribution(3.0, 0.0, num_elements=30)
        data = true.sampler(seed=1).sample(400)
        fitted = fit(data, true.estimator(), max_its=1, rng=np.random.RandomState(0), print_iter=0)
        self.assertAlmostEqual(fitted.alpha, 3.0, delta=0.5)
        self.assertEqual(fitted.discount, 0.0)

    def test_estimator_recovers_discount_when_enabled(self):
        true = PitmanYorProcessDistribution(1.0, 0.5, num_elements=40)
        data = true.sampler(seed=2).sample(500)
        fitted = fit(data, PitmanYorProcessEstimator(estimate_discount=True), max_its=1, print_iter=0)
        self.assertAlmostEqual(fitted.discount, 0.5, delta=0.12)
        self.assertAlmostEqual(fitted.alpha, 1.0, delta=0.6)

    def test_estimator_reports_boundary_for_no_new_blocks(self):
        data = [[0, 0, 0], [1, 1]]
        estimator = PitmanYorProcessEstimator(discount=0.0)
        accumulator = estimator.accumulator_factory().make()
        for row in data:
            accumulator.update(row, 1.0, None)
        fitted = estimator.estimate(None, accumulator.value())
        default = PitmanYorProcessDistribution(1.0, 0.0)

        self.assertLess(fitted.alpha, 1.0e-6)
        self.assertEqual(fitted.fit_metadata["boundary"]["alpha"], "lower")
        self.assertTrue(fitted.fit_metadata["converged"])
        self.assertGreaterEqual(
            sum(fitted.log_density(x) for x in data),
            sum(default.log_density(x) for x in data),
        )

    def test_estimator_returns_default_when_unestimated(self):
        estimator = PitmanYorProcessEstimator(discount=0.25)
        fitted = estimator.estimate(None, (0.0, {}, {}, {}))

        self.assertEqual(fitted.alpha, 1.0)
        self.assertEqual(fitted.discount, 0.25)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            PitmanYorProcessDistribution(1.0, 1.0)  # discount must be < 1
        with self.assertRaises(ValueError):
            PitmanYorProcessDistribution(-0.5, 0.3)  # alpha must be > -discount


class PitmanYorContractTestCase(unittest.TestCase):
    def test_fractional_and_string_labels_preserve_partition_identity(self):
        dist = PitmanYorProcessDistribution(1.5, 0.3)
        reference = dist.log_density([0, 1, 0])
        self.assertAlmostEqual(dist.log_density([0.1, 0.9, 0.1]), reference)
        self.assertAlmostEqual(dist.log_density(["left", "right", "left"]), reference)
        encoded = dist.dist_to_encoder().seq_encode([[0.1, 0.9, 0.1], ["left", "right", "left"]])
        np.testing.assert_array_equal(encoded[0], [2, 1])
        np.testing.assert_array_equal(encoded[1], [2, 1])

    def test_raw_partitions_reject_non_sequences_and_invalid_label_equality(self):
        dist = PitmanYorProcessDistribution()
        for value in (1, "abc", [[0], [0]], [np.nan, np.nan]):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                dist.log_density(value)

    def test_encoded_partitions_require_canonical_positive_integer_sizes(self):
        dist = PitmanYorProcessDistribution()
        invalid = (
            [[0, 1]],
            [[1.5, 1]],
            [[True, 1]],
            [[1, 2]],
            [[[1, 2]]],
        )
        for value in invalid:
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                dist.seq_log_density(value)
        np.testing.assert_allclose(
            dist.seq_log_density([np.asarray([2, 1]), np.asarray([], dtype=int)]),
            [dist.log_density([0, 0, 1]), 0.0],
        )

    def test_sequence_accumulation_requires_exact_finite_nonnegative_weights(self):
        acc = PitmanYorProcessAccumulator()
        encoded = [np.asarray([2, 1]), np.asarray([3])]
        for weights in ([1.0], [[1.0], [1.0]], [1.0, -1.0], [1.0, np.nan]):
            with self.subTest(weights=repr(weights)), self.assertRaises(ValueError):
                acc.seq_update(encoded, weights, None)
        self.assertEqual(acc.value(), (0.0, {}, {}, {}))
        acc.seq_update(encoded, np.asarray([0.5, 2.0]), None)
        scalar = PitmanYorProcessAccumulator()
        scalar.update(["a", "a", "b"], 0.5, None)
        scalar.update([0, 0, 0], 2.0, None)
        self.assertEqual(acc.value(), scalar.value())

    def test_accumulation_rejects_invalid_weights_atomically(self):
        acc = PitmanYorProcessAccumulator()
        before = acc.value()
        for weight in (-1.0, np.nan, [1.0]):
            with self.subTest(weight=repr(weight)), self.assertRaises(ValueError):
                acc.update([0, 0, 1], weight, None)
            self.assertEqual(acc.value(), before)

    def test_serialized_histograms_are_validated_atomically(self):
        acc = PitmanYorProcessAccumulator()
        acc.update([0, 0, 1], 1.0, None)
        before = acc.value()
        malformed = (
            (-1.0, {}, {}, {}),
            (1.0, {0: 1.0}, {}, {1: 1.0}),
            (1.0, {1: np.nan}, {}, {1: 1.0}),
            (1.0, {1: 1.0}, {}, {}),
            (1.0, {1: 0.5, 2: 1.0}, {}, {1: 1.0, 2: 0.5}),
            (1.0, {1: 1.0}, {1: 2.0}, {}),
        )
        for value in malformed:
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                acc.combine(value)
            self.assertEqual(acc.value(), before)
            with self.assertRaises(ValueError):
                PitmanYorProcessAccumulator().from_value(value)

    def test_estimator_validates_controls_and_statistics(self):
        for kwargs in (
            {"discount": np.nan},
            {"discount": 1.0},
            {"estimate_discount": 1},
            {"max_alpha": 0.0},
            {"max_alpha": -1.0},
            {"max_alpha": np.inf},
        ):
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises((TypeError, ValueError)):
                PitmanYorProcessEstimator(**kwargs)
        estimator = PitmanYorProcessEstimator()
        with self.assertRaises(ValueError):
            estimator.estimate(None, (1.0, {1: 1.0}, {}, {}))

    def test_estimator_reports_uninformative_and_upper_boundary_results(self):
        empty = PitmanYorProcessEstimator().estimate(None, (3.0, {}, {}, {}))
        self.assertEqual(empty.alpha, 1.0)
        self.assertEqual(empty.fit_metadata["solver"], "none")
        self.assertTrue(empty.fit_metadata["converged"])
        self.assertEqual(empty.fit_metadata["repairs"], ())

        accumulator = PitmanYorProcessAccumulator()
        for _ in range(4):
            accumulator.update([0, 1, 2], 1.0, None)
        upper = PitmanYorProcessEstimator(max_alpha=2.0).estimate(None, accumulator.value())
        self.assertEqual(upper.alpha, 2.0)
        self.assertEqual(upper.fit_metadata["boundary"]["alpha"], "upper")
        self.assertTrue(np.isfinite(upper.fit_metadata["objective"]))

    def test_distribution_and_sampler_validate_sizes(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=repr(value)), self.assertRaises((TypeError, ValueError)):
                PitmanYorProcessDistribution(num_elements=value)
        sampler = PitmanYorProcessDistribution(num_elements=3).sampler(seed=1)
        for value in (-1, 1.5, True):
            with self.subTest(value=repr(value)), self.assertRaises((TypeError, ValueError)):
                sampler.sample(value)
        with self.assertRaises(ValueError):
            PitmanYorProcessDistribution().estimator(pseudo_count=1.0)


if __name__ == "__main__":
    unittest.main()
