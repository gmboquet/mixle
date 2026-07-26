"""Chinese Restaurant Process: Ewens normalization, sequential sampling, concentration MLE."""

import unittest

import numpy as np
from scipy.special import digamma

from mixle.inference import estimate
from mixle.stats import ChineseRestaurantProcessDistribution


def _set_partitions(collection):
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for rest in _set_partitions(collection[1:]):
        for i, block in enumerate(rest):
            yield rest[:i] + [[first] + block] + rest[i + 1 :]
        yield [[first]] + rest


def _labels(partition, n):
    z = np.empty(n, dtype=int)
    for label, block in enumerate(partition):
        for item in block:
            z[item] = label
    return z


class ChineseRestaurantProcessTest(unittest.TestCase):
    def setUp(self):
        self.n = 5
        self.alpha = 1.7
        self.d = ChineseRestaurantProcessDistribution(self.alpha, self.n)

    def test_density_sums_to_one_over_partitions(self):
        parts = list(_set_partitions(list(range(self.n))))
        self.assertEqual(len(parts), 52)  # Bell(5)
        total = sum(self.d.density(_labels(p, self.n)) for p in parts)
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_relabeling_invariant(self):
        z = np.array([0, 0, 1, 2, 2])
        relabeled = np.array([2, 2, 0, 1, 1])  # same partition, different labels
        self.assertAlmostEqual(self.d.log_density(z), self.d.log_density(relabeled))
        self.assertEqual(self.d.log_density(np.array([0, 0, 1, 2])), -np.inf)  # wrong n

    def test_sampler_expected_blocks_matches_theory(self):
        s = self.d.sampler(seed=0).sample(40000)
        ek_emp = float(np.mean([len(np.unique(z)) for z in s]))
        ek_theory = self.alpha * float(digamma(self.alpha + self.n) - digamma(self.alpha))
        self.assertAlmostEqual(ek_emp, ek_theory, delta=0.04)

    def test_mle_recovers_alpha(self):
        est = estimate(list(self.d.sampler(seed=1).sample(20000)), self.d.estimator())
        self.assertAlmostEqual(est.alpha, self.alpha, delta=0.1)

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            ChineseRestaurantProcessDistribution(0.0, 5)

    def test_partition_size_and_shape_are_exact(self):
        with self.assertRaises(TypeError):
            ChineseRestaurantProcessDistribution(1.0, 2.9)
        d = ChineseRestaurantProcessDistribution(1.0, 2)
        self.assertEqual(d.log_density(np.array([[0, 1]])), -np.inf)
        with self.assertRaises(ValueError):
            d.dist_to_encoder().seq_encode([np.array([[0, 1]])])

    def test_mixed_labels_and_encoding_preserve_partition(self):
        d = ChineseRestaurantProcessDistribution(2.0, 2)
        labels = [1, "one"]
        scalar = d.log_density(labels)
        encoded = d.dist_to_encoder().seq_encode([labels])
        self.assertAlmostEqual(d.seq_log_density(encoded)[0], scalar)
        floats = [0.1, 0.2]
        self.assertAlmostEqual(
            d.seq_log_density(d.dist_to_encoder().seq_encode([floats]))[0],
            d.log_density(floats),
        )
        np.testing.assert_array_equal(encoded[0], [0, 1])

    def test_accumulation_and_statistics_fail_closed(self):
        estimator = self.d.estimator()
        accumulator = estimator.accumulator_factory().make()
        with self.assertRaises(ValueError):
            accumulator.update([0, 1], 1.0, None)
        with self.assertRaises(ValueError):
            accumulator.update([0, 0, 1, 2, 2], -1.0, None)
        with self.assertRaises(ValueError):
            accumulator.seq_update(
                [[0, 0, 1, 2, 2], [0, 1, 2, 3, 4]],
                [1.0],
                None,
            )
        self.assertEqual(accumulator.value().schema_version, 1)
        with self.assertRaises(ValueError):
            estimator.estimate(None, (100.0, 1.0, self.n))
        with self.assertRaises(ValueError):
            estimator.estimate(None, (0.0, 0.0, self.n))

    def test_nonzero_implicit_pseudo_count_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            self.d.estimator(pseudo_count=1.0)


if __name__ == "__main__":
    unittest.main()
