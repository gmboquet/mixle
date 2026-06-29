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


if __name__ == "__main__":
    unittest.main()
