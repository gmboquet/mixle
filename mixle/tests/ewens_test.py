"""Ewens permutation distribution: theta^cycles law, exact sampling, theta recovery."""

import itertools
import math
import unittest

import numpy as np

from mixle.stats import EwensDistribution


def _cycles(p):
    n, seen, c = len(p), [False] * len(p), 0
    for i in range(n):
        if not seen[i]:
            c += 1
            j = i
            while not seen[j]:
                seen[j], j = True, p[j]
    return c


class EwensTest(unittest.TestCase):
    def test_density_matches_theta_power_cycles(self):
        for n, theta in ((4, 0.5), (5, 2.0), (6, 1.0)):
            d = EwensDistribution(n, theta)
            z = math.prod(theta + i for i in range(n))
            for p in itertools.permutations(range(n)):
                self.assertAlmostEqual(d.density(list(p)), theta ** _cycles(p) / z, places=12)

    def test_density_sums_to_one(self):
        d = EwensDistribution(6, 1.7)
        self.assertAlmostEqual(sum(d.density(list(p)) for p in itertools.permutations(range(6))), 1.0, places=9)

    def test_uniform_at_theta_one(self):
        d = EwensDistribution(5, 1.0)
        vals = [d.density(list(p)) for p in itertools.permutations(range(5))]
        np.testing.assert_allclose(vals, 1.0 / math.factorial(5), atol=1e-12)

    def test_sampler_mean_cycle_count(self):
        for theta in (0.4, 1.0, 3.0):
            d = EwensDistribution(8, theta)
            samp = d.sampler(seed=1).sample(15000)
            exp = sum(theta / (theta + i) for i in range(8))
            self.assertAlmostEqual(np.mean([_cycles(s) for s in samp]), exp, delta=0.1)

    def test_theta_recovery(self):
        for theta in (0.4, 1.0, 3.0):
            true = EwensDistribution(8, theta)
            samp = true.sampler(seed=2).sample(20000)
            acc = true.estimator().accumulator_factory().make()
            acc.seq_update(true.dist_to_encoder().seq_encode(samp), np.ones(len(samp)), None)
            fit = true.estimator().estimate(len(samp), acc.value())
            self.assertAlmostEqual(fit.theta, theta, delta=0.15 * theta + 0.05)

    def test_validation(self):
        with self.assertRaises(ValueError):
            EwensDistribution(5, 0.0)  # theta > 0
        with self.assertRaises(ValueError):
            EwensDistribution(1, 1.0)  # dim >= 2


if __name__ == "__main__":
    unittest.main()
