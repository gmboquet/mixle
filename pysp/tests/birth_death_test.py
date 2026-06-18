"""Tests for the general birth-death-sampling process (fossilized birth-death is a special case)."""

import math
import unittest

import numpy as np

from pysp.stats import BirthDeathSamplingDistribution, BirthDeathSamplingEstimator
from pysp.utils.estimation import optimize


class BirthDeathSamplingTest(unittest.TestCase):
    def test_log_density_closed_form(self):
        d = BirthDeathSamplingDistribution(0.5, 0.3, 0.2)
        # n0=2: birth at t=1 (n:2->3), sampling at t=2 (n=3), death at t=4 (n:3->2); horizon T=5.
        traj = (2, 5.0, [(1.0, 0), (2.0, 2), (4.0, 1)])
        # integral n dt = 2*1 + 3*(2-1) + 3*(4-2) + 2*(5-4) = 2 + 3 + 6 + 2 = 13
        # sum log n at events = log2 + log3 + log3
        expected = (
            (math.log(2) + math.log(3) + math.log(3))
            + 1 * math.log(0.5)  # one birth
            + 1 * math.log(0.3)  # one death
            + 1 * math.log(0.2)  # one sampling
            - (0.5 + 0.3 + 0.2) * 13.0
        )
        self.assertAlmostEqual(d.log_density(traj), expected, places=10)

    def test_seq_matches_scalar(self):
        d = BirthDeathSamplingDistribution(0.7, 0.4, 0.3, initial_population=3, horizon=4.0)
        trajs = d.sampler(0).sample(20)
        enc = d.dist_to_encoder().seq_encode(trajs)
        seq = np.asarray(d.seq_log_density(enc))
        scalar = np.array([d.log_density(t) for t in trajs])
        np.testing.assert_allclose(seq, scalar, atol=1e-9)

    def test_recovers_rates(self):
        true = BirthDeathSamplingDistribution(0.8, 0.4, 0.3, initial_population=6, horizon=4.0)
        data = true.sampler(1).sample(500)
        fit = optimize(data, BirthDeathSamplingEstimator(), max_its=1, rng=np.random.RandomState(0), out=None)
        self.assertAlmostEqual(fit.birth_rate, 0.8, delta=0.12)
        self.assertAlmostEqual(fit.death_rate, 0.4, delta=0.12)
        self.assertAlmostEqual(fit.sampling_rate, 0.3, delta=0.12)

    def test_pure_birth_death_no_sampling(self):
        # sampling_rate = 0: no sampling events, recovered sampling rate stays 0, no NaN/inf.
        true = BirthDeathSamplingDistribution(0.7, 0.5, 0.0, initial_population=8, horizon=4.0)
        data = true.sampler(2).sample(400)
        for traj in data:
            self.assertFalse(any(etype == 2 for _, etype in traj[2]))
            self.assertTrue(math.isfinite(true.log_density(traj)))
        fit = optimize(data, BirthDeathSamplingEstimator(), max_its=1, rng=np.random.RandomState(0), out=None)
        self.assertEqual(fit.sampling_rate, 0.0)
        self.assertAlmostEqual(fit.birth_rate, 0.7, delta=0.12)


if __name__ == "__main__":
    unittest.main()
