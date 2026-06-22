"""Tests for partial / top-m ranking MLE in Plackett-Luce (generalized MM estimator, WS-M)."""

import unittest

import numpy as np
from numpy.random import RandomState

from pysp.inference.estimation import optimize
from pysp.stats.graph.plackett_luce import (
    PlackettLuceAccumulator,
    PlackettLuceDistribution,
    PlackettLucePartialAccumulator,
    PlackettLucePartialEstimator,
)


class PlackettLucePartialMleTest(unittest.TestCase):
    def test_full_ranking_stats_match_full_accumulator(self):
        # On full rankings the generalized (partial) MM statistics must equal the vectorized
        # full-ranking accumulator's, exactly -- so partial estimation is a strict superset.
        k = 5
        true = PlackettLuceDistribution(np.log([0.35, 0.25, 0.2, 0.13, 0.07]))
        data = true.sampler(0).sample(400)
        estimate = PlackettLuceDistribution(np.log(np.full(k, 1.0 / k)))

        full_acc = PlackettLuceAccumulator(dim=k)
        full_acc.seq_update(np.asarray([list(o) for o in data], dtype=int), np.ones(len(data)), estimate)

        part_acc = PlackettLucePartialAccumulator(dim=k)
        part_acc.seq_update([np.asarray(o, dtype=int) for o in data], np.ones(len(data)), estimate)

        np.testing.assert_allclose(part_acc.num, full_acc.num, atol=1e-9)
        np.testing.assert_allclose(part_acc.den, full_acc.den, atol=1e-9)

    def test_recovers_top_worths_from_partial_rankings(self):
        k = 5
        true = PlackettLuceDistribution(np.log([0.40, 0.25, 0.18, 0.12, 0.05]))
        full = true.sampler(1).sample(4000)
        partial = [list(o[:3]) for o in full]  # observe only each ranking's top 3

        fit = optimize(partial, PlackettLucePartialEstimator(dim=k), max_its=60, rng=RandomState(0), out=None)

        # The top-3 worths are well-identified by top-3 data; their order should match the truth.
        self.assertEqual(int(np.argmax(fit.log_w)), 0)
        self.assertGreater(fit.log_w[0], fit.log_w[1])
        self.assertGreater(fit.log_w[1], fit.log_w[2])

    def test_optimize_runs_and_normalizes(self):
        k = 4
        true = PlackettLuceDistribution(np.log([0.4, 0.3, 0.2, 0.1]))
        partial = [list(o[:2]) for o in true.sampler(2).sample(1500)]
        fit = optimize(partial, PlackettLucePartialEstimator(dim=k), max_its=40, rng=RandomState(0), out=None)
        self.assertAlmostEqual(float(np.sum(np.exp(fit.log_w))), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
