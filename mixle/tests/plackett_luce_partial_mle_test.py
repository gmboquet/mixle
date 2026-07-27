"""Tests for partial / top-m ranking MLE in Plackett-Luce (generalized MM estimator, WS-M)."""

import unittest

import numpy as np
from numpy.random import RandomState

from mixle.inference.estimation import optimize
from mixle.stats.rankings.plackett_luce import (
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

    def test_partial_fit_encoder_scores_its_own_training_data(self):
        # A distribution fitted via PlackettLucePartialEstimator must remain usable through the
        # standard fitted.dist_to_encoder().seq_encode(data) / seq_log_density(...) pattern used
        # everywhere else in this codebase (e.g. optimize()'s scoring loop) -- dist_to_encoder()
        # must not unconditionally hand back the full-ranking-only encoder regardless of what the
        # distribution was actually fit on, since that encoder can't build a dense array from
        # ragged/partial rows and previously raised a numpy shape error on this exact call.
        data = [[0, 2], [1, 0], [2, 1, 0]]  # ragged partial rankings, dim=3
        est = PlackettLucePartialEstimator(dim=3)
        acc = est.accumulator_factory().make()
        for row in data:
            acc.update(row, 1.0, None)
        fitted = est.estimate(len(data), acc.value())

        encoded = fitted.dist_to_encoder().seq_encode(data)
        seq_ll = fitted.seq_log_density(encoded)
        # log_density is the ground truth for partial-ranking scoring (it already includes the
        # unranked-item denominator term); seq_log_density on the batch encoding must agree.
        scalar_ll = np.array([fitted.log_density(row) for row in data])
        np.testing.assert_allclose(seq_ll, scalar_ll, rtol=1e-12, atol=1e-12)

    def test_partial_estimator_optimize_warm_start_scores_correctly(self):
        # optimize()'s prev_estimate warm-start path derives its data encoder from
        # prev_estimate.dist_to_encoder() rather than the estimator's own accumulator factory --
        # the other call site (besides direct accumulator use) where a full-ranking-only encoder
        # would have broken on ragged partial data.
        k = 4
        true = PlackettLuceDistribution(np.log([0.4, 0.3, 0.2, 0.1]))
        partial = [list(o[:2]) for o in true.sampler(3).sample(500)]
        proto = PlackettLuceDistribution(np.log(np.full(k, 1.0 / k)), allow_partial=True)
        fit = optimize(
            partial, PlackettLucePartialEstimator(dim=k), prev_estimate=proto, max_its=10, rng=RandomState(0), out=None
        )
        self.assertAlmostEqual(float(np.sum(np.exp(fit.log_w))), 1.0, places=6)

    def test_partial_accumulator_rejects_corrupt_evidence_and_copies_state(self):
        accumulator = PlackettLucePartialAccumulator(3)
        with self.assertRaises(ValueError):
            accumulator.update([0, 0], 1.0, None)
        with self.assertRaises(ValueError):
            accumulator.seq_update([np.asarray([0, 1])], np.asarray([-1.0]), None)
        with self.assertRaises(ValueError):
            accumulator.seq_update([np.asarray([0, 1])], np.asarray([1.0, 2.0]), None)
        extreme = PlackettLuceDistribution([-1.0e308, 0.0, 1.0e308], allow_partial=True)
        accumulator.update([0, 1], 1.0, extreme)
        receipt = accumulator.value()
        receipt[1][:] = 0.0
        receipt[2][:] = 0.0
        self.assertGreater(accumulator.value()[1].sum(), 0.0)
        with self.assertRaises(ValueError):
            PlackettLucePartialEstimator(3).estimate(None, (1.0, np.zeros(2), np.zeros(3)))
        with self.assertRaises(ValueError):
            PlackettLucePartialEstimator(3, pseudo_count=-1.0)


if __name__ == "__main__":
    unittest.main()
