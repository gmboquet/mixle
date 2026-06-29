"""Bradley-Terry paired-comparison model: normalization, MM fit, and worth recovery."""

import unittest

import numpy as np

from mixle.stats import BradleyTerryDistribution


class BradleyTerryTest(unittest.TestCase):
    def test_density_sums_to_one_over_ordered_pairs(self):
        d = BradleyTerryDistribution([2.0, 0.5, -1.0, 0.3, -0.8])
        tot = sum(d.density((w, ell)) for w in range(5) for ell in range(5) if w != ell)
        self.assertAlmostEqual(tot, 1.0, places=10)

    def test_seq_matches_scalar(self):
        d = BradleyTerryDistribution([1.0, 0.0, -1.0, 0.5])
        pairs = np.array([(w, ell) for w in range(4) for ell in range(4) if w != ell])
        np.testing.assert_allclose(d.seq_log_density(pairs), [d.log_density(p) for p in pairs], atol=1e-12)

    def test_win_probability_formula(self):
        d = BradleyTerryDistribution([1.0, -1.0])  # centered -> [1, -1]
        # P(0 beats 1) / P(1 beats 0) = exp(2)
        p01 = np.exp(d.log_density((0, 1)))
        p10 = np.exp(d.log_density((1, 0)))
        self.assertAlmostEqual(p01 / p10, np.exp(2.0), places=9)

    def test_mm_recovers_worths(self):
        true = BradleyTerryDistribution([2.0, 1.0, 0.0, -1.0, -2.0])
        samp = true.sampler(seed=1).sample(40000)
        acc = true.estimator().accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(samp), np.ones(len(samp)), None)
        fit = true.estimator().estimate(len(samp), acc.value())
        np.testing.assert_allclose(fit.log_w, true.log_w, atol=0.15)
        self.assertEqual(list(np.argsort(-fit.log_w)), list(np.argsort(-true.log_w)))

    def test_combine_equals_single_shard(self):
        true = BradleyTerryDistribution([1.5, 0.0, -1.5])
        enc = true.dist_to_encoder().seq_encode(true.sampler(seed=2).sample(2000))
        est = true.estimator()

        def shard(rows):
            a = est.accumulator_factory().make()
            a.seq_update(rows, np.ones(len(rows)), None)
            return a

        a = shard(enc[:1200])
        a.combine(shard(enc[1200:]).value())
        full = shard(enc)
        np.testing.assert_allclose(
            est.estimate(2000, a.value()).log_w, est.estimate(2000, full.value()).log_w, atol=1e-9
        )

    def test_pseudo_count_keeps_never_winners_finite(self):
        # item 2 never wins; without smoothing its worth is -inf, with smoothing it stays finite
        enc = np.array([[0, 2], [0, 1], [1, 2], [0, 2], [1, 2]])
        fit = BradleyTerryDistribution(np.zeros(3)).estimator(pseudo_count=0.5)
        acc = fit.accumulator_factory().make()
        acc.seq_update(enc, np.ones(len(enc)), None)
        d = fit.estimate(len(enc), acc.value())
        self.assertTrue(np.all(np.isfinite(d.log_w)))
        self.assertEqual(int(np.argmin(d.log_w)), 2)  # the never-winner is ranked last

    def test_validation(self):
        with self.assertRaises(ValueError):
            BradleyTerryDistribution([1.0])  # K must be >= 2
        with self.assertRaises(ValueError):
            BradleyTerryDistribution([0.0, 0.0]).dist_to_encoder().seq_encode([(1, 1)])  # winner == loser


if __name__ == "__main__":
    unittest.main()
