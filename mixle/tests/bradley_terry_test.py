"""Bradley-Terry paired-comparison model: normalization, MM fit, and worth recovery."""

import unittest
import warnings

import numpy as np

from mixle.stats import BradleyTerryDistribution
from mixle.stats.rankings.bradley_terry import BradleyTerryDataEncoder, BradleyTerryEstimator


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

    def test_every_scoring_boundary_rejects_invalid_comparisons(self):
        dist = BradleyTerryDistribution(np.zeros(3))
        invalid = ((0, 0), (-1, 0), (0, 3), (0.5, 1), (0, 1, 2))
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    dist.log_density(value)
                with self.assertRaises((TypeError, ValueError)):
                    dist.seq_log_density(np.asarray([value]))
                with self.assertRaises((TypeError, ValueError)):
                    dist.dist_to_encoder().seq_encode([value])

    def test_encoder_identity_includes_support_dimension(self):
        self.assertEqual(BradleyTerryDataEncoder(3), BradleyTerryDataEncoder(3))
        self.assertNotEqual(BradleyTerryDataEncoder(3), BradleyTerryDataEncoder(4))
        self.assertIn("dim=3", str(BradleyTerryDataEncoder(3)))

    def test_extreme_finite_worths_sample_without_overflow(self):
        dist = BradleyTerryDistribution([-1.0e308, 0.0, 1.0e308])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            draws = dist.sampler(seed=1).sample(100)
        self.assertTrue(all(0 <= winner < 3 and 0 <= loser < 3 and winner != loser for winner, loser in draws))

    def test_accumulator_rejects_corrupt_evidence_and_copies_state(self):
        dist = BradleyTerryDistribution(np.zeros(3))
        accumulator = dist.estimator().accumulator_factory().make()
        with self.assertRaises(ValueError):
            accumulator.seq_update(np.asarray([[0, 1], [1, 2]]), np.ones(1), None)
        with self.assertRaises(ValueError):
            accumulator.update((0, 1), -1.0, None)

        accumulator.update((0, 1), 2.0, None)
        value = accumulator.value()
        value[1][0, 1] += 100.0
        self.assertEqual(accumulator.wins[0, 1], 2.0)
        restored = dist.estimator().accumulator_factory().make().from_value(accumulator.value())
        restored.wins[0, 1] += 100.0
        self.assertEqual(accumulator.wins[0, 1], 2.0)

    def test_estimator_controls_and_statistics_are_validated(self):
        for kwargs in (
            {"dim": 1},
            {"dim": 3.5},
            {"dim": 3, "pseudo_count": -1.0},
            {"dim": 3, "max_iter": 0},
            {"dim": 3, "max_iter": 2.5},
            {"dim": 3, "tol": 0.0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    BradleyTerryEstimator(**kwargs)
        estimator = BradleyTerryEstimator(3)
        with self.assertRaises(ValueError):
            estimator.estimate(None, (1.0, np.zeros((2, 2))))
        with self.assertRaises(ValueError):
            estimator.estimate(None, (1.0, np.zeros((3, 3))))

    def test_unidentified_and_boundary_fits_fail_closed_without_prior(self):
        estimator = BradleyTerryEstimator(3)
        disconnected = np.zeros((3, 3))
        disconnected[0, 1] = disconnected[1, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "disconnected"):
            estimator.estimate(None, (2.0, disconnected))

        separated = np.zeros((3, 3))
        separated[0, 1] = separated[0, 2] = separated[1, 2] = 1.0
        with self.assertRaisesRegex(ValueError, "finite interior MLE"):
            estimator.estimate(None, (3.0, separated))

    def test_explicit_prior_regularizes_boundary_fit_and_reports_diagnostics(self):
        separated = np.zeros((3, 3))
        separated[0, 1] = separated[0, 2] = separated[1, 2] = 1.0
        fitted = BradleyTerryEstimator(3, pseudo_count=0.5).estimate(None, (3.0, separated))
        self.assertTrue(fitted.fit_diagnostics.converged)
        self.assertTrue(fitted.fit_diagnostics.regularized)
        self.assertEqual(fitted.fit_diagnostics.pseudo_count, 0.5)
        self.assertGreater(fitted.fit_diagnostics.iterations, 0)
        self.assertTrue(np.all(np.isfinite(fitted.log_w)))

    def test_invalid_diagonal_counts_and_nonconvergence_fail_closed(self):
        diagonal = np.zeros((3, 3))
        diagonal[0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "zero diagonal"):
            BradleyTerryEstimator(3, pseudo_count=0.5).estimate(None, (1.0, diagonal))

        cyclic = np.zeros((3, 3))
        cyclic[0, 1], cyclic[1, 0] = 10.0, 1.0
        cyclic[1, 2], cyclic[2, 1] = 10.0, 1.0
        cyclic[2, 0], cyclic[0, 2] = 2.0, 1.0
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            BradleyTerryEstimator(3, max_iter=1, tol=1e-15).estimate(None, (25.0, cyclic))


if __name__ == "__main__":
    unittest.main()
