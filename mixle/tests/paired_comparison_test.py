"""Paired-comparison models: Thurstone-Mosteller (Gaussian) and Bradley-Terry with ties (Davidson, Rao-Kupper)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mixle.stats import DavidsonDistribution, RaoKupperDistribution, ThurstoneMostellerDistribution
from mixle.stats.rankings.paired_comparison import PairDataEncoder, _TieEncoder


class ThurstoneMostellerTest(unittest.TestCase):
    def test_density_sums_to_one(self):
        d = ThurstoneMostellerDistribution([1.5, 0.5, -0.5, -1.5])
        tot = sum(d.density((w, ell)) for w in range(4) for ell in range(4) if w != ell)
        self.assertAlmostEqual(tot, 1.0, places=9)

    def test_probit_win_probability(self):
        from scipy.special import ndtr

        d = ThurstoneMostellerDistribution([1.0, -1.0])
        # P(0 beats 1) = Phi((mu0-mu1)/sqrt2) = Phi(2/sqrt2)
        self.assertAlmostEqual(np.exp(d.log_density((0, 1)) + d.log_pairs), ndtr(2.0 / np.sqrt(2.0)), places=9)

    def test_mu_recovery(self):
        true = ThurstoneMostellerDistribution([2.0, 1.0, 0.0, -1.0, -2.0])
        samp = true.sampler(seed=1).sample(30000)
        acc = true.estimator().accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(samp), np.ones(len(samp)), None)
        fit = true.estimator().estimate(len(samp), acc.value())
        np.testing.assert_allclose(fit.mu, true.mu, atol=0.2)
        self.assertEqual(list(np.argsort(-fit.mu)), list(np.argsort(-true.mu)))
        self.assertTrue(fit.fit_diagnostics.graph_connected)
        self.assertFalse(fit.fit_diagnostics.exact_mle)


class DavidsonTest(unittest.TestCase):
    def test_density_sums_to_one(self):
        d = DavidsonDistribution([1.0, 0.0, -1.0], nu=1.5)
        tot = sum(d.density((i, j, o)) for i in range(3) for j in range(i + 1, 3) for o in range(3))
        self.assertAlmostEqual(tot, 1.0, places=9)

    def test_canonicalization_flips_outcome(self):
        d = DavidsonDistribution([1.0, 0.0, -1.0], nu=1.0)
        self.assertAlmostEqual(d.log_density((1, 0, 0)), d.log_density((0, 1, 1)))  # i-wins(1,0) == j-wins(0,1)
        self.assertAlmostEqual(d.log_density((1, 0, 2)), d.log_density((0, 1, 2)))  # ties symmetric

    def test_recovers_worths_and_tie_parameter(self):
        true = DavidsonDistribution([2.0, 0.7, -0.7, -2.0], nu=1.2)
        samp = true.sampler(seed=2).sample(40000)
        acc = true.estimator().accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(samp), np.ones(len(samp)), None)
        fit = true.estimator().estimate(len(samp), acc.value())
        np.testing.assert_allclose(fit.log_w, true.log_w, atol=0.2)
        self.assertAlmostEqual(fit.nu, 1.2, delta=0.2)
        self.assertTrue(fit.fit_diagnostics.converged)
        self.assertTrue(fit.fit_diagnostics.graph_connected)

    def test_validation(self):
        with self.assertRaises(ValueError):
            DavidsonDistribution([0.0, 0.0], nu=-0.5)


class RaoKupperTest(unittest.TestCase):
    def test_density_sums_to_one(self):
        d = RaoKupperDistribution([1.0, 0.0, -1.0], nu=1.8)
        tot = sum(d.density((i, j, o)) for i in range(3) for j in range(i + 1, 3) for o in range(3))
        self.assertAlmostEqual(tot, 1.0, places=9)

    def test_recovers_worths_and_threshold(self):
        true = RaoKupperDistribution([2.0, 0.7, -0.7, -2.0], nu=1.6)
        samp = true.sampler(seed=3).sample(40000)
        acc = true.estimator().accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(samp), np.ones(len(samp)), None)
        fit = true.estimator().estimate(len(samp), acc.value())
        np.testing.assert_allclose(fit.log_w, true.log_w, atol=0.2)
        self.assertAlmostEqual(fit.nu, 1.6, delta=0.25)
        self.assertTrue(fit.fit_diagnostics.converged)

    def test_validation(self):
        with self.assertRaises(ValueError):
            RaoKupperDistribution([0.0, 0.0], nu=0.5)  # nu must be >= 1


class PairedComparisonContractTest(unittest.TestCase):
    def test_pair_and_tie_encoders_include_dimension_in_identity(self):
        self.assertEqual(PairDataEncoder(3), PairDataEncoder(3))
        self.assertNotEqual(PairDataEncoder(3), PairDataEncoder(4))
        self.assertIn("dim=3", str(PairDataEncoder(3)))
        self.assertEqual(_TieEncoder(3), _TieEncoder(3))
        self.assertNotEqual(_TieEncoder(3), _TieEncoder(4))
        self.assertIn("dim=3", str(_TieEncoder(3)))

    def test_every_scoring_boundary_rejects_invalid_pairs(self):
        pair_dist = ThurstoneMostellerDistribution(np.zeros(3))
        invalid_pairs = ((0, 0), (-1, 0), (0, 3), (0.5, 1), (0, 1, 2))
        for value in invalid_pairs:
            with self.subTest(value=repr(value)):
                with self.assertRaises((TypeError, ValueError)):
                    pair_dist.log_density(value)
                with self.assertRaises((TypeError, ValueError)):
                    pair_dist.seq_log_density(np.asarray([value]))
                with self.assertRaises((TypeError, ValueError)):
                    pair_dist.dist_to_encoder().seq_encode([value])

        tie_dist = DavidsonDistribution(np.zeros(3))
        invalid_ties = ((0, 0, 0), (-1, 1, 0), (0, 3, 0), (0, 1, 3), (0, 1, 0.5))
        for value in invalid_ties:
            with self.subTest(value=repr(value)):
                with self.assertRaises((TypeError, ValueError)):
                    tie_dist.log_density(value)
                with self.assertRaises((TypeError, ValueError)):
                    tie_dist.seq_log_density(np.asarray([value]))
                with self.assertRaises((TypeError, ValueError)):
                    tie_dist.dist_to_encoder().seq_encode([value])

    def test_accumulators_validate_weights_alignment_and_copy_state(self):
        pair_dist = ThurstoneMostellerDistribution(np.zeros(3))
        accumulator = pair_dist.estimator().accumulator_factory().make()
        with self.assertRaises(ValueError):
            accumulator.seq_update(np.asarray([[0, 1], [1, 2]]), np.ones(1), None)
        with self.assertRaises(ValueError):
            accumulator.update((0, 1), -1.0, None)
        accumulator.update((0, 1), 2.0, None)
        value = accumulator.value()
        value[1][0, 1] += 100.0
        self.assertEqual(accumulator.wins[0, 1], 2.0)
        restored = pair_dist.estimator().accumulator_factory().make().from_value(accumulator.value())
        restored.wins[0, 1] += 100.0
        self.assertEqual(accumulator.wins[0, 1], 2.0)

        tie_dist = DavidsonDistribution(np.zeros(3))
        tie_accumulator = tie_dist.estimator().accumulator_factory().make()
        with self.assertRaises(ValueError):
            tie_accumulator.seq_update(np.asarray([[0, 1, 2], [1, 2, 0]]), np.ones(1), None)
        tie_accumulator.update((0, 1, 2), 1.0, None)
        tie_value = tie_accumulator.value()
        tie_value[2][0, 1] += 100.0
        self.assertEqual(tie_accumulator.ties[0, 1], 1.0)

    def test_regularization_controls_are_honored_and_validated(self):
        self.assertEqual(ThurstoneMostellerDistribution(np.zeros(3)).estimator(0.5).pseudo_count, 0.5)
        self.assertEqual(DavidsonDistribution(np.zeros(3)).estimator(0.5).pseudo_count, 0.5)
        self.assertEqual(RaoKupperDistribution(np.zeros(3)).estimator(0.5).pseudo_count, 0.5)
        for factory in (
            lambda: ThurstoneMostellerDistribution(np.zeros(3)).estimator(-1.0),
            lambda: DavidsonDistribution(np.zeros(3)).estimator(-1.0),
            lambda: RaoKupperDistribution(np.zeros(3)).estimator(-1.0),
            lambda: DavidsonDistribution(np.zeros(3), nu=np.nan),
            lambda: RaoKupperDistribution(np.zeros(3), nu=np.inf),
        ):
            with self.subTest(factory=repr(factory)):
                with self.assertRaises((TypeError, ValueError)):
                    factory()

    def test_sampling_rejects_negative_or_fractional_sizes(self):
        for distribution in (
            ThurstoneMostellerDistribution(np.zeros(3)),
            DavidsonDistribution(np.zeros(3)),
            RaoKupperDistribution(np.zeros(3)),
        ):
            for size in (-1, 1.5):
                with self.subTest(distribution=type(distribution).__name__, size=repr(size)):
                    with self.assertRaises((TypeError, ValueError)):
                        distribution.sampler(seed=1).sample(size)

    def test_unidentified_graphs_fail_closed_unless_regularized(self):
        pair = ThurstoneMostellerDistribution(np.zeros(3))
        pair_accumulator = pair.estimator().accumulator_factory().make()
        pair_accumulator.update((0, 1), 1.0, None)
        with self.assertRaisesRegex(ValueError, "disconnected"):
            pair.estimator().estimate(1.0, pair_accumulator.value())
        regularized_pair = pair.estimator(0.5).estimate(1.0, pair_accumulator.value())
        self.assertTrue(regularized_pair.fit_diagnostics.regularized)

        tie = DavidsonDistribution(np.zeros(3))
        tie_accumulator = tie.estimator().accumulator_factory().make()
        tie_accumulator.update((0, 1, 2), 1.0, None)
        with self.assertRaisesRegex(ValueError, "disconnected"):
            tie.estimator().estimate(1.0, tie_accumulator.value())

    def test_failed_tie_optimizer_is_never_returned_as_a_fit(self):
        distribution = DavidsonDistribution(np.zeros(2))
        accumulator = distribution.estimator().accumulator_factory().make()
        accumulator.update((0, 1, 0), 1.0, None)
        accumulator.update((0, 1, 1), 1.0, None)
        accumulator.update((0, 1, 2), 1.0, None)
        failed = SimpleNamespace(
            success=False,
            status=2,
            message="line search failed",
            x=np.asarray([0.0, 1.0]),
            fun=1.0,
            jac=np.asarray([0.0, 0.0]),
            nit=1,
        )
        with patch("mixle.stats.rankings.paired_comparison.optimize.minimize", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "line search failed"):
                distribution.estimator().estimate(3.0, accumulator.value())

    def test_extreme_tie_parameters_are_scored_in_log_space(self):
        for distribution in (
            DavidsonDistribution([1.0e308, -1.0e308], nu=1.0e308),
            RaoKupperDistribution([1.0e308, -1.0e308], nu=1.0e308),
        ):
            probabilities = [distribution.density((0, 1, outcome)) * 1.0 for outcome in range(3)]
            self.assertTrue(np.all(np.isfinite(probabilities)))
            self.assertAlmostEqual(sum(probabilities), 1.0, places=12)

    def test_statistic_structure_and_nobs_are_validated(self):
        pair = ThurstoneMostellerDistribution(np.zeros(3))
        pair_accumulator = pair.estimator().accumulator_factory().make()
        pair_accumulator.update((0, 1), 1.0, None)
        with self.assertRaises(ValueError):
            pair.estimator().estimate(2.0, pair_accumulator.value())
        corrupt_wins = pair_accumulator.value()[1]
        corrupt_wins[0, 0] = 1.0
        with self.assertRaises(ValueError):
            pair_accumulator.from_value((2.0, corrupt_wins))

        tie = DavidsonDistribution(np.zeros(3))
        tie_accumulator = tie.estimator().accumulator_factory().make()
        corrupt_ties = np.zeros((3, 3))
        corrupt_ties[1, 0] = 1.0
        with self.assertRaises(ValueError):
            tie_accumulator.from_value((1.0, np.zeros((3, 3)), corrupt_ties))


if __name__ == "__main__":
    unittest.main()
