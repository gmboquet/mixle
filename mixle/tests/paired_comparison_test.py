"""Paired-comparison models: Thurstone-Mosteller (Gaussian) and Bradley-Terry with ties (Davidson, Rao-Kupper)."""

import unittest

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
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    pair_dist.log_density(value)
                with self.assertRaises((TypeError, ValueError)):
                    pair_dist.seq_log_density(np.asarray([value]))
                with self.assertRaises((TypeError, ValueError)):
                    pair_dist.dist_to_encoder().seq_encode([value])

        tie_dist = DavidsonDistribution(np.zeros(3))
        invalid_ties = ((0, 0, 0), (-1, 1, 0), (0, 3, 0), (0, 1, 3), (0, 1, 0.5))
        for value in invalid_ties:
            with self.subTest(value=value):
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
            with self.subTest(factory=factory):
                with self.assertRaises((TypeError, ValueError)):
                    factory()

    def test_sampling_rejects_negative_or_fractional_sizes(self):
        for distribution in (
            ThurstoneMostellerDistribution(np.zeros(3)),
            DavidsonDistribution(np.zeros(3)),
            RaoKupperDistribution(np.zeros(3)),
        ):
            for size in (-1, 1.5):
                with self.subTest(distribution=type(distribution).__name__, size=size):
                    with self.assertRaises((TypeError, ValueError)):
                        distribution.sampler(seed=1).sample(size)


if __name__ == "__main__":
    unittest.main()
