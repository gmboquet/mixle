"""Focused contracts for exact count encoding and fixed-support fitting."""

import unittest

import numpy as np

from mixle.stats import BetaBinomialDistribution, BinomialDataEncoder, BinomialDistribution


class BetaBinomialEvidenceContractTest(unittest.TestCase):
    def test_encoder_and_accumulator_reject_counts_outside_fixed_support(self):
        distribution = BetaBinomialDistribution(2, 1.5, 2.5)
        encoder = distribution.dist_to_encoder()
        accumulator = distribution.estimator().accumulator_factory().make()
        np.testing.assert_array_equal(encoder.seq_encode([0, 1, 2]), np.asarray([0, 1, 2]))

        for invalid in (-1, 1.5, 3, np.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                encoder.seq_encode([invalid])
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                accumulator.update(invalid, 1.0, distribution)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                accumulator.seq_update(np.asarray([invalid]), np.ones(1), distribution)

    def test_encoder_identity_includes_trial_count(self):
        self.assertNotEqual(
            BetaBinomialDistribution(2, 1.0, 1.0).dist_to_encoder(),
            BetaBinomialDistribution(3, 1.0, 1.0).dist_to_encoder(),
        )


class BinomialEvidenceContractTest(unittest.TestCase):
    def test_model_owned_estimators_preserve_shifted_support_and_identity(self):
        distribution = BinomialDistribution(0.4, 10, min_val=5, name="counts", keys="shared")
        statistics = (3.0, 21.0, 7, 7)
        for estimator in (distribution.estimator(), distribution.estimator(pseudo_count=2.0)):
            with self.subTest(pseudo_count=estimator.pseudo_count):
                fitted = estimator.estimate(None, statistics)
                self.assertEqual((fitted.n, fitted.min_val), (10, 5))
                self.assertEqual((fitted.name, fitted.keys), ("counts", "shared"))

    def test_shifted_encoder_accepts_negative_support_and_rejects_outside_it(self):
        encoder = BinomialDistribution(0.5, 2, min_val=-2).dist_to_encoder()
        _, _, values, minimum, maximum = encoder.seq_encode([-2, -1, 0])
        np.testing.assert_array_equal(values, np.asarray([-2, -1, 0], dtype=np.int64))
        self.assertEqual((minimum, maximum), (-2, 0))
        for invalid in (-3, 1, np.inf, 0.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                encoder.seq_encode([invalid])

    def test_generic_encoder_uses_checked_int64_without_value_corruption(self):
        value = 2**40
        encoded = BinomialDataEncoder().seq_encode([value])
        self.assertEqual(encoded[2].dtype, np.dtype(np.int64))
        self.assertEqual(int(encoded[2][0]), value)
        for invalid in (2**63, -(2**63) - 1, np.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                BinomialDataEncoder().seq_encode([invalid])

    def test_zero_weight_does_not_expand_inferred_support(self):
        accumulator = BinomialDataEncoder().seq_encode([2, 100])
        estimator = BinomialDistribution(0.5, 2).estimator()
        learned = estimator.accumulator_factory().make()
        learned.seq_update(accumulator, np.asarray([1.0, 0.0]), None)
        self.assertEqual(learned.value(), (1.0, 2.0, 0, 2))


if __name__ == "__main__":
    unittest.main()
