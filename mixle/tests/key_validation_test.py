import unittest

import numpy as np

from mixle.inference import seq_estimate
from mixle.stats import (
    CompositeEstimator,
    DiagonalGaussianEstimator,
    GaussianDistribution,
    GaussianEstimator,
    KeyValidationError,
    PoissonEstimator,
    seq_encode,
    validate_estimator_keys,
)
from mixle.stats.univariate.continuous.gaussian import GaussianAccumulator


class KeyValidationTestCase(unittest.TestCase):
    def test_same_family_same_settings_key_passes(self):
        est = CompositeEstimator(
            (
                GaussianEstimator(keys="shared_gaussian"),
                GaussianEstimator(keys="shared_gaussian"),
            )
        )
        validate_estimator_keys(est)

    def test_cross_family_key_fails(self):
        est = CompositeEstimator(
            (
                GaussianEstimator(keys="bad_key"),
                PoissonEstimator(keys="bad_key"),
            )
        )
        with self.assertRaises(KeyValidationError):
            validate_estimator_keys(est)

    def test_same_family_different_settings_key_fails(self):
        est = CompositeEstimator(
            (
                GaussianEstimator(pseudo_count=(1.0, 1.0), suff_stat=(0.0, 1.0), keys="bad_key"),
                GaussianEstimator(pseudo_count=(2.0, 1.0), suff_stat=(0.0, 1.0), keys="bad_key"),
            )
        )
        with self.assertRaises(KeyValidationError):
            validate_estimator_keys(est)

    def test_seq_estimate_validates_before_combining_stats(self):
        model = GaussianDistribution(0.0, 1.0)
        enc = seq_encode(model.sampler(seed=1).sample(10), model=model)
        est = CompositeEstimator(
            (
                GaussianEstimator(keys="bad_key"),
                PoissonEstimator(keys="bad_key"),
            )
        )
        with self.assertRaises(KeyValidationError):
            seq_estimate(enc, est, model)

    def test_array_contents_and_private_configuration_are_part_of_signature(self):
        array_config = CompositeEstimator(
            (
                DiagonalGaussianEstimator(
                    dim=2,
                    suff_stat=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
                    keys="shared",
                ),
                DiagonalGaussianEstimator(
                    dim=2,
                    suff_stat=(np.array([1.0, 0.0]), np.array([1.0, 1.0])),
                    keys="shared",
                ),
            )
        )
        with self.assertRaises(KeyValidationError):
            validate_estimator_keys(array_config)

        left = GaussianEstimator(keys="private")
        right = GaussianEstimator(keys="private")
        left._regularization_mode = "left"
        right._regularization_mode = "right"
        with self.assertRaises(KeyValidationError):
            validate_estimator_keys(CompositeEstimator((left, right)))

    def test_keys_are_type_stable_and_nan_is_canonical(self):
        nan_keyed = CompositeEstimator(
            (
                GaussianEstimator(keys=float("nan")),
                GaussianEstimator(keys=float("nan")),
            )
        )
        validate_estimator_keys(nan_keyed)

        # Python considers True == 1, but typed key identity must not pool them.
        distinct = CompositeEstimator(
            (
                GaussianEstimator(keys=True),
                PoissonEstimator(keys=1),
            )
        )
        validate_estimator_keys(distinct)

    def test_estimator_and_accumulator_key_sites_are_cross_checked(self):
        estimator = GaussianEstimator(keys="estimator-key")

        class MismatchedFactory:
            @staticmethod
            def make():
                return GaussianAccumulator(keys="accumulator-key")

        estimator.accumulator_factory = lambda: MismatchedFactory()
        with self.assertRaisesRegex(KeyValidationError, "Estimator/accumulator keyed sites disagree"):
            validate_estimator_keys(estimator)

    def test_unsupported_key_types_fail_instead_of_being_ignored(self):
        with self.assertRaisesRegex(KeyValidationError, "Key values must be scalar"):
            validate_estimator_keys(GaussianEstimator(keys={"not": "a scalar key"}))


if __name__ == "__main__":
    unittest.main()
