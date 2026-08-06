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


class SharedEstimatorObjectTest(unittest.TestCase):
    """Reusing ONE keyed estimator across slots is the natural way to say "share these parameters".

    The traversal used to mark estimator objects visited for the whole walk, so a shared object
    registered one keyed site while its factory necessarily produced one accumulator per slot --
    and the site-count reconciliation rejected the model. The identical model built from separately
    constructed estimators carrying the same key was accepted, which is how the inconsistency
    showed up: two spellings of one model, only one of which could be fitted.
    """

    def test_shared_object_and_distinct_objects_are_both_accepted(self):
        from mixle.stats import MixtureEstimator

        shared = GaussianEstimator(keys="tied")
        validate_estimator_keys(MixtureEstimator([shared] * 3))
        validate_estimator_keys(MixtureEstimator([GaussianEstimator(keys="tied") for _ in range(3)]))

    def test_a_genuine_site_count_mismatch_is_still_reported(self):
        from mixle.stats import MixtureEstimator

        estimator = MixtureEstimator([GaussianEstimator(keys="tied") for _ in range(3)])
        real_factory = estimator.accumulator_factory

        class FewerAccumulators:
            @staticmethod
            def make():
                accumulator = real_factory().make()
                accumulator.accumulators = accumulator.accumulators[:1]  # one merge site, not three
                return accumulator

        estimator.accumulator_factory = lambda: FewerAccumulators()
        with self.assertRaisesRegex(KeyValidationError, "accumulator merge sites"):
            validate_estimator_keys(estimator)


class FamilyOwnedKeyAttributeTest(unittest.TestCase):
    """Every shipped family's own key-attribute names must be visible to the validator.

    A family holds its keys under its own attribute names (LDA: alpha_key/topics_key, pLSI:
    dc_key/sc_key/wc_key, heterogeneous PCFG: emission_key/rule_key). A name the collector does not
    know is not a harmless omission: the estimator half still declares the key through its `keys`
    tuple, so reconciliation reports it as estimator-only and the family's documented `keys=`
    argument fails for EVERY non-None value. All three families were unusable that way.
    """

    def test_lda_accepts_each_key_position(self):
        from mixle.stats import IntegerCategoricalEstimator
        from mixle.stats.latent.lda import LDAEstimator

        base = IntegerCategoricalEstimator(min_val=0, max_val=5, pseudo_count=1e-3)
        for keys in ((None, None), (None, "topics"), ("alpha", None), ("alpha", "topics")):
            with self.subTest(keys=keys):
                validate_estimator_keys(LDAEstimator([base] * 2, keys=keys))

    def test_every_shipped_key_attribute_is_collected(self):
        from mixle.stats.compute.pdist import _KEY_ATTRS

        # The names below are the ones shipped families assign; missing any of them silently
        # disables that family's keys= argument, so the list is pinned rather than discovered.
        for attribute in ("alpha_key", "topics_key", "dc_key", "sc_key", "wc_key", "emission_key", "rule_key"):
            with self.subTest(attribute=attribute):
                self.assertIn(attribute, _KEY_ATTRS)


if __name__ == "__main__":
    unittest.main()
