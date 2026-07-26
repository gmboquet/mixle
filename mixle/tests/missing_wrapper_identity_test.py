import unittest

from mixle.stats import (
    CompositeDistribution,
    CompositeEstimator,
    GaussianDistribution,
    GaussianEstimator,
    OptionalDistribution,
    marginalized,
)
from mixle.stats.missing import marginalize_estimator_leaves, unwrap_marginalized


class MissingWrapperIdentityTest(unittest.TestCase):
    def test_unwrap_preserves_explicitly_modeled_missingness(self):
        modeled = OptionalDistribution(
            GaussianDistribution(1.0, 2.0),
            p=0.25,
            missing_value=None,
            name="informative",
        )
        self.assertIs(unwrap_marginalized(modeled), modeled)
        self.assertTrue(unwrap_marginalized(modeled).has_p)
        self.assertEqual(unwrap_marginalized(modeled).p, 0.25)

        unmarked_mar = OptionalDistribution(
            GaussianDistribution(1.0, 2.0),
            p=None,
            missing_value=None,
        )
        self.assertIs(unwrap_marginalized(unmarked_mar), unmarked_mar)

    def test_only_helper_provenance_is_unwrapped(self):
        child = GaussianDistribution(1.0, 2.0)
        wrapped = marginalized(child)
        self.assertIs(unwrap_marginalized(wrapped), child)

        estimator = wrapped.estimator()
        fitted = estimator.estimate(
            2.0,
            ([0.0, 2.0], (2.0, 2.5, 2.0, 2.0)),
        )
        self.assertIsInstance(fitted, OptionalDistribution)
        self.assertFalse(fitted.has_p)
        self.assertIsInstance(unwrap_marginalized(fitted), GaussianDistribution)

    def test_composite_estimator_keys_and_concrete_identity_survive(self):
        class TaggedCompositeEstimator(CompositeEstimator):
            pass

        estimator = TaggedCompositeEstimator(
            (GaussianEstimator(keys="leaf"),),
            keys="composite",
        )
        estimator.tag = object()
        wrapped = marginalize_estimator_leaves(estimator)
        self.assertIsInstance(wrapped, TaggedCompositeEstimator)
        self.assertIsNot(wrapped, estimator)
        self.assertEqual(wrapped.keys, "composite")
        self.assertIs(wrapped.tag, estimator.tag)
        self.assertIsInstance(wrapped.estimators, tuple)
        self.assertTrue(wrapped.estimators[0]._marginalized_by_helper)

    def test_composite_distribution_type_metadata_and_container_survive(self):
        class TaggedCompositeDistribution(CompositeDistribution):
            pass

        child = GaussianDistribution(1.0, 2.0)
        original = TaggedCompositeDistribution((marginalized(child),))
        original.tag = object()
        result = unwrap_marginalized(original)

        self.assertIsInstance(result, TaggedCompositeDistribution)
        self.assertIsNot(result, original)
        self.assertIs(result.tag, original.tag)
        self.assertIsInstance(result.dists, tuple)
        self.assertIs(result.dists[0], child)
        self.assertIsInstance(original.dists[0], OptionalDistribution)


if __name__ == "__main__":
    unittest.main()
