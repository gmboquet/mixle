import unittest

import numpy as np

from mixle.inference import squarem_packer
from mixle.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    ExponentialDistribution,
    GaussianDistribution,
    LaplaceDistribution,
    MixtureDistribution,
    PoissonDistribution,
)


class TaggedMixture(MixtureDistribution):
    pass


class TaggedComposite(CompositeDistribution):
    pass


class SquaremPackerContractTest(unittest.TestCase):
    def test_round_trip_preserves_boundaries_metadata_and_container_types(self):
        component = TaggedComposite(
            (
                GaussianDistribution(1.5, 2.0, name="normal", keys="normal-key"),
                ExponentialDistribution(3.0, name="exponential", keys="exponential-key"),
                PoissonDistribution(4.0, name="poisson", keys="poisson-key"),
                LaplaceDistribution(-2.0, 0.75, name="laplace", keys="laplace-key"),
                CategoricalDistribution(
                    {1: 1.0, "impossible": 0.0},
                    default_value=0.125,
                    name="categorical",
                    keys="categorical-key",
                    scoring_only=True,
                ),
            )
        )
        component.tag = "component metadata"
        model = TaggedMixture(
            [
                component,
                CategoricalDistribution(
                    {"impossible": 0.0, 1: 1.0},
                    name="inactive",
                    keys="inactive-key",
                ),
            ],
            [1.0, 0.0],
            name="outer",
        )
        model.tag = "mixture metadata"

        pack, unpack = squarem_packer(model)
        rebuilt = unpack(pack(model))

        self.assertIsInstance(rebuilt, TaggedMixture)
        self.assertIsInstance(rebuilt.components, list)
        self.assertEqual(rebuilt.tag, "mixture metadata")
        self.assertEqual(rebuilt.name, "outer")
        np.testing.assert_array_equal(rebuilt.w, [1.0, 0.0])
        np.testing.assert_array_equal(rebuilt.zw, [False, True])
        self.assertEqual(rebuilt.log_w[1], -np.inf)

        rebuilt_component = rebuilt.components[0]
        self.assertIsInstance(rebuilt_component, TaggedComposite)
        self.assertIsInstance(rebuilt_component.dists, tuple)
        self.assertEqual(rebuilt_component.tag, "component metadata")
        expected_fields = [
            ("normal", "normal-key"),
            ("exponential", "exponential-key"),
            ("poisson", "poisson-key"),
            ("laplace", "laplace-key"),
            ("categorical", "categorical-key"),
        ]
        self.assertEqual([(dist.name, dist.keys) for dist in rebuilt_component.dists], expected_fields)
        categorical = rebuilt_component.dists[-1]
        self.assertEqual(categorical.pmap, {1: 1.0, "impossible": 0.0})
        self.assertEqual(categorical.default_value, 0.125)
        self.assertTrue(categorical.scoring_only)

    def test_pack_rejects_changed_support_boundary_or_metadata(self):
        model = MixtureDistribution(
            [
                CategoricalDistribution(
                    {"a": 0.75, "b": 0.25, "unused": 0.0},
                    name="labels",
                    keys="shared",
                ),
                GaussianDistribution(0.0, 1.0, name="normal", keys="normal-key"),
            ],
            [1.0, 0.0],
            name="mixture",
        )
        pack, _ = squarem_packer(model)

        changed_labels = MixtureDistribution(
            [
                CategoricalDistribution(
                    {"a": 0.75, "c": 0.25, "unused": 0.0},
                    name="labels",
                    keys="shared",
                ),
                GaussianDistribution(0.0, 1.0, name="normal", keys="normal-key"),
            ],
            [1.0, 0.0],
            name="mixture",
        )
        changed_category_boundary = MixtureDistribution(
            [
                CategoricalDistribution(
                    {"a": 0.75, "b": 0.0, "unused": 0.25},
                    name="labels",
                    keys="shared",
                ),
                GaussianDistribution(0.0, 1.0, name="normal", keys="normal-key"),
            ],
            [1.0, 0.0],
            name="mixture",
        )
        changed_mixture_boundary = MixtureDistribution(model.components, [0.5, 0.5], name="mixture")
        changed_metadata = MixtureDistribution(
            [
                CategoricalDistribution(
                    {"a": 0.75, "b": 0.25, "unused": 0.0},
                    name="different",
                    keys="shared",
                ),
                model.components[1],
            ],
            [1.0, 0.0],
            name="mixture",
        )

        for changed in (
            changed_labels,
            changed_category_boundary,
            changed_mixture_boundary,
            changed_metadata,
        ):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, "structure or fixed support"):
                    pack(changed)

    def test_invalid_coordinate_vectors_and_generated_values_are_rejected(self):
        model = MixtureDistribution([GaussianDistribution(0.0, 1.0)], [1.0])
        pack, unpack = squarem_packer(model)
        theta = pack(model)

        invalid = (
            theta[:-1],
            theta.reshape(1, -1),
            np.full_like(theta, np.nan),
            np.full_like(theta, np.inf),
            ["not", "numeric", "coordinates"],
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    unpack(values)

        overflow = theta.copy()
        overflow[1] = 1.0e308
        with self.assertRaisesRegex(ValueError, "overflowed"):
            unpack(overflow)

        model.components[0].mu = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            pack(model)

    def test_non_simplex_categorical_models_are_rejected(self):
        model = MixtureDistribution(
            [CategoricalDistribution({"a": 2.0, "b": 1.0}, scoring_only=True)],
            [1.0],
        )
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            squarem_packer(model)


if __name__ == "__main__":
    unittest.main()
