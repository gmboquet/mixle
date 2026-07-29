import unittest

import numpy as np

from mixle.stats import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagonalGaussianDistribution,
    ExponentialDistribution,
    GammaDistribution,
    GaussianDistribution,
    GeometricDistribution,
    HalfNormalDistribution,
    IntegerCategoricalDistribution,
    InverseGammaDistribution,
    InverseGaussianDistribution,
    LogGaussianDistribution,
    MultivariateGaussianDistribution,
    NegativeBinomialDistribution,
    ParetoDistribution,
    PoissonDistribution,
    RayleighDistribution,
    VonMisesDistribution,
    conjugate_posterior,
    mixture_conjugate_posterior,
)


class ConjugateGeometryContractTest(unittest.TestCase):
    def test_univariate_data_and_weights_require_exact_vector_geometry(self):
        dist = PoissonDistribution(1.0)
        invalid = (
            (np.array([[1.0], [2.0]]), None),
            (np.array([1.0, 2.0]), np.ones((2, 1))),
            (np.array([1.0, 2.0]), np.ones((2, 2))),
            (np.array([1.0, 2.0]), np.ones(1)),
        )
        for data, weights in invalid:
            with self.subTest(data_shape=repr(data.shape), weights=repr(None if weights is None else weights.shape)):
                with self.assertRaises(ValueError):
                    conjugate_posterior(dist, data, weights=weights)

    def test_sampling_cardinality_requires_a_non_negative_integer(self):
        posterior = conjugate_posterior(BernoulliDistribution(0.5), [0, 1])
        for size in (True, 1.5, -1):
            with self.subTest(size=repr(size)):
                with self.assertRaises((TypeError, ValueError)):
                    posterior.sample(size)
                with self.assertRaises((TypeError, ValueError)):
                    posterior.sampler(seed=1).sample(size)
        self.assertEqual(posterior.sample(np.int64(0))["p"].shape, (0,))

    def test_multivariate_geometry_and_spd_prior_are_exact(self):
        dist = MultivariateGaussianDistribution(np.zeros(2), np.eye(2))
        invalid_calls = (
            ([1.0, 2.0], None, None),
            ([[1.0, 2.0]], [[1.0]], None),
            ([[1.0, 2.0]], None, {"m": [0.0]}),
            ([[1.0, 2.0]], None, {"kappa": 0.0}),
            ([[1.0, 2.0]], None, {"nu": 1.0}),
            ([[1.0, 2.0]], None, {"psi": [[1.0, 0.5], [0.0, 1.0]]}),
            ([[1.0, 2.0]], None, {"psi": [[1.0, 2.0], [2.0, 1.0]]}),
        )
        for data, weights, prior in invalid_calls:
            with self.subTest(data=repr(data), weights=repr(weights), prior=repr(prior)):
                with self.assertRaises(ValueError):
                    conjugate_posterior(dist, data, weights=weights, prior=prior)

        prior_only = conjugate_posterior(
            dist,
            [],
            prior={"m": [1.0, -1.0], "kappa": 2.0, "nu": 4.0, "psi": np.eye(2)},
        )
        np.testing.assert_array_equal(prior_only.m, [1.0, -1.0])
        np.testing.assert_array_equal(prior_only.psi, np.eye(2))

    def test_diagonal_geometry_and_per_dimension_priors_are_validated(self):
        dist = DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            conjugate_posterior(dist, [[1.0]])
        with self.assertRaises(ValueError):
            conjugate_posterior(dist, [[1.0, 2.0]], weights=[[1.0]])
        with self.assertRaises(ValueError):
            conjugate_posterior(dist, [[1.0, 2.0]], prior={"m": [0.0]})
        with self.assertRaises(ValueError):
            conjugate_posterior(dist, [[1.0, 2.0]], prior={"a": [1.0, 0.0]})

        posterior = conjugate_posterior(
            dist,
            [],
            prior={
                "m": [1.0, -1.0],
                "kappa": [2.0, 3.0],
                "a": [4.0, 5.0],
                "b": [6.0, 7.0],
            },
        )
        np.testing.assert_array_equal(posterior.mean()["mu"], [1.0, -1.0])


class ConjugateSupportAndPriorContractTest(unittest.TestCase):
    def test_each_likelihood_rejects_observations_outside_its_support(self):
        invalid = (
            (GeometricDistribution(0.5), [0]),
            (GeometricDistribution(0.5), [1.5]),
            (PoissonDistribution(1.0), [-1]),
            (PoissonDistribution(1.0), [1.5]),
            (ExponentialDistribution(1.0), [-1.0]),
            (LogGaussianDistribution(0.0, 1.0), [0.0]),
            (RayleighDistribution(1.0), [-1.0]),
            (HalfNormalDistribution(1.0), [-1.0]),
            (GammaDistribution(2.0, 1.0), [0.0]),
            (InverseGammaDistribution(2.0, 1.0), [0.0]),
            (InverseGaussianDistribution(1.0, 1.0), [0.0]),
            (ParetoDistribution(2.0, 3.0), [1.999]),
            (NegativeBinomialDistribution(2.0, 0.5), [-1]),
            (NegativeBinomialDistribution(2.0, 0.5), [0.5]),
        )
        for dist, data in invalid:
            with self.subTest(distribution=type(dist).__name__, data=repr(data)):
                with self.assertRaises(ValueError):
                    conjugate_posterior(dist, data)

    def test_positive_prior_hyperparameters_are_enforced(self):
        invalid = (
            (PoissonDistribution(1.0), [1], {"shape": 0.0}),
            (ExponentialDistribution(1.0), [1.0], {"rate": np.inf}),
            (GaussianDistribution(0.0, 1.0), [0.0], {"m": np.nan}),
            (GaussianDistribution(0.0, 1.0), [0.0], {"kappa": 0.0}),
            (GaussianDistribution(0.0, 1.0), [0.0], {"a": -1.0}),
            (RayleighDistribution(1.0), [1.0], {"b": 0.0}),
            (GammaDistribution(2.0, 1.0), [1.0], {"shape": np.nan}),
            (VonMisesDistribution(0.0, 1.0), [0.0], {"m": np.inf}),
            (VonMisesDistribution(0.0, 1.0), [0.0], {"R": -1.0}),
        )
        for dist, data, prior in invalid:
            with self.subTest(distribution=type(dist).__name__, prior=repr(prior)):
                with self.assertRaises(ValueError):
                    conjugate_posterior(dist, data, prior=prior)

    def test_categorical_updates_validate_alignment_support_and_prior(self):
        dist = CategoricalDistribution({"a": 0.5, "b": 0.5})
        with self.assertRaises(ValueError):
            conjugate_posterior(dist, ["a", "b"], weights=[1.0])
        with self.assertRaises(ValueError):
            conjugate_posterior(dist, ["a", "b"], weights=[1.0, -1.0])
        with self.assertRaises(ValueError):
            conjugate_posterior(dist, ["a", "outside"])
        with self.assertRaises(ValueError):
            conjugate_posterior(dist, ["a"], prior={"alpha": 0.0})
        with self.assertRaises(ValueError):
            conjugate_posterior(dist, ["a"], prior={"alpha": [1.0]})

        empty = conjugate_posterior(
            IntegerCategoricalDistribution(3, [0.5, 0.5]),
            [],
        )
        self.assertEqual(empty.support, [3, 4])

    def test_mixed_categorical_label_types_are_supported(self):
        posterior = conjugate_posterior(
            CategoricalDistribution({}),
            [1, "one", 1, "one"],
        )
        self.assertEqual(set(posterior.support), {1, "one"})


class ConjugatePredictiveAndMixtureContractTest(unittest.TestCase):
    def test_plugin_predictives_carry_machine_readable_receipts(self):
        cases = (
            conjugate_posterior(GeometricDistribution(0.5), [1, 2]),
            conjugate_posterior(ExponentialDistribution(1.0), [1.0, 2.0]),
            conjugate_posterior(LogGaussianDistribution(0.0, 1.0), [1.0, 2.0]),
            conjugate_posterior(RayleighDistribution(1.0), [1.0, 2.0]),
            conjugate_posterior(GammaDistribution(2.0, 1.0), [1.0, 2.0]),
            conjugate_posterior(
                DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0]),
                [[1.0, 2.0], [2.0, 3.0]],
            ),
            conjugate_posterior(VonMisesDistribution(0.0, 1.0), [0.0, 1.0]),
        )
        for posterior in cases:
            with self.subTest(family=repr(posterior.family)):
                predictive = posterior.posterior_predictive()
                self.assertFalse(predictive.is_exact_posterior_predictive)
                self.assertEqual(predictive.posterior_predictive_kind, "posterior_mean_plugin")
                self.assertEqual(
                    predictive.approximation_receipt["method"],
                    "posterior_mean_plugin",
                )

    def test_mixture_rejects_bad_probabilities_and_all_zero_evidence(self):
        priors = [{"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 2.0}]
        invalid_weights = ([1.0], [-1.0, 2.0], [0.0, 0.0], [np.nan, 1.0])
        for weights in invalid_weights:
            with self.subTest(weights=repr(weights)):
                with self.assertRaises(ValueError):
                    mixture_conjugate_posterior(
                        BernoulliDistribution(0.5),
                        [0, 1],
                        priors,
                        prior_weights=weights,
                    )

        with np.errstate(divide="ignore"):
            with self.assertRaisesRegex(ValueError, "zero evidence"):
                mixture_conjugate_posterior(
                    RayleighDistribution(1.0),
                    [0.0],
                    [{"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 2.0}],
                )

    def test_mixture_posterior_constructor_validates_all_aligned_vectors(self):
        from mixle.stats.bayes.conjugate import MixtureConjugatePosterior

        component = conjugate_posterior(BernoulliDistribution(0.5), [0, 1])
        invalid = (
            ([], [], [], []),
            ([component], [1.0, 0.0], [1.0], [0.0]),
            ([component], [1.0], [1.0], [0.0, 1.0]),
            ([component], [1.0], [-1.0], [0.0]),
            ([component], [1.0], [1.0], [np.nan]),
        )
        for components, post_weights, prior_weights, evidence in invalid:
            with self.subTest(
                component_count=len(components),
                post_weights=post_weights,
                prior_weights=prior_weights,
                evidence=evidence,
            ):
                with self.assertRaises(ValueError):
                    MixtureConjugatePosterior(
                        components,
                        post_weights,
                        prior_weights,
                        evidence,
                    )

    def test_one_shot_data_is_replayed_and_mapping_means_are_averaged(self):
        priors = [{"alpha": [8.0, 2.0]}, {"alpha": [2.0, 8.0]}]
        dist = CategoricalDistribution({"a": 0.5, "b": 0.5})
        from_iterator = mixture_conjugate_posterior(
            dist,
            iter(["a", "a", "b"]),
            priors,
            prior_weights=[0.25, 0.75],
        )
        from_list = mixture_conjugate_posterior(
            dist,
            ["a", "a", "b"],
            priors,
            prior_weights=[0.25, 0.75],
        )
        np.testing.assert_allclose(from_iterator.weights, from_list.weights)
        self.assertEqual(from_iterator.mean()["map"].keys(), {"a", "b"})
        for label in ("a", "b"):
            expected = sum(
                weight * component.mean()["map"][label]
                for weight, component in zip(from_iterator.weights, from_iterator.components)
            )
            self.assertAlmostEqual(from_iterator.mean()["map"][label], expected)


if __name__ == "__main__":
    unittest.main()
