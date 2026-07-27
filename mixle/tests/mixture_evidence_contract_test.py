"""Canonical mixture evidence behavior across host and accelerated paths."""

import unittest
from types import SimpleNamespace

import numpy as np

import mixle.stats as stats
from mixle.engines import NUMPY_ENGINE
from mixle.stats.compute.fused_kernels import CompiledMixture
from mixle.stats.compute.kernel import GeneratedNumbaKernel
from mixle.stats.compute.mixture_evidence import (
    IMPOSSIBLE_EVIDENCE_POLICY,
    ImpossibleEvidencePolicy,
    InvalidMixtureEvidenceError,
    normalize_engine_mixture_log_scores,
    normalize_mixture_log_scores,
)
from mixle.stats.compute.posterior import ImpossiblePosteriorError
from mixle.stats.compute.stacked import StackedMixtureKernel
from mixle.stats.latent.mixture import MixtureAccumulator
from mixle.stats.univariate.continuous.gaussian import GaussianAccumulator


class MixtureEvidenceContractTest(unittest.TestCase):
    def test_canonical_nonfinite_rows(self):
        scores = np.array(
            [
                [0.0, -1.0],
                [-np.inf, -np.inf],
                [np.inf, -3.0],
            ]
        )
        result = normalize_mixture_log_scores(scores)
        self.assertEqual(IMPOSSIBLE_EVIDENCE_POLICY, ImpossibleEvidencePolicy.ZERO_RESPONSIBILITY)
        self.assertTrue(np.isneginf(result.log_evidence[1]))
        self.assertTrue(np.isposinf(result.log_evidence[2]))
        np.testing.assert_array_equal(result.responsibilities[1], [0.0, 0.0])
        np.testing.assert_array_equal(result.responsibilities[2], [1.0, 0.0])
        self.assertAlmostEqual(result.responsibilities[0].sum(), 1.0)

        engine_result = normalize_engine_mixture_log_scores(NUMPY_ENGINE.asarray(scores), NUMPY_ENGINE)
        np.testing.assert_array_equal(engine_result.log_evidence, result.log_evidence)
        np.testing.assert_array_equal(engine_result.responsibilities, result.responsibilities)

    def test_nan_and_ambiguous_positive_infinity_raise_typed_error(self):
        for scores in (
            np.array([[np.nan, 0.0]]),
            np.array([[np.inf, np.inf]]),
        ):
            with self.subTest(scores=scores):
                with self.assertRaises(InvalidMixtureEvidenceError):
                    normalize_mixture_log_scores(scores)
                with self.assertRaises(InvalidMixtureEvidenceError):
                    normalize_engine_mixture_log_scores(scores, NUMPY_ENGINE)

    def test_host_mixture_does_not_accumulate_impossible_rows(self):
        model = stats.MixtureDistribution(
            [
                stats.CategoricalDistribution({"a": 1.0, "b": 0.0}),
                stats.CategoricalDistribution({"a": 0.0, "b": 1.0}),
            ],
            [0.5, 0.5],
        )
        enc = model.dist_to_encoder().seq_encode(["a", "outside"])
        posterior = model.seq_posterior(enc)
        np.testing.assert_array_equal(posterior[1], [0.0, 0.0])
        self.assertTrue(np.isneginf(model.seq_log_density(enc)[1]))

        accumulator = model.estimator().accumulator_factory().make()
        accumulator.seq_update(enc, np.ones(2), model)
        self.assertAlmostEqual(float(accumulator.comp_counts.sum()), 1.0)
        with self.assertRaises(ImpossiblePosteriorError):
            model.latent_posterior(["outside"])

    def test_constructor_owns_weights_and_component_container(self):
        components = [stats.GaussianDistribution(0.0, 1.0), stats.GaussianDistribution(2.0, 1.0)]
        weights = np.asarray([0.25, 0.75])
        model = stats.MixtureDistribution(components, weights)
        original_score = model.log_density(0.0)

        weights[:] = [0.9, 0.1]
        components.clear()

        np.testing.assert_array_equal(model.w, [0.25, 0.75])
        self.assertEqual(len(model.components), 2)
        self.assertEqual(model.log_density(0.0), original_score)

    def test_specialized_mixture_constructors_enforce_owned_probability_geometry(self):
        components = [
            stats.CategoricalDistribution({"a": 1.0}),
            stats.CategoricalDistribution({"b": 1.0}),
        ]
        weights = np.asarray([0.25, 0.75])
        heterogeneous = stats.HeterogeneousMixtureDistribution(components, weights)
        semi_supervised = stats.SemiSupervisedMixtureDistribution(components, weights)
        topics = list(components)
        outer = np.asarray([0.4, 0.6])
        inner = np.asarray([[0.8, 0.2], [0.1, 0.9]])
        hierarchical = stats.HierarchicalMixtureDistribution(topics, outer, inner)

        weights[:] = [0.9, 0.1]
        components.clear()
        outer[:] = [0.7, 0.3]
        inner[:] = [[1.0, 0.0], [1.0, 0.0]]
        topics.clear()

        np.testing.assert_array_equal(heterogeneous.w, [0.25, 0.75])
        np.testing.assert_array_equal(semi_supervised.w, [0.25, 0.75])
        np.testing.assert_array_equal(hierarchical.w, [0.4, 0.6])
        np.testing.assert_array_equal(hierarchical.taus, [[0.8, 0.2], [0.1, 0.9]])
        self.assertEqual(len(heterogeneous.components), 2)
        self.assertEqual(len(semi_supervised.components), 2)
        self.assertEqual(len(hierarchical.topics), 2)

        for mixture_type in (
            stats.HeterogeneousMixtureDistribution,
            stats.SemiSupervisedMixtureDistribution,
        ):
            with self.subTest(mixture_type=mixture_type.__name__):
                with self.assertRaises(ValueError):
                    mixture_type([stats.CategoricalDistribution({"a": 1.0})], [0.5, 0.5])
                with self.assertRaises(ValueError):
                    mixture_type([stats.CategoricalDistribution({"a": 1.0})], [np.nan])

        with self.assertRaises(ValueError):
            stats.HierarchicalMixtureDistribution(
                [stats.CategoricalDistribution({"a": 1.0})],
                [0.5, 0.5],
                [[1.0], [0.5]],
            )

    def test_scoring_only_categorical_cannot_become_a_generative_mixture_component(self):
        scorer = stats.CategoricalDistribution({"a": 0.5}, scoring_only=True)
        for constructor in (
            lambda: stats.MixtureDistribution([scorer], [1.0]),
            lambda: stats.HeterogeneousMixtureDistribution([scorer], [1.0]),
            lambda: stats.SemiSupervisedMixtureDistribution([scorer], [1.0]),
            lambda: stats.HierarchicalMixtureDistribution([scorer], [1.0], [[1.0]]),
        ):
            with self.subTest(constructor=constructor):
                with self.assertRaisesRegex(TypeError, "generative probability laws"):
                    constructor()

    def test_specialized_mixtures_give_impossible_evidence_zero_responsibility(self):
        components = [
            stats.CategoricalDistribution({"a": 1.0}),
            stats.CategoricalDistribution({"b": 1.0}),
        ]
        heterogeneous = stats.HeterogeneousMixtureDistribution(components, [0.5, 0.5])
        h_enc = heterogeneous.dist_to_encoder().seq_encode(["outside"])
        np.testing.assert_array_equal(heterogeneous.posterior("outside"), [0.0, 0.0])
        np.testing.assert_array_equal(heterogeneous.seq_posterior(h_enc), [[0.0, 0.0]])
        h_acc = heterogeneous.estimator().accumulator_factory().make()
        h_acc.seq_update(h_enc, np.ones(1), heterogeneous)
        np.testing.assert_array_equal(h_acc.comp_counts, [0.0, 0.0])

        semi_supervised = stats.SemiSupervisedMixtureDistribution(components, [0.5, 0.5])
        observation = ("outside", [(0, 1.0)])
        s_enc = semi_supervised.dist_to_encoder().seq_encode([observation])
        np.testing.assert_array_equal(semi_supervised.posterior(observation), [0.0, 0.0])
        np.testing.assert_array_equal(semi_supervised.seq_posterior(s_enc), [[0.0, 0.0]])
        s_acc = semi_supervised.estimator().accumulator_factory().make()
        s_acc.seq_update(s_enc, np.ones(1), semi_supervised)
        np.testing.assert_array_equal(s_acc.comp_counts, [0.0, 0.0])

        hierarchical = stats.HierarchicalMixtureDistribution(
            components,
            [0.4, 0.6],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        impossible = hierarchical.dist_to_encoder().seq_encode([["outside"]])
        empty = hierarchical.dist_to_encoder().seq_encode([[]])
        np.testing.assert_array_equal(hierarchical.seq_posterior(impossible), [[0.0, 0.0]])
        np.testing.assert_allclose(hierarchical.seq_posterior(empty), [[0.4, 0.6]])
        hierarchical_acc = hierarchical.estimator().accumulator_factory().make()
        hierarchical_acc.seq_update(impossible, np.ones(1), hierarchical)
        np.testing.assert_array_equal(hierarchical_acc.w_counts, [0.0, 0.0])
        np.testing.assert_array_equal(hierarchical_acc.comp_counts, np.zeros((2, 2)))

    def test_specialized_accumulator_snapshots_own_their_counts(self):
        components = [
            stats.CategoricalDistribution({"a": 1.0}),
            stats.CategoricalDistribution({"b": 1.0}),
        ]
        models = [
            stats.HeterogeneousMixtureDistribution(components, [0.5, 0.5]),
            stats.SemiSupervisedMixtureDistribution(components, [0.5, 0.5]),
            stats.HierarchicalMixtureDistribution(components, [0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]]),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):
                accumulator = model.estimator().accumulator_factory().make()
                accumulator.comp_counts[...] = 2.0
                snapshot = accumulator.value()
                snapshot[0][...] = 99.0
                np.testing.assert_array_equal(accumulator.comp_counts, np.full_like(accumulator.comp_counts, 2.0))

    def test_semi_supervised_prior_requires_exact_finite_entries(self):
        model = stats.SemiSupervisedMixtureDistribution(
            [stats.CategoricalDistribution({"a": 1.0}), stats.CategoricalDistribution({"b": 1.0})],
            [0.5, 0.5],
        )
        invalid_priors = [[(0.5, 1.0)], [(True, 1.0)], [(0, np.nan)], [(0, np.inf)]]
        for prior in invalid_priors:
            with self.subTest(prior=prior):
                with self.assertRaises((TypeError, ValueError)):
                    model.posterior(("a", prior))
                with self.assertRaises((TypeError, ValueError)):
                    model.dist_to_encoder().seq_encode([("a", prior)])

    def test_mixture_accumulator_owns_count_snapshots(self):
        accumulator = (
            stats.MixtureDistribution(
                [stats.GaussianDistribution(0.0, 1.0), stats.GaussianDistribution(1.0, 1.0)],
                [0.5, 0.5],
            )
            .estimator()
            .accumulator_factory()
            .make()
        )
        accumulator.comp_counts[:] = [1.0, 2.0]

        snapshot = accumulator.value()
        snapshot[0][0] = 99.0
        self.assertEqual(accumulator.comp_counts[0], 1.0)

        donor_counts = np.asarray([3.0, 4.0])
        accumulator.from_value((donor_counts, snapshot[1]))
        donor_counts[0] = 88.0
        self.assertEqual(accumulator.comp_counts[0], 3.0)

    def test_keyed_component_pooling_restores_independent_accumulators(self):
        first = MixtureAccumulator([GaussianAccumulator()], keys=(None, "shared-components"))
        second = MixtureAccumulator([GaussianAccumulator()], keys=(None, "shared-components"))
        first.accumulators[0].update(1.0, 1.0, None)
        second.accumulators[0].update(3.0, 1.0, None)

        pooled = {}
        first.key_merge(pooled)
        second.key_merge(pooled)
        first.key_replace(pooled)
        second.key_replace(pooled)

        self.assertIsNot(first.accumulators, second.accumulators)
        self.assertIsNot(first.accumulators[0], second.accumulators[0])
        self.assertEqual(first.accumulators[0].value(), second.accumulators[0].value())
        first.accumulators[0].update(100.0, 1.0, None)
        self.assertNotEqual(first.accumulators[0].value(), second.accumulators[0].value())

    def test_object_and_compiled_fast_paths_share_the_policy(self):
        weighted = np.array(
            [
                [0.0, -1.0],
                [-np.inf, -np.inf],
                [np.inf, -2.0],
                [-3.0, np.inf],
            ]
        )
        model = SimpleNamespace(log_w=np.array([0.0, -np.inf]), zw=np.array([False, True]))

        generated = object.__new__(GeneratedNumbaKernel)
        generated.components = (object(), object())
        generated.dist = model
        generated.component_scores = lambda _enc: weighted.copy()
        np.testing.assert_array_equal(generated.posteriors(None)[1], [0.0, 0.0])
        np.testing.assert_array_equal(generated.posteriors(None)[2], [1.0, 0.0])
        np.testing.assert_array_equal(generated.posteriors(None)[3], [1.0, 0.0])

        compiled = object.__new__(CompiledMixture)
        compiled.model = model
        compiled.is_mixture = True
        compiled.seq_component_log_density = lambda _enc, _model: weighted.copy()
        np.testing.assert_array_equal(compiled.posteriors(None, model)[1], [0.0, 0.0])
        np.testing.assert_array_equal(compiled.posteriors(None, model)[2], [1.0, 0.0])
        np.testing.assert_array_equal(compiled.posteriors(None, model)[3], [1.0, 0.0])

    def test_stacked_kernel_uses_zero_for_impossible_rows(self):
        model = stats.MixtureDistribution(
            [
                stats.CategoricalDistribution({"a": 1.0, "b": 0.0}),
                stats.CategoricalDistribution({"a": 0.0, "b": 1.0}),
            ],
            [0.5, 0.5],
        )
        enc = model.dist_to_encoder().seq_encode(["a", "outside"])
        kernel = StackedMixtureKernel(model, NUMPY_ENGINE)
        self.assertTrue(np.isneginf(kernel.score(enc)[1]))
        np.testing.assert_array_equal(kernel.posteriors(enc)[1], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
