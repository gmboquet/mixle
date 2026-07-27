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

    def test_scoring_only_categorical_cannot_become_a_generative_mixture_component(self):
        scorer = stats.CategoricalDistribution({"a": 0.5}, scoring_only=True)
        with self.assertRaisesRegex(TypeError, "generative probability laws"):
            stats.MixtureDistribution([scorer], [1.0])

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
