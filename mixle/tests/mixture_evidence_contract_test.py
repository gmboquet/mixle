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
