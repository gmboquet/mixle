import unittest
import io
import copy

import numpy as np

from pysp.engines import NUMPY_ENGINE, NumpyEngine
from pysp.stats import AffineTransform, BetaDistribution, BinomialDistribution, CategoricalDistribution, \
    CategoricalEstimator, CompositeDistribution, CompositeEstimator, DiagonalGaussianDistribution, GaussianDistribution, \
    GaussianEstimator, \
    MixtureDistribution, MixtureEstimator, MultivariateGaussianDistribution, OptionalDistribution, PoissonDistribution, \
    SequenceDistribution, SpearmanRankingDistribution, TransformDistribution, WeightedDistribution, EncodedData, \
    GeneratedNumbaKernel, \
    EngineNotSupportedError, NumbaKernelFactory, \
    encoded_nbytes, generated_log_density, generated_numba_log_density, generated_numba_stacked_log_density, \
    generated_stacked_log_density, generated_stacked_params, generated_sufficient_statistics, kernel_for, \
    seq_estimate, seq_log_density_sum
from pysp.stats.stacked import StackedMixtureKernel, estimate_component_shard_value, tie_component_shard_values
from pysp.stats.kernel import GenericKernel, KernelFactory
from pysp.utils.estimation import optimize


def _assert_stats_close(test_case, actual, expected):
    if isinstance(actual, (tuple, list)):
        test_case.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            _assert_stats_close(test_case, a, e)
        return
    np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                               rtol=1.0e-12, atol=1.0e-12)


def _assert_mixture_close(test_case, actual, expected):
    np.testing.assert_allclose(actual.w, expected.w, rtol=1.0e-12, atol=1.0e-12)
    test_case.assertEqual(len(actual.components), len(expected.components))
    for a, e in zip(actual.components, expected.components):
        test_case.assertIsInstance(a, GaussianDistribution)
        np.testing.assert_allclose([a.mu, a.sigma2], [e.mu, e.sigma2], rtol=1.0e-12, atol=1.0e-12)


def _apply_global_key_merge(*accumulators):
    stats_dict = {}
    for acc in accumulators:
        acc.key_merge(stats_dict)
    for acc in accumulators:
        acc.key_replace(stats_dict)


def _split_component_shards(value):
    counts, comp_stats = value
    return tuple(
        (idx, (np.asarray(counts[idx:idx + 1], dtype=np.float64).copy(), (copy.deepcopy(comp_stats[idx]),)))
        for idx in range(len(counts))
    )


def _merge_component_shards(shards):
    ordered = sorted(shards, key=lambda item: item[0])
    counts = np.concatenate([value[0] for _, value in ordered])
    stats = tuple(value[1][0] for _, value in ordered)
    return counts, stats


class ComputeKernelTestCase(unittest.TestCase):

    class FakeTorchEngine(NumpyEngine):
        name = 'torch'

    def test_gaussian_kernel_matches_legacy_seq_paths(self):
        dist = GaussianDistribution(0.5, 2.0)
        est = GaussianEstimator()
        data = np.asarray([-2.0, -0.5, 0.0, 1.0, 2.5])
        weights = np.asarray([0.2, 1.0, 0.7, 1.3, 0.4])
        enc = dist.dist_to_encoder().seq_encode(data)

        kernel = dist.kernel(estimator=est)
        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(enc), rtol=0, atol=0)

        legacy_acc = est.accumulator_factory().make()
        legacy_acc.seq_update(enc, weights, dist)
        _assert_stats_close(self, kernel.accumulate(enc, weights), legacy_acc.value())

    def test_mixture_kernel_matches_component_scores_and_suff_stats(self):
        dist = MixtureDistribution(
            [GaussianDistribution(-2.0, 0.8), GaussianDistribution(2.5, 1.4)],
            [0.35, 0.65],
        )
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        data = dist.sampler(seed=3).sample(size=80)
        weights = np.linspace(0.2, 1.4, len(data))
        enc = dist.dist_to_encoder().seq_encode(data)

        kernel = kernel_for(dist, estimator=est)
        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(enc), rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(kernel.component_scores(enc), dist.seq_component_log_density(enc),
                                   rtol=1.0e-12, atol=1.0e-12)

        legacy_acc = est.accumulator_factory().make()
        legacy_acc.seq_update(enc, weights, dist)
        _assert_stats_close(self, kernel.accumulate(enc, weights), legacy_acc.value())

    def test_component_shard_key_tying_matches_child_key_merge(self):
        dist = MixtureDistribution(
            [GaussianDistribution(-2.0, 0.8), GaussianDistribution(2.5, 1.4)],
            [0.35, 0.65],
        )
        est = MixtureEstimator([
            GaussianEstimator(keys='shared_gaussian'),
            GaussianEstimator(keys='shared_gaussian'),
        ])
        data = dist.sampler(seed=30).sample(size=70)
        weights = np.linspace(0.2, 1.4, len(data))
        enc = dist.dist_to_encoder().seq_encode(data)

        acc = est.accumulator_factory().make()
        acc.seq_update(enc, weights, dist)
        expected = est.accumulator_factory().make().from_value(acc.value())
        _apply_global_key_merge(expected)

        tied = tie_component_shard_values(est, _split_component_shards(acc.value()))
        expected_model = est.estimate(None, expected.value())

        _assert_stats_close(self, _merge_component_shards(tied), expected.value())
        for start, value in tied:
            shard = estimate_component_shard_value(est, start, value, total_count=float(expected.value()[0].sum()))
            _assert_stats_close(self, shard.weights, expected_model.w[start:start + 1])
            _assert_stats_close(
                self,
                [shard.components[0].mu, shard.components[0].sigma2],
                [expected_model.components[start].mu, expected_model.components[start].sigma2],
            )

    def test_component_shard_key_tying_matches_mixture_weight_and_component_keys(self):
        dist = MixtureDistribution(
            [GaussianDistribution(-2.0, 0.8), GaussianDistribution(2.5, 1.4)],
            [0.35, 0.65],
        )
        est = MixtureEstimator([
            GaussianEstimator(),
            GaussianEstimator(),
        ], keys=('shared_weights', 'shared_components'))
        data = dist.sampler(seed=31).sample(size=80)
        left, right = data[:35], data[35:]
        enc_left = dist.dist_to_encoder().seq_encode(left)
        enc_right = dist.dist_to_encoder().seq_encode(right)

        acc_left = est.accumulator_factory().make()
        acc_right = est.accumulator_factory().make()
        acc_left.seq_update(enc_left, np.linspace(0.3, 1.1, len(left)), dist)
        acc_right.seq_update(enc_right, np.linspace(0.4, 1.2, len(right)), dist)
        shard_values = _split_component_shards(acc_left.value()) + _split_component_shards(acc_right.value())
        expected_left = est.accumulator_factory().make().from_value(acc_left.value())
        expected_right = est.accumulator_factory().make().from_value(acc_right.value())
        _apply_global_key_merge(expected_left, expected_right)

        tied = tie_component_shard_values(est, shard_values)
        expected = expected_left.value()
        expected_model = est.estimate(None, expected)

        for start, value in tied:
            _assert_stats_close(self, value[0], expected[0][start:start + 1])
            _assert_stats_close(self, value[1], (expected[1][start],))
            shard = estimate_component_shard_value(est, start, value, total_count=float(expected[0].sum()))
            _assert_stats_close(self, shard.weights, expected_model.w[start:start + 1])
            _assert_stats_close(
                self,
                [shard.components[0].mu, shard.components[0].sigma2],
                [expected_model.components[start].mu, expected_model.components[start].sigma2],
            )

    def test_kernel_refresh_updates_parameters_without_rebuilding(self):
        dist = GaussianDistribution(0.0, 1.0)
        replacement = GaussianDistribution(3.0, 0.5)
        enc = dist.dist_to_encoder().seq_encode(np.asarray([-1.0, 0.0, 1.0]))
        kernel = dist.kernel()

        before = kernel.score(enc)
        kernel.refresh(replacement)
        after = kernel.score(enc)

        np.testing.assert_allclose(after, replacement.seq_log_density(enc), rtol=0, atol=0)
        self.assertFalse(np.allclose(before, after))

    def test_registered_factory_overrides_generic_kernel(self):
        class TaggedKernel(GenericKernel):
            tag = 'custom'

        class TaggedFactory(KernelFactory):
            def build(self, dist, engine, estimator=None):
                return TaggedKernel(dist, engine=engine, estimator=estimator)

        class TaggedGaussian(GaussianDistribution):
            pass

        from pysp.stats import register_kernel_factory
        register_kernel_factory(TaggedGaussian, TaggedFactory())

        kernel = TaggedGaussian(0.0, 1.0).kernel()
        self.assertIsInstance(kernel, TaggedKernel)

    def test_generic_kernel_rejects_unsupported_engine(self):
        class NumpyOnlyGaussian(GaussianDistribution):
            engine_ready = ('numpy',)

        dist = NumpyOnlyGaussian(0.0, 1.0)
        self.assertEqual(dist.supported_engines(), ('numpy',))
        self.assertTrue(dist.supports_engine(NUMPY_ENGINE))
        self.assertFalse(dist.supports_engine(self.FakeTorchEngine()))

        with self.assertRaises(EngineNotSupportedError):
            dist.kernel(engine=self.FakeTorchEngine())

    def test_generic_kernel_allows_distribution_engine_opt_in(self):
        class TorchReadyGaussian(GaussianDistribution):
            engine_ready = ('numpy', 'torch')

        dist = TorchReadyGaussian(0.0, 1.0)
        enc = dist.dist_to_encoder().seq_encode(np.asarray([-1.0, 0.0, 1.0]))
        kernel = dist.kernel(engine=self.FakeTorchEngine())

        self.assertIsInstance(kernel, GenericKernel)
        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(enc), rtol=0, atol=0)

    def test_optimize_with_numpy_engine_matches_legacy_mixture_em(self):
        truth = MixtureDistribution(
            [GaussianDistribution(-2.0, 0.7), GaussianDistribution(2.0, 1.1)],
            [0.4, 0.6],
        )
        start = MixtureDistribution(
            [GaussianDistribution(-1.0, 2.0), GaussianDistribution(1.0, 2.0)],
            [0.5, 0.5],
        )
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        data = truth.sampler(seed=12).sample(size=80)

        legacy = optimize(data, est, max_its=3, delta=None, prev_estimate=start, out=io.StringIO())
        engine_fit = optimize(data, est, max_its=3, delta=None, prev_estimate=start, engine=NUMPY_ENGINE,
                              out=io.StringIO())

        _assert_mixture_close(self, engine_fit, legacy)

    def test_optimize_uses_engine_kernel_and_rejects_unsupported_families(self):
        dist = GaussianDistribution(0.0, 1.0)
        est = GaussianEstimator()
        data = np.asarray([-1.0, 0.0, 1.0, 2.0])
        fitted = optimize(data, est, max_its=1, delta=None, prev_estimate=dist,
                          engine=self.FakeTorchEngine(), out=io.StringIO())
        self.assertIsInstance(fitted, GaussianDistribution)

        class NumpyOnlyGaussian(GaussianDistribution):
            engine_ready = ('numpy',)

        numpy_only = NumpyOnlyGaussian(0.0, 1.0)
        with self.assertRaises(EngineNotSupportedError):
            optimize(data, est, max_its=1, delta=None,
                     prev_estimate=numpy_only, engine=self.FakeTorchEngine(), out=io.StringIO())

    def test_default_kernel_dispatch_selects_generated_numba_on_numpy(self):
        # the default dispatch (model.kernel()) must auto-select generated numba
        # on the numpy engine for mixtures of declared exp-family leaves, and the
        # score must match the legacy seq path on the legacy seq_encode payload
        # that the engine estimation path feeds kernels
        from pysp.stats import seq_encode, seq_log_density_sum
        from pysp.stats.kernel import GenericKernel
        mix = MixtureDistribution([GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0)], [0.5, 0.5])
        comp = MixtureDistribution([
            CompositeDistribution((GaussianDistribution(-2.0, 1.0), PoissonDistribution(3.0))),
            CompositeDistribution((GaussianDistribution(2.0, 1.0), PoissonDistribution(8.0))),
        ], [0.5, 0.5])
        for model in (mix, comp):
            kernel = model.kernel()
            self.assertIsInstance(kernel, GeneratedNumbaKernel)
            enc = seq_encode(model.sampler(seed=5).sample(60), model=model)
            for _, payload in enc:
                np.testing.assert_allclose(kernel.score(payload), model.seq_log_density(payload),
                                           rtol=1.0e-10, atol=1.0e-10)

        # a non-exponential-family model keeps the guaranteed generic fallback
        spearman = SpearmanRankingDistribution(np.asarray([0.4, 0.35, 0.25]))
        self.assertIsInstance(spearman.kernel(), GenericKernel)

    def test_numba_kernel_factory_prefers_generated_declarations(self):
        dist = MixtureDistribution(
            [GaussianDistribution(-1.0, 0.7), GaussianDistribution(1.5, 1.2)],
            [0.4, 0.6],
        )
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        data = dist.sampler(seed=11).sample(size=40)
        weights = np.linspace(0.3, 1.1, len(data))

        kernel = NumbaKernelFactory().build(dist, NUMPY_ENGINE, estimator=est)
        self.assertIsInstance(kernel, GeneratedNumbaKernel)
        enc = kernel.encode(data)
        legacy_enc = dist.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(legacy_enc),
                                   rtol=1.0e-10, atol=1.0e-10)
        np.testing.assert_allclose(kernel.component_scores(enc), dist.seq_component_log_density(legacy_enc),
                                   rtol=1.0e-10, atol=1.0e-10)

        legacy_acc = est.accumulator_factory().make()
        legacy_acc.seq_update(legacy_enc, weights, dist)
        _assert_stats_close(self, kernel.accumulate(enc, weights), legacy_acc.value())

    def test_numba_kernel_factory_falls_back_for_support_mismatched_generated_family(self):
        dist = MixtureDistribution([
            BinomialDistribution(0.25, 4),
            BinomialDistribution(0.60, 6),
        ], [0.5, 0.5])
        data = [0, 1, 2, 4]

        kernel = NumbaKernelFactory().build(dist, NUMPY_ENGINE)
        self.assertNotIsInstance(kernel, GeneratedNumbaKernel)
        enc = kernel.encode(data)
        legacy_enc = dist.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(legacy_enc),
                                   rtol=1.0e-10, atol=1.0e-10)

    def test_declaration_generated_stacked_scores_match_leaf_scores(self):
        engine = self.FakeTorchEngine()
        components = [GaussianDistribution(-1.0, 0.7), GaussianDistribution(2.0, 1.3)]
        enc = components[0].dist_to_encoder().seq_encode(np.asarray([-2.0, 0.0, 1.5]))

        params = generated_stacked_params(components, engine)
        scores = generated_stacked_log_density(enc, params, engine)
        expected = np.column_stack([component.seq_log_density(enc) for component in components])

        np.testing.assert_allclose(scores, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_declaration_generated_leaf_scores_are_generic_kernel_fallback(self):
        class DeclarationOnlyBeta(BetaDistribution):
            backend_seq_log_density = None
            backend_log_density_from_params = None

        dist = DeclarationOnlyBeta(2.5, 3.25)
        enc = dist.dist_to_encoder().seq_encode(np.asarray([0.2, 0.4, 0.8]))
        engine = self.FakeTorchEngine()

        expected = dist.seq_log_density(enc)
        np.testing.assert_allclose(generated_log_density(dist, enc, engine), expected,
                                   rtol=1.0e-12, atol=1.0e-12)

        kernel = kernel_for(dist, engine=engine)
        self.assertIsInstance(kernel, GenericKernel)
        np.testing.assert_allclose(kernel.score(enc), expected, rtol=1.0e-12, atol=1.0e-12)

    def test_declaration_generated_numba_leaf_scores_match_legacy(self):
        dist = BetaDistribution(2.5, 3.25)
        data = np.asarray([0.2, 0.4, 0.8])
        enc = dist.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(generated_numba_log_density(dist, enc), dist.seq_log_density(enc),
                                   rtol=1.0e-12, atol=1.0e-12)

        kernel = NumbaKernelFactory().build(dist, NUMPY_ENGINE, estimator=dist.estimator())
        self.assertIsInstance(kernel, GeneratedNumbaKernel)
        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(enc),
                                   rtol=1.0e-12, atol=1.0e-12)

        weights = np.asarray([0.5, 1.25, 0.75])
        legacy = dist.estimator().accumulator_factory().make()
        legacy.seq_update(enc, weights, dist)
        _assert_stats_close(self, kernel.accumulate(enc, weights), legacy.value())

    def test_declaration_generated_numba_stacked_scores_match_legacy(self):
        components = [BetaDistribution(2.5, 3.25), BetaDistribution(4.0, 1.75)]
        enc = components[0].dist_to_encoder().seq_encode(np.asarray([0.2, 0.4, 0.8]))
        params = generated_stacked_params(components, NUMPY_ENGINE)

        expected = np.column_stack([component.seq_log_density(enc) for component in components])
        np.testing.assert_allclose(generated_numba_stacked_log_density(enc, params), expected,
                                   rtol=1.0e-12, atol=1.0e-12)

    def test_declaration_generated_numba_vector_leaf_scores_match_legacy(self):
        dist = DiagonalGaussianDistribution([0.0, 1.0], [1.5, 2.5])
        data = np.asarray([[0.5, 1.5], [-0.25, 2.0], [1.25, -0.5]])
        enc = dist.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(generated_numba_log_density(dist, enc), dist.seq_log_density(enc),
                                   rtol=1.0e-12, atol=1.0e-12)

        components = [
            DiagonalGaussianDistribution([0.0, 1.0], [1.5, 2.5]),
            DiagonalGaussianDistribution([2.0, -0.5], [0.75, 1.25]),
        ]
        params = generated_stacked_params(components, NUMPY_ENGINE)
        expected = np.column_stack([component.seq_log_density(enc) for component in components])
        np.testing.assert_allclose(generated_numba_stacked_log_density(enc, params), expected,
                                   rtol=1.0e-12, atol=1.0e-12)

    def test_declaration_generated_numba_vector_mixture_kernel_matches_legacy(self):
        class DeclarationOnlyDiagonalGaussian(DiagonalGaussianDistribution):
            pass

        components = [
            DeclarationOnlyDiagonalGaussian([0.0, 1.0], [1.5, 2.5]),
            DeclarationOnlyDiagonalGaussian([2.0, -0.5], [0.75, 1.25]),
        ]
        dist = MixtureDistribution(components, [0.35, 0.65])
        est = MixtureEstimator([component.estimator() for component in components])
        data = np.asarray([[0.5, 1.5], [-0.25, 2.0], [1.25, -0.5], [1.0, 0.25]])
        weights = np.asarray([0.5, 1.25, 0.75, 1.1])

        kernel = NumbaKernelFactory().build(dist, NUMPY_ENGINE, estimator=est)
        self.assertIsInstance(kernel, GeneratedNumbaKernel)
        enc = kernel.encode(data)
        legacy_enc = dist.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(kernel.component_scores(enc), dist.seq_component_log_density(legacy_enc),
                                   rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(legacy_enc),
                                   rtol=1.0e-12, atol=1.0e-12)

        legacy_acc = est.accumulator_factory().make()
        legacy_acc.seq_update(legacy_enc, weights, dist)
        _assert_stats_close(self, kernel.accumulate(enc, weights), legacy_acc.value())

    def test_declaration_generated_numba_matrix_leaf_scores_match_legacy(self):
        dist = MultivariateGaussianDistribution([0.5, -1.0], [[1.5, 0.3], [0.3, 2.0]])
        data = np.asarray([[0.5, 1.5], [-0.25, 2.0], [1.25, -0.5]])
        enc = dist.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(generated_numba_log_density(dist, enc), dist.seq_log_density(enc),
                                   rtol=1.0e-12, atol=1.0e-12)

        components = [
            MultivariateGaussianDistribution([0.0, 1.0], [[1.5, 0.2], [0.2, 2.5]]),
            MultivariateGaussianDistribution([2.0, -0.5], [[0.75, -0.1], [-0.1, 1.25]]),
        ]
        params = generated_stacked_params(components, NUMPY_ENGINE)
        expected = np.column_stack([component.seq_log_density(enc) for component in components])
        np.testing.assert_allclose(generated_numba_stacked_log_density(enc, params), expected,
                                   rtol=1.0e-12, atol=1.0e-12)

    def test_declaration_generated_numba_matrix_mixture_kernel_matches_legacy(self):
        components = [
            MultivariateGaussianDistribution([0.0, 1.0], [[1.5, 0.2], [0.2, 2.5]]),
            MultivariateGaussianDistribution([2.0, -0.5], [[0.75, -0.1], [-0.1, 1.25]]),
        ]
        dist = MixtureDistribution(components, [0.35, 0.65])
        est = MixtureEstimator([component.estimator() for component in components])
        data = np.asarray([[0.5, 1.5], [-0.25, 2.0], [1.25, -0.5], [1.0, 0.25]])
        weights = np.asarray([0.5, 1.25, 0.75, 1.1])

        kernel = NumbaKernelFactory().build(dist, NUMPY_ENGINE, estimator=est)
        self.assertIsInstance(kernel, GeneratedNumbaKernel)
        enc = kernel.encode(data)
        legacy_enc = dist.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(kernel.component_scores(enc), dist.seq_component_log_density(legacy_enc),
                                   rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(legacy_enc),
                                   rtol=1.0e-12, atol=1.0e-12)

        legacy_acc = est.accumulator_factory().make()
        legacy_acc.seq_update(legacy_enc, weights, dist)
        _assert_stats_close(self, kernel.accumulate(enc, weights), legacy_acc.value())

    def test_declaration_generated_numba_mixture_kernel_matches_legacy(self):
        components = [BetaDistribution(2.5, 3.25), BetaDistribution(4.0, 1.75)]
        dist = MixtureDistribution(components, [0.4, 0.6])
        est = MixtureEstimator([component.estimator() for component in components])
        data = np.asarray([0.2, 0.4, 0.8, 0.55])
        weights = np.asarray([0.5, 1.25, 0.75, 1.1])

        kernel = NumbaKernelFactory().build(dist, NUMPY_ENGINE, estimator=est)
        self.assertIsInstance(kernel, GeneratedNumbaKernel)
        enc = kernel.encode(data)
        legacy_enc = dist.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(kernel.component_scores(enc), dist.seq_component_log_density(legacy_enc),
                                   rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(legacy_enc),
                                   rtol=1.0e-12, atol=1.0e-12)

        legacy_acc = est.accumulator_factory().make()
        legacy_acc.seq_update(legacy_enc, weights, dist)
        _assert_stats_close(self, kernel.accumulate(enc, weights), legacy_acc.value())

    def test_declaration_generated_numba_composite_mixture_kernel_matches_legacy(self):
        components = [
            CompositeDistribution((BetaDistribution(2.5, 3.25), BetaDistribution(1.75, 4.0))),
            CompositeDistribution((BetaDistribution(4.0, 1.75), BetaDistribution(3.5, 2.25))),
        ]
        dist = MixtureDistribution(components, [0.35, 0.65])
        est = MixtureEstimator([
            CompositeEstimator(tuple(child.estimator() for child in component.dists))
            for component in components
        ])
        data = [(0.2, 0.7), (0.4, 0.5), (0.8, 0.25), (0.55, 0.35)]
        weights = np.asarray([0.5, 1.25, 0.75, 1.1])

        kernel = NumbaKernelFactory().build(dist, NUMPY_ENGINE, estimator=est)
        self.assertIsInstance(kernel, GeneratedNumbaKernel)
        enc = kernel.encode(data)
        legacy_enc = dist.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(kernel.component_scores(enc), dist.seq_component_log_density(legacy_enc),
                                   rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(legacy_enc),
                                   rtol=1.0e-12, atol=1.0e-12)

        legacy_acc = est.accumulator_factory().make()
        legacy_acc.seq_update(legacy_enc, weights, dist)
        _assert_stats_close(self, kernel.accumulate(enc, weights), legacy_acc.value())

    def test_declaration_generated_numba_optional_mixture_kernel_matches_legacy(self):
        components = [
            OptionalDistribution(BetaDistribution(2.5, 3.25), p=0.20, missing_value=None),
            OptionalDistribution(BetaDistribution(4.0, 1.75), p=0.45, missing_value=None),
        ]
        dist = MixtureDistribution(components, [0.35, 0.65])
        est = MixtureEstimator([component.estimator() for component in components])
        data = [None, 0.2, 0.4, None, 0.8, 0.55]
        weights = np.asarray([0.5, 1.25, 0.75, 1.1, 0.9, 0.6])

        kernel = NumbaKernelFactory().build(dist, NUMPY_ENGINE, estimator=est)
        self.assertIsInstance(kernel, GeneratedNumbaKernel)
        enc = kernel.encode(data)
        legacy_enc = dist.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(kernel.component_scores(enc), dist.seq_component_log_density(legacy_enc),
                                   rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(legacy_enc),
                                   rtol=1.0e-12, atol=1.0e-12)

        legacy_acc = est.accumulator_factory().make()
        legacy_acc.seq_update(legacy_enc, weights, dist)
        _assert_stats_close(self, kernel.accumulate(enc, weights), legacy_acc.value())

    def test_declaration_generated_numba_sequence_mixture_kernel_matches_legacy(self):
        components = [
            SequenceDistribution(BetaDistribution(2.5, 3.25), len_dist=PoissonDistribution(1.8)),
            SequenceDistribution(BetaDistribution(4.0, 1.75), len_dist=PoissonDistribution(2.4)),
        ]
        dist = MixtureDistribution(components, [0.35, 0.65])
        est = MixtureEstimator([component.estimator() for component in components])
        data = [[0.2, 0.4], [0.8], [], [0.55, 0.3, 0.7], [0.45]]
        weights = np.asarray([0.5, 1.25, 0.75, 1.1, 0.6])

        kernel = NumbaKernelFactory().build(dist, NUMPY_ENGINE, estimator=est)
        self.assertIsInstance(kernel, GeneratedNumbaKernel)
        enc = kernel.encode(data)
        legacy_enc = dist.dist_to_encoder().seq_encode(data)

        np.testing.assert_allclose(kernel.component_scores(enc), dist.seq_component_log_density(legacy_enc),
                                   rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(legacy_enc),
                                   rtol=1.0e-12, atol=1.0e-12)

        legacy_acc = est.accumulator_factory().make()
        legacy_acc.seq_update(legacy_enc, weights, dist)
        _assert_stats_close(self, kernel.accumulate(enc, weights), legacy_acc.value())

    def test_numba_factory_uses_stacked_route_for_weighted_mixture_when_fused_builder_declines(self):
        components = [
            WeightedDistribution(BetaDistribution(2.5, 3.25)),
            WeightedDistribution(BetaDistribution(4.0, 1.75)),
        ]
        dist = MixtureDistribution(components, [0.35, 0.65])
        est = MixtureEstimator([component.estimator() for component in components])
        data = [(0.2, 1.5), (0.4, 0.7), (0.8, 2.0), (0.55, 0.3)]
        weights = np.asarray([0.5, 1.25, 0.75, 1.1])

        kernel = NumbaKernelFactory().build(dist, NUMPY_ENGINE, estimator=est)
        self.assertIsInstance(kernel, StackedMixtureKernel)
        enc = kernel.encode(data)

        np.testing.assert_allclose(kernel.component_scores(enc), dist.seq_component_log_density(enc),
                                   rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(enc),
                                   rtol=1.0e-12, atol=1.0e-12)

        legacy_acc = est.accumulator_factory().make()
        legacy_acc.seq_update(enc, weights, dist)
        _assert_stats_close(self, kernel.accumulate(enc, weights), legacy_acc.value())

    def test_numba_factory_uses_stacked_route_for_transform_mixture_when_fused_builder_declines(self):
        transform = AffineTransform(loc=0.0, scale=2.0)
        components = [
            TransformDistribution(BetaDistribution(2.5, 3.25), transform=transform),
            TransformDistribution(BetaDistribution(4.0, 1.75), transform=transform),
        ]
        dist = MixtureDistribution(components, [0.35, 0.65])
        est = MixtureEstimator([component.estimator() for component in components])
        data = [0.4, 0.8, 1.6, 1.1]
        weights = np.asarray([0.5, 1.25, 0.75, 1.1])

        kernel = NumbaKernelFactory().build(dist, NUMPY_ENGINE, estimator=est)
        self.assertIsInstance(kernel, StackedMixtureKernel)
        enc = kernel.encode(data)

        np.testing.assert_allclose(kernel.component_scores(enc), dist.seq_component_log_density(enc),
                                   rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(enc),
                                   rtol=1.0e-12, atol=1.0e-12)

        legacy_acc = est.accumulator_factory().make()
        legacy_acc.seq_update(enc, weights, dist)
        _assert_stats_close(self, kernel.accumulate(enc, weights), legacy_acc.value())

    def test_declaration_generated_leaf_stats_match_legacy_accumulators(self):
        dist = BetaDistribution(2.5, 3.25)
        est = dist.estimator()
        data = np.asarray([0.2, 0.4, 0.8])
        weights = np.asarray([0.5, 1.25, 0.75])
        enc = dist.dist_to_encoder().seq_encode(data)
        legacy = est.accumulator_factory().make()
        legacy.seq_update(enc, weights, dist)

        generated = generated_sufficient_statistics(dist, enc, weights, self.FakeTorchEngine())
        _assert_stats_close(self, generated, legacy.value())

        class NoAccumulatorEstimator(object):
            def accumulator_factory(self):
                raise AssertionError('generated stats should avoid host accumulator fallback')

        kernel = GenericKernel(dist, engine=self.FakeTorchEngine(), estimator=NoAccumulatorEstimator())
        _assert_stats_close(self, kernel.accumulate(enc, weights), legacy.value())

    def test_declaration_generated_leaf_vector_stats_match_legacy_accumulators(self):
        dist = DiagonalGaussianDistribution([0.0, 1.0], [1.5, 2.5])
        est = dist.estimator()
        data = np.asarray([[0.5, 1.5], [-0.25, 2.0], [1.25, -0.5]])
        weights = np.asarray([0.4, 1.1, 0.8])
        enc = dist.dist_to_encoder().seq_encode(data)
        legacy = est.accumulator_factory().make()
        legacy.seq_update(enc, weights, dist)

        generated = generated_sufficient_statistics(dist, enc, weights, self.FakeTorchEngine())
        _assert_stats_close(self, generated, legacy.value())

    def test_stacked_kernel_uses_declaration_generated_fallback(self):
        class DeclarationOnlyGaussian(GaussianDistribution):
            backend_stacked_params = None
            backend_stacked_log_density = None

        dist = MixtureDistribution(
            [DeclarationOnlyGaussian(-1.0, 0.7), DeclarationOnlyGaussian(2.0, 1.3)],
            [0.4, 0.6],
        )
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        data = np.asarray([-2.0, 0.0, 1.5])
        enc = dist.dist_to_encoder().seq_encode(data)

        kernel = kernel_for(dist, engine=self.FakeTorchEngine(), estimator=est)

        self.assertIsInstance(kernel, StackedMixtureKernel)
        np.testing.assert_allclose(kernel.component_scores(enc), dist.seq_component_log_density(enc),
                                   rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(kernel.score(enc), dist.seq_log_density(enc),
                                   rtol=1.0e-12, atol=1.0e-12)

    def test_encoder_nbytes_reports_nested_payload_size(self):
        enc = (np.asarray([1.0, 2.0], dtype=np.float64),
               {'idx': np.asarray([1, 2, 3], dtype=np.int32)})
        expected = enc[0].nbytes + enc[1]['idx'].nbytes + len('idx'.encode('utf-8'))
        self.assertEqual(encoded_nbytes(enc), expected)

        data = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        encoder = GaussianDistribution(0.0, 1.0).dist_to_encoder()
        payload = encoder.seq_encode(data)
        self.assertEqual(encoder.nbytes(payload), payload.nbytes)

    def test_encoded_data_wrapper_behaves_like_one_seq_chunk(self):
        dist = GaussianDistribution(0.0, 1.0)
        est = GaussianEstimator()
        data = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        encoder = dist.dist_to_encoder()
        wrapped = EncodedData.from_data(data, encoder)

        self.assertEqual(wrapped.count, len(data))
        self.assertIs(wrapped.encoder, encoder)
        self.assertEqual(wrapped.engine.name, 'numpy')
        self.assertEqual(wrapped.nbytes, wrapped.payload.nbytes)
        self.assertEqual(wrapped.as_seq_chunk()[0], len(data))

        chunked = [(len(data), wrapped.payload)]
        np.testing.assert_allclose(seq_log_density_sum(wrapped, dist),
                                   seq_log_density_sum(chunked, dist),
                                   rtol=0, atol=0)

        wrapped_model = seq_estimate(wrapped, est, dist)
        chunked_model = seq_estimate(chunked, est, dist)
        self.assertAlmostEqual(wrapped_model.mu, chunked_model.mu)
        self.assertAlmostEqual(wrapped_model.sigma2, chunked_model.sigma2)


if __name__ == '__main__':
    unittest.main()
