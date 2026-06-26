"""Source-generated fused numba kernels (fused_codegen): correctness + fusibility gating."""

import unittest

import numpy as np

import pysp.stats as stats
from pysp.stats.compute.fused_codegen import analyze, fused_seq_log_density, fusible


def _ll_close(model, data):
    enc = model.dist_to_encoder().seq_encode(data)
    return np.allclose(fused_seq_log_density(model, enc), model.seq_log_density(enc), rtol=1e-9, atol=1e-12)


class FusibilityTest(unittest.TestCase):
    def test_cheap_leaf_structures_are_fusible(self):
        g = stats.GaussianDistribution(0.0, 1.0)
        self.assertTrue(fusible(g))
        self.assertTrue(fusible(stats.CompositeDistribution((g, stats.ExponentialDistribution(1.0)))))
        self.assertTrue(fusible(stats.MixtureDistribution([g, g], [0.5, 0.5])))

    def test_matrix_leaves_are_fusible(self):
        # MVGaussian fuses via BLAS `@` (matmul quad form + weighted Gram), not a scalar loop
        self.assertTrue(fusible(stats.MultivariateGaussianDistribution([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])))

    def test_unsupported_leaf_is_not_fusible(self):
        # a leaf with no template falls back to numpy
        self.assertFalse(fusible(stats.CategoricalDistribution({"a": 0.5, "b": 0.5})))

    def test_mixture_mixing_scalar_and_matrix_leaves_is_fusible(self):
        # a composite of a cheap scalar leaf + a BLAS matrix leaf fuses into one njit
        def comp(mu):
            return stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(mu, 1.0),
                    stats.MultivariateGaussianDistribution([mu, mu], [[1.0, 0.0], [0.0, 1.0]]),
                )
            )

        self.assertTrue(fusible(stats.MixtureDistribution([comp(0.0), comp(1.0)], [0.5, 0.5])))


class CorrectnessTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)

    def test_single_leaf(self):
        data = [float(x) for x in self.rng.randn(500)]
        self.assertTrue(_ll_close(stats.GaussianDistribution(0.4, 1.3), data))

    def test_composite_of_leaves(self):
        c = stats.CompositeDistribution((stats.GaussianDistribution(0.0, 1.0), stats.ExponentialDistribution(2.0)))
        data = [(float(self.rng.randn()), float(abs(self.rng.randn()) + 0.1)) for _ in range(500)]
        self.assertTrue(_ll_close(c, data))

    def test_mixture_of_leaves(self):
        m = stats.MixtureDistribution([stats.GaussianDistribution(float(i) - 1, 1.0) for i in range(3)], [1 / 3] * 3)
        data = [float(self.rng.randn() + 2 * self.rng.randint(3)) for _ in range(500)]
        self.assertTrue(_ll_close(m, data))

    def test_mixture_of_composite_heterogeneous_leaves(self):
        m = stats.MixtureDistribution(
            [
                stats.CompositeDistribution(
                    (stats.GaussianDistribution(float(k), 1.0), stats.ExponentialDistribution(float(k) + 1.0))
                )
                for k in range(4)
            ],
            [0.25] * 4,
        )
        data = [(float(self.rng.randn()), float(abs(self.rng.randn()) + 0.1)) for _ in range(500)]
        self.assertTrue(_ll_close(m, data))

    def test_compiled_kernel_is_cached_by_signature(self):
        from pysp.stats.compute.fused_codegen import _compile

        m1 = stats.MixtureDistribution([stats.GaussianDistribution(float(i), 1.0) for i in range(3)], [1 / 3] * 3)
        m2 = stats.MixtureDistribution([stats.GaussianDistribution(float(i) + 9, 2.0) for i in range(3)], [1 / 3] * 3)
        self.assertIs(_compile(analyze(m1)), _compile(analyze(m2)))  # same structure -> same compiled fn


class FusedEStepTest(unittest.TestCase):
    """The fused E-step (score + responsibilities + per-leaf weighted statistics) matches the numpy fit."""

    def setUp(self):
        self.rng = np.random.RandomState(0)

    def _matches(self, model, est, data):
        from pysp.stats.compute.fused_codegen import fused_accumulate

        enc = model.dist_to_encoder().seq_encode(data)
        w = np.ones(len(data))
        m_fused = est.estimate(float(len(data)), fused_accumulate(model, enc, w))
        acc = est.accumulator_factory().make()
        acc.seq_update(enc, w, model)
        d = {}
        acc.key_merge(d)
        acc.key_replace(d)
        m_np = est.estimate(float(len(data)), acc.value())

        def ll(m):
            return float(np.sum(m.seq_log_density(enc)))

        return np.isclose(ll(m_fused), ll(m_np), rtol=1e-9)

    def test_single_leaf_estep(self):
        data = [float(self.rng.randn() * 1.3 + 0.4) for _ in range(2000)]
        self.assertTrue(self._matches(stats.GaussianDistribution(0.0, 1.0), stats.GaussianEstimator(), data))

    def test_composite_estep(self):
        c = stats.CompositeDistribution((stats.GaussianDistribution(0.0, 1.0), stats.ExponentialDistribution(1.0)))
        e = stats.CompositeEstimator((stats.GaussianEstimator(), stats.ExponentialEstimator()))
        data = [(float(self.rng.randn()), float(abs(self.rng.randn()) + 0.1)) for _ in range(2000)]
        self.assertTrue(self._matches(c, e, data))

    def test_mixture_estep(self):
        m = stats.MixtureDistribution([stats.GaussianDistribution(float(i) - 1, 1.0) for i in range(3)], [1 / 3] * 3)
        e = stats.MixtureEstimator([stats.GaussianEstimator() for _ in range(3)])
        data = [float(self.rng.randn() + 3 * (self.rng.randint(3) - 1)) for _ in range(2000)]
        self.assertTrue(self._matches(m, e, data))

    def test_mixture_of_composite_estep(self):
        m = stats.MixtureDistribution(
            [
                stats.CompositeDistribution(
                    (stats.GaussianDistribution(float(k), 1.0), stats.ExponentialDistribution(float(k) + 1.0))
                )
                for k in range(3)
            ],
            [1 / 3] * 3,
        )
        e = stats.MixtureEstimator(
            [stats.CompositeEstimator((stats.GaussianEstimator(), stats.ExponentialEstimator())) for _ in range(3)]
        )
        data = [(float(abs(self.rng.randn()) * 2), float(abs(self.rng.randn()) + 0.1)) for _ in range(2000)]
        self.assertTrue(self._matches(m, e, data))

    def test_mvgaussian_gmm_estep(self):
        K, D = 4, 5
        rng = np.random.RandomState(1)
        m = stats.MixtureDistribution(
            [stats.MultivariateGaussianDistribution((rng.randn(D)).tolist(), np.eye(D).tolist()) for _ in range(K)],
            [1 / K] * K,
        )
        e = stats.MixtureEstimator([stats.MultivariateGaussianEstimator(dim=D) for _ in range(K)])
        data = [(rng.randn(D) + rng.randint(K)).tolist() for _ in range(3000)]
        self.assertTrue(self._matches(m, e, data))

    def test_mixture_of_composite_scalar_and_matrix_estep(self):
        K, D = 3, 4
        rng = np.random.RandomState(2)

        def comp(k):
            return stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(float(k), 1.0),
                    stats.MultivariateGaussianDistribution((rng.randn(D)).tolist(), np.eye(D).tolist()),
                )
            )

        m = stats.MixtureDistribution([comp(k) for k in range(K)], [1 / K] * K)
        e = stats.MixtureEstimator(
            [
                stats.CompositeEstimator((stats.GaussianEstimator(), stats.MultivariateGaussianEstimator(dim=D)))
                for _ in range(K)
            ]
        )
        data = [(float(rng.randn()), (rng.randn(D)).tolist()) for _ in range(3000)]
        self.assertTrue(self._matches(m, e, data))

    def test_estep_gating(self):
        from pysp.stats.compute.fused_codegen import fusible_estep

        self.assertTrue(fusible_estep(stats.GaussianDistribution(0.0, 1.0)))
        self.assertTrue(fusible_estep(stats.MultivariateGaussianDistribution([0.0], [[1.0]])))
        # a Categorical leaf has no template -> not fusible
        self.assertFalse(fusible_estep(stats.CategoricalDistribution({"a": 0.5, "b": 0.5})))


class LeafFamilyTest(unittest.TestCase):
    """Scalar families beyond Gaussian/Exponential: Poisson, Gamma (arity-2), Geometric, Bernoulli."""

    def setUp(self):
        self.rng = np.random.RandomState(7)

    def _mix(self, leaves):
        return stats.MixtureDistribution(leaves, [1.0 / len(leaves)] * len(leaves))

    def test_new_families_score(self):
        cases = [
            (
                self._mix([stats.PoissonDistribution(2.0), stats.PoissonDistribution(9.0)]),
                [int(x) for x in self.rng.poisson(5, 400)],
            ),
            (
                self._mix([stats.GammaDistribution(2.0, 1.0), stats.GammaDistribution(5.0, 2.0)]),
                [float(abs(x)) + 0.1 for x in self.rng.gamma(3, 2, 400)],
            ),
            (
                self._mix([stats.GeometricDistribution(0.3), stats.GeometricDistribution(0.7)]),
                [int(x) + 1 for x in self.rng.geometric(0.4, 400)],
            ),
            (
                self._mix([stats.BernoulliDistribution(0.3), stats.BernoulliDistribution(0.8)]),
                [int(x) for x in (self.rng.rand(400) < 0.5)],
            ),
        ]
        for model, data in cases:
            self.assertTrue(fusible(model))
            self.assertTrue(_ll_close(model, data), type(model.components[0]).__name__)

    def test_new_families_estep(self):
        specs = [
            (
                stats.PoissonDistribution,
                stats.PoissonEstimator,
                (2.0, 9.0),
                lambda: [int(x) for x in self.rng.poisson(5, 1500)],
            ),
            (
                stats.GeometricDistribution,
                stats.GeometricEstimator,
                (0.3, 0.7),
                lambda: [int(x) + 1 for x in self.rng.geometric(0.4, 1500)],
            ),
            (
                stats.BernoulliDistribution,
                stats.BernoulliEstimator,
                (0.3, 0.8),
                lambda: [int(x) for x in (self.rng.rand(1500) < 0.5)],
            ),
        ]
        for dist_cls, est_cls, (a, b), gen in specs:
            model = self._mix([dist_cls(a), dist_cls(b)])
            est = stats.MixtureEstimator([est_cls(), est_cls()])
            self.assertTrue(FusedEStepTest._matches(self, model, est, gen()), dist_cls.__name__)

    def test_gamma_estep(self):
        model = self._mix([stats.GammaDistribution(2.0, 1.0), stats.GammaDistribution(5.0, 2.0)])
        est = stats.MixtureEstimator([stats.GammaEstimator(), stats.GammaEstimator()])
        data = [float(abs(x)) + 0.1 for x in self.rng.gamma(3, 2, 1500)]
        self.assertTrue(FusedEStepTest._matches(self, model, est, data))

    def test_binomial(self):
        # tabulated leaf: a (K,max+1) log-pmf table + map-reducible global min/max over x
        model = self._mix([stats.BinomialDistribution(0.3, 10), stats.BinomialDistribution(0.7, 10)])
        est = stats.MixtureEstimator([stats.BinomialEstimator(), stats.BinomialEstimator()])
        data = [int(x) for x in self.rng.binomial(10, 0.4, 1200)]
        self.assertTrue(fusible(model))
        self.assertTrue(_ll_close(model, data))
        self.assertTrue(FusedEStepTest._matches(self, model, est, data))

    def test_negbinomial(self):
        # tabulated leaf with a per-component weighted histogram feeding the iterative dispersion MLE
        model = self._mix([stats.NegativeBinomialDistribution(3.0, 0.4), stats.NegativeBinomialDistribution(8.0, 0.6)])
        est = stats.MixtureEstimator([stats.NegativeBinomialEstimator(), stats.NegativeBinomialEstimator()])
        data = [int(x) for x in self.rng.negative_binomial(5, 0.5, 1200)]
        self.assertTrue(fusible(model))
        self.assertTrue(_ll_close(model, data))
        self.assertTrue(FusedEStepTest._matches(self, model, est, data))

    def test_continuous_and_directional_families(self):
        rng = np.random.RandomState(11)
        pos = [abs(x) + 0.3 for x in rng.randn(1500)]
        unit = [min(max(x, 0.02), 0.98) for x in rng.rand(1500)]
        ang = [float(x) for x in rng.uniform(-3.14, 3.14, 1500)]
        ge = [int(x) + 1 for x in rng.geometric(0.3, 1500)]
        cases = [
            ([stats.HalfNormalDistribution(1.0), stats.HalfNormalDistribution(2.0)], stats.HalfNormalEstimator, pos),
            ([stats.RayleighDistribution(1.0), stats.RayleighDistribution(2.0)], stats.RayleighEstimator, pos),
            (
                [stats.InverseGaussianDistribution(1.0, 2.0), stats.InverseGaussianDistribution(2.0, 3.0)],
                stats.InverseGaussianEstimator,
                pos,
            ),
            (
                [stats.InverseGammaDistribution(3.0, 2.0), stats.InverseGammaDistribution(2.0, 1.0)],
                stats.InverseGammaEstimator,
                pos,
            ),
            ([stats.BetaDistribution(2.0, 3.0), stats.BetaDistribution(4.0, 2.0)], stats.BetaEstimator, unit),
            (
                [stats.VonMisesDistribution(0.0, 2.0), stats.VonMisesDistribution(1.0, 1.0)],
                stats.VonMisesEstimator,
                ang,
            ),
            (
                [stats.WrappedCauchyDistribution(0.0, 0.5), stats.WrappedCauchyDistribution(1.0, 0.3)],
                stats.WrappedCauchyEstimator,
                ang,
            ),
            ([stats.LogSeriesDistribution(0.4), stats.LogSeriesDistribution(0.7)], stats.LogSeriesEstimator, ge),
            # Pareto: scale xm is the global min over x (a map-reducible reduction); all data >= xm
            (
                [stats.ParetoDistribution(0.5, 2.0), stats.ParetoDistribution(0.5, 4.0)],
                stats.ParetoEstimator,
                [0.5 + abs(x) + 0.05 for x in rng.randn(1500)],
            ),
        ]
        for comps, est_cls, data in cases:
            model = self._mix(comps)
            est = stats.MixtureEstimator([est_cls() for _ in comps])
            name = type(comps[0]).__name__
            self.assertTrue(_ll_close(model, data), f"{name} score")
            self.assertTrue(FusedEStepTest._matches(self, model, est, data), f"{name} estep")

    def test_loggaussian(self):
        model = self._mix([stats.LogGaussianDistribution(0.0, 1.0), stats.LogGaussianDistribution(1.0, 0.5)])
        est = stats.MixtureEstimator([stats.LogGaussianEstimator(), stats.LogGaussianEstimator()])
        data = [float(abs(x)) + 0.1 for x in self.rng.lognormal(0, 1, 1500)]
        self.assertTrue(_ll_close(model, data))
        self.assertTrue(FusedEStepTest._matches(self, model, est, data))

    def test_diagonal_gaussian_gmm(self):
        # the vector-leaf path: 2-D data, inline per-dim loop, (K,D) accumulators -- the workhorse GMM
        D, K = 6, 4
        rng = np.random.RandomState(3)
        model = self._mix(
            [
                stats.DiagonalGaussianDistribution((rng.randn(D)).tolist(), (np.abs(rng.randn(D)) + 0.5).tolist())
                for _ in range(K)
            ]
        )
        est = stats.MixtureEstimator([stats.DiagonalGaussianEstimator(dim=D) for _ in range(K)])
        data = [(rng.randn(D) + rng.randint(K)).tolist() for _ in range(2000)]
        self.assertTrue(fusible(model))
        self.assertTrue(_ll_close(model, data))
        self.assertTrue(FusedEStepTest._matches(self, model, est, data))

    def test_composite_scalar_and_vector_leaves(self):
        D = 3
        rng = np.random.RandomState(4)

        def comp(k):
            return stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(float(k), 1.0),
                    stats.DiagonalGaussianDistribution([float(k)] * D, [1.0] * D),
                )
            )

        model = self._mix([comp(0), comp(1), comp(2)])
        est = stats.MixtureEstimator(
            [
                stats.CompositeEstimator((stats.GaussianEstimator(), stats.DiagonalGaussianEstimator(dim=D)))
                for _ in range(3)
            ]
        )
        data = [(float(rng.randn()), (rng.randn(D)).tolist()) for _ in range(1500)]
        self.assertTrue(FusedEStepTest._matches(self, model, est, data))

    def test_heterogeneous_composite_scalar_multiarray_and_matrix(self):
        # one fused kernel mixing a scalar leaf, two arity-2 leaves, and a BLAS matrix leaf
        D = 3

        def comp(k):
            return stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(float(k), 1.0),
                    stats.PoissonDistribution(2.0 + k),
                    stats.GammaDistribution(2.0, 1.0 + k),
                    stats.MultivariateGaussianDistribution([float(k)] * D, np.eye(D).tolist()),
                )
            )

        model = self._mix([comp(0), comp(1), comp(2)])
        est = stats.MixtureEstimator(
            [
                stats.CompositeEstimator(
                    (
                        stats.GaussianEstimator(),
                        stats.PoissonEstimator(),
                        stats.GammaEstimator(),
                        stats.MultivariateGaussianEstimator(dim=D),
                    )
                )
                for _ in range(3)
            ]
        )
        data = [
            (
                float(self.rng.randn()),
                int(self.rng.poisson(3)),
                float(abs(self.rng.randn()) + 0.1),
                (self.rng.randn(D)).tolist(),
            )
            for _ in range(1500)
        ]
        self.assertTrue(FusedEStepTest._matches(self, model, est, data))


class DispatchTest(unittest.TestCase):
    """optimize(engine=<numba>) routes fusible models to the FusedKernel and matches the numpy fit."""

    def test_optimize_on_numba_engine_matches_numpy_for_fusible_model(self):
        from pysp.engines import FUSED_NUMPY_ENGINE
        from pysp.inference import optimize
        from pysp.stats.compute.fused_codegen import FusedKernel

        K, L = 6, 4
        init = stats.MixtureDistribution(
            [
                stats.CompositeDistribution(
                    tuple(stats.GaussianDistribution(float(k + l) * 0.1, 1.0) for l in range(L))
                )
                for k in range(K)
            ],
            [1 / K] * K,
        )
        est = stats.MixtureEstimator(
            [stats.CompositeEstimator(tuple(stats.GaussianEstimator() for _ in range(L))) for _ in range(K)]
        )
        self.assertIsInstance(init.kernel(engine=FUSED_NUMPY_ENGINE, estimator=est), FusedKernel)
        rng = np.random.RandomState(0)
        data = [tuple(float(rng.randn()) for _ in range(L)) for _ in range(4000)]
        base = optimize(data, est, prev_estimate=init, max_its=8, out=None)
        fused = optimize(data, est, prev_estimate=init, max_its=8, out=None, engine=FUSED_NUMPY_ENGINE)
        ll = lambda m: float(np.sum(m.seq_log_density(m.dist_to_encoder().seq_encode(data))))  # noqa: E731
        self.assertTrue(np.isclose(ll(base), ll(fused), rtol=1e-7))

    def test_non_fusible_model_falls_back_on_numba_engine(self):
        from pysp.engines import FUSED_NUMPY_ENGINE
        from pysp.stats.compute.fused_codegen import FusedKernel

        # a Categorical mixture has no leaf template -> must keep its existing engine kernel
        m = stats.MixtureDistribution(
            [stats.CategoricalDistribution({"a": 0.7, "b": 0.3}), stats.CategoricalDistribution({"a": 0.2, "b": 0.8})],
            [0.5, 0.5],
        )
        e = stats.MixtureEstimator([stats.CategoricalEstimator() for _ in range(2)])
        self.assertNotIsInstance(m.kernel(engine=FUSED_NUMPY_ENGINE, estimator=e), FusedKernel)


if __name__ == "__main__":
    unittest.main()
