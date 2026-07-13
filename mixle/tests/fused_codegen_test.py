"""Source-generated fused numba kernels (fused_codegen): correctness + fusibility gating."""

import unittest

import numpy as np
import pytest

import mixle.stats as stats
from mixle.stats.compute.fused_codegen import analyze, fused_seq_log_density, fusible
from mixle.utils.optional_deps import HAS_NUMBA

# The fused codegen path JIT-compiles with numba; without it these tests cannot run. They are also
# marked ("numba", "optional") in conftest so the CI fast/full (no-numba) gates skip collection and the
# optional-extras job (which installs numba) exercises them.
pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="fused codegen requires numba")


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
        # Laplace keeps the raw observations (its MLE needs the weighted median, not a fixed reduction),
        # so it has no template and falls back to numpy
        self.assertFalse(fusible(stats.LaplaceDistribution(0.0, 1.0)))

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
        from mixle.stats.compute.fused_codegen import _compile

        m1 = stats.MixtureDistribution([stats.GaussianDistribution(float(i), 1.0) for i in range(3)], [1 / 3] * 3)
        m2 = stats.MixtureDistribution([stats.GaussianDistribution(float(i) + 9, 2.0) for i in range(3)], [1 / 3] * 3)
        self.assertIs(_compile(analyze(m1)), _compile(analyze(m2)))  # same structure -> same compiled fn


class FusedEStepTest(unittest.TestCase):
    """The fused E-step (score + responsibilities + per-leaf weighted statistics) matches the numpy fit."""

    def setUp(self):
        self.rng = np.random.RandomState(0)

    def _matches(self, model, est, data):
        from mixle.stats.compute.fused_codegen import fused_accumulate

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
        from mixle.stats.compute.fused_codegen import fusible_estep

        self.assertTrue(fusible_estep(stats.GaussianDistribution(0.0, 1.0)))
        self.assertTrue(fusible_estep(stats.MultivariateGaussianDistribution([0.0], [[1.0]])))
        self.assertTrue(fusible_estep(stats.CategoricalDistribution({"a": 0.5, "b": 0.5})))
        # Laplace keeps raw observations (weighted-median MLE) -> not fusible
        self.assertFalse(fusible_estep(stats.LaplaceDistribution(0.0, 1.0)))


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

    def test_categorical(self):
        # categorical kind: data is already a category index, scored from a (K,C) log-prob table; the only
        # sufficient statistic is the per-category weighted count histogram
        model = self._mix(
            [
                stats.CategoricalDistribution({"a": 0.6, "b": 0.3, "c": 0.1}),
                stats.CategoricalDistribution({"a": 0.1, "b": 0.4, "c": 0.5}),
            ]
        )
        est = stats.MixtureEstimator([stats.CategoricalEstimator(), stats.CategoricalEstimator()])
        data = [["a", "b", "c"][i] for i in self.rng.randint(3, size=1500)]
        self.assertTrue(_ll_close(model, data))
        self.assertTrue(FusedEStepTest._matches(self, model, est, data))

    def test_integer_categorical(self):
        model = self._mix(
            [
                stats.IntegerCategoricalDistribution(0, [0.6, 0.3, 0.1]),
                stats.IntegerCategoricalDistribution(0, [0.1, 0.4, 0.5]),
            ]
        )
        est = stats.MixtureEstimator([stats.IntegerCategoricalEstimator(), stats.IntegerCategoricalEstimator()])
        data = [int(i) for i in self.rng.randint(3, size=1500)]
        self.assertTrue(_ll_close(model, data))
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
        reals = [float(x) for x in rng.randn(1500)]
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
            # Pareto: scale xm is the PER-COMPONENT min over rows with responsibility > 0 (the wmin
            # accumulator); with shared supports it coincides with the global min. Differing supports
            # are pinned by fused_out_of_support_test.
            (
                [stats.ParetoDistribution(0.5, 2.0), stats.ParetoDistribution(0.5, 4.0)],
                stats.ParetoEstimator,
                [0.5 + abs(x) + 0.05 for x in rng.randn(1500)],
            ),
            # location-scale with moment-matched (sumx, sumx^2, n) statistics + closed-form densities
            ([stats.GumbelDistribution(0.0, 1.0), stats.GumbelDistribution(2.0, 1.5)], stats.GumbelEstimator, reals),
            (
                [stats.LogisticDistribution(0.0, 1.0), stats.LogisticDistribution(2.0, 1.5)],
                stats.LogisticEstimator,
                reals,
            ),
            (
                [stats.StudentTDistribution(5.0, 0.0, 1.0), stats.StudentTDistribution(8.0, 2.0, 1.5)],
                stats.StudentTEstimator,
                reals,
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


class NestedTest(unittest.TestCase):
    """Arbitrarily nested Composite/Mixture trees of scalar leaves fuse via the recursive path."""

    def setUp(self):
        self.rng = np.random.RandomState(5)
        self.G = stats.GaussianDistribution

    def _estep_ok(self, model, est, data):
        from mixle.stats.compute.fused_codegen import fused_accumulate

        enc = model.dist_to_encoder().seq_encode(data)
        w = np.abs(self.rng.randn(len(data))) + 0.1
        m_f = est.estimate(float(len(data)), fused_accumulate(model, enc, w))
        acc = est.accumulator_factory().make()
        acc.seq_update(enc, w, model)
        d = {}
        acc.key_merge(d)
        acc.key_replace(d)
        m_n = est.estimate(float(len(data)), acc.value())
        ll = lambda m: float(np.sum(m.seq_log_density(enc)))  # noqa: E731
        return np.isclose(ll(m_f), ll(m_n), rtol=1e-7)

    def test_composite_with_a_mixture_factor(self):
        G = self.G
        model = stats.CompositeDistribution(
            (G(1.0, 2.0), stats.MixtureDistribution([G(0.0, 1.0), G(4.0, 1.5)], [0.5, 0.5]))
        )
        est = stats.CompositeEstimator(
            (stats.GaussianEstimator(), stats.MixtureEstimator([stats.GaussianEstimator(), stats.GaussianEstimator()]))
        )
        self.assertTrue(fusible(model))
        data = [
            (float(self.rng.randn() * 2 + 1), float(self.rng.randn() + 4 * self.rng.randint(2))) for _ in range(2500)
        ]
        self.assertTrue(_ll_close(model, data))
        self.assertTrue(self._estep_ok(model, est, data))

    def test_mixture_of_mixtures(self):
        G = self.G
        model = stats.MixtureDistribution(
            [
                stats.MixtureDistribution([G(0.0, 1.0), G(2.0, 1.0)], [0.5, 0.5]),
                stats.MixtureDistribution([G(8.0, 1.0), G(10.0, 1.0)], [0.5, 0.5]),
            ],
            [0.5, 0.5],
        )
        est = stats.MixtureEstimator(
            [
                stats.MixtureEstimator([stats.GaussianEstimator(), stats.GaussianEstimator()]),
                stats.MixtureEstimator([stats.GaussianEstimator(), stats.GaussianEstimator()]),
            ]
        )
        self.assertTrue(fusible(model))
        data = [float(self.rng.randn() + self.rng.choice([0, 2, 8, 10])) for _ in range(2500)]
        self.assertTrue(_ll_close(model, data))
        self.assertTrue(self._estep_ok(model, est, data))

    def test_mixture_of_composite_with_nested_mixture(self):
        G = self.G

        def comp(k):
            return stats.CompositeDistribution(
                (G(float(k), 1.0), stats.MixtureDistribution([G(0.0, 1.0), G(3.0, 1.0)], [0.5, 0.5]))
            )

        model = stats.MixtureDistribution([comp(0), comp(1)], [0.5, 0.5])
        est = stats.MixtureEstimator(
            [
                stats.CompositeEstimator(
                    (
                        stats.GaussianEstimator(),
                        stats.MixtureEstimator([stats.GaussianEstimator(), stats.GaussianEstimator()]),
                    )
                )
                for _ in range(2)
            ]
        )
        self.assertTrue(fusible(model))
        data = [
            (float(self.rng.randn() + k), float(self.rng.randn() + 3 * self.rng.randint(2)))
            for k in self.rng.randint(2, size=2500)
        ]
        self.assertTrue(_ll_close(model, data))
        self.assertTrue(self._estep_ok(model, est, data))

    def test_heterogeneous_mixture(self):
        # different-typed components (the engine cannot stack these) -- the recursive path unrolls them
        model = stats.MixtureDistribution([self.G(1.0, 1.0), stats.GammaDistribution(2.0, 1.0)], [0.5, 0.5])
        est = stats.MixtureEstimator([stats.GaussianEstimator(), stats.GammaEstimator()])
        self.assertTrue(fusible(model))
        data = [float(abs(self.rng.randn()) + 0.3) for _ in range(2500)]
        self.assertTrue(_ll_close(model, data))
        self.assertTrue(self._estep_ok(model, est, data))

    def test_heterogeneous_nested_mixture(self):
        # Mixture[Gaussian, Mixture[Gamma, Exponential]] -- heterogeneous AND nested
        model = stats.MixtureDistribution(
            [
                self.G(1.0, 1.0),
                stats.MixtureDistribution(
                    [stats.GammaDistribution(2.0, 1.0), stats.ExponentialDistribution(1.0)], [0.5, 0.5]
                ),
            ],
            [0.5, 0.5],
        )
        est = stats.MixtureEstimator(
            [stats.GaussianEstimator(), stats.MixtureEstimator([stats.GammaEstimator(), stats.ExponentialEstimator()])]
        )
        self.assertTrue(fusible(model))
        data = [float(abs(self.rng.randn()) + 0.3) for _ in range(2500)]
        self.assertTrue(_ll_close(model, data))
        self.assertTrue(self._estep_ok(model, est, data))

    def test_nested_with_a_matrix_leaf_is_not_recursively_fused(self):
        # the recursive path is scalar-only; a nested model with an MVGaussian leaf falls back to numpy
        from mixle.stats.compute.fused_nested import fusible_nested

        model = stats.CompositeDistribution(
            (
                stats.GaussianDistribution(0.0, 1.0),
                stats.MixtureDistribution(
                    [stats.MultivariateGaussianDistribution([0.0, 0.0], np.eye(2).tolist())] * 2, [0.5, 0.5]
                ),
            )
        )
        self.assertFalse(fusible_nested(model))


class DispatchTest(unittest.TestCase):
    """optimize(engine=<numba>) routes fusible models to the FusedKernel and matches the numpy fit."""

    def test_optimize_on_numba_engine_matches_numpy_for_fusible_model(self):
        from mixle.engines import FUSED_NUMPY_ENGINE
        from mixle.inference import optimize
        from mixle.stats.compute.fused_codegen import FusedKernel

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
        from mixle.engines import FUSED_NUMPY_ENGINE
        from mixle.stats.compute.fused_codegen import FusedKernel

        # a Laplace mixture has no leaf template -> must keep its existing engine kernel
        m = stats.MixtureDistribution(
            [stats.LaplaceDistribution(0.0, 1.0), stats.LaplaceDistribution(2.0, 1.5)],
            [0.5, 0.5],
        )
        e = stats.MixtureEstimator([stats.LaplaceEstimator() for _ in range(2)])
        self.assertNotIsInstance(m.kernel(engine=FUSED_NUMPY_ENGINE, estimator=e), FusedKernel)


if __name__ == "__main__":
    unittest.main()


class AutoFusionCostModelTest(unittest.TestCase):
    """optimize(engine=None) auto-switches a large fusible composite fit to the fused kernel — and the
    result is parity-identical to the host path (the cost model only changes speed, never the answer)."""

    def _mix(self):
        G, P, Mix, Comp = (
            stats.GaussianDistribution,
            stats.PoissonDistribution,
            stats.MixtureDistribution,
            stats.CompositeDistribution,
        )
        return Mix([Comp((G(-2, 1), P(2.0))), Comp((G(2, 1), P(9.0)))], [0.5, 0.5])

    def test_gate(self):
        from mixle.inference.fusion_policy import should_auto_fuse

        mix = self._mix()
        self.assertTrue(should_auto_fuse(mix, [(60000, None)], 30))  # large composite-mixture -> fuse
        self.assertFalse(should_auto_fuse(mix, [(2000, None)], 30))  # small -> stay on host
        self.assertFalse(should_auto_fuse(stats.GaussianDistribution(0, 1), [(10**7, None)], 100))  # bare leaf

    def test_auto_fusion_is_parity_identical(self):
        from mixle.engines import NUMPY_ENGINE
        from mixle.inference import optimize

        rng = np.random.RandomState(0)
        m = self._mix()
        data = [(float(rng.normal(0, 2)), float(rng.poisson(5))) for _ in range(60000)]  # 60k*30 >= 1.5e6
        auto = optimize(data, m.estimator(), prev_estimate=m, max_its=30, out=None)  # auto-fuse
        host = optimize(data, m.estimator(), prev_estimate=m, max_its=30, engine=NUMPY_ENGINE, out=None)
        self.assertTrue(np.allclose(sorted(auto.w), sorted(host.w), atol=1e-8))
        am = sorted(c.dists[0].mu for c in auto.components)
        hm = sorted(c.dists[0].mu for c in host.components)
        self.assertTrue(np.allclose(am, hm, atol=1e-8))


class ReducedPrecisionTest(unittest.TestCase):
    """Opt-in float32 row arithmetic with float64 accumulation (DeepSeek low-precision-compute pattern).

    ``compute_dtype`` down-casts the floating data/params so the row math runs in reduced precision while
    every accumulator stays float64, so the log-sum-exp and the E-step statistics do not drift with N or K.
    ``None`` (the default, and the auto-fusion engine's dtype) is byte-identical to the legacy path.
    """

    def _gmm(self, rng, k=8, fields=8):
        comps = [
            stats.CompositeDistribution(
                tuple(stats.GaussianDistribution(float(rng.randn()), float(0.5 + rng.rand())) for _ in range(fields))
            )
            for _ in range(k)
        ]
        return stats.MixtureDistribution(comps, list(rng.dirichlet(np.ones(k))))

    def test_default_dtype_is_byte_identical(self):
        rng = np.random.RandomState(0)
        m = self._gmm(rng)
        enc = m.dist_to_encoder().seq_encode(m.sampler(1).sample(5000))
        # compute_dtype=None and compute_dtype=float64 must both equal the unparameterized fused path
        base = fused_seq_log_density(m, enc)
        self.assertTrue(np.array_equal(base, fused_seq_log_density(m, enc, None)))
        self.assertTrue(np.array_equal(base, fused_seq_log_density(m, enc, np.float64)))

    def test_float32_scores_match_float64_within_tolerance(self):
        rng = np.random.RandomState(1)
        m = self._gmm(rng)
        enc = m.dist_to_encoder().seq_encode(m.sampler(2).sample(20000))
        f64 = fused_seq_log_density(m, enc)
        f32 = fused_seq_log_density(m, enc, np.float32)
        self.assertTrue(np.allclose(f32, f64, rtol=1e-4, atol=1e-4))

    def test_float64_accumulation_does_not_drift_over_large_n(self):
        # The point of high-precision accumulation: summing N reduced-precision rows stays float64-accurate.
        # A naive float32 reduction of ~2e5 terms (sum magnitude ~1e5) would drift by ~N*eps_f32 (rel ~1e-2);
        # here the reduction is float64 so the relative error must be near float32 *element* precision.
        rng = np.random.RandomState(2)
        m = self._gmm(rng)
        enc = m.dist_to_encoder().seq_encode(m.sampler(3).sample(200000))
        s64 = float(fused_seq_log_density(m, enc).sum())
        s32 = float(fused_seq_log_density(m, enc, np.float32).sum())
        self.assertLess(abs(s32 - s64) / abs(s64), 1e-6)

    def test_float32_estep_statistics_match_float64(self):
        from mixle.stats.compute.fused_codegen import fused_accumulate

        rng = np.random.RandomState(3)
        m = self._gmm(rng)
        enc = m.dist_to_encoder().seq_encode(m.sampler(4).sample(20000))
        w = np.ones(20000)
        (counts64, _comp64), ll64 = fused_accumulate(m, enc, w, return_ll=True)
        (counts32, _comp32), ll32 = fused_accumulate(m, enc, w, return_ll=True, compute_dtype=np.float32)
        self.assertLess(abs(ll32 - ll64) / abs(ll64), 1e-6)  # data-LL is the float64-accumulated normalizer
        self.assertTrue(np.allclose(np.asarray(counts32), np.asarray(counts64), rtol=1e-3, atol=1e-2))

    def test_integer_index_leaves_are_not_downcast(self):
        # Categorical / tabulated leaves carry INTEGER index arrays; reduced precision must leave them intact
        # (only floating arrays are cast). A Poisson + Categorical mixture exercises both.
        rng = np.random.RandomState(4)
        k = 4
        comps = [
            stats.CompositeDistribution(
                (
                    stats.PoissonDistribution(float(1 + 5 * rng.rand())),
                    stats.CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2}),
                )
            )
            for _ in range(k)
        ]
        m = stats.MixtureDistribution(comps, list(rng.dirichlet(np.ones(k))))
        enc = m.dist_to_encoder().seq_encode(m.sampler(5).sample(10000))
        f64 = fused_seq_log_density(m, enc)
        f32 = fused_seq_log_density(m, enc, np.float32)
        self.assertTrue(np.allclose(f32, f64, rtol=1e-4, atol=1e-4))

    def test_fused_kernel_takes_precision_from_engine(self):
        from mixle.engines import NumpyEngine
        from mixle.stats.compute.fused_codegen import FusedKernel

        rng = np.random.RandomState(6)
        m = self._gmm(rng)
        enc = m.dist_to_encoder().seq_encode(m.sampler(7).sample(5000))
        # default fused engine -> None (auto-fusion never silently lowers precision); float32 engine -> float32
        k_default = FusedKernel(m, NumpyEngine(prefer_fused=True))
        k_f32 = FusedKernel(m, NumpyEngine(dtype="float32", prefer_fused=True))
        self.assertIsNone(k_default.compute_dtype)
        self.assertEqual(np.dtype(k_f32.compute_dtype), np.float32)
        self.assertTrue(np.array_equal(k_default.score(enc), fused_seq_log_density(m, enc)))
        self.assertTrue(np.allclose(k_f32.score(enc), fused_seq_log_density(m, enc), rtol=1e-4, atol=1e-4))

    def test_float32_summed_ll_robust_on_danger_zone_families(self):
        # The validated fp32 safety envelope: across ill-conditioned families the f64-accumulated
        # summed log-likelihood (what EM actually consumes) stays accurate to ~1e-6 relative, even
        # though per-row error widens. This pins the band the quantization review said was untested.
        rng = np.random.RandomState(7)

        def summed_rel_err(model):
            enc = model.dist_to_encoder().seq_encode(model.sampler(11).sample(20000))
            s64 = float(fused_seq_log_density(model, enc).sum())
            s32 = float(fused_seq_log_density(model, enc, np.float32).sum())
            return abs(s32 - s64) / max(abs(s64), 1e-12)

        danger = {
            "tiny-variance": stats.MixtureDistribution(
                [stats.GaussianDistribution(0.0, 1e-6), stats.GaussianDistribution(1.0, 1e-6)], [0.5, 0.5]
            ),
            "studentt-heavy-tail": stats.MixtureDistribution(
                [stats.StudentTDistribution(1.5, 0.0, 1.0), stats.StudentTDistribution(1.5, 3.0, 1.0)], [0.5, 0.5]
            ),
            "pareto-heavy-tail": stats.MixtureDistribution(
                [stats.ParetoDistribution(1.2, 1.0), stats.ParetoDistribution(1.2, 2.0)], [0.5, 0.5]
            ),
            "near-degenerate": stats.MixtureDistribution(
                [stats.GaussianDistribution(0.0, 1.0), stats.GaussianDistribution(1e-4, 1.0)], [0.5, 0.5]
            ),
        }
        for name, model in danger.items():
            self.assertLess(summed_rel_err(model), 1e-6, "fp32 summed-LL drifted on %s" % name)

    def test_fused_rejects_non_float32_reduced_precision(self):
        # numba cannot compile float16/bfloat16 or sub-byte formats on CPU -> clear error, not a numba crash.
        rng = np.random.RandomState(8)
        m = self._gmm(rng)
        enc = m.dist_to_encoder().seq_encode(m.sampler(9).sample(2000))
        with self.assertRaises(ValueError) as ctx:
            fused_seq_log_density(m, enc, np.float16)
        self.assertIn("float32", str(ctx.exception))
