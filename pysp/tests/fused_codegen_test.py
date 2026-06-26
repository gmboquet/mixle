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

    def test_blas_and_unsupported_leaves_are_not_fusible(self):
        # MVGaussian is BLAS-bound (numpy gemm beats a scalar loop) -> deliberately not fused
        self.assertFalse(fusible(stats.MultivariateGaussianDistribution([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])))
        # a leaf with no template falls back to numpy
        self.assertFalse(fusible(stats.CategoricalDistribution({"a": 0.5, "b": 0.5})))

    def test_mixture_with_a_blas_leaf_is_not_fusible(self):
        # a composite factor that is BLAS-bound (MVGaussian) makes the whole mixture fall back to numpy
        def comp(mu):
            return stats.CompositeDistribution(
                (
                    stats.GaussianDistribution(mu, 1.0),
                    stats.MultivariateGaussianDistribution([mu, mu], [[1.0, 0.0], [0.0, 1.0]]),
                )
            )

        self.assertFalse(fusible(stats.MixtureDistribution([comp(0.0), comp(1.0)], [0.5, 0.5])))


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


if __name__ == "__main__":
    unittest.main()
