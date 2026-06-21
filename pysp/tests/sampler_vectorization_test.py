"""Phase A: vectorized samplers must be bit-identical to the per-draw (batched=False) reference."""

import unittest

import numpy as np

from pysp.stats import (
    CategoricalDistribution,
    DiagonalGaussianDistribution,
    GaussianDistribution,
    HeterogeneousMixtureDistribution,
    PoissonDistribution,
)
from pysp.stats.latent.gaussian_mixture import GaussianMixtureDistribution


def _identical(a, b):
    return len(a) == len(b) and all(np.array_equal(np.asarray(a[i]), np.asarray(b[i])) for i in range(len(a)))


class SamplerVectorizationTest(unittest.TestCase):
    def test_diagonal_gaussian_matches_loop(self):
        d = DiagonalGaussianDistribution([1.0, -2.0, 0.5], [2.0, 0.5, 1.0])
        vec = d.sampler(seed=7).sample(50)
        s = d.sampler(seed=7)
        loop = [s.sample() for _ in range(50)]
        self.assertTrue(_identical(vec, loop))

    def test_gaussian_mixture_batched_identical(self):
        gm = GaussianMixtureDistribution(
            np.array([[0.0, 0.0], [5.0, 5.0]]), np.array([np.eye(2), np.eye(2)]), [0.6, 0.4]
        )
        a = gm.sampler(seed=3).sample(100, batched=True)
        b = gm.sampler(seed=3).sample(100, batched=False)
        self.assertTrue(_identical(a, b))

    def test_heterogeneous_mixture_batched_identical(self):
        hm = HeterogeneousMixtureDistribution([GaussianDistribution(0, 1), PoissonDistribution(3.0)], [0.5, 0.5])
        a = hm.sampler(seed=3).sample(100, batched=True)
        b = hm.sampler(seed=3).sample(100, batched=False)
        self.assertTrue(_identical(a, b))

    def test_heterogeneous_categorical_batched_identical(self):
        hm = HeterogeneousMixtureDistribution(
            [CategoricalDistribution({"a": 0.5, "b": 0.5}), CategoricalDistribution({"a": 0.2, "b": 0.8})], [0.5, 0.5]
        )
        a = hm.sampler(seed=5).sample(100, batched=True)
        b = hm.sampler(seed=5).sample(100, batched=False)
        self.assertTrue(_identical(a, b))


if __name__ == "__main__":
    unittest.main()
