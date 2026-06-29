"""Phase A: vectorized samplers must be bit-identical to the per-draw (batched=False) reference."""

import unittest

import numpy as np

from mixle.stats import (
    CategoricalDistribution,
    DiagonalGaussianDistribution,
    GaussianDistribution,
    HeterogeneousMixtureDistribution,
    PoissonDistribution,
    TruncatedDistribution,
)
from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution


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


class TruncatedSamplerTest(unittest.TestCase):
    def test_batched_rejection_only_yields_allowed(self):
        # low acceptance + numeric support exercises the np.isin fast path
        d = TruncatedDistribution(PoissonDistribution(2.0), allowed=[8, 9, 10])
        x = d.sampler(seed=1).sample(500)
        self.assertEqual(len(x), 500)
        self.assertTrue(all(v in (8, 9, 10) for v in x))

    def test_zero_truncated_and_reference_path(self):
        d = TruncatedDistribution(PoissonDistribution(2.0), forbidden=[0])
        self.assertTrue(all(v != 0 for v in d.sampler(seed=2).sample(2000)))  # batched
        self.assertTrue(all(v != 0 for v in d.sampler(seed=2).sample(200, batched=False)))  # per-draw


if __name__ == "__main__":
    unittest.main()
