"""Regression: stacked MVN accumulation must not materialize an (N, K, dim, dim) tensor.

A fused/GPU mixture E-step over full-covariance Gaussians used to build a per-sample,
per-component outer-product tensor before reducing it — N*K*dim*dim, ~20 GB at n=2e4,
k=8, dim=128, which OOMs a GPU. The weighted moments are now contracted as gemms whose
peak working set is O(N*dim + K*dim*dim). These tests lock in both the numerics (equal to
the old broadcast-and-reduce reference) and the memory bound (a dense intermediate would
dwarf the asserted ceiling).
"""

import tracemalloc
import unittest

import numpy as np

from mixle.engines import NumpyEngine
from mixle.stats.compute.declarations import _generated_exp_family_pair_term
from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution as MVN


class StackedSuffStatTest(unittest.TestCase):
    def test_matches_broadcast_reference(self):
        # the new gemm accumulation must equal the original broadcast-and-reduce formula
        eng = NumpyEngine()
        rng = np.random.RandomState(0)
        n, k, dim = 500, 4, 7
        x = rng.randn(n, dim)
        w = rng.dirichlet(np.ones(k), size=n)  # (n, k) responsibilities
        sum_x, sum_xx, counts = MVN.backend_stacked_sufficient_statistics(x, w, {}, eng)
        ref_sum_x = np.sum(w[:, :, None] * x[:, None, :], axis=0)
        outer = x[:, :, None] * x[:, None, :]
        ref_sum_xx = np.sum(w[:, :, None, None] * outer[:, None, :, :], axis=0)
        ref_counts = np.sum(w, axis=0)
        np.testing.assert_allclose(np.asarray(sum_x), ref_sum_x, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(np.asarray(sum_xx), ref_sum_xx, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(np.asarray(counts), ref_counts, rtol=1e-12, atol=1e-12)
        self.assertEqual(np.asarray(sum_xx).shape, (k, dim, dim))

    def test_pair_term_matches_broadcast_reference(self):
        # the stacked forward log-density contraction <T_n, eta_k> over feature axes
        eng = NumpyEngine()
        rng = np.random.RandomState(1)
        n, k, dim = 300, 5, 8
        stat = rng.randn(n, dim, dim)  # per-row matrix sufficient statistic
        eta = rng.randn(k, dim, dim)  # per-component natural parameter
        got = _generated_exp_family_pair_term(stat, eta, eng, stacked=True)
        ref = np.einsum("nij,kij->nk", stat, eta)
        self.assertEqual(np.asarray(got).shape, (n, k))
        np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-11, atol=1e-11)

    def test_accumulation_peak_memory_is_bounded(self):
        # a config whose dense (n, k, dim, dim) intermediate would be ~2.3 GB; the gemm
        # accumulation must stay orders of magnitude below that. tracemalloc tracks numpy
        # allocations, so a regression that rebuilds the dense tensor spikes the peak.
        eng = NumpyEngine()
        rng = np.random.RandomState(2)
        n, k, dim = 4000, 8, 96
        dense_bytes = n * k * dim * dim * 8
        x = rng.randn(n, dim)
        w = rng.dirichlet(np.ones(k), size=n)
        tracemalloc.start()
        tracemalloc.reset_peak()
        MVN.backend_stacked_sufficient_statistics(x, w, {}, eng)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertGreater(dense_bytes, 2_000_000_000)  # the intermediate we refuse to build
        self.assertLess(peak, 150_000_000, f"peak {peak / 1e6:.0f} MB — dense would be {dense_bytes / 1e9:.1f} GB")


if __name__ == "__main__":
    unittest.main()
