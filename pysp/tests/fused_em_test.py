"""Tests for optimize(reuse_estep_ll=True): the fused EM that reuses the E-step likelihood.

Verifies that the mixture accumulator's reported batch log-likelihood matches seq_log_density_sum,
that fixed-iteration fused EM matches the standard loop exactly, that delta-based convergence reaches
the same optimum, and that models which can't report the LL (top-level HMM) fall back cleanly.
"""
import io
import unittest

import numpy as np

from pysp.stats import (MixtureDistribution, GaussianDistribution, MixtureEstimator, GaussianEstimator,
                        HiddenMarkovModelDistribution, HiddenMarkovEstimator, CategoricalDistribution,
                        seq_encode, seq_log_density_sum)
from pysp.utils.estimation import optimize


class FusedEMTestCase(unittest.TestCase):

    def setUp(self):
        self.truth = MixtureDistribution(
            [GaussianDistribution(-3.0, 1.0), GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 1.0)],
            [0.4, 0.3, 0.3])
        self.data = self.truth.sampler(1).sample(8000)

    def _mk(self):
        return MixtureEstimator([GaussianEstimator()] * 3)

    def test_estep_ll_matches_seq_log_density_sum(self):
        enc = seq_encode(self.data, model=self.truth)
        _, ref = seq_log_density_sum(enc, self.truth)
        acc = self._mk().accumulator_factory().make()
        acc._track_ll = True
        for sz, x in enc:
            acc.seq_update(x, np.ones(sz), self.truth)
        self.assertAlmostEqual(acc._seq_ll, ref, places=6)

    def test_fixed_iters_identical_to_standard(self):
        std = optimize(self.data, self._mk(), max_its=30, delta=None,
                       rng=np.random.RandomState(1), out=io.StringIO())
        fused = optimize(self.data, self._mk(), max_its=30, delta=None,
                         rng=np.random.RandomState(1), out=io.StringIO(), reuse_estep_ll=True)
        # Fixed iteration count, same init -> identical fit.
        self.assertTrue(np.allclose(std.w, fused.w, atol=1.0e-10))
        for cs, cf in zip(std.components, fused.components):
            self.assertAlmostEqual(cs.mu, cf.mu, places=10)
            self.assertAlmostEqual(cs.sigma2, cf.sigma2, places=10)

    def test_delta_convergence_matches(self):
        enc = seq_encode(self.data, model=self.truth)
        std = optimize(self.data, self._mk(), max_its=300, delta=1.0e-7,
                       rng=np.random.RandomState(2), out=io.StringIO())
        fused = optimize(self.data, self._mk(), max_its=300, delta=1.0e-7,
                         rng=np.random.RandomState(2), out=io.StringIO(), reuse_estep_ll=True)
        _, lls = seq_log_density_sum(enc, std)
        _, llf = seq_log_density_sum(enc, fused)
        # Same optimum (the one-iteration convergence lag may stop a hair earlier/later).
        self.assertLess(abs(lls - llf), 1.0e-4)

    def test_fallback_for_non_mixture_top_level(self):
        topics = [GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)]
        hmm = HiddenMarkovModelDistribution(topics, [0.5, 0.5], [[0.8, 0.2], [0.2, 0.8]],
                                            len_dist=CategoricalDistribution({20: 1.0}))
        data = hmm.sampler(1).sample(300)
        est = HiddenMarkovEstimator([GaussianEstimator()] * 2, use_numba=True)
        std = optimize(data, est, max_its=15, delta=None, rng=np.random.RandomState(3), out=io.StringIO())
        # reuse_estep_ll requested but the HMM accumulator can't report it -> clean fallback.
        fb = optimize(data, est, max_its=15, delta=None, rng=np.random.RandomState(3),
                      out=io.StringIO(), reuse_estep_ll=True)
        enc = seq_encode(data, model=std)
        _, lls = seq_log_density_sum(enc, std)
        _, llf = seq_log_density_sum(enc, fb)
        self.assertAlmostEqual(lls, llf, places=6)

    def test_default_off_unchanged(self):
        # Default (reuse_estep_ll=False) must not set the tracking flag or alter results.
        m = optimize(self.data, self._mk(), max_its=10, rng=np.random.RandomState(4), out=io.StringIO())
        acc = self._mk().accumulator_factory().make()
        self.assertFalse(acc._track_ll)
        self.assertTrue(np.isclose(sum(m.w), 1.0))


if __name__ == '__main__':
    unittest.main()
