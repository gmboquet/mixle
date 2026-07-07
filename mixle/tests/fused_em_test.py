"""Tests for optimize(reuse_estep_ll=True): the fused EM that reuses the E-step likelihood.

Verifies that the mixture accumulator's reported batch log-likelihood matches seq_log_density_sum,
that fixed-iteration fused EM matches the standard loop exactly, that delta-based convergence reaches
the same optimum, and that models which can't report the LL (top-level HMM) fall back cleanly.
"""

import io
import unittest

import numpy as np

from mixle.inference.estimation import optimize
from mixle.stats import (
    GaussianDistribution,
    GaussianEstimator,
    HiddenMarkovEstimator,
    HiddenMarkovModelDistribution,
    MixtureDistribution,
    MixtureEstimator,
    PoissonDistribution,
    PoissonEstimator,
    SequenceDistribution,
    SequenceEstimator,
    seq_encode,
    seq_log_density_sum,
)


class FusedEMTestCase(unittest.TestCase):
    def setUp(self):
        self.truth = MixtureDistribution(
            [GaussianDistribution(-3.0, 1.0), GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 1.0)],
            [0.4, 0.3, 0.3],
        )
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
        std = optimize(self.data, self._mk(), max_its=30, delta=None, rng=np.random.RandomState(1), out=io.StringIO())
        fused = optimize(
            self.data,
            self._mk(),
            max_its=30,
            delta=None,
            rng=np.random.RandomState(1),
            out=io.StringIO(),
            reuse_estep_ll=True,
        )
        # Fixed iteration count, same init -> identical fit.
        self.assertTrue(np.allclose(std.w, fused.w, atol=1.0e-10))
        for cs, cf in zip(std.components, fused.components):
            self.assertAlmostEqual(cs.mu, cf.mu, places=10)
            self.assertAlmostEqual(cs.sigma2, cf.sigma2, places=10)

    def test_delta_convergence_matches(self):
        enc = seq_encode(self.data, model=self.truth)
        std = optimize(
            self.data, self._mk(), max_its=300, delta=1.0e-7, rng=np.random.RandomState(2), out=io.StringIO()
        )
        fused = optimize(
            self.data,
            self._mk(),
            max_its=300,
            delta=1.0e-7,
            rng=np.random.RandomState(2),
            out=io.StringIO(),
            reuse_estep_ll=True,
        )
        _, lls = seq_log_density_sum(enc, std)
        _, llf = seq_log_density_sum(enc, fused)
        # Same optimum (the one-iteration convergence lag may stop a hair earlier/later).
        self.assertLess(abs(lls - llf), 1.0e-4)

    def test_hmm_estep_ll_matches_and_fixed_iters_identical(self):
        topics = [GaussianDistribution(-4.0, 1.0), GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 1.0)]
        A = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        for use_numba in (True, False):
            hmm = HiddenMarkovModelDistribution(topics, [1 / 3.0] * 3, A, len_dist=PoissonDistribution(8.0))
            data = hmm.sampler(1).sample(1000)
            enc = seq_encode(data, model=hmm)
            est = HiddenMarkovEstimator(
                [GaussianEstimator()] * 3, len_estimator=PoissonEstimator(), use_numba=use_numba
            )
            with self.subTest(use_numba=use_numba):
                # E-step reported LL == seq_log_density_sum.
                acc = est.accumulator_factory().make()
                acc._track_ll = True
                for sz, x in enc:
                    acc.seq_update(x, np.ones(sz), hmm)
                _, ref = seq_log_density_sum(enc, hmm)
                self.assertAlmostEqual(acc._seq_ll, ref, places=5)
                # Fixed-iteration fused == standard.
                std = optimize(data, est, max_its=12, delta=None, rng=np.random.RandomState(3), out=io.StringIO())
                fused = optimize(
                    data,
                    est,
                    max_its=12,
                    delta=None,
                    rng=np.random.RandomState(3),
                    out=io.StringIO(),
                    reuse_estep_ll=True,
                )
                _, ls = seq_log_density_sum(enc, std)
                _, lf = seq_log_density_sum(enc, fused)
                self.assertAlmostEqual(ls, lf, places=6)

    def test_fallback_for_unsupported_top_level(self):
        # A SequenceDistribution iterates under EM (its inner mixture has latent components) but the
        # top-level SequenceAccumulator never records the E-step LL, so reuse_estep_ll hits the
        # `_seq_ll is None` fallback in _local_fused_step and scores the model itself. The fused and
        # standard loops must still produce the same result.
        # (Note: hmixture, HMMs, and the other latent families now DO report _seq_ll and take the
        # fused fast path -- they're covered by the tests above; this one exercises the fallback.)
        inner = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.5, 0.5])
        truth = SequenceDistribution(inner, len_dist=PoissonDistribution(5.0))
        data = truth.sampler(1).sample(400)
        est = SequenceEstimator(MixtureEstimator([GaussianEstimator()] * 2), len_estimator=PoissonEstimator())

        # Confirm the top-level accumulator genuinely does NOT report _seq_ll, so the fallback runs.
        enc = seq_encode(data, model=truth)
        acc = est.accumulator_factory().make()
        acc._track_ll = True
        for sz, x in enc:
            acc.seq_update(x, np.ones(sz), truth)
        self.assertIsNone(getattr(acc, "_seq_ll", None))

        std = optimize(data, est, max_its=15, delta=None, rng=np.random.RandomState(4), out=io.StringIO())
        fb = optimize(
            data, est, max_its=15, delta=None, rng=np.random.RandomState(4), out=io.StringIO(), reuse_estep_ll=True
        )
        enc = seq_encode(data, model=std)
        _, lls = seq_log_density_sum(enc, std)
        _, llf = seq_log_density_sum(enc, fb)
        self.assertAlmostEqual(lls, llf, places=6)

    def test_best_of_reuse_estep_ll(self):
        # best_of forwards reuse_estep_ll to each trial's optimize; same trials/seed -> same result.
        from mixle.inference.estimation import best_of

        std = best_of(self.data, None, self._mk(), 3, 20, 0.1, None, np.random.RandomState(1), out=io.StringIO())
        fused = best_of(
            self.data,
            None,
            self._mk(),
            3,
            20,
            0.1,
            None,
            np.random.RandomState(1),
            out=io.StringIO(),
            reuse_estep_ll=True,
        )
        self.assertAlmostEqual(std[0], fused[0], places=6)

    def test_default_off_unchanged(self):
        # Default (reuse_estep_ll=False) must not set the tracking flag or alter results.
        m = optimize(self.data, self._mk(), max_its=10, rng=np.random.RandomState(4), out=io.StringIO())
        acc = self._mk().accumulator_factory().make()
        self.assertFalse(acc._track_ll)
        self.assertTrue(np.isclose(sum(m.w), 1.0))

    def test_on_step_reports_the_model_paired_with_its_own_log_density(self):
        # Regression: the fused loop's on_step used to report EMStep(i+1, nxt, ll_model, dll) where
        # ll_model is the log-likelihood of the OLD model (this iteration's input, per the loop's own
        # docstring), not of nxt -- a caller that checkpoints (model, log_density) pairs together (as
        # the EMStep docstring recommends) would persist a mismatched pair. Every reported step's
        # model.log_density(x) summed over the data must equal the step's own reported log_density.
        g = GaussianDistribution(0.0, 1.0)
        data = list(np.random.RandomState(0).randn(200))
        est = GaussianEstimator()
        steps = []
        optimize(
            data,
            est,
            max_its=8,
            delta=1.0e-9,
            on_step=steps.append,
            reuse_estep_ll=True,
            out=None,
        )
        self.assertGreater(len(steps), 0)
        for s in steps:
            direct_ll = sum(s.model.log_density(x) for x in data)
            with self.subTest(iter=s.iter):
                self.assertAlmostEqual(direct_ll, s.log_density, places=6)


if __name__ == "__main__":
    unittest.main()
