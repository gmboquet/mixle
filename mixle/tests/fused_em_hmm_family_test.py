"""Fused-EM (optimize(reuse_estep_ll=True)) support for the HMM-style families.

For each family this verifies two things:
  * PARITY: with _track_ll enabled the accumulator's reported batch log-likelihood (_seq_ll)
    equals seq_log_density_sum() for the same data and distribution -- including the
    length-distribution term and the family's own -inf/bad-row handling.
  * FUSED: EM run to convergence with reuse_estep_ll=True reaches the same optimum as the standard
    loop (same init/seed). The fused loop's best-model selection lags the standard loop by one
    iteration by design, so the families are compared at the converged fixed point rather than at a
    fixed iteration count.

Families covered: SegmentalHiddenMarkov, TreeHiddenMarkov (numpy + numba encodings),
LookbackHiddenMarkovModel.
"""

import io
import unittest

import numpy as np

import mixle.stats.latent.lookback_hidden_markov_model as look_back_mod
from mixle.inference.estimation import optimize
from mixle.stats import (
    CategoricalDistribution,
    CategoricalEstimator,
    GaussianDistribution,
    GaussianEstimator,
    SequenceDistribution,
    SequenceEstimator,
    seq_encode,
    seq_log_density_sum,
)
from mixle.stats.latent.segmental_hidden_markov_model import SegmentalHiddenMarkovModelDistribution
from mixle.stats.latent.tree_hidden_markov_model import TreeHiddenMarkovEstimator, TreeHiddenMarkovModelDistribution
from mixle.stats.sequences.integer_markov_chain import IntegerMarkovChainDistribution, IntegerMarkovChainEstimator
from mixle.stats.univariate.discrete.integer_categorical import (
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
)


class FusedEMHmmFamilyTestCase(unittest.TestCase):
    def _parity(self, dist, est, data):
        enc = seq_encode(data, model=dist)
        _, ref = seq_log_density_sum(enc, dist)
        acc = est.accumulator_factory().make()
        acc._track_ll = True
        for sz, x in enc:
            acc.seq_update(x, np.ones(sz), dist)
        self.assertAlmostEqual(acc._seq_ll, ref, delta=1e-5 * max(1.0, abs(ref)))

    def _fused(self, est, data, max_its=200):
        std = optimize(
            data,
            est,
            max_its=max_its,
            delta=1e-8,
            rng=np.random.RandomState(7),
            out=io.StringIO(),
            reuse_estep_ll=False,
        )
        fused = optimize(
            data, est, max_its=max_its, delta=1e-8, rng=np.random.RandomState(7), out=io.StringIO(), reuse_estep_ll=True
        )
        _, ls = seq_log_density_sum(seq_encode(data, model=std), std)
        _, lf = seq_log_density_sum(seq_encode(data, model=fused), fused)
        self.assertAlmostEqual(ls, lf, delta=1e-4 * max(1.0, abs(ls)))

    def _default_off(self, est):
        acc = est.accumulator_factory().make()
        self.assertFalse(acc._track_ll)
        self.assertEqual(acc._seq_ll, 0.0)

    # ----------------------------------------------------------------- segmental

    def _segmental(self):
        dist = SegmentalHiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 1.0)],
            [0.6, 0.4],
            [[0.7, 0.3], [0.4, 0.6]],
            len_dist=CategoricalDistribution({3: 0.5, 4: 0.5}),
        )
        return dist, dist.estimator(pseudo_count=1e-6), dist.sampler(seed=1).sample(25)

    def test_segmental(self):
        dist, est, data = self._segmental()
        self._parity(dist, est, data)
        # max_its=200 never actually reaches the delta=1e-8 stopping criterion on this fixture (still
        # improving by ~2e-4/iteration at 200), so the standard/fused comparison is really "same fixed
        # iteration count", not "same converged optimum" -- verified the fused/standard log-likelihoods
        # stay bit-identical (not just within tolerance) all the way down to max_its=15; 50 keeps a large
        # margin above that floor while cutting this EM loop's cost ~4x.
        self._fused(est, data, max_its=50)
        self._default_off(est)

    # (the semi-supervised HMM is numpy-only and has no fused-EM fast path; see semi_supervised_hmm_test.py)

    # ------------------------------------------------------------------ tree_hmm

    _TREES = [
        [((0, -1), 0.1), ((1, 0), 0.2), ((2, 1), 9.9)],
        [((0, -1), 0.1), ((1, 0), 0.2), ((2, 1), 9.9), ((3, 2), 0.3)],
        [((0, -1), 0.1), ((1, 0), 0.2), ((2, 0), 9.9)],
        [((0, -1), 9.5), ((1, 0), 9.7), ((2, 0), 0.1), ((3, 1), 9.9), ((4, 1), 0.2)],
        [((0, -1), 0.4)],
    ]

    def _tree_gauss_dist(self, use_numba):
        topics = [GaussianDistribution(mu=0.0, sigma2=1.0), GaussianDistribution(mu=10.0, sigma2=1.0)]
        return TreeHiddenMarkovModelDistribution(
            topics=topics,
            w=np.array([0.6, 0.4]),
            transitions=np.array([[0.7, 0.3], [0.2, 0.8]]),
            len_dist=CategoricalDistribution({0: 0.3, 1: 0.4, 2: 0.3}),
            terminal_level=6,
            use_numba=use_numba,
        )

    def _tree_gauss_est(self, use_numba):
        ge = lambda: GaussianEstimator(pseudo_count=(1.0, 1.0), suff_stat=(0.0, 1.0))
        return TreeHiddenMarkovEstimator(
            [ge(), ge()], len_estimator=CategoricalEstimator(pseudo_count=0.1), use_numba=use_numba
        )

    def _tree_cat_dist(self, use_numba):
        topics = [
            CategoricalDistribution({"a": 0.8, "b": 0.1, "c": 0.1}),
            CategoricalDistribution({"a": 0.1, "b": 0.1, "c": 0.8}),
        ]
        return TreeHiddenMarkovModelDistribution(
            topics=topics,
            w=np.array([0.6, 0.4]),
            transitions=np.array([[0.7, 0.3], [0.2, 0.8]]),
            len_dist=CategoricalDistribution({0: 0.3, 1: 0.4, 2: 0.3}),
            terminal_level=5,
            use_numba=use_numba,
        )

    def _tree_cat_est(self, use_numba):
        return TreeHiddenMarkovEstimator(
            [CategoricalEstimator(pseudo_count=0.5), CategoricalEstimator(pseudo_count=0.5)],
            len_estimator=CategoricalEstimator(pseudo_count=0.1),
            use_numba=use_numba,
        )

    def test_tree_numpy(self):
        # Parity (bit-for-bit) on the curated trees; fused on categorical-emission sampled trees,
        # where the tree EM is numerically stable (Gaussian tree EM on the tiny curated trees is
        # degenerate on both fused and standard paths, unrelated to _seq_ll tracking).
        self._parity(self._tree_gauss_dist(False), self._tree_gauss_est(False), self._TREES)
        cat_data = self._tree_cat_dist(False).sampler(seed=2).sample(60)
        self._fused(self._tree_cat_est(False), cat_data)
        self._default_off(self._tree_gauss_est(False))

    def test_tree_numba(self):
        self._parity(self._tree_gauss_dist(True), self._tree_gauss_est(True), self._TREES)
        cat_data = self._tree_cat_dist(True).sampler(seed=2).sample(60)
        self._fused(self._tree_cat_est(True), cat_data)
        self._default_off(self._tree_gauss_est(True))

    # ----------------------------------------------------------------- lookback

    def _lookback_dist(self, mod):
        d0 = IntegerCategoricalDistribution(0, [0.4, 0.3, 0.3])
        dist1 = IntegerMarkovChainDistribution(3, [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        dist2 = IntegerMarkovChainDistribution(3, [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
        dist3 = IntegerMarkovChainDistribution(3, [[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
        init_dists = [SequenceDistribution(d0, CategoricalDistribution({1: 1.0}))] * 3
        states = [dist1, dist2, dist3]
        len_dist = CategoricalDistribution({7: 0.5, 8: 0.25, 9: 0.25})
        transition = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        w = [0.4, 0.3, 0.3]
        return mod.LookbackHiddenMarkovModelDistribution(
            states, w=w, transitions=transition, lag=1, init_dist=init_dists, len_dist=len_dist
        )

    def _lookback_est(self, mod):
        est0 = SequenceEstimator(
            IntegerCategoricalEstimator(pseudo_count=0.1), len_estimator=CategoricalEstimator(pseudo_count=0.1)
        )
        est1 = IntegerMarkovChainEstimator(3, pseudo_count=0.1)
        return mod.LookbackHiddenMarkovModelEstimator(
            [est1] * 3,
            lag=1,
            init_estimators=[est0] * 3,
            len_estimator=CategoricalEstimator(pseudo_count=0.1),
            pseudo_count=(1.0, 1.0),
        )

    def test_lookback_typed(self):
        dist = self._lookback_dist(look_back_mod)
        data = dist.sampler(seed=1).sample(60)
        self._parity(dist, self._lookback_est(look_back_mod), data)
        self._fused(self._lookback_est(look_back_mod), data)
        self._default_off(self._lookback_est(look_back_mod))


if __name__ == "__main__":
    unittest.main()
