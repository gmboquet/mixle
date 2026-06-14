"""Fused-EM (reuse_estep_ll) parity tests for the association families.

Each family that reports the E-step data log-likelihood through the
``_track_ll``/``_seq_ll`` accumulator contract is checked two ways:

  (a) PARITY: with ``_track_ll = True`` the accumulated ``_seq_ll`` over a
      ``seq_update`` pass must equal ``seq_log_density_sum`` for the same data.
  (b) FUSED EM: ``optimize(..., reuse_estep_ll=True)`` for a fixed number of
      iterations must reach the same final log-likelihood as the standard loop.

The integer-hidden-association numba branch does not expose the per-observation
normalizer; it reports ``_seq_ll = None`` so the fused loop falls back to a
separate scoring pass. The fused result must still match the standard loop.
"""

import io
import unittest

import numpy as np

from pysp.stats import seq_encode, seq_log_density_sum
from pysp.stats.categorical import CategoricalDistribution, CategoricalEstimator
from pysp.stats.composite import CompositeDistribution, CompositeEstimator
from pysp.stats.conditional import ConditionalDistribution, ConditionalDistributionEstimator
from pysp.stats.hidden_association import HiddenAssociationDistribution, HiddenAssociationEstimator
from pysp.stats.ibp import IndianBuffetProcessDistribution, IndianBuffetProcessEstimator
from pysp.stats.int_hidden_association import IntegerHiddenAssociationDistribution, IntegerHiddenAssociationEstimator
from pysp.stats.int_multinomial import IntegerMultinomialDistribution, IntegerMultinomialEstimator
from pysp.stats.markov_transform import MarkovTransformDistribution, MarkovTransformEstimator
from pysp.stats.sparse_markov_transform import SparseMarkovAssociationDistribution, SparseMarkovAssociationEstimator
from pysp.utils.estimation import optimize


def _devnull():
    return io.StringIO()


class FusedEmAssociationTestCase(unittest.TestCase):
    def _check_parity(self, dist, est, data):
        enc = seq_encode(data, dist.dist_to_encoder())
        _, ref = seq_log_density_sum(enc, dist)
        acc = est.accumulator_factory().make()
        acc._track_ll = True
        for sz, x in enc:
            acc.seq_update(x, np.ones(sz), dist)
        got = getattr(acc, "_seq_ll", None)
        self.assertIsNotNone(got, "accumulator did not report _seq_ll")
        if np.isneginf(ref):
            self.assertTrue(np.isneginf(got))
        else:
            self.assertAlmostEqual(got, ref, delta=1.0e-5 * max(1.0, abs(ref)))

    def _check_fused(self, est, data, max_its=6):
        m0 = optimize(
            data, est, max_its=max_its, delta=None, rng=np.random.RandomState(3), reuse_estep_ll=False, out=_devnull()
        )
        m1 = optimize(
            data, est, max_its=max_its, delta=None, rng=np.random.RandomState(3), reuse_estep_ll=True, out=_devnull()
        )
        _, l0 = seq_log_density_sum(seq_encode(data, m0.dist_to_encoder()), m0)
        _, l1 = seq_log_density_sum(seq_encode(data, m1.dist_to_encoder()), m1)
        self.assertAlmostEqual(l0, l1, delta=1.0e-5 * max(1.0, abs(l0)))

    # ---- hidden_association ----------------------------------------------
    def test_hidden_association(self):
        dist = HiddenAssociationDistribution(
            cond_dist=ConditionalDistribution(
                {
                    "a": CategoricalDistribution({"x": 0.80, "y": 0.20}),
                    "b": CategoricalDistribution({"x": 0.25, "y": 0.75}),
                }
            ),
            len_dist=CategoricalDistribution({0.0: 0.10, 2.0: 0.25, 3.0: 0.35, 4.0: 0.30}),
        )
        data = [
            ([("a", 2.0), ("b", 1.0)], [("x", 1.0), ("y", 2.0)]),
            ([("b", 3.0)], [("y", 2.0)]),
            ([("a", 1.0)], []),
            ([("a", 1.0), ("b", 2.0)], [("x", 3.0), ("y", 1.0)]),
        ]
        est = HiddenAssociationEstimator(
            cond_estimator=ConditionalDistributionEstimator({"a": CategoricalEstimator(), "b": CategoricalEstimator()}),
            len_estimator=CategoricalEstimator(),
        )
        self._check_parity(dist, est, data)
        self._check_fused(est, data)

    # ---- int_hidden_association ------------------------------------------
    def test_int_hidden_association_numpy_parity(self):
        cond = np.asarray([[0.80, 0.20], [0.25, 0.75], [0.55, 0.45]])
        state = np.asarray([[0.70, 0.20, 0.10], [0.15, 0.25, 0.60]])
        data = [
            ([(0, 2.0), (1, 1.0)], [(0, 3.0), (1, 2.0)]),
            ([(1, 1.0), (2, 2.0)], [(2, 1.0)]),
            ([(0, 1.0)], [(0, 1.0), (1, 1.0), (2, 2.0)]),
            ([(2, 3.0), (0, 1.0)], [(1, 2.0), (2, 1.0)]),
        ]
        dist = IntegerHiddenAssociationDistribution(
            state_prob_mat=state, cond_weights=cond, alpha=0.30, use_numba=False
        )
        est = IntegerHiddenAssociationEstimator(num_vals=[3, 3], num_states=2, alpha=0.30, use_numba=False)
        self._check_parity(dist, est, data)

    def test_int_hidden_association_numba_reports_none(self):
        cond = np.asarray([[0.80, 0.20], [0.25, 0.75], [0.55, 0.45]])
        state = np.asarray([[0.70, 0.20, 0.10], [0.15, 0.25, 0.60]])
        data = [([(0, 2.0), (1, 1.0)], [(0, 3.0), (1, 2.0)]), ([(2, 3.0), (0, 1.0)], [(1, 2.0), (2, 1.0)])]
        dist = IntegerHiddenAssociationDistribution(state_prob_mat=state, cond_weights=cond, alpha=0.30, use_numba=True)
        est = IntegerHiddenAssociationEstimator(num_vals=[3, 3], num_states=2, alpha=0.30, use_numba=True)
        enc = seq_encode(data, dist.dist_to_encoder())
        acc = est.accumulator_factory().make()
        acc._track_ll = True
        for sz, x in enc:
            acc.seq_update(x, np.ones(sz), dist)
        # numba branch cannot report the normalizer: must signal fallback (None), not 0.0.
        self.assertIsNone(acc._seq_ll)

    def _int_hidden_fused_data(self):
        init_prob = [0.2, 0.2, 0.3, 0.3]
        state_prob = [[0.7, 0.1, 0.1, 0.1], [0.0, 0.3, 0.4, 0.3]]
        cond_w = [[0.8, 0.2], [0.4, 0.6], [0.2, 0.8], [0.5, 0.5]]
        len_dist = CategoricalDistribution({3: 1.0})
        init_dist = IntegerMultinomialDistribution(0, init_prob, len_dist=len_dist)
        gdist = IntegerHiddenAssociationDistribution(state_prob, cond_w, prev_dist=init_dist, len_dist=len_dist)
        return gdist.sampler(1).sample(150)

    def test_int_hidden_association_fused_numpy(self):
        data = self._int_hidden_fused_data()
        len_est = CategoricalEstimator()
        prev_est = IntegerMultinomialEstimator(min_val=0, len_estimator=len_est)
        est = IntegerHiddenAssociationEstimator(4, 2, prev_estimator=prev_est, len_estimator=len_est, use_numba=False)
        self._check_fused(est, data)

    def test_int_hidden_association_fused_numba_fallback(self):
        data = self._int_hidden_fused_data()
        len_est = CategoricalEstimator()
        prev_est = IntegerMultinomialEstimator(min_val=0, len_estimator=len_est)
        est = IntegerHiddenAssociationEstimator(4, 2, prev_estimator=prev_est, len_estimator=len_est, use_numba=True)
        # fused loop must fall back gracefully and match the standard loop.
        self._check_fused(est, data)

    # ---- ibp -------------------------------------------------------------
    def test_ibp(self):
        K = 5
        rng = np.random.RandomState(1)
        data = [list(np.flatnonzero(rng.rand(K) < 0.4)) for _ in range(40)]
        dist = IndianBuffetProcessDistribution(K, alpha=2.0, data_format="sparse")
        est = IndianBuffetProcessEstimator(K, alpha=2.0, data_format="sparse")
        self._check_parity(dist, est, data)
        self._check_fused(est, data)

    # ---- markov_transform ------------------------------------------------
    def test_markov_transform(self):
        nw = 3
        init_prob = np.asarray([0.5, 0.3, 0.2])
        rng = np.random.RandomState(7)
        cond_prob = rng.rand(nw * nw, nw) + 0.1
        cond_prob /= cond_prob.sum(axis=1, keepdims=True)
        len_dist = CompositeDistribution(
            (
                CategoricalDistribution({2: 0.5, 3: 0.5}),
                CategoricalDistribution({2: 0.5, 3: 0.5}),
                CategoricalDistribution({3: 0.6, 4: 0.4}),
            )
        )
        dist = MarkovTransformDistribution(init_prob, cond_prob, alpha=0.05, len_dist=len_dist)
        data = dist.sampler(seed=11).sample(size=25)
        len_est = CompositeEstimator((CategoricalEstimator(), CategoricalEstimator(), CategoricalEstimator()))
        est = MarkovTransformEstimator(3, alpha=0.05, len_estimator=len_est)
        self._check_parity(dist, est, data)
        self._check_fused(est, data)

    # ---- sparse_markov_transform -----------------------------------------
    def _sparse_setup(self, low_memory):
        nw = 3
        init_prob = np.asarray([0.5, 0.3, 0.2])
        rng = np.random.RandomState(11)
        cond_prob = rng.rand(nw, nw) + 0.1
        cond_prob /= cond_prob.sum(axis=1, keepdims=True)
        len_dist = CompositeDistribution(
            (CategoricalDistribution({2: 0.5, 3: 0.5}), CategoricalDistribution({3: 0.6, 4: 0.4}))
        )
        dist = SparseMarkovAssociationDistribution(
            init_prob, cond_prob, alpha=0.1, len_dist=len_dist, low_memory=low_memory
        )
        data = dist.sampler(seed=5).sample(size=30)
        len_est = CompositeEstimator((CategoricalEstimator(), CategoricalEstimator()))
        est = SparseMarkovAssociationEstimator(num_vals=nw, alpha=0.1, len_estimator=len_est, low_memory=low_memory)
        return dist, est, data

    def test_sparse_markov_transform_lowmem(self):
        dist, est, data = self._sparse_setup(low_memory=True)
        self._check_parity(dist, est, data)
        self._check_fused(est, data)

    def test_sparse_markov_transform_flat(self):
        dist, est, data = self._sparse_setup(low_memory=False)
        self._check_parity(dist, est, data)
        self._check_fused(est, data)


if __name__ == "__main__":
    unittest.main()
