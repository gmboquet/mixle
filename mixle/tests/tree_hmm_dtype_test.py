"""Dtype pinning at TreeHMM's explicitly-signed numba kernel boundaries.

The tree Baum-Welch/initialize kernels carry EXPLICIT eager signatures (int32 encoder arrays, int64
states, float64 weights): eager dispatch has no widening, so caller-supplied integer weights (an
``np.ones(n, dtype=int)``, a list of ints) used to arrive as int64 and fail with numba's
"No matching definition for argument type(s)". Seen on Python 3.14 where the estimation path
produced integer weights; the call sites now coerce at the boundary. This test drives BOTH weighted
kernel entry points (seq_initialize and the Baum-Welch seq_update) with deliberately-integer weights
and must never raise, on any Python.
"""

import unittest

import numpy as np

from mixle.utils.optional_deps import HAS_NUMBA

if HAS_NUMBA:
    from mixle.stats import GaussianDistribution, IntegerCategoricalDistribution, TreeHiddenMarkovModelDistribution


@unittest.skipUnless(HAS_NUMBA, "the tree HMM kernels require numba")
class TreeHmmKernelDtypeTest(unittest.TestCase):
    def _model(self):
        topics = [GaussianDistribution(mu=float(10 * s), sigma2=1.0) for s in range(3)]
        trans = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
        len_dist = IntegerCategoricalDistribution(min_val=0, p_vec=np.array([0.25, 0.25, 0.5]))
        return TreeHiddenMarkovModelDistribution(
            topics=topics, w=np.ones(3) / 3, transitions=trans, len_dist=len_dist, terminal_level=2
        )

    def test_integer_weights_survive_both_weighted_kernel_boundaries(self):
        model = self._model()
        rng = np.random.RandomState(0)
        data = model.sampler(seed=1).sample(60)
        enc = model.dist_to_encoder().seq_encode(data)
        est = model.estimator()
        acc = est.accumulator_factory().make()

        int_weights = np.ones(len(data), dtype=np.int64)  # the exact shape of the 3.14 failure
        acc.seq_initialize(enc, int_weights, rng)  # numba_initialize boundary
        acc2 = est.accumulator_factory().make()
        acc2.seq_update(enc, int_weights, model)  # numba_baum_welch boundary

        new_model = est.estimate(len(data), acc2.value())
        self.assertTrue(np.isfinite(np.sum(new_model.seq_log_density(enc))))

    def test_float_weights_unchanged(self):
        model = self._model()
        data = model.sampler(seed=2).sample(40)
        enc = model.dist_to_encoder().seq_encode(data)
        est = model.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(enc, np.ones(len(data)), model)
        self.assertTrue(np.isfinite(np.sum(model.seq_log_density(enc))))


if __name__ == "__main__":
    unittest.main()
