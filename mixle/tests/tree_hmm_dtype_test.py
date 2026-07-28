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

    def test_initialize_conserves_training_mass_under_every_thread_count(self):
        """seq_initialize must count every node exactly once, however many threads numba uses.

        The kernel accumulated straight into the shared ``(K,)``/``(K, K)`` statistic arrays from
        inside a ``prange``. That is an unsynchronized read-modify-write: two threads reading the
        same element before either wrote lost one update, so training mass silently vanished. A
        200-record fixture with 752 nodes accumulated 674-749 of them, differing run to run, and
        every tree-HMM fit began from statistics missing several percent of the evidence at random.
        Assert the conservation law directly -- one unit of initial-state mass per tree plus one
        unit of transition mass per edge must equal the total state mass, which equals the node
        count -- and assert it is bit-identical across thread counts so a reintroduced race cannot
        hide behind a single-threaded CI runner.
        """
        import numba

        model = self._model()
        data = model.sampler(seed=3).sample(200)
        nodes = sum(len(record) for record in data)
        roots = sum(1 for record in data for node in record if node[0][1] == -1)
        enc = model.dist_to_encoder().seq_encode(data)
        est = model.estimator()

        available = numba.get_num_threads()
        try:
            for threads in {1, available}:
                numba.set_num_threads(threads)
                for trial in range(4):
                    acc = est.accumulator_factory().make()
                    acc.seq_initialize(enc, np.ones(len(data)), np.random.RandomState(trial))
                    state = float(acc.state_counts.sum())
                    assigned = float(acc.init_counts.sum() + acc.trans_counts.sum())
                    self.assertEqual(state, float(nodes), f"lost state mass at {threads} thread(s)")
                    self.assertEqual(float(acc.init_counts.sum()), float(roots))
                    self.assertEqual(float(acc.trans_counts.sum()), float(nodes - roots))
                    self.assertEqual(assigned, state, f"mass not conserved at {threads} thread(s)")
        finally:
            numba.set_num_threads(available)


if __name__ == "__main__":
    unittest.main()
