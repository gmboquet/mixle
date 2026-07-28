"""Distributed heterogeneous EM executor (mixle.inference.heterogeneous_executor): sharded tree-reduce."""

import unittest

import numpy as np

import mixle.stats as st
from mixle.engines.heterogeneous import Worker, plan_heterogeneous
from mixle.inference.heterogeneous_executor import (
    heterogeneous_em_step,
    heterogeneous_fit,
    shards_from_plan,
    tree_reduce_values,
)


def _gmm(rng, k=3):
    comps = [st.GaussianDistribution(float(6 * rng.randn()), float(0.5 + rng.rand())) for _ in range(k)]
    return st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(k))))


def _cat_mixture():
    return st.MixtureDistribution(
        [st.CategoricalDistribution({"a": 0.6, "b": 0.4}), st.CategoricalDistribution({"a": 0.2, "b": 0.8})],
        [0.5, 0.5],
    )


class TreeReduceTest(unittest.TestCase):
    def test_integer_stats_bit_identical_to_linear_fold(self):
        rng = np.random.RandomState(0)
        m = _cat_mixture()
        data = m.sampler(1).sample(3000)
        est = m.estimator()
        fac = est.accumulator_factory()
        # 9 shard values
        vals = []
        for lo in range(0, 2700, 300):
            shard = data[lo : lo + 300]
            acc = fac.make()
            acc.seq_update(m.dist_to_encoder().seq_encode(shard), np.ones(len(shard)), m)
            vals.append(acc.value())
        # linear fold
        lin = fac.make().from_value(vals[0])
        for v in vals[1:]:
            lin.combine(v)
        # tree fold at several arities -> same estimated model (integer counts -> bit-identical)
        for branch in (2, 3, 4):
            tree = tree_reduce_values(vals, fac, branch=branch)
            mt = est.estimate(2700, tree)
            ml = est.estimate(2700, lin.value())
            self.assertTrue(np.allclose(sorted(mt.w), sorted(ml.w)))


class ShardingInvarianceTest(unittest.TestCase):
    def test_distributing_does_not_change_the_em_step(self):
        rng = np.random.RandomState(1)
        m = _gmm(rng, 3)
        data = m.sampler(2).sample(4000)
        est = m.estimator()
        serial = heterogeneous_em_step(est, m, data, n_shards=1)  # the serial baseline
        for k in (2, 8, 17):
            dist = heterogeneous_em_step(est, m, data, n_shards=k)
            self.assertTrue(np.allclose(sorted(serial.w), sorted(dist.w), atol=1e-9))
            sm = sorted(c.mu for c in serial.components)
            dm = sorted(c.mu for c in dist.components)
            self.assertTrue(np.allclose(sm, dm, atol=1e-9))

    def test_fit_converges(self):
        rng = np.random.RandomState(2)
        m = _gmm(rng, 3)
        data = m.sampler(3).sample(4000)
        fit1 = heterogeneous_fit(m, data, max_its=15, n_shards=1)
        fit8 = heterogeneous_fit(m, data, max_its=15, n_shards=8)
        self.assertTrue(np.allclose(sorted(fit1.w), sorted(fit8.w), atol=1e-8))


class HeterogeneousPrecisionTest(unittest.TestCase):
    def test_mismatched_shard_sizes_raises_instead_of_silently_dropping_rows(self):
        # _shard_bounds never checked sum(shard_sizes) == len(data); rows past the last bound were
        # silently excluded from every shard -- reachable in practice via shards_from_plan() if a
        # precomputed plan is applied to a differently-sized dataset than it was sized against.
        rng = np.random.RandomState(5)
        m = _gmm(rng, 2)
        data = m.sampler(6).sample(1000)
        est = m.estimator()
        with self.assertRaises(ValueError):
            heterogeneous_em_step(est, m, data, n_shards=2, shard_sizes=[300, 400])  # sums to 700, not 1000

    def test_per_shard_float32_runs_and_stays_close(self):
        rng = np.random.RandomState(3)
        m = _gmm(rng, 2)
        data = m.sampler(4).sample(4000)
        est = m.estimator()
        f64 = heterogeneous_em_step(est, m, data, n_shards=4)
        mixed = heterogeneous_em_step(est, m, data, n_shards=4, shard_precisions=[np.float32, None, np.float32, None])
        self.assertTrue(np.allclose(sorted(f64.w), sorted(mixed.w), atol=1e-3))

    def test_plan_drives_the_executor(self):
        rng = np.random.RandomState(4)
        m = _gmm(rng, 3)
        data = m.sampler(5).sample(3000)
        workers = [
            Worker("g0", "gpu", ("float32", "float64")),
            Worker("c0", "cpu", ("float32", "float64")),
            Worker("c1", "cpu", ("float32", "float64")),
        ]
        plan = plan_heterogeneous(workers, len(data), target_rel_error=None)
        sizes, precisions = shards_from_plan(plan)
        self.assertEqual(sum(sizes), len(data))
        fit = heterogeneous_fit(
            m, data, max_its=10, n_shards=len(sizes), shard_sizes=sizes, shard_precisions=precisions
        )
        serial = heterogeneous_fit(m, data, max_its=10, n_shards=1)
        self.assertTrue(np.allclose(sorted(fit.w), sorted(serial.w), atol=1e-2))  # close despite mixed precision


class MultiProcessExecutorTest(unittest.TestCase):
    def test_real_worker_processes_match_serial(self):
        # actual OS processes: sufficient-statistic payloads cross the process boundary by pickling and
        # combine() folds those freshly-unpickled copies -> result identical to the serial executor.
        from concurrent.futures import ProcessPoolExecutor

        rng = np.random.RandomState(7)
        m = _gmm(rng, 3)
        data = m.sampler(8).sample(2000)
        est = m.estimator()
        serial = heterogeneous_em_step(est, m, data, n_shards=4)
        with ProcessPoolExecutor(max_workers=2) as pool:
            parallel = heterogeneous_em_step(est, m, data, n_shards=4, pool=pool)
        self.assertTrue(np.allclose(sorted(serial.w), sorted(parallel.w), atol=1e-9))
        sm = sorted(c.mu for c in serial.components)
        pm = sorted(c.mu for c in parallel.components)
        self.assertTrue(np.allclose(sm, pm, atol=1e-9))


class ReductionBranchTest(unittest.TestCase):
    """MXR-080-1633: branch=1 rebuilds an equal-length level forever, hanging the public EM step."""

    class _Factory:
        class _Acc:
            def from_value(self, v):
                self.v = v
                return self

            def combine(self, other):
                self.v += other

            def value(self):
                return self.v

        def make(self):
            return self._Acc()

    def test_unary_branch_is_rejected_instead_of_looping_forever(self):
        with self.assertRaises(ValueError) as ctx:
            tree_reduce_values([1, 2, 3], self._Factory(), branch=1)
        self.assertIn("never terminates", str(ctx.exception))

    def test_non_positive_and_non_integer_branches_are_rejected(self):
        for bad in (0, -2):
            with self.assertRaises(ValueError):
                tree_reduce_values([1, 2, 3], self._Factory(), branch=bad)
        for bad in (2.5, True):
            with self.assertRaises(TypeError):
                tree_reduce_values([1, 2, 3], self._Factory(), branch=bad)

    def test_single_payload_needs_no_reduction(self):
        # Nothing to reduce, so an unused branch must not fail the run.
        self.assertEqual(tree_reduce_values([7], self._Factory(), branch=1), 7)

    def test_valid_branch_still_reduces(self):
        self.assertEqual(tree_reduce_values([1, 2, 3, 4], self._Factory(), branch=2), 10)

    def test_fit_rejects_a_non_positive_iteration_count(self):
        m = _cat_mixture()
        data = m.sampler(1).sample(50)
        for bad in (0, -3):
            with self.assertRaises(ValueError):
                heterogeneous_fit(m, data, max_its=bad, n_shards=2)


class ShardPartitionTest(unittest.TestCase):
    """MXR-080-1634: shard sizes must be a real partition, not merely sum to len(data)."""

    def setUp(self):
        self.model = _cat_mixture()
        self.data = self.model.sampler(1).sample(5)
        self.est = self.model.estimator()

    def test_negative_sizes_that_sum_correctly_are_rejected(self):
        # [-2, -2, 9] sums to 5 but negative slicing makes shards [0,1,2], [], [1,2,3,4]:
        # rows 1 and 2 are processed twice and the E-step sees seven observations for five rows.
        with self.assertRaises(ValueError):
            heterogeneous_em_step(self.est, self.model, self.data, shard_sizes=[-2, -2, 9])

    def test_fractional_sizes_are_rejected(self):
        with self.assertRaises(TypeError):
            heterogeneous_em_step(self.est, self.model, self.data, shard_sizes=[2.5, 2.5])

    def test_empty_shard_size_list_is_rejected(self):
        with self.assertRaises(ValueError):
            heterogeneous_em_step(self.est, self.model, self.data, shard_sizes=[])

    def test_non_positive_shard_count_is_rejected(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                heterogeneous_em_step(self.est, self.model, self.data, n_shards=bad)

    def test_valid_partition_still_runs_and_matches_one_shard(self):
        one = heterogeneous_em_step(self.est, self.model, self.data, n_shards=1)
        split = heterogeneous_em_step(self.est, self.model, self.data, shard_sizes=[2, 3])
        self.assertTrue(np.allclose(sorted(one.w), sorted(split.w), atol=1e-9))

    def test_zero_row_shards_are_still_accepted(self):
        # plan_heterogeneous validates assignment rows as NONNEGATIVE and a slow worker can round to
        # zero rows, so shards_from_plan legitimately emits a 0. Rejecting it would break real plans.
        one = heterogeneous_em_step(self.est, self.model, self.data, n_shards=1)
        with_empty = heterogeneous_em_step(self.est, self.model, self.data, shard_sizes=[2, 0, 3])
        self.assertTrue(np.allclose(sorted(one.w), sorted(with_empty.w), atol=1e-9))


if __name__ == "__main__":
    unittest.main()
