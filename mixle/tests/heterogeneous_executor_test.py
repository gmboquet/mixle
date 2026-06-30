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


if __name__ == "__main__":
    unittest.main()
