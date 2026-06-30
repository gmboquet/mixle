"""Precision-aware heterogeneous planning (mixle.engines.heterogeneous)."""

import math
import unittest

from mixle.engines.heterogeneous import Worker, plan_heterogeneous


class HeterogeneousPlanTest(unittest.TestCase):
    def _pool(self):
        return [
            Worker("g0", "gpu", ("fp8", "bfloat16", "float16", "float32", "float64")),
            Worker("g1", "gpu", ("fp8", "bfloat16", "float16", "float32", "float64")),
            Worker("c0", "cpu", ("float32", "float64", "dd")),
        ]

    def test_assigns_all_rows_exactly(self):
        plan = plan_heterogeneous(self._pool(), 1_000_000)
        self.assertEqual(plan.total_rows(), 1_000_000)

    def test_picks_fastest_precision_per_device_when_unconstrained(self):
        plan = plan_heterogeneous(self._pool(), 1_000_000, target_rel_error=None)
        by = {a.name: a for a in plan.assignments}
        self.assertEqual(by["g0"].precision, "fp8")  # GPU goes lowest precision when accuracy is free
        self.assertEqual(by["c0"].precision, "float32")  # CPU's fastest real compute precision

    def test_tight_accuracy_forces_high_precision_everywhere(self):
        plan = plan_heterogeneous(self._pool(), 1_000_000, target_rel_error=1e-12, op_count=1000)
        for a in plan.assignments:
            self.assertEqual(a.precision, "float64")  # only float64 meets the budget

    def test_load_balances_toward_faster_workers(self):
        plan = plan_heterogeneous(self._pool(), 1_000_000, target_rel_error=None)
        by = {a.name: a for a in plan.assignments}
        # fp8 GPUs (throughput 4.0) get more rows than the fp32 CPU (1.4)
        self.assertGreater(by["g0"].rows, by["c0"].rows)

    def test_reduce_depth_is_logarithmic(self):
        workers = [Worker("w%d" % i, "cpu", ("float32", "float64")) for i in range(1000)]
        plan = plan_heterogeneous(workers, 10_000_000)
        self.assertEqual(plan.reduce_depth, max(1, math.ceil(math.log2(1000) / 2)))
        self.assertGreaterEqual(plan.reduce_depth, 4)  # not a single-root fan-in

    def test_single_worker_pool(self):
        plan = plan_heterogeneous([Worker("solo", "cpu", ("float32", "float64"))], 500)
        self.assertEqual(plan.total_rows(), 500)
        self.assertEqual(len(plan.assignments), 1)

    def test_empty_pool_raises(self):
        with self.assertRaises(ValueError):
            plan_heterogeneous([], 100)


if __name__ == "__main__":
    unittest.main()
