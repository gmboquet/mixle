"""Precision-aware heterogeneous planning (mixle.engines.heterogeneous)."""

import math
import unittest

from mixle.engines.heterogeneous import (
    HeterogeneousPlan,
    InfeasiblePrecisionError,
    Worker,
    WorkerAssignment,
    plan_heterogeneous,
)


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
        plan = plan_heterogeneous(
            self._pool(), 1_000_000, target_rel_error=1e-12, op_count=1000, allow_infeasible=True
        )
        for a in plan.assignments:
            self.assertEqual(a.precision, "float64")  # only float64 meets the budget
            self.assertFalse(a.meets_target)  # the format score is heuristic, not a workload certificate

    def test_load_balances_toward_faster_workers(self):
        plan = plan_heterogeneous(self._pool(), 1_000_000, target_rel_error=None)
        by = {a.name: a for a in plan.assignments}
        # fp8 GPUs (throughput 4.0) get more rows than the fp32 CPU (1.4)
        self.assertGreater(by["g0"].rows, by["c0"].rows)

    def test_reduce_depth_is_logarithmic(self):
        workers = [Worker("w%d" % i, "cpu", ("float32", "float64")) for i in range(1000)]
        plan = plan_heterogeneous(workers, 10_000_000)
        self.assertEqual(plan.reduce_depth, math.ceil(math.log2(1000) / 2))
        self.assertGreaterEqual(plan.reduce_depth, 4)  # not a single-root fan-in

    def test_single_worker_pool(self):
        plan = plan_heterogeneous([Worker("solo", "cpu", ("float32", "float64"))], 500)
        self.assertEqual(plan.total_rows(), 500)
        self.assertEqual(len(plan.assignments), 1)
        self.assertEqual(plan.reduce_depth, 0)

    def test_empty_pool_raises(self):
        with self.assertRaises(ValueError):
            plan_heterogeneous([], 100)


class InfeasiblePrecisionTest(unittest.TestCase):
    """MXR-080-0133: an unachievable target_rel_error must never be silently downgraded."""

    def test_impossible_target_raises_by_default(self):
        # Audit's exact repro: a worker that only supports float32 (~6e-8 relative roundoff per op)
        # cannot come within twelve orders of magnitude of a 1e-20 target.
        worker = Worker(name="w0", device="cpu", precisions=("float32",))
        with self.assertRaises(InfeasiblePrecisionError) as ctx:
            plan_heterogeneous([worker], 100, target_rel_error=1e-20)
        message = str(ctx.exception)
        self.assertIn("w0", message)
        self.assertIn("1e-20", message)

    def test_allow_infeasible_returns_explicit_infeasible_plan_with_achieved_bound(self):
        worker = Worker(name="w0", device="cpu", precisions=("float32",))
        plan = plan_heterogeneous([worker], 100, target_rel_error=1e-20, allow_infeasible=True)
        self.assertFalse(plan.is_feasible())
        infeasible = plan.infeasible_assignments()
        self.assertEqual(len(infeasible), 1)
        self.assertEqual(infeasible[0].name, "w0")
        self.assertFalse(infeasible[0].meets_target)
        # Quantified: float32's unit roundoff is 2**-24 at op_count=1000 -> ~5.96e-5, far above the 1e-20 ask.
        product = 1000 * 2.0**-24
        self.assertAlmostEqual(infeasible[0].achieved_rel_error, product / (1.0 - product), places=12)
        self.assertGreater(infeasible[0].achieved_rel_error, 1e-20)
        # The plan is still otherwise usable -- rows are assigned normally, not zeroed out or omitted.
        self.assertEqual(plan.total_rows(), 100)

    def test_endpoint_score_below_target_remains_explicitly_heuristic(self):
        worker = Worker(name="w0", device="cpu", precisions=("float32", "float64"))
        with self.assertRaises(InfeasiblePrecisionError):
            plan_heterogeneous([worker], 1000, target_rel_error=1e-12, op_count=1000)
        plan = plan_heterogeneous(
            [worker], 1000, target_rel_error=1e-12, op_count=1000, allow_infeasible=True
        )
        self.assertFalse(plan.is_feasible())
        self.assertFalse(plan.assignments[0].meets_target)
        self.assertEqual(plan.assignments[0].evidence_kind, "heuristic")
        self.assertEqual(plan.assignments[0].precision, "float64")
        self.assertIsNotNone(plan.assignments[0].achieved_rel_error)
        self.assertLessEqual(plan.assignments[0].achieved_rel_error, 1e-12)

    def test_unconstrained_plan_is_always_feasible(self):
        worker = Worker(name="w0", device="cpu", precisions=("float32", "float64"))
        plan = plan_heterogeneous([worker], 1000, target_rel_error=None)
        self.assertTrue(plan.is_feasible())
        self.assertTrue(plan.assignments[0].meets_target)

    def test_fallback_prefers_the_most_accurate_option_not_the_fastest(self):
        # A GPU worker with both float16 (throughput 2.5, coarse) and float64 (throughput 1.0, tightest):
        # when NEITHER meets an impossible target, the reported fallback must be the most accurate one
        # (float64), not whichever is fastest -- otherwise the "achieved bound" would understate the true
        # gap, and a plan.is_feasible()==False caller reading achieved_rel_error would be misled.
        worker = Worker(name="g", device="gpu", precisions=("float16", "float64"))
        plan = plan_heterogeneous(
            [worker], 100, allowed_precisions=("float16", "float64"), target_rel_error=1e-20, allow_infeasible=True
        )
        self.assertEqual(plan.assignments[0].precision, "float64")
        self.assertFalse(plan.assignments[0].meets_target)


class WorkerValidationTest(unittest.TestCase):
    """MXR-080-0134: a Worker must reject malformed capacities/capabilities at construction."""

    def test_unknown_device_rejected(self):
        with self.assertRaises(ValueError):
            Worker(name="w0", device="tpu", precisions=("float32",))

    def test_empty_precision_set_rejected(self):
        with self.assertRaises(ValueError):
            Worker(name="w0", device="cpu", precisions=())

    def test_unsupported_device_precision_pair_rejected(self):
        # bfloat16 is only modeled on gpu in _THROUGHPUT, not cpu.
        with self.assertRaises(ValueError):
            Worker(name="w0", device="cpu", precisions=("bfloat16",))

    def test_zero_throughput_rejected_before_plan_heterogeneous_can_divide_by_zero(self):
        # The finding's exact failure mode: multiple zero-throughput workers used to reach
        # plan_heterogeneous's row-split division (total_eff == 0) and raise an opaque ZeroDivisionError
        # there instead of a clear, intentional validation error. Now no zero-throughput worker can even
        # be constructed, so plan_heterogeneous never sees one.
        with self.assertRaises(ValueError):
            Worker(name="w0", device="cpu", precisions=("float32",), base_throughput=0.0)

    def test_negative_throughput_rejected(self):
        with self.assertRaises(ValueError):
            Worker(name="w0", device="cpu", precisions=("float32",), base_throughput=-1.0)

    def test_non_finite_throughput_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf"), True):
            with self.subTest(base_throughput=bad):
                with self.assertRaises(ValueError):
                    Worker(name="w0", device="cpu", precisions=("float32",), base_throughput=bad)

    def test_worker_copies_mutable_precision_sequence(self):
        precisions = ["float32"]
        worker = Worker(name="w0", device="cpu", precisions=precisions)
        precisions.append("bogus")
        self.assertEqual(worker.precisions, ("float32",))


class WorkerAssignmentValidationTest(unittest.TestCase):
    """MXR-080-0134: output-side sanity checks, independent of plan_heterogeneous's own input validation --
    a WorkerAssignment must reject an internally-inconsistent allocation regardless of how it is built."""

    def test_negative_rows_rejected(self):
        with self.assertRaises(ValueError):
            WorkerAssignment(name="w0", rows=-1, precision="float32", effective_throughput=1.0)

    def test_non_int_rows_rejected(self):
        with self.assertRaises(ValueError):
            WorkerAssignment(name="w0", rows=1.5, precision="float32", effective_throughput=1.0)

    def test_nonpositive_or_non_finite_effective_throughput_rejected(self):
        for bad in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.subTest(effective_throughput=bad):
                with self.assertRaises(ValueError):
                    WorkerAssignment(name="w0", rows=10, precision="float32", effective_throughput=bad)

    def test_forged_precision_boolean_or_error_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            WorkerAssignment(name="w", rows=1, precision="bogus", effective_throughput=1.0)
        with self.assertRaises(ValueError):
            WorkerAssignment(name="w", rows=1, precision="float32", effective_throughput=1.0, meets_target="false")
        with self.assertRaises(ValueError):
            WorkerAssignment(
                name="w",
                rows=1,
                precision="float32",
                effective_throughput=1.0,
                meets_target=False,
                achieved_rel_error=float("nan"),
                evidence_kind="heuristic",
            )
        with self.assertRaises(ValueError):
            WorkerAssignment(
                name="w",
                rows=1,
                precision="float32",
                effective_throughput=1.0,
                meets_target=True,
                achieved_rel_error=0.1,
                evidence_kind="heuristic",
            )

    def test_plan_record_requires_assignments_and_exact_reduction_depth(self):
        with self.assertRaises(ValueError):
            HeterogeneousPlan((), 0)
        assignment = WorkerAssignment("w", 1, "float32", 1.0)
        with self.assertRaises(ValueError):
            HeterogeneousPlan((assignment,), -1)
        with self.assertRaises(ValueError):
            HeterogeneousPlan((assignment,), 1)


class PlanInputValidationTest(unittest.TestCase):
    """MXR-080-0134: plan_heterogeneous itself must reject malformed rows/op_count/precisions/target."""

    def _worker(self):
        return Worker(name="w0", device="cpu", precisions=("float32", "float64"))

    def test_negative_rows_rejected(self):
        with self.assertRaises(ValueError):
            plan_heterogeneous([self._worker()], -100)

    def test_negative_op_count_rejected(self):
        with self.assertRaises(ValueError):
            plan_heterogeneous([self._worker()], 100, op_count=-5)

    def test_empty_allowed_precisions_rejected(self):
        with self.assertRaises(ValueError):
            plan_heterogeneous([self._worker()], 100, allowed_precisions=())

    def test_invalid_target_rel_error_rejected(self):
        for bad in (-1e-6, 0.0, float("nan"), float("inf"), True):
            with self.subTest(target_rel_error=bad):
                with self.assertRaises(ValueError):
                    plan_heterogeneous([self._worker()], 100, target_rel_error=bad)

    def test_allow_infeasible_requires_actual_boolean(self):
        with self.assertRaises(ValueError):
            plan_heterogeneous(
                [self._worker()], 100, target_rel_error=1e-12, allow_infeasible="false"
            )

    def test_million_operation_float32_score_uses_gamma_and_remains_uncertified(self):
        worker = Worker("w", "cpu", ("float32",))
        plan = plan_heterogeneous(
            [worker],
            1,
            target_rel_error=0.06,
            op_count=1_000_000,
            allow_infeasible=True,
        )
        estimate = plan.assignments[0].achieved_rel_error
        self.assertGreater(estimate, 0.06)
        self.assertFalse(plan.is_feasible())

    def test_well_posed_pool_still_plans_with_sensible_nonnegative_rows(self):
        # Negative control: ordinary, valid workers of differing throughput must still plan normally --
        # no negative or fractional row counts anywhere, and the total is exact.
        workers = [
            Worker(name="fast", device="cpu", precisions=("float32",), base_throughput=5.0),
            Worker(name="slow", device="cpu", precisions=("float32",), base_throughput=0.1),
        ]
        plan = plan_heterogeneous(workers, 777)
        self.assertEqual(plan.total_rows(), 777)
        for a in plan.assignments:
            self.assertIsInstance(a.rows, int)
            self.assertGreaterEqual(a.rows, 0)
        by_name = {a.name: a.rows for a in plan.assignments}
        self.assertGreater(by_name["fast"], by_name["slow"])


if __name__ == "__main__":
    unittest.main()
