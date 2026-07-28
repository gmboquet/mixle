"""The pool plane (H1): round-trip artifact, budget + confirm rails, telemetry integration."""

import unittest

from mixle.pool import LocalBackend, PoolJob, PoolResult, submit
from mixle.telemetry import Telemetry


class RoundTripTest(unittest.TestCase):
    def test_local_backend_runs_and_returns_the_artifact(self):
        res = submit(PoolJob(run=lambda: {"weights": [1, 2, 3]}, kind="block"), LocalBackend())
        self.assertIsInstance(res, PoolResult)
        self.assertTrue(res.ok)
        self.assertEqual(res.artifact, {"weights": [1, 2, 3]})

    def test_default_backend_is_local(self):
        res = submit(PoolJob(run=lambda: 42))
        self.assertTrue(res.ok)
        self.assertEqual(res.artifact, 42)

    def test_a_failing_job_is_a_result_not_a_crash(self):
        def boom():
            raise ValueError("kaboom")

        res = submit(PoolJob(run=boom))
        self.assertEqual(res.status, "error")
        self.assertIn("kaboom", res.reason)

    def test_duration_is_measured(self):
        clock = iter([10.0, 12.5])
        res = submit(PoolJob(run=lambda: 1), LocalBackend(clock=lambda: next(clock)))
        self.assertEqual(res.duration_s, 2.5)


class BudgetRailTest(unittest.TestCase):
    def test_over_budget_job_is_rejected_before_running(self):
        ran = {"v": False}

        def work():
            ran["v"] = True
            return 1

        res = submit(PoolJob(run=work, est_cost=5.0, budget=1.0))
        self.assertEqual(res.status, "rejected")
        self.assertFalse(ran["v"])  # never executed
        self.assertIn("exceeds budget", res.reason)

    def test_within_budget_runs(self):
        res = submit(PoolJob(run=lambda: 1, est_cost=0.5, budget=1.0))
        self.assertTrue(res.ok)

    def test_negative_est_cost_is_rejected(self):
        # A negative cost would "recover" budget that was never actually spent.
        ran = {"v": False}

        def work():
            ran["v"] = True
            return 1

        with self.assertRaises(ValueError):
            submit(PoolJob(run=work, est_cost=-1_000_000.0, budget=1.0))
        self.assertFalse(ran["v"])  # never executed

    def test_nan_est_cost_is_rejected(self):
        # NaN compares False against everything, so `est_cost > budget` would silently pass.
        with self.assertRaises(ValueError):
            submit(PoolJob(run=lambda: 1, est_cost=float("nan"), budget=0.0))

    def test_infinite_est_cost_is_rejected(self):
        # An infinite cost against the default infinite budget would otherwise slip through.
        with self.assertRaises(ValueError):
            submit(PoolJob(run=lambda: 1, est_cost=float("inf")))

    def test_negative_infinite_est_cost_is_rejected(self):
        with self.assertRaises(ValueError):
            submit(PoolJob(run=lambda: 1, est_cost=float("-inf"), budget=0.0))

    def test_nan_budget_is_rejected(self):
        # A NaN budget breaks the comparison from the other side: any cost passes it.
        with self.assertRaises(ValueError):
            submit(PoolJob(run=lambda: 1, est_cost=1_000_000.0, budget=float("nan")))

    def test_negative_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            submit(PoolJob(run=lambda: 1, est_cost=0.0, budget=-5.0))

    def test_infinite_budget_is_a_legitimate_no_ceiling(self):
        res = submit(PoolJob(run=lambda: 1, est_cost=1_000_000.0, budget=float("inf")))
        self.assertTrue(res.ok)


class ConfirmRailTest(unittest.TestCase):
    class _FakeGPU:
        billable = True

        def __init__(self):
            self.ran = False

        def submit(self, job):
            self.ran = True
            return PoolResult(job.id, "done", artifact=job.run())

    def test_billable_backend_requires_confirm(self):
        gpu = self._FakeGPU()
        res = submit(PoolJob(run=lambda: 1, est_cost=0.5, budget=10.0), gpu, confirm=False)
        self.assertEqual(res.status, "rejected")
        self.assertFalse(gpu.ran)  # spend never incurred implicitly

    def test_billable_backend_runs_with_confirm(self):
        gpu = self._FakeGPU()
        res = submit(PoolJob(run=lambda: 7, est_cost=0.5, budget=10.0), gpu, confirm=True)
        self.assertTrue(res.ok)
        self.assertTrue(gpu.ran)

    def test_local_backend_needs_no_confirm(self):
        self.assertTrue(submit(PoolJob(run=lambda: 1)).ok)

    def test_truthy_non_boolean_confirmation_does_not_authorize_spend(self):
        # MXR-080-1738: the gate used ordinary truthiness, so the string "false" -- an entirely
        # ordinary way to spell "no" across JSON/CLI/form boundaries -- confirmed billable execution.
        for value in ("false", "0", "no", 1, [0], object()):
            with self.subTest(confirm=value):
                gpu = self._FakeGPU()
                res = submit(PoolJob(run=lambda: 1, est_cost=0.5, budget=10.0), gpu, confirm=value)
                self.assertEqual(res.status, "rejected")
                self.assertFalse(gpu.ran)


class BackendSettlementTest(unittest.TestCase):
    """A backend's response is third-party data: submit() settles it before the caller trusts it."""

    class _Backend:
        billable = False

        def __init__(self, result):
            self._result = result

        def submit(self, job):
            return self._result(job) if callable(self._result) else self._result

    def test_realized_cost_over_budget_is_not_a_successful_result(self):
        # MXR-080-1739: only est_cost was ever compared with budget, so a backend could bill any
        # amount it liked for an under-estimated job and the overrun was accepted unchanged.
        job = PoolJob(run=lambda: 1, est_cost=0.5, budget=1.0)
        res = submit(job, self._Backend(lambda j: PoolResult(j.id, "done", artifact=1, cost=999.0)))
        self.assertFalse(res.ok)
        self.assertEqual(res.status, "error")
        self.assertIn("exceeds budget", res.reason)
        self.assertEqual(res.cost, 999.0)  # the spend really happened; it stays reconcilable

    def test_realized_cost_exactly_at_budget_settles(self):
        job = PoolJob(run=lambda: 1, est_cost=0.5, budget=1.0)
        res = submit(job, self._Backend(lambda j: PoolResult(j.id, "done", artifact=1, cost=1.0)))
        self.assertTrue(res.ok)

    def test_result_for_a_different_job_is_rejected(self):
        # MXR-080-1740: submit() returned whatever the backend supplied, including a result whose
        # job_id named an entirely different job.
        job = PoolJob(run=lambda: 1, est_cost=0.5, budget=1.0, id="real-job")
        res = submit(job, self._Backend(PoolResult("wrong-job", "done", artifact="x")))
        self.assertFalse(res.ok)
        self.assertEqual(res.job_id, "real-job")
        self.assertIn("wrong-job", res.reason)

    def test_unknown_status_is_rejected(self):
        job = PoolJob(run=lambda: 1)
        res = submit(job, self._Backend(lambda j: PoolResult(j.id, "finished", artifact="x")))
        self.assertEqual(res.status, "error")
        self.assertIn("unknown status", res.reason)

    def test_non_finite_duration_is_rejected(self):
        job = PoolJob(run=lambda: 1)
        res = submit(job, self._Backend(lambda j: PoolResult(j.id, "done", duration_s=float("nan"))))
        self.assertEqual(res.status, "error")
        self.assertIn("duration_s", res.reason)

    def test_negative_cost_is_rejected(self):
        # A negative realized cost is a credit for work that was never refunded.
        job = PoolJob(run=lambda: 1)
        res = submit(job, self._Backend(lambda j: PoolResult(j.id, "done", cost=-5.0)))
        self.assertEqual(res.status, "error")
        self.assertIn("cost", res.reason)

    def test_non_poolresult_response_is_rejected(self):
        res = submit(PoolJob(run=lambda: 1), self._Backend({"status": "done"}))
        self.assertEqual(res.status, "error")
        self.assertIn("not a PoolResult", res.reason)

    def test_settlement_failure_is_still_telemetered(self):
        tel = Telemetry()
        job = PoolJob(run=lambda: 1, est_cost=0.5, budget=1.0)
        submit(job, self._Backend(lambda j: PoolResult(j.id, "done", cost=999.0)), telemetry=tel)
        events = list(tel.events(kind="pool_job"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].choice, "error")
        self.assertEqual(events[0].outcome["cost"], 999.0)


class TelemetryTest(unittest.TestCase):
    def test_every_submission_emits_a_pool_job_event(self):
        tel = Telemetry()
        submit(PoolJob(run=lambda: 1, kind="verb", reason="demo"), telemetry=tel)
        submit(PoolJob(run=lambda: 1, est_cost=9.0, budget=1.0), telemetry=tel)  # rejected
        events = list(tel.events(kind="pool_job"))
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].choice, "done")
        self.assertEqual(events[1].choice, "rejected")
        self.assertEqual(events[0].features["kind"], "verb")


if __name__ == "__main__":
    unittest.main()
