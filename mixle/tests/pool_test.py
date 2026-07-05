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
