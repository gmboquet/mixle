"""solve_structured: dict-valued routines decomposed per field onto the calibrated shapes."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _triage(t):
    """The rigid enricher: a ticket gets a queue AND a priority score."""
    queue = (
        "finance"
        if (t["kind"] == "refund" and t["amount"] > 500)
        else ("billing" if t["kind"] in ("refund", "billing") else "support")
    )
    priority = 10.0 + 0.02 * t["amount"] + (25.0 if t["region"] == "eu" else 0.0)
    return {"queue": queue, "priority": priority}


def _tickets(n, seed=0):
    rng = np.random.RandomState(seed)
    kinds = ["refund", "billing", "question"]
    return [
        {
            "kind": kinds[rng.randint(0, 3)],
            "amount": float(rng.uniform(0, 1000)),
            "region": ["us", "eu"][rng.randint(0, 2)],
        }
        for _ in range(n)
    ]


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class SolveStructuredTest(unittest.TestCase):
    def test_all_fields_or_escalate(self):
        from mixle.task import solve_structured

        sol = solve_structured(_triage, _tickets(500), tol={"priority": 3.0}, alpha=0.1, seed=0, epochs=300)
        self.assertEqual(sol.schema, {"queue": "categorical", "priority": "numeric"})

        fresh = _tickets(200, seed=9)
        local = wrong_queue = 0
        for t in fresh:
            got = sol(t)
            want = _triage(t)
            self.assertEqual(set(got), {"queue", "priority"})  # the schema always comes back whole
            if sol.try_local(t) is not None:
                local += 1
                wrong_queue += int(got["queue"] != want["queue"])
                self.assertLess(abs(got["priority"] - want["priority"]), 3.0 * 2)  # within the qhat<=tol regime
            else:
                self.assertEqual(got, want)  # escalations are the teacher's exact dict
        self.assertGreater(local, 50)  # the students carry real traffic
        self.assertLess(wrong_queue / max(local, 1), 0.2)

        rep = sol.report()
        self.assertEqual(rep["requests"], 200)  # only __call__ counts; try_local probes are free
        self.assertEqual(rep["harvested"], rep["escalated"])

    def test_numeric_field_requires_tol(self):
        from mixle.task import solve_structured

        with self.assertRaises(ValueError):
            solve_structured(_triage, _tickets(50), seed=0, epochs=20)

    def test_improve_pushes_harvest_into_every_field(self):
        from mixle.task import solve_structured

        sol = solve_structured(_triage, _tickets(200), tol={"priority": 2.0}, alpha=0.15, seed=0, epochs=150)
        for t in _tickets(200, seed=3):
            sol(t)
        if sol.harvested_inputs:
            sol.improve()
            self.assertEqual(len(sol.harvested_inputs), 0)  # harvest consumed by the field buffers


if __name__ == "__main__":
    unittest.main()
