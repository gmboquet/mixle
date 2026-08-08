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
        self.assertEqual(rep["coverage_contract"], "joint_structured")
        self.assertTrue(np.isfinite(rep["joint_qhat"]))
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
            with self.assertRaisesRegex(RuntimeError, "fresh base inputs"):
                sol.improve()
            self.assertGreater(len(sol.harvested_inputs), 0)  # unsafe reuse does not consume evidence

    def test_schema_is_exact_and_duplicate_inputs_keep_row_identity(self):
        from mixle.task import solve_structured

        rows = ["same input"] * 60

        def batched_teacher(batch):
            return [{"label": "even" if i % 2 == 0 else "odd"} for i in range(len(batch))]

        sol = solve_structured(
            batched_teacher,
            rows,
            schema={"label": "categorical"},
            alpha=0.1,
            seed=0,
            epochs=20,
        )
        self.assertEqual(sol.schema, {"label": "categorical"})

        broken = _tickets(40)
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            solve_structured(
                lambda batch: [_triage(row) if i else {"queue": "billing"} for i, row in enumerate(batch)],
                broken,
                schema={"queue": "categorical", "priority": "numeric"},
                tol={"priority": 2.0},
                epochs=10,
            )


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class StructuredPersistenceTest(unittest.TestCase):
    def test_save_load_serves_identically(self):
        import tempfile

        from mixle.task import StructuredSolution, solve_structured

        sol = solve_structured(_triage, _tickets(300), tol={"priority": 3.0}, alpha=0.1, seed=0, epochs=200)
        fresh = _tickets(60, seed=5)
        want = [sol.try_local(t) for t in fresh]
        with tempfile.TemporaryDirectory() as d:
            path = sol.save(d + "/triage")
            back = StructuredSolution.load(path, _triage)
            self.assertEqual(back.schema, sol.schema)
            got = [back.try_local(t) for t in fresh]
        for g, w in zip(got, want):
            if w is None:
                self.assertIsNone(g)
            else:
                self.assertEqual(g["queue"], w["queue"])
                self.assertAlmostEqual(g["priority"], w["priority"], places=4)
        # escalation on the loaded artifact still runs the PARENT teacher exactly
        hard = {"kind": "refund", "amount": 505.0, "region": "eu"}
        self.assertEqual(set(back(hard)), {"queue", "priority"})


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class StructuredResolveTest(unittest.TestCase):
    """prelabeled= fans harvested dicts down into every field's TRAINING split, teacher-free."""

    def test_prelabeled_trains_every_field_without_teacher_calls(self):
        from mixle.task import solve_structured

        calls = {"n": 0}

        def counting_teacher(t):
            if isinstance(t, list):  # the batched probe, not a real label
                raise TypeError("per-item teacher")
            calls["n"] += 1
            return _triage(t)

        base = _tickets(100, seed=0)
        harvested = _tickets(60, seed=11)
        pre_outs = [_triage(t) for t in harvested]
        broken = list(pre_outs)
        broken[0] = {"queue": broken[0]["queue"]}
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            solve_structured(
                counting_teacher,
                base,
                tol=1e6,
                alpha=0.1,
                prelabeled=(harvested, broken),
                seed=0,
                epochs=20,
            )
        calls["n"] = 0

        sol = solve_structured(
            counting_teacher, base, tol=1e6, alpha=0.1, prelabeled=(harvested, pre_outs), seed=0, epochs=150
        )
        self.assertEqual(calls["n"], len(base))  # prelabeled pairs came in free
        self.assertEqual(sol.schema, {"queue": "categorical", "priority": "numeric"})
        num = sol.fields_num["priority"]
        cat = sol.fields_cat["queue"]
        joint_count = sol.calibration_receipt["calibration_count"]
        # the structured level ALSO reserves a disjoint record-level evaluation slice for the
        # answered-slice measurement (STAT-RR16-2); those rows never reach any sub-model
        eval_count = sol.calibration_receipt["evaluation_count"]
        self.assertEqual(eval_count, sol.eval_rows)
        self.assertGreaterEqual(eval_count, 2)
        # BOTH sub-solution shapes reserve TWO holdout roles -- conformal calibration (cal_*) and
        # selection (sel_*) -- so the rows withheld from training are cal + sel, not cal alone
        # (MXR-080-1891 for solve(); STAT-RR11-1 gave solve_regression the same split).
        num_reserved = len(num.cal_inputs) + len(num.sel_inputs)
        self.assertEqual(len(num.train_inputs), len(base) - joint_count - eval_count - num_reserved + len(harvested))
        cat_reserved = len(cat.cal_inputs) + len(cat.sel_inputs)
        self.assertEqual(len(cat.train_inputs), len(base) - joint_count - eval_count - cat_reserved + len(harvested))
        for t in harvested:
            self.assertNotIn(repr(t), [repr(c) for c in num.cal_inputs])
            self.assertNotIn(repr(t), [repr(c) for c in cat.cal_inputs])
            self.assertNotIn(repr(t), [repr(c) for c in cat.sel_inputs])


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class StructuredAnsweredSliceTest(unittest.TestCase):
    def test_measurement_matches_the_real_joint_gate_and_round_trips(self):
        # STAT-RR16-2: the answered-slice numbers must be the REAL joint gate (try_local) run
        # over the disjoint record-level evaluation slice, with record-correct meaning
        # categoricals exact and numerics within their tolerance.
        import json
        import tempfile
        from pathlib import Path

        from mixle.task import StructuredSolution, solve_structured

        base = _tickets(200)
        sol = solve_structured(_triage, base, tol={"priority": 3.0}, alpha=0.1, seed=0, epochs=150)
        receipt = sol.calibration_receipt
        self.assertEqual(receipt["evaluation_count"], sol.eval_rows)
        self.assertGreaterEqual(sol.eval_rows, 2)
        self.assertTrue(set(receipt["evaluation_indices"]).isdisjoint(receipt["calibration_indices"]))
        self.assertLessEqual(sol.answered_eval_correct, sol.answered_eval_n)
        self.assertLessEqual(sol.answered_eval_n, sol.eval_rows)

        answered = correct = 0
        for i in receipt["evaluation_indices"]:
            truth = _triage(base[i])
            got = sol.try_local(base[i])
            if got is None:
                continue
            answered += 1
            ok_queue = str(got["queue"]) == str(truth["queue"])
            ok_priority = abs(float(got["priority"]) - float(truth["priority"])) <= 3.0
            correct += int(ok_queue and ok_priority)
        self.assertEqual(answered, sol.answered_eval_n)
        self.assertEqual(correct, sol.answered_eval_correct)

        block = sol.report()["answered_slice"]
        if sol.answered_eval_n == 0:
            self.assertIsNone(block)
        else:
            self.assertEqual(block["n_answered"], sol.answered_eval_n)
            self.assertEqual(block["n_evaluated"], sol.eval_rows)
            self.assertAlmostEqual(block["agreement"], sol.answered_eval_correct / sol.answered_eval_n, places=4)
            low, high = block["ci95"]
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)
            self.assertLessEqual(low, block["agreement"] + 1e-4)
            self.assertGreaterEqual(high, block["agreement"] - 1e-4)

        with tempfile.TemporaryDirectory() as d:
            path = sol.save(d + "/triage")
            back = StructuredSolution.load(path, _triage)
            self.assertEqual(back.eval_rows, sol.eval_rows)
            self.assertEqual(back.answered_eval_n, sol.answered_eval_n)
            self.assertEqual(back.answered_eval_correct, sol.answered_eval_correct)
            # an artifact WITHOUT a measurement member is refused, never defaulted to
            # "measured nothing" (the STAT-RR14-1 mechanism)
            sj = Path(path) / "structured.json"
            doc = json.loads(sj.read_text())
            del doc["answered_eval_n"]
            sj.write_text(json.dumps(doc))
            with self.assertRaises(KeyError):
                StructuredSolution.load(path, _triage)


if __name__ == "__main__":
    unittest.main()
