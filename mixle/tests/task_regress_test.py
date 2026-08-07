"""solve_regression: coverage-guaranteed numeric replacement with a precision escalation rule."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _price(item):
    """The rigid pricing routine: base by kind + smooth size effect."""
    base = {"basic": 20.0, "pro": 80.0, "max": 150.0}[item["kind"]]
    return base + 0.5 * item["size"] + 0.001 * item["size"] ** 2


def _items(n, seed=0):
    rng = np.random.RandomState(seed)
    kinds = ["basic", "pro", "max"]
    return [{"kind": kinds[rng.randint(0, 3)], "size": float(rng.uniform(0, 100))} for _ in range(n)]


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class SolveRegressionTest(unittest.TestCase):
    def test_coverage_holds_and_precision_gate_answers(self):
        from mixle.task import solve_regression

        sol = solve_regression(_price, _items(500), tol=20.0, alpha=0.1, seed=0, epochs=400)
        self.assertTrue(sol.answers_locally)  # the calibrated width met the tolerance

        # finite-sample conformal coverage on FRESH exchangeable inputs
        fresh = _items(400, seed=7)
        covered = 0
        for it in fresh:
            yhat, lo, hi = sol.interval(it)
            covered += int(lo <= _price(it) <= hi)
        self.assertGreater(covered / len(fresh), 0.85)  # >= 1 - alpha, minus finite-sample slack

        # locally answered values honor the calibrated precision in aggregate (the err q90 sits AT the
        # calibrated quantile by construction; allow interpolation slack)
        errs = [abs(sol(it) - _price(it)) for it in fresh]
        self.assertLess(np.quantile(errs, 0.9), sol.qhat * 1.1 + 1e-9)
        self.assertLess(sol.qhat, 1.0)  # and the width is genuinely tight on this smooth task

    def test_impossible_tolerance_escalates_everything(self):
        from mixle.task import solve_regression

        sol = solve_regression(_price, _items(200), tol=0.01, alpha=0.1, seed=0, epochs=100)
        self.assertFalse(sol.answers_locally)  # honest failure: the student can't hit +/-0.01
        it = {"kind": "pro", "size": 42.0}
        self.assertAlmostEqual(sol(it), _price(it), places=9)  # every request runs the real code
        self.assertEqual(sol.report()["escalated"], 1)

    def test_invalid_data_and_calibration_parameters_fail_closed(self):
        from mixle.task import solve_regression

        data = _items(40)
        with self.assertRaisesRegex(ValueError, "alpha"):
            solve_regression(_price, data, tol=1.0, alpha=float("nan"))
        with self.assertRaisesRegex(ValueError, "tol"):
            solve_regression(_price, data, tol=-1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            solve_regression(lambda xs: [float("nan")] * len(xs), data, tol=1.0)
        with self.assertRaisesRegex(ValueError, "requires at least"):
            solve_regression(_price, data, tol=1.0, alpha=0.01, holdout=0.25)

    def test_improve_promotes_on_selection_rows_and_calibrates_on_fresh_harvest(self):
        from mixle.task import solve_regression

        sol = solve_regression(_price, _items(150), tol=0.01, alpha=0.1, seed=0, epochs=150)
        mae0 = sol.holdout_mae
        solve_cal = [repr(c) for c in sol.cal_inputs]
        harvest = _items(300, seed=3)
        for it in harvest:  # harvest real labels while escalating
            sol(it)
        reserved = [repr(c) for c in sol.harvested_inputs[-max(9, len(sol.harvested_inputs) // 4) :]]
        promoted = sol.improve()
        # every promotion decision reads the selection rows (STAT-RR11-1: promoting on the
        # calibrated width took a running minimum of qhat over one fixed slice)
        self.assertEqual(sol.selection_uses, 1)
        self.assertTrue(np.isfinite(sol.qhat))
        if promoted:
            # the promotion metric is held-out selection error, not the calibrated width
            self.assertLessEqual(sol.holdout_mae, mae0 + 1e-12)
            # STAT-RR12-1: the certifying rows POSTDATE the harvest -- they are the reserved
            # harvest tail, never the solve-time calibration rows (which gated the harvest and
            # therefore helped construct the candidate), and never candidate training rows
            self.assertEqual([repr(c) for c in sol.cal_inputs], reserved)
            self.assertEqual(sol.calibration_evidence, "fresh-harvest")
            train_reprs = {repr(c) for c in sol.train_inputs}
            self.assertEqual({repr(c) for c in sol.cal_inputs} & train_reprs, set())
        else:
            self.assertEqual([repr(c) for c in sol.cal_inputs], solve_cal)
            self.assertEqual(sol.calibration_evidence, "solve-split")
        # the two roles never overlap
        overlap = {repr(c) for c in sol.cal_inputs} & {repr(c) for c in sol.sel_inputs}
        self.assertEqual(overlap, set())

    def test_improve_waits_until_the_harvest_funds_a_fresh_calibration_slice(self):
        from mixle.task import solve_regression

        sol = solve_regression(_price, _items(150), tol=0.01, alpha=0.1, seed=0, epochs=150)
        for it in _items(9, seed=13):  # nine harvested pairs cannot fund 9 calibration + 1 training
            sol(it)
        self.assertFalse(sol.improve())
        self.assertEqual(sol.calibration_evidence, "solve-split")
        self.assertEqual(len(sol.harvested_inputs), 9)  # the harvest keeps accumulating

    def test_improve_accepts_fresh_evidence_inputs_for_calibration(self):
        from mixle.task import solve_regression

        sol = solve_regression(_price, _items(150), tol=0.01, alpha=0.1, seed=0, epochs=150)
        for it in _items(120, seed=3):
            sol(it)
        evidence = _items(30, seed=21)
        promoted = sol.improve(evidence_inputs=evidence)
        if promoted:
            self.assertEqual(sol.calibration_evidence, "fresh-evidence")
            self.assertEqual([repr(c) for c in sol.cal_inputs], [repr(c) for c in evidence])

    def test_selection_coverage_is_measured_on_rows_the_quantile_never_touches(self):
        from mixle.task import solve_regression

        sol = solve_regression(_price, _items(200), tol=20.0, alpha=0.1, seed=0, epochs=200)
        rep = sol.report()
        # a fresh (single-use) measurement of the deployed interval, honestly labeled
        self.assertEqual(rep["selection_uses"], 0)
        self.assertEqual(rep["calibration_evidence"], "solve-split")
        self.assertIsNotNone(rep["selection_coverage"])
        self.assertGreaterEqual(rep["selection_coverage"], 0.75)  # near 1 - alpha, minus slack
        # STAT-RR12-2: the proportion travels with its denominator and an exact interval
        self.assertEqual(rep["selection_coverage_n"], len(sol.sel_inputs))
        low, high = rep["selection_coverage_ci95"]
        self.assertLessEqual(0.0, low)
        self.assertLessEqual(low, rep["selection_coverage"])
        self.assertLessEqual(rep["selection_coverage"], high)
        self.assertLessEqual(high, 1.0)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class RegressionPersistenceTest(unittest.TestCase):
    def test_save_load_serves_identically(self):
        import tempfile

        from mixle.task import RegressionSolution, solve_regression

        sol = solve_regression(_price, _items(300), tol=20.0, alpha=0.1, seed=0, epochs=200)
        fresh = _items(60, seed=5)
        want = [sol.interval(it) for it in fresh]
        with tempfile.TemporaryDirectory() as d:
            path = sol.save(d + "/pricer")
            back = RegressionSolution.load(path, _price)
            got = [back.interval(it) for it in fresh]
        for a, b in zip(got, want):
            self.assertAlmostEqual(a[0], b[0], places=5)
            self.assertAlmostEqual(a[1], b[1], places=5)
        self.assertEqual(back.answers_locally, sol.answers_locally)
        back.harvested_inputs.append(fresh[0])
        back.harvested_ys.append(1.0)
        with self.assertRaises(RuntimeError):
            back.improve()  # loaded artifacts serve; improving needs the original data

    def test_save_load_round_trips_the_honesty_ledgers(self):
        # STAT-RR13-2: selection_uses reset to 0 on load, presenting spent selection evidence
        # as fresh; the calibration regime must survive too
        import tempfile

        from mixle.task import RegressionSolution, solve_regression

        sol = solve_regression(_price, _items(150), tol=20.0, alpha=0.1, seed=0, epochs=100)
        sol.selection_uses = 4
        sol.calibration_evidence = "fresh-harvest"
        with tempfile.TemporaryDirectory() as d:
            back = RegressionSolution.load(sol.save(d + "/pricer"), _price)
        self.assertEqual(back.calibration_evidence, "fresh-harvest")
        self.assertEqual(back.selection_uses, 4)
        self.assertEqual(back.report()["selection_uses"], 4)
        self.assertEqual(back.report()["calibration_evidence"], "fresh-harvest")


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class RegressionResolveTest(unittest.TestCase):
    """prelabeled= closes the serving loop: harvested pairs retrain WITHOUT re-calling the teacher."""

    def test_prelabeled_trains_only_and_calibration_stays_fresh(self):
        from mixle.task import solve_regression

        calls = {"n": 0}

        def counting_teacher(item):
            if isinstance(item, list):  # the batched probe, not a real label
                raise TypeError("per-item teacher")
            calls["n"] += 1
            return _price(item)

        base_items = _items(100, seed=0)
        harvested = _items(60, seed=11)
        pre = (harvested, [_price(it) for it in harvested])

        sol = solve_regression(counting_teacher, base_items, tol=1e6, alpha=0.1, prelabeled=pre, seed=0, epochs=150)
        # the teacher labeled ONLY the base inputs; prelabeled pairs came in free
        self.assertEqual(calls["n"], len(base_items))
        # prelabeled landed in training, never calibration or selection
        held_out = len(sol.cal_inputs) + len(sol.sel_inputs)
        self.assertEqual(len(sol.train_inputs), len(base_items) - held_out + len(harvested))
        self.assertLessEqual(held_out, len(base_items))
        for it in harvested:
            self.assertNotIn(repr(it), [repr(c) for c in sol.cal_inputs])
            self.assertNotIn(repr(it), [repr(c) for c in sol.sel_inputs])

    def test_prelabeled_data_does_not_degrade_the_fit(self):
        from mixle.task import solve_regression

        small = _items(80, seed=0)  # holdout 20 -> 10 calibration + 10 selection at alpha=0.1
        extra = _items(400, seed=5)
        pre = (extra, [_price(it) for it in extra])
        lone = solve_regression(_price, small, tol=1e6, alpha=0.1, seed=0, epochs=200)
        fed = solve_regression(_price, small, tol=1e6, alpha=0.1, prelabeled=pre, seed=0, epochs=200)
        # the 10-point selection split makes MAE noisy; pin non-degradation, not strict tightening
        self.assertLess(fed.holdout_mae, lone.holdout_mae * 1.5)
        self.assertTrue(np.isfinite(fed.qhat))


if __name__ == "__main__":
    unittest.main()
