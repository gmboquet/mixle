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

    def test_improve_promotes_only_tighter_calibration(self):
        from mixle.task import solve_regression

        sol = solve_regression(_price, _items(150), tol=0.01, alpha=0.1, seed=0, epochs=150)
        q0 = sol.qhat
        for it in _items(300, seed=3):  # harvest real labels while escalating
            sol(it)
        sol.improve()
        self.assertLessEqual(sol.qhat, q0 + 1e-12)  # anti-regression on the calibrated width


if __name__ == "__main__":
    unittest.main()
