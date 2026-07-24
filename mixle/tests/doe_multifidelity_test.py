"""Cost-aware multi-fidelity Bayesian optimization (mixle.doe.multifidelity)."""

import importlib.util
import unittest
import warnings

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
class MultiFidelityTest(unittest.TestCase):
    def test_uses_both_fidelities_and_finds_target_optimum(self):
        from mixle.doe import multi_fidelity_minimize

        opt = np.array([0.3, -0.4, 0.1])

        def obj(x, s):
            base = float(np.sum((x - opt) ** 2))
            return base if s == 1.0 else base + 0.05 * np.sin(8 * x[0])  # cheap, slightly biased low fidelity

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = multi_fidelity_minimize(
                obj,
                [(-1.0, 1.0)] * 3,
                fidelities=(0.5, 1.0),
                costs=(1.0, 5.0),
                n_init=5,
                max_cost=80.0,
                n_candidates=200,
                seed=0,
            )
        n_low = int(np.sum(res["X"][:, -1] == 0.5))
        n_high = int(np.sum(res["X"][:, -1] == 1.0))
        self.assertGreater(n_low, 5)  # spent cheap evaluations exploring
        self.assertGreater(n_high, 5)  # and expensive ones refining
        self.assertLess(np.linalg.norm(res["x"] - opt), 0.25)  # reached the target optimum
        # MXR-080-0182: cost must never overshoot max_cost now (was <= 90.0, a 10-unit overshoot
        # allowance for the old "spend, then discover the overshoot" bug).
        self.assertLessEqual(res["cost"], 80.0)
        self.assertTrue(res["target_evaluated"])  # a genuine target-fidelity result, not a fallback


# --------------------------------------------------------------------------- MXR-080-0181
# `target` need not have been a member of `fidelities`. Such a target was never evaluated (the
# fidelity-selection loop only ever picks from `fidelities`), and the final fallback silently returned
# the best observation at ANY fidelity, mislabeled as the target-fidelity result: with fidelities
# (0.2, 0.5) and target=1, the function returned the 0.2-fidelity value as if it answered the
# target(=1.0)-fidelity question. `target` must now be validated as a member of `fidelities` up front.
class TargetFidelityValidationTest(unittest.TestCase):
    def test_target_outside_fidelities_is_rejected(self):
        from mixle.doe import multi_fidelity_minimize

        def obj(x, s):
            return float(np.sum(x))

        with self.assertRaisesRegex(ValueError, "target fidelity"):
            multi_fidelity_minimize(
                obj, [(0.0, 1.0)], fidelities=(0.2, 0.5), target=1, n_init=1, max_cost=3.0, n_candidates=8, seed=0
            )

    @unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
    def test_target_in_fidelities_still_works_and_is_genuinely_evaluated(self):
        from mixle.doe import multi_fidelity_minimize

        opt = np.array([0.3])

        def obj(x, s):
            base = float(np.sum((x - opt) ** 2))
            return base if s == 1.0 else base + 0.05

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = multi_fidelity_minimize(
                obj,
                [(-1.0, 1.0)],
                fidelities=(0.5, 1.0),
                target=1.0,
                n_init=3,
                max_cost=12.0,
                n_candidates=32,
                seed=0,
            )
        self.assertTrue(res["target_evaluated"])
        self.assertIsNotNone(res["x"])
        self.assertIsNotNone(res["y"])
        self.assertIn(1.0, res["X"][:, -1].tolist())  # the target fidelity was genuinely queried


# A budget that never affords the target fidelity at all (not even during initialization) exercises the
# honest side of MXR-080-0181's fix once combined with MXR-080-0182's budget enforcement below: `target`
# being a valid member of `fidelities` does not guarantee it was ever reachable under the budget. `x`/`y`
# must come back `None` (with `target_evaluated=False`) rather than silently standing in a lower-fidelity
# observation for the target-fidelity answer.
class TargetNeverAffordableTest(unittest.TestCase):
    def test_budget_exhausted_before_target_ever_affordable_reports_honestly(self):
        from mixle.doe import multi_fidelity_minimize

        def obj(x, s):
            return float(np.sum(x))

        # cheapest fidelity (0.1) affords exactly one init point; the target (5.0, the default
        # max(fidelities)) costs more than the entire budget and is never affordable even once -- init
        # leaves spent == max_cost exactly, so the main loop never even starts.
        res = multi_fidelity_minimize(
            obj, [(0.0, 1.0)], fidelities=(0.1, 5.0), n_init=1, max_cost=0.1, n_candidates=8, seed=0
        )
        self.assertFalse(res["target_evaluated"])
        self.assertIsNone(res["x"])
        self.assertIsNone(res["y"])
        self.assertNotIn(5.0, res["X"][:, -1].tolist())


# --------------------------------------------------------------------------- MXR-080-0182
# Initial seeding ignored `max_cost` entirely (a max_cost=0 run still spent during initialization) and
# each later sequential evaluation could overshoot the remaining budget (spend first, discover the
# overshoot after the fact). Fidelity/cost configuration -- empty, duplicate, non-finite fidelities, and
# a cost schedule whose length doesn't match `fidelities` -- was also unchecked. Every evaluation must
# now reserve its cost against the remaining budget before calling `objective`.
class BudgetEnforcementTest(unittest.TestCase):
    def _obj(self, x, s):
        return float(np.sum(x))

    def test_max_cost_zero_is_rejected_not_silently_overspent(self):
        from mixle.doe import multi_fidelity_minimize

        with self.assertRaisesRegex(ValueError, "cannot afford a single evaluation"):
            multi_fidelity_minimize(
                self._obj, [(0.0, 1.0)], fidelities=(0.5, 1.0), n_init=1, max_cost=0.0, n_candidates=8, seed=0
            )

    @unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
    def test_sequential_evaluation_never_overshoots_max_cost(self):
        from mixle.doe import multi_fidelity_minimize

        # init totals exactly 1.5 (one 0.5-cost point + one 1.0-cost point); 0.1 remains after that --
        # less than either fidelity's own cost, so the pre-fix code's next pick always overshot by
        # however much that fidelity cost.
        res = multi_fidelity_minimize(
            self._obj,
            [(0.0, 1.0)],
            fidelities=(0.5, 1.0),
            costs=(0.5, 1.0),
            n_init=1,
            max_cost=1.6,
            n_candidates=8,
            seed=0,
        )
        self.assertLessEqual(res["cost"], 1.6)

    def test_empty_fidelities_rejected(self):
        from mixle.doe import multi_fidelity_minimize

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            multi_fidelity_minimize(self._obj, [(0.0, 1.0)], fidelities=(), n_init=1, max_cost=5.0)

    def test_duplicate_fidelities_rejected(self):
        from mixle.doe import multi_fidelity_minimize

        with self.assertRaisesRegex(ValueError, "duplicates"):
            multi_fidelity_minimize(self._obj, [(0.0, 1.0)], fidelities=(0.5, 0.5, 1.0), n_init=1, max_cost=5.0)

    def test_non_finite_fidelities_rejected(self):
        from mixle.doe import multi_fidelity_minimize

        with self.assertRaisesRegex(ValueError, "finite"):
            multi_fidelity_minimize(
                self._obj, [(0.0, 1.0)], fidelities=(0.5, float("nan"), 1.0), n_init=1, max_cost=5.0
            )

    def test_cost_length_mismatch_rejected(self):
        from mixle.doe import multi_fidelity_minimize

        with self.assertRaisesRegex(ValueError, "one entry per fidelity"):
            multi_fidelity_minimize(
                self._obj, [(0.0, 1.0)], fidelities=(0.5, 1.0), costs=(1.0, 2.0, 3.0), n_init=1, max_cost=5.0
            )


if __name__ == "__main__":
    unittest.main()
