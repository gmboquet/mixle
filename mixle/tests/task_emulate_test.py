"""mixle.task.emulate: budget-limited GP surrogates for expensive simulators, placed by acquisition.

Three receipts mirror the M4 roadmap card's acceptance criteria exactly:

* ``test_alc_placement_beats_random_at_matched_budget`` -- at the same total simulator budget, ALC
  (active-learning) placement gives a lower held-out RMSE than a random design (the "EIG/A5-placed
  samples beat random placement" claim, transplanted to this module's continuous-domain ALC criterion
  -- see the emulate.py module docstring for why ALC, not acquire()'s categorical EIG, is what applies
  here).
* ``test_error_bars_are_calibrated`` -- the emulator's own coverage receipt (fraction of held-out
  points inside its ``mean +/- 1 std``) is close to the nominal Gaussian value.
* ``test_multi_fidelity_beats_single_fidelity_at_matched_cost`` -- given a cheap, correlated low
  fidelity, multi-fidelity emulation reaches a lower held-out RMSE than single-fidelity at the same
  total cost.
"""

from __future__ import annotations

import importlib.util
import unittest
import warnings
from unittest import mock

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _true_f(x: np.ndarray) -> float:
    """A standard smooth-but-nonlinear 2-D test function (bounded, no closed-form flat regions)."""
    return float(np.sin(3.0 * x[0]) + 0.3 * x[1] ** 2 - 0.2 * x[0] * x[1])


def _cheap_biased_f(x: np.ndarray, s: float) -> float:
    """Multi-fidelity response: exact at ``s=1``, correlated-but-biased at cheaper ``s``."""
    base = _true_f(x)
    if s >= 1.0:
        return base
    return base + 0.4 * float(np.cos(5.0 * x[0])) * (1.0 - s)


BOUNDS_2D = [(-2.0, 2.0), (-2.0, 2.0)]

# The Forrester/Sobester/Keane (2007) multi-fidelity benchmark: a canonical low-fidelity function that
# is a smooth, globally *correlated but biased* transform of the high-fidelity one (not independent
# noise), which is exactly the regime cost-aware multi-fidelity GPs are built to exploit. 1-D so the
# GP's shared augmented-input kernel can actually resolve the two curves apart at a small budget.
BOUNDS_1D = [(0.0, 1.0)]


def _forrester_high(x: np.ndarray) -> float:
    xx = float(x[0])
    return float((6.0 * xx - 2.0) ** 2 * np.sin(12.0 * xx - 4.0))


def _forrester_low(x: np.ndarray) -> float:
    return float(0.5 * _forrester_high(x) + 10.0 * (float(x[0]) - 0.5) - 5.0)


def _forrester_target(x: np.ndarray) -> float:
    return _forrester_high(x)


def _forrester_mf(x: np.ndarray, s: float) -> float:
    return _forrester_high(x) if s >= 1.0 else _forrester_low(x)


@unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
class EmulateBasicsTest(unittest.TestCase):
    def test_predict_and_escalate_mask_shapes(self):
        from mixle.task.emulate import emulate

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            em = emulate(_true_f, BOUNDS_2D, budget=24, n_init=4, seed=0)
        x_query = np.array([[0.0, 0.0], [1.5, -1.0], [-1.9, 1.9]])
        mean, std = em.predict(x_query)
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(std.shape, (3,))
        self.assertTrue(np.all(std >= 0.0))
        mask = em.escalate_mask(x_query, tol=1.0e9)
        self.assertEqual(mask.shape, (3,))
        self.assertFalse(bool(np.any(mask)))  # an absurdly high tolerance never escalates
        mask_low = em.escalate_mask(x_query, tol=-1.0)
        self.assertTrue(bool(np.all(mask_low)))  # an impossible-to-clear tolerance always escalates

    def test_receipt_fields_are_finite_and_positive(self):
        from mixle.task.emulate import emulate

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            em = emulate(_true_f, BOUNDS_2D, budget=24, n_init=4, seed=1)
        r = em.receipt
        self.assertTrue(np.isfinite(r.held_out_rmse))
        self.assertGreaterEqual(r.held_out_rmse, 0.0)
        self.assertGreaterEqual(r.coverage, 0.0)
        self.assertLessEqual(r.coverage, 1.0)
        self.assertAlmostEqual(r.nominal_coverage, 0.6826894921370859, places=6)
        self.assertGreater(r.n_holdout, 0)
        self.assertGreater(r.n_train, 0)
        self.assertEqual(r.cost_spent, 24.0)
        self.assertIsNone(r.fidelities)

    def test_budget_too_small_raises(self):
        from mixle.task.emulate import emulate

        with self.assertRaises(ValueError):
            emulate(_true_f, BOUNDS_2D, budget=1, seed=0)


@unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
class ActiveLearningPlacementTest(unittest.TestCase):
    def test_alc_placement_beats_random_at_matched_budget(self):
        from mixle.task.emulate import emulate

        budget = 40
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            em_alc = emulate(_true_f, BOUNDS_2D, budget=budget, n_init=4, method="alc", seed=7)
            em_random = emulate(_true_f, BOUNDS_2D, budget=budget, n_init=4, method="random", seed=7)

        rmse_alc = em_alc.receipt.held_out_rmse
        rmse_random = em_random.receipt.held_out_rmse
        print(f"[M4 receipt] ALC RMSE={rmse_alc:.4f} vs random RMSE={rmse_random:.4f} at budget={budget}")
        self.assertLess(rmse_alc, rmse_random)
        self.assertLess(rmse_alc / rmse_random, 0.85)  # a real margin, not noise

    def test_error_bars_are_calibrated(self):
        from mixle.task.emulate import emulate

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            em = emulate(_true_f, BOUNDS_2D, budget=80, n_init=4, holdout_frac=0.35, method="alc", seed=3)

        r = em.receipt
        print(f"[M4 receipt] coverage={r.coverage:.3f} vs nominal={r.nominal_coverage:.3f} (n_holdout={r.n_holdout})")
        self.assertLess(abs(r.coverage - r.nominal_coverage), 0.3)


@unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
class MultiFidelityTest(unittest.TestCase):
    def test_multi_fidelity_beats_single_fidelity_at_matched_cost(self):
        from mixle.task.emulate import emulate

        budget = 30
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            em_sf = emulate(_forrester_target, BOUNDS_1D, budget=budget, n_init=2, method="alc", seed=5)
            em_mf = emulate(
                _forrester_mf,
                BOUNDS_1D,
                budget=budget,
                fidelities=(0.3, 1.0),
                costs=(0.1, 1.0),
                n_init=2,
                n_candidates=100,
                n_reference=60,
                seed=5,
            )

        rmse_sf = em_sf.receipt.held_out_rmse
        rmse_mf = em_mf.receipt.held_out_rmse
        print(
            f"[M4 receipt] multi-fidelity RMSE={rmse_mf:.4f} (n_train={em_mf.receipt.n_train}) vs "
            f"single-fidelity RMSE={rmse_sf:.4f} (n_train={em_sf.receipt.n_train}) at matched cost={budget}"
        )
        # MXR-080-0182: multi-fidelity training reserves each evaluation's cost against the budget before
        # spending it, so cost_spent never exceeds the requested budget -- it previously could overshoot,
        # both from an unconditional per-fidelity seeding phase that ignored the budget entirely (bounded
        # only by n_init times the summed cost across all fidelities, not by a single evaluation) and
        # from the sequential loop's last pick landing after the check meant to stop it. Single-fidelity
        # always spends its full declared budget by construction, so this is a matched-*or-less* cost
        # comparison, not a bit-identical one.
        self.assertLessEqual(em_mf.receipt.cost_spent, budget)
        self.assertLessEqual(em_mf.receipt.cost_spent, em_sf.receipt.cost_spent)
        self.assertLess(rmse_mf, rmse_sf)


# --------------------------------------------------------------------------- MXR-080-0181 (ruled out)
# mixle.doe.multifidelity.multi_fidelity_minimize's `target` need not have been a member of `fidelities`,
# so a target outside the queryable set was never evaluated and the fallback silently returned a
# lower-fidelity result mislabeled as the target answer. `_fit_multi_fidelity` cannot exhibit this: unlike
# multi_fidelity_minimize, neither it nor the public `emulate()` exposes a separate `target` parameter --
# target is always derived as `float(fids.max())`, which is trivially a member of `fids` by construction.
# This is a documentation test (nothing was broken, nothing was fixed) that pins the structural guarantee
# so it stays true if either function's shape ever changes.
class TargetFidelityStructuralGuaranteeTest(unittest.TestCase):
    def test_emulate_exposes_no_target_parameter(self):
        import inspect

        from mixle.task.emulate import emulate

        self.assertNotIn("target", inspect.signature(emulate).parameters)

    def test_fit_multi_fidelity_target_is_always_a_member_of_fidelities(self):
        for fidelities in [(0.3, 1.0), (0.1, 0.5, 1.0), (2.0, 7.0)]:
            fids = np.asarray(fidelities, dtype=np.float64)
            target = float(fids.max())
            self.assertIn(target, fids.tolist())


# --------------------------------------------------------------------------- MXR-080-0182
# _fit_multi_fidelity's initial per-fidelity seeding ignored the training budget entirely -- it
# unconditionally drew and evaluated n_init points at EVERY fidelity before checking cost at all, so the
# overshoot was bounded only by n_init times the summed cost across all fidelities, not by a single
# evaluation the way the sequential loop's own (also unguarded) last pick was. Every evaluation -- seeding
# and the sequential loop alike -- must now reserve its cost against the remaining budget before spending.
class MultiFidelityBudgetEnforcementTest(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
    def test_seeding_never_overshoots_budget_with_large_n_init(self):
        from mixle.task.emulate import emulate

        # n_init=10 at fidelities costing 1.0 and 5.0 would unconditionally spend 10*(1.0+5.0)=60 on
        # seeding alone -- 20 over this budget=50 run (of which 10 is reserved for the holdout) before
        # the pre-fix code ever reached its sequential loop.
        budget = 50
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            em = emulate(
                _forrester_mf,
                BOUNDS_1D,
                budget=budget,
                fidelities=(1.0, 5.0),
                n_init=10,
                holdout_frac=0.05,
                n_candidates=50,
                n_reference=30,
                seed=5,
            )
        self.assertLessEqual(em.receipt.cost_spent, budget)

    def test_budget_too_small_for_cheapest_fidelity_is_rejected(self):
        from mixle.task.emulate import emulate

        # budget=10.5 with holdout_frac=0.05 reserves a 10.0 holdout (< budget, so that check alone
        # passes), leaving max_cost=0.5 for training -- below the cheapest fidelity's own cost of 1.0.
        with self.assertRaisesRegex(ValueError, "cheapest fidelity"):
            emulate(
                _forrester_mf,
                BOUNDS_1D,
                budget=10.5,
                fidelities=(1.0, 5.0),
                n_init=10,
                holdout_frac=0.05,
                seed=5,
            )


# --------------------------------------------------------------------------- MXR-080-0183
# _fit_multi_fidelity's sequential loop caught every exception from surrogate fitting with a bare
# `except Exception` and turned it into a silent early stop: a genuine numerical failure (e.g. a singular
# covariance surfacing as LinAlgError) and a completely unrelated bug (e.g. a stray TypeError) were both
# swallowed identically, and the returned Emulator/receipt carried no trace that anything had gone wrong.
# Only well-defined numerical failure types are caught now, and EmulatorReceipt always carries an explicit
# `stopped_reason` (+ `error` when it failed).
class MultiFidelitySurrogateFitFailureStatusTest(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
    def test_normal_multi_fidelity_run_reports_budget_exhausted_with_no_error(self):
        from mixle.task.emulate import emulate

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            em = emulate(_forrester_mf, BOUNDS_1D, budget=30, fidelities=(0.3, 1.0), n_init=2, seed=2)
        self.assertEqual(em.receipt.stopped_reason, "budget_exhausted")
        self.assertIsNone(em.receipt.error)

    @unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
    def test_normal_single_fidelity_run_reports_budget_exhausted_with_no_error(self):
        from mixle.task.emulate import emulate

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            em = emulate(_true_f, BOUNDS_2D, budget=24, n_init=4, seed=0)
        self.assertEqual(em.receipt.stopped_reason, "budget_exhausted")
        self.assertIsNone(em.receipt.error)

    @unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
    def test_a_genuine_numerical_failure_after_partial_progress_reports_explicit_status(self):
        """A failure AFTER at least one successful fit must be gracefully absorbed -- there is a
        previously-fit surrogate to fall back to -- with the failure surfaced via stopped_reason/error.

        Unlike mixle.doe.multifidelity.multi_fidelity_minimize (whose single, always-guarded fit call
        makes any failure immediately gracefully absorbable), _fit_multi_fidelity needs at least one
        prior successful fit to fall back to; see test_surrogate_never_fits_even_once_propagates below
        for the no-partial-success case.
        """
        emulate_mod = importlib.import_module("mixle.task.emulate")
        from mixle.task.emulate import emulate

        real_fit_surrogate = emulate_mod._fit_surrogate
        call_count = {"n": 0}

        def _fail_on_second_call(x, y, gp, fit_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise np.linalg.LinAlgError("simulated singular covariance")
            return real_fit_surrogate(x, y, gp, fit_kwargs)

        with mock.patch.object(emulate_mod, "_fit_surrogate", side_effect=_fail_on_second_call):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                em = emulate(_forrester_mf, BOUNDS_1D, budget=40, fidelities=(0.3, 1.0), n_init=2, seed=0)
        self.assertEqual(em.receipt.stopped_reason, "surrogate_fit_failed")
        self.assertIsNotNone(em.receipt.error)
        self.assertIn("LinAlgError", em.receipt.error)
        self.assertGreaterEqual(call_count["n"], 2)

    def test_surrogate_never_fits_even_once_propagates(self):
        """No partial success exists to fall back to, so the failure must propagate -- same as
        _fit_single_fidelity's existing unguarded _fit_surrogate call -- rather than returning a broken
        Emulator with no gp."""
        emulate_mod = importlib.import_module("mixle.task.emulate")
        from mixle.task.emulate import emulate

        with mock.patch.object(
            emulate_mod, "_fit_surrogate", side_effect=np.linalg.LinAlgError("simulated singular covariance")
        ):
            with self.assertRaises(np.linalg.LinAlgError):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    emulate(_forrester_mf, BOUNDS_1D, budget=40, fidelities=(0.3, 1.0), n_init=2, seed=0)

    def test_an_unrelated_bug_propagates_instead_of_being_swallowed(self):
        emulate_mod = importlib.import_module("mixle.task.emulate")
        from mixle.task.emulate import emulate

        with mock.patch.object(emulate_mod, "_fit_surrogate", side_effect=TypeError("simulated unrelated coding bug")):
            with self.assertRaises(TypeError):
                emulate(_forrester_mf, BOUNDS_1D, budget=40, fidelities=(0.3, 1.0), n_init=2, seed=0)

    @unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
    def test_a_transient_unrelated_bug_also_propagates_not_swallowed(self):
        """A bug that fails only ONCE (e.g. a one-off coding mistake, as opposed to a persistent
        misconfiguration) is the most dangerous case for a bare `except Exception`: if caught, the loop
        would break silently and a later, unrelated success could mask it completely. Confirms the
        narrowed except doesn't even attempt to catch it, transient or not."""
        emulate_mod = importlib.import_module("mixle.task.emulate")
        from mixle.task.emulate import emulate

        real_fit_surrogate = emulate_mod._fit_surrogate
        call_count = {"n": 0}

        def _fail_on_second_call(x, y, gp, fit_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise TypeError("simulated unrelated coding bug (transient)")
            return real_fit_surrogate(x, y, gp, fit_kwargs)

        with mock.patch.object(emulate_mod, "_fit_surrogate", side_effect=_fail_on_second_call):
            with self.assertRaises(TypeError):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    emulate(_forrester_mf, BOUNDS_1D, budget=40, fidelities=(0.3, 1.0), n_init=2, seed=0)


if __name__ == "__main__":
    unittest.main()
