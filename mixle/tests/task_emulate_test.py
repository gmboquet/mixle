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
        # Matched budget, not bit-identical cost: the multi-fidelity loop's last pick can overshoot
        # max_cost by less than one fidelity's cost before the `spent < max_cost` check stops it (same
        # semantics as mixle.doe.multifidelity.multi_fidelity_minimize).
        self.assertAlmostEqual(em_mf.receipt.cost_spent, em_sf.receipt.cost_spent, delta=1.0)
        self.assertLess(rmse_mf, rmse_sf)


if __name__ == "__main__":
    unittest.main()
