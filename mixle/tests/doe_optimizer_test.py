"""Tests for the ask-tell BayesianOptimizer (WS-E).

The ask-tell mechanics (initial space-filling design, tell/best bookkeeping, validation) are
torch-free; the GP-acquisition phase requires torch and is skipped without it.
"""

import importlib.util
import unittest

import numpy as np

from mixle.doe import BayesianOptimizer

HAS_TORCH = importlib.util.find_spec("torch") is not None


class AskTellMechanicsTest(unittest.TestCase):
    bounds = [(-2.0, 2.0), (0.0, 5.0)]

    def _in_bounds(self, x):
        b = np.asarray(self.bounds, dtype=float)
        return bool(np.all(x >= b[:, 0] - 1e-9) and np.all(x <= b[:, 1] + 1e-9))

    def test_ask_returns_point_in_bounds(self):
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        x = opt.ask()
        self.assertEqual(x.shape, (2,))
        self.assertTrue(self._in_bounds(x))

    def test_initial_asks_are_space_filling_and_distinct(self):
        # Within the n_init budget, asks come from a Latin-hypercube design (no GP, no torch needed).
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        pts = np.array([opt.ask() for _ in range(4)])
        self.assertEqual(pts.shape, (4, 2))
        self.assertEqual(len({tuple(p) for p in pts}), 4)  # all distinct
        self.assertTrue(all(self._in_bounds(p) for p in pts))

    def test_batch_ask_in_init_phase(self):
        opt = BayesianOptimizer(self.bounds, n_init=6, seed=1)
        batch = opt.ask(q=3)
        self.assertEqual(batch.shape, (3, 2))
        self.assertTrue(all(self._in_bounds(p) for p in batch))

    def test_tell_records_and_best_tracks_minimum(self):
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        opt.tell([0.0, 1.0], 5.0).tell([1.0, 2.0], 2.0).tell([-1.0, 4.0], 9.0)
        self.assertEqual(opt.n_observations, 3)
        self.assertEqual(opt.x.shape, (3, 2))
        self.assertEqual(opt.y.shape, (3,))
        self.assertEqual(opt.best.best_y, 2.0)
        np.testing.assert_array_equal(opt.best.best_x, [1.0, 2.0])

    def test_tell_accepts_batches(self):
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        opt.tell([[0.0, 1.0], [1.0, 2.0]], [3.0, 1.0])
        self.assertEqual(opt.n_observations, 2)
        self.assertEqual(opt.best.best_y, 1.0)

    def test_maximize_flips_incumbent(self):
        opt = BayesianOptimizer(self.bounds, maximize=True, n_init=4, seed=0)
        opt.tell([[0.0, 1.0], [1.0, 2.0]], [3.0, 1.0])
        self.assertEqual(opt.best.best_y, 3.0)

    def test_validation(self):
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        with self.assertRaises(ValueError):
            opt.ask(q=0)
        with self.assertRaises(ValueError):
            opt.tell([0.0], 1.0)  # wrong dimension
        with self.assertRaises(ValueError):
            opt.tell([[0.0, 1.0], [1.0, 2.0]], [1.0])  # x/y length mismatch
        with self.assertRaises(ValueError):
            _ = BayesianOptimizer(self.bounds, n_init=4, seed=0).best  # no observations yet


class InitialDesignBoundaryTest(unittest.TestCase):
    """Regression coverage for MXR-080-0187.

    ask()'s initial-design loop used to gate on ``self._init_used + len(points) < self.n_init``,
    but ``_next_init_point()`` already increments ``self._init_used`` as its own side effect --
    double-counting every point dispensed by the loop (once via ``_init_used``'s own increment,
    once via ``points`` growing) and making it believe it had dispensed twice as many init points
    as it actually had. A single batched ``ask(5)`` with ``n_init=5`` stopped after 3 points and
    fell through to GP batch acquisition with zero observations, which fails. This class targets
    every batch size that lands on, splits around, or overshoots that boundary -- an off-by-one /
    double-counting bug only shows up at specific boundary conditions, so boundary coverage is the
    real proof of the fix.
    """

    bounds = [(-2.0, 2.0), (0.0, 5.0)]

    def _in_bounds(self, x):
        b = np.asarray(self.bounds, dtype=float)
        return bool(np.all(x >= b[:, 0] - 1e-9) and np.all(x <= b[:, 1] + 1e-9))

    def test_ask_n_equal_n_init_dispenses_every_initial_point_in_one_call(self):
        # The audit's exact repro: ask(5) with n_init=5 used to stop after 3 points and fall through
        # to GP batch acquisition with zero observations.
        opt = BayesianOptimizer(self.bounds, n_init=5, seed=0)
        batch = opt.ask(5)
        self.assertEqual(batch.shape, (5, 2))
        self.assertEqual(opt._init_used, 5)
        self.assertTrue(all(self._in_bounds(p) for p in batch))
        self.assertEqual(len({tuple(p) for p in batch}), 5)  # all distinct

    def test_ask_exactly_at_a_different_boundary_still_works(self):
        # Same shape of bug, a different n_init, to confirm the fix is not specific to n_init=5.
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=2)
        batch = opt.ask(4)
        self.assertEqual(batch.shape, (4, 2))
        self.assertEqual(opt._init_used, 4)

    def test_splitting_a_batch_across_two_asks_matches_asking_it_all_at_once(self):
        # ask(3) then ask(2) (both still inside the n_init=5 budget) must dispense the same 5
        # points, in the same order, as a single ask(5) -- the initial design is a fixed,
        # precomputed array indexed by _init_used, so splitting a request must never change which
        # points come out or strand any of them behind a double-counted boundary check.
        split = BayesianOptimizer(self.bounds, n_init=5, seed=0)
        combined = np.vstack([split.ask(3), split.ask(2)])
        direct = BayesianOptimizer(self.bounds, n_init=5, seed=0).ask(5)
        np.testing.assert_array_equal(combined, direct)
        self.assertEqual(split._init_used, 5)

    def test_overshooting_the_boundary_in_one_call_dispenses_the_full_init_design_first(self):
        # n_init=4, ask(6) with nothing told yet: the first 4 slots must still come from the
        # initial design (verified via _init_used, since the zero-observations GP error below
        # discards the returned batch); only the remaining 2 attempt -- and, with no observations
        # yet, fail -- GP acquisition. Must be 4, not fewer, despite the double-counting bug.
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=1)
        with self.assertRaises(ValueError):
            opt.ask(6)
        self.assertEqual(opt._init_used, 4)

    def test_batch_overshoot_after_the_design_is_already_exhausted_raises_cleanly(self):
        # n_init=3: ask(3) exhausts the design (remaining=0, no GP call at all); a second, batched
        # ask(2) with still nothing told hits the `remaining > 1` (propose_batch) branch -- not just
        # the `remaining == 1` (propose_next) branch the pre-existing doe_stability2_test.py
        # coverage exercises -- and must raise the same documented zero-observations error, not
        # silently wrap around _init_design via `_init_used % n_init`.
        opt = BayesianOptimizer(self.bounds, n_init=3, seed=3)
        first = opt.ask(3)
        self.assertEqual(first.shape, (3, 2))
        with self.assertRaises(ValueError):
            opt.ask(2)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_batch_spanning_the_boundary_dispenses_remaining_init_then_a_gp_point(self):
        # n_init=5: ask(3) dispenses 3 init points; telling them makes a GP fit possible; ask(3)
        # again must dispense the 2 remaining init-design points first, then exactly 1 real
        # GP-acquired point -- the boundary-spanning case named in the audit finding.
        opt = BayesianOptimizer(self.bounds, n_init=5, seed=0, n_candidates=64, fit_kwargs={"max_its": 30})
        first = opt.ask(3)
        opt.tell(first, [float(np.sum(p**2)) for p in first])
        second = opt.ask(3)
        self.assertEqual(second.shape, (3, 2))
        np.testing.assert_array_equal(second[0], opt._init_design[3])
        np.testing.assert_array_equal(second[1], opt._init_design[4])
        self.assertTrue(self._in_bounds(second[2]))
        self.assertEqual(opt._init_used, 5)

    def test_negative_control_well_within_the_boundary_is_unaffected(self):
        # n_init=10: ordinary ask/tell cycles that never approach the boundary. Every point must
        # still come from the initial design, distinct, in bounds -- the fix must not disturb
        # ordinary, non-boundary-spanning usage. (A negative control entirely PAST the boundary
        # already exists above: AskTellOptimizationTest exercises full ask/tell campaigns well
        # beyond n_init with a real GP the whole way.)
        opt = BayesianOptimizer(self.bounds, n_init=10, seed=4)
        all_points = []
        for q in (3, 4, 2):
            batch = opt.ask(q)
            opt.tell(batch, [float(np.sum(p**2)) for p in batch])
            all_points.append(batch)
        all_points = np.vstack(all_points)
        self.assertEqual(all_points.shape, (9, 2))
        self.assertEqual(opt._init_used, 9)
        self.assertEqual(len({tuple(p) for p in all_points}), 9)
        self.assertTrue(all(self._in_bounds(p) for p in all_points))
        self.assertEqual(opt.n_observations, 9)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class AskTellOptimizationTest(unittest.TestCase):
    def test_loop_converges_on_a_bowl(self):
        target = np.array([0.5, -1.0])
        bounds = [(-3.0, 3.0), (-3.0, 3.0)]

        def objective(p):
            return float(np.sum((p - target) ** 2))

        opt = BayesianOptimizer(bounds, n_init=6, n_candidates=256, seed=0, fit_kwargs={"max_its": 60})
        for _ in range(24):
            x = opt.ask()
            opt.tell(x, objective(x))
        self.assertEqual(opt.n_observations, 24)
        self.assertLess(opt.best.best_y, 0.5)  # beats the coarse initial design, nears the optimum

    def test_batch_ask_after_data_returns_distinct_points(self):
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]

        def objective(p):
            return float(np.sum(p**2))

        opt = BayesianOptimizer(bounds, n_init=5, n_candidates=128, seed=0, fit_kwargs={"max_its": 50})
        for _ in range(5):  # exhaust the init design
            x = opt.ask()
            opt.tell(x, objective(x))
        batch = opt.ask(q=3)  # now a GP kriging-believer batch
        self.assertEqual(batch.shape, (3, 2))
        self.assertGreater(len({tuple(np.round(p, 6)) for p in batch}), 1)


if __name__ == "__main__":
    unittest.main()
