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
        # tell() only accepts points this optimizer's own ask() actually returned (MXR-080-0188), so
        # -- unlike the arbitrary hand-picked x's this test used before that fix -- these must be
        # real, currently-pending proposals; ask() one at a time and tell() each back.
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        p0, p1, p2 = opt.ask(), opt.ask(), opt.ask()
        opt.tell(p0, 5.0).tell(p1, 2.0).tell(p2, 9.0)
        self.assertEqual(opt.n_observations, 3)
        self.assertEqual(opt.x.shape, (3, 2))
        self.assertEqual(opt.y.shape, (3,))
        self.assertEqual(opt.best.best_y, 2.0)
        np.testing.assert_array_equal(opt.best.best_x, p1)

    def test_tell_accepts_batches(self):
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        batch = opt.ask(2)  # tell() requires real, currently-pending proposals (MXR-080-0188)
        opt.tell(batch, [3.0, 1.0])
        self.assertEqual(opt.n_observations, 2)
        self.assertEqual(opt.best.best_y, 1.0)

    def test_maximize_flips_incumbent(self):
        opt = BayesianOptimizer(self.bounds, maximize=True, n_init=4, seed=0)
        batch = opt.ask(2)  # tell() requires real, currently-pending proposals (MXR-080-0188)
        opt.tell(batch, [3.0, 1.0])
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

    def test_constructor_and_ask_controls_are_exact_and_finite(self):
        for invalid in ("false", 1, np.bool_(True)):
            with self.assertRaises(TypeError):
                BayesianOptimizer(self.bounds, maximize=invalid)
        for invalid in (np.nan, np.inf, -1.0):
            with self.assertRaises(ValueError):
                BayesianOptimizer(self.bounds, xi=invalid)
        for invalid in (0, 2.5, True, np.bool_(True)):
            with self.assertRaises((TypeError, ValueError)):
                BayesianOptimizer(self.bounds, n_candidates=invalid)
        with self.assertRaises(ValueError):
            BayesianOptimizer(self.bounds, acq_kwargs={"kappa": np.nan})

        optimizer = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        for invalid in (True, np.bool_(True), 1.5, 0):
            with self.assertRaises((TypeError, ValueError)):
                optimizer.ask(invalid)


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

    def test_overshooting_the_boundary_in_one_call_is_atomic_and_safe_to_retry(self):
        # n_init=4, ask(6) with nothing told yet: 4 init points are available but the remaining 2
        # require GP acquisition, which fails with zero observations. ask() is atomic (MXR-080-0188):
        # a failed call leaves _init_used/pending untouched rather than burning the 4 real
        # init-design points the caller never received, so a subsequent in-budget ask(4) must still
        # return the full, uncorrupted initial design -- not fewer than 4 due to the double-counting
        # bug, and not skipped over due to a partially-committed failed call.
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=1)
        with self.assertRaises(ValueError):
            opt.ask(6)
        self.assertEqual(opt._init_used, 0)
        self.assertEqual(opt.n_pending, 0)
        retry = opt.ask(4)
        self.assertEqual(retry.shape, (4, 2))
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


class NInitValidationTest(unittest.TestCase):
    """Regression coverage for MXR-080-0188: an invalid n_init (zero, negative, non-integer) used to
    be silently raised to 1 (``max(1, int(n_init))``) instead of rejected -- a caller typo produced a
    working-looking but wrong-sized initial design rather than a clear error.
    """

    bounds = [(-2.0, 2.0), (0.0, 5.0)]

    def test_zero_n_init_is_rejected(self):
        with self.assertRaises(ValueError):
            BayesianOptimizer(self.bounds, n_init=0)

    def test_negative_n_init_is_rejected(self):
        with self.assertRaises(ValueError):
            BayesianOptimizer(self.bounds, n_init=-3)

    def test_fractional_n_init_is_rejected(self):
        with self.assertRaises(ValueError):
            BayesianOptimizer(self.bounds, n_init=2.5)

    def test_bool_n_init_is_rejected(self):
        # bool is a technically-valid int subclass in Python but never a meaningful count.
        with self.assertRaises(TypeError):
            BayesianOptimizer(self.bounds, n_init=True)

    def test_none_n_init_still_uses_the_documented_default(self):
        # n_init=None is the documented "use 2*dim+1" default, not a validation-rejection case.
        opt = BayesianOptimizer(self.bounds, n_init=None)
        self.assertEqual(opt.n_init, 2 * opt.dim + 1)

    def test_valid_n_init_still_works(self):
        opt = BayesianOptimizer(self.bounds, n_init=7)
        self.assertEqual(opt.n_init, 7)


class TellValidationTest(unittest.TestCase):
    """Regression coverage for MXR-080-0188: tell() used to accept out-of-bounds/non-finite points,
    NaN/Inf outcomes, duplicate tells, and observations for points never asked, all without
    validation. Every case here stays within the n_init budget (no torch/GP fit needed): the
    validation being tested runs identically regardless of which phase a point came from.
    """

    bounds = [(-2.0, 2.0), (0.0, 5.0)]

    def test_out_of_bounds_point_is_rejected(self):
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        opt.ask()
        with self.assertRaises(ValueError):
            opt.tell([10.0, 10.0], 1.0)  # well outside both dimensions' bounds

    def test_non_finite_point_is_rejected(self):
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        opt.ask()
        with self.assertRaises(ValueError):
            opt.tell([float("nan"), 1.0], 1.0)

    def test_non_finite_outcome_is_rejected(self):
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        p = opt.ask()
        with self.assertRaises(ValueError):
            opt.tell(p, float("nan"))
        with self.assertRaises(ValueError):
            opt.tell(p, float("inf"))

    def test_unsolicited_point_is_rejected(self):
        # A point this optimizer's own ask() never returned -- a likely caller bug (wrong array, a
        # stale point from a different optimizer) -- is rejected by default rather than silently
        # accepted.
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        opt.ask()
        with self.assertRaises(ValueError):
            opt.tell([0.0, 1.0], 1.0)

    def test_duplicate_tell_for_an_already_told_point_is_rejected(self):
        # tell() is write-once per point; there is no "update an existing observation" path.
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        p = opt.ask()
        opt.tell(p, 1.0)
        with self.assertRaises(ValueError):
            opt.tell(p, 2.0)

    def test_duplicate_point_repeated_within_one_tell_call_is_rejected(self):
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        p = opt.ask()
        with self.assertRaises(ValueError):
            opt.tell([p, p], [1.0, 2.0])

    def test_invalid_tell_does_not_partially_apply(self):
        # row 0 is a real pending point, row 1 is unsolicited -- the whole call must be rejected
        # and neither row recorded (validation covers the whole batch before anything is applied).
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        p = opt.ask()
        with self.assertRaises(ValueError):
            opt.tell([p, [0.0, 1.0]], [1.0, 2.0])
        self.assertEqual(opt.n_observations, 0)
        self.assertEqual(opt.n_pending, 1)  # p is still pending, untouched by the failed call

    def test_valid_tell_resolves_the_matching_pending_proposal(self):
        opt = BayesianOptimizer(self.bounds, n_init=4, seed=0)
        p = opt.ask()
        self.assertEqual(opt.n_pending, 1)
        opt.tell(p, 1.0)
        self.assertEqual(opt.n_pending, 0)
        self.assertEqual(opt.n_observations, 1)

    def test_large_distinct_points_resolve_their_exact_pending_ids(self):
        bounds = [(1.0e9, 1.0e9 + 10_000.0)]
        opt = BayesianOptimizer(bounds, n_init=2, seed=0)
        opt._init_design = np.array([[1.0e9 + 1_000.0], [1.0e9 + 3_000.0]])
        first = opt.ask()
        second = opt.ask()
        self.assertFalse(opt._close(first, second))
        opt.tell(second, 2.0)
        self.assertEqual(opt.n_pending, 1)
        np.testing.assert_array_equal(opt.pending[0], first)
        np.testing.assert_array_equal(opt.x[0], second)


class PendingProposalTrackingTest(unittest.TestCase):
    """Regression coverage for MXR-080-0188: beyond the initial design, outstanding (asked-but-not-
    told) points were not tracked at all, so a second, overlapping ask() before any tell() had no way
    to avoid re-proposing a point the first ask() already handed out.
    """

    bounds = [(-2.0, 2.0), (0.0, 5.0)]

    def test_ask_tracks_its_return_value_as_pending_until_told(self):
        opt = BayesianOptimizer(self.bounds, n_init=3, seed=0)
        p0 = opt.ask()
        self.assertEqual(opt.n_pending, 1)
        np.testing.assert_array_equal(opt.pending[0], p0)
        opt.tell(p0, 1.0)
        self.assertEqual(opt.n_pending, 0)
        self.assertEqual(opt.pending.shape, (0, 2))

    def test_pending_points_from_a_prior_ask_are_folded_into_the_next_gp_fit_as_fantasy_observations(self):
        # White-box wiring proof, independent of torch/randomness: patch out the surrogate-fit and
        # acquisition-proposal seams ask() calls through with a fake GP and a spy, and confirm the
        # SECOND overlapping ask() (before any tell()) feeds the surrogate one more row than the
        # first call did -- the first ask()'s still-pending point, fantasized in.
        import mixle.doe.optimizer as optimizer_module

        class _FakeGP:
            def predict(self, x, y, query, return_cov=False):
                return np.zeros(query.shape[0])

        captured_y_lengths: list[int] = []

        def _fake_fit_surrogate(x, y, gp, fit_kwargs):
            return _FakeGP()

        def _spy_propose_next(x, y, bounds, **kwargs):
            captured_y_lengths.append(len(y))
            return np.array([0.1, 0.1])

        opt = BayesianOptimizer(self.bounds, n_init=1, seed=0)
        opt.tell(opt.ask(), 1.0)  # one real observation, nothing pending; init design now exhausted

        original_fit, original_propose = optimizer_module._fit_surrogate, optimizer_module.propose_next
        optimizer_module._fit_surrogate = _fake_fit_surrogate
        optimizer_module.propose_next = _spy_propose_next
        try:
            opt.ask()  # remaining=1, GP phase, nothing pending yet -> plain y (1 real observation)
            opt.ask()  # the previous ask()'s point is now pending -> fantasy-augmented y (1 + 1)
        finally:
            optimizer_module._fit_surrogate = original_fit
            optimizer_module.propose_next = original_propose
        self.assertEqual(captured_y_lengths, [1, 2])

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_two_overlapping_asks_before_any_tell_do_not_return_duplicate_points(self):
        # Integration-level companion to the white-box test above, with a real GP: exhaust the init
        # design and tell() it, then ask() twice more with nothing told in between (the parallel/
        # async-campaign scenario the finding names) and confirm the two GP-acquired proposals are
        # not the same point.
        bounds = [(-2.0, 2.0), (-2.0, 2.0)]

        def objective(p):
            return float(np.sum(p**2))

        opt = BayesianOptimizer(bounds, n_init=3, seed=0, n_candidates=256, fit_kwargs={"max_its": 40})
        init = opt.ask(3)
        opt.tell(init, [objective(p) for p in init])
        first = opt.ask()
        second = opt.ask()
        self.assertEqual(opt.n_pending, 2)
        self.assertFalse(np.allclose(first, second))


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class NormalUsageEndToEndTest(unittest.TestCase):
    """Negative control for MXR-080-0188: an ordinary, well-behaved ask/tell campaign -- every
    proposal matched to its correct tell(), no duplicates, no invalid inputs -- must still work
    end to end across both the initial-design and GP-acquisition phases with the new pending
    tracking and tell() validation in place.
    """

    def test_a_well_behaved_ask_tell_campaign_still_works_end_to_end(self):
        target = np.array([0.5, -1.0])
        bounds = [(-3.0, 3.0), (-3.0, 3.0)]

        def objective(p):
            return float(np.sum((p - target) ** 2))

        opt = BayesianOptimizer(bounds, n_init=4, n_candidates=128, seed=0, fit_kwargs={"max_its": 40})
        seen = []
        for _ in range(10):
            x = opt.ask()
            self.assertEqual(opt.n_pending, 1)
            opt.tell(x, objective(x))
            self.assertEqual(opt.n_pending, 0)
            seen.append(tuple(x))
        self.assertEqual(len(seen), len(set(seen)))  # no duplicates across the whole campaign
        self.assertEqual(opt.n_observations, 10)
        self.assertIsNotNone(opt.best)

        # A batched tail, spanning nothing (already well past n_init): still resolves cleanly.
        batch = opt.ask(3)
        opt.tell(batch, [objective(p) for p in batch])
        self.assertEqual(opt.n_observations, 13)
        self.assertEqual(opt.n_pending, 0)


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


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class ThompsonSamplingBayesianOptimizerReproducibilityTest(unittest.TestCase):
    """``thompson_sampling`` used to fall back to a fresh, unseeded ``np.random.RandomState()``
    whenever the proposal loop's own seeded ``rng`` was not manually re-threaded into ``acq_kwargs``.
    ``BayesianOptimizer(acq="thompson", seed=2)`` therefore reproduced its Latin-hypercube init design
    bit-for-bit across runs (that path always used ``self.rng`` correctly) but drew a DIFFERENT
    GP-guided proposal every time, since Thompson's own draw came from OS entropy rather than
    ``self.rng``. ``mixle.doe.bayesopt._propose_one`` now threads its ``rng`` into every acquisition
    call, so a seeded ``BayesianOptimizer`` is fully reproducible under ``acq="thompson"`` too, exactly
    like it already was for ``acq="ei"``/``"pi"``/``"ucb"``. See also
    ``doe_bayesopt_test.ThompsonSamplingRngThreadingTest`` for the torch-free proof at the
    ``propose_next``/acquisition-function level.
    """

    bounds = [(-2.0, 2.0), (0.0, 5.0)]

    @staticmethod
    def _objective(p):
        return float(np.sum(np.asarray(p) ** 2))

    def _run(self, seed=2, n_init=4, n_gp=3):
        opt = BayesianOptimizer(self.bounds, acq="thompson", seed=seed, n_init=n_init, n_candidates=64)
        init_pts = [opt.ask() for _ in range(n_init)]
        for p in init_pts:
            opt.tell(p, self._objective(p))
        gp_pts = [opt.ask() for _ in range(n_gp)]  # now in GP-acquisition (thompson) territory
        return np.array(init_pts), np.array(gp_pts)

    def test_gp_guided_thompson_proposals_are_bit_identical_across_runs(self):
        # The exact reported reproduction: two seed=2 runs.
        init_a, gp_a = self._run()
        init_b, gp_b = self._run()
        np.testing.assert_array_equal(init_a, init_b)  # already true pre-fix -- the LHS design was fine
        np.testing.assert_array_equal(gp_a, gp_b)  # the actual bug: this used to differ run to run

    def test_bit_identical_even_with_unrelated_optimizer_calls_interleaved(self):
        # Burn-in: drive several unrelated BayesianOptimizer instances (different seeds, an unseeded
        # one, and a different acquisition) through full ask/tell cycles between the two seed=2 runs,
        # to rule out any hidden shared/global state -- not merely that a single instance's own rng is
        # self-consistent.
        init_a, gp_a = self._run()

        for burn_seed in (None, 5, 9):
            burn = BayesianOptimizer(self.bounds, acq="ei", seed=burn_seed, n_init=3, n_candidates=32)
            for _ in range(4):
                p = burn.ask()
                burn.tell(p, self._objective(p))

        unrelated_thompson = BayesianOptimizer(self.bounds, acq="thompson", seed=123, n_init=3, n_candidates=32)
        for _ in range(3):
            p = unrelated_thompson.ask()
            unrelated_thompson.tell(p, self._objective(p))
        unrelated_thompson.ask(2)  # its own GP-guided thompson draws, deliberately left un-told

        init_b, gp_b = self._run()
        np.testing.assert_array_equal(init_a, init_b)
        np.testing.assert_array_equal(gp_a, gp_b)

    def test_default_acq_ei_path_is_unaffected_by_the_rng_threading_change(self):
        # Negative control at the BayesianOptimizer level: the default acquisition takes no rng at
        # all, so it must remain exactly as reproducible (and only as reproducible) as before.
        def run_ei(seed):
            opt = BayesianOptimizer(self.bounds, acq="ei", seed=seed, n_init=4, n_candidates=64)
            init_pts = [opt.ask() for _ in range(4)]
            for p in init_pts:
                opt.tell(p, self._objective(p))
            return np.array([opt.ask() for _ in range(3)])

        ei_a = run_ei(2)
        ei_b = run_ei(2)
        np.testing.assert_array_equal(ei_a, ei_b)


if __name__ == "__main__":
    unittest.main()
