"""Global sensitivity analysis: Sobol indices vs the Ishigami analytic values, Morris screening (Phase 4).

MXR-080-0192 (shared bounds/count/name/model-output validation contract) and MXR-080-0193 (raw Sobol
estimates plus bootstrap uncertainty) are covered below the original Phase 4 coverage.
"""

import unittest
import warnings

import numpy as np

from mixle.doe import dgsm, fast_indices, morris_screening, sobol_indices


def ishigami(x, a=7.0, b=0.1):
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


BOUNDS = [(-np.pi, np.pi)] * 3


class SobolTest(unittest.TestCase):
    def setUp(self):
        self.res = sobol_indices(ishigami, BOUNDS, n=16384, seed=0, names=["x1", "x2", "x3"])

    def test_first_order_matches_analytic_ishigami(self):
        np.testing.assert_allclose(self.res["S1"], [0.314, 0.442, 0.0], atol=0.03)

    def test_total_order_matches_analytic_ishigami(self):
        np.testing.assert_allclose(self.res["ST"], [0.557, 0.442, 0.244], atol=0.03)

    def test_x3_is_pure_interaction(self):
        self.assertLess(self.res["S1"][2], 0.05)  # no main effect
        self.assertGreater(self.res["ST"][2] - self.res["S1"][2], 0.1)  # but interacts (with x1)

    def test_additive_linear_model(self):
        res = sobol_indices(lambda x: x[:, 0] + 2 * x[:, 1] + 3 * x[:, 2], [(0, 1)] * 3, n=8192)
        self.assertAlmostEqual(res["S1"].sum(), 1.0, delta=0.02)  # additive -> first orders partition
        np.testing.assert_allclose(res["S1"], res["ST"], atol=0.02)  # no interactions
        np.testing.assert_allclose(res["S1"] / res["S1"][0], [1.0, 4.0, 9.0], atol=0.2)  # variance ~ coef^2

    def test_constant_output_is_all_zero(self):
        res = sobol_indices(lambda x: np.ones(len(x)), [(0, 1)] * 2, n=512)
        np.testing.assert_array_equal(res["S1"], [0.0, 0.0])
        np.testing.assert_array_equal(res["ST"], [0.0, 0.0])


class _NoisyNull:
    """Output is independent noise: true S1=ST=0 for every input, exactly. The RNG stream persists
    across calls (not reseeded per call), so ``ya``/``yb``/``yab`` still differ from one another --
    unlike a literally constant-output model, this is *not* the ``var <= 0`` special case, so the
    finite-sample estimate has genuine, reproducible Monte Carlo noise around 0.
    """

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)

    def __call__(self, x):
        return self._rng.standard_normal(x.shape[0])


class SobolUncertaintyTest(unittest.TestCase):
    """MXR-080-0193: sobol_indices must always return the raw (unclipped) estimate plus a bootstrap
    standard error and confidence interval, with any clipped convenience value kept separate.
    """

    def test_raw_estimate_is_not_clamped_to_the_boundary(self):
        # Small n against a genuinely null (noise-only) model makes MC noise dominate: some raw S1
        # values must land below 0 (ST's Jansen form is a mean of squares and can't itself go
        # negative, so S1 is where clipping-to-false-certainty would show up). If this ever regressed
        # to clipping S1 in place, this loop would never find one.
        saw_negative_raw = any(
            np.any(sobol_indices(_NoisyNull(seed), [(0, 1)] * 3, n=32, seed=seed)["S1"] < 0.0) for seed in range(30)
        )
        self.assertTrue(saw_negative_raw, "raw S1 should sometimes be negative under pure MC noise")

    def test_clipped_field_is_separate_and_is_a_real_clip_of_the_raw_field(self):
        res = sobol_indices(_NoisyNull(0), [(0, 1)] * 3, n=32, seed=0)
        np.testing.assert_allclose(res["S1_clipped"], np.clip(res["S1"], 0.0, 1.0))
        np.testing.assert_allclose(res["ST_clipped"], np.clip(res["ST"], 0.0, None))
        # this specific (seeded, reproducible) fixture has a genuinely negative raw S1 entry -- confirm
        # it survives un-clamped under "S1" and only "S1_clipped" zeroes it out.
        self.assertLess(res["S1"][1], 0.0)
        self.assertEqual(res["S1_clipped"][1], 0.0)

    def test_standard_error_and_confidence_interval_fields_are_well_formed(self):
        res = sobol_indices(ishigami, BOUNDS, n=4096, seed=0)
        for key in ("S1_standard_error", "ST_standard_error"):
            self.assertTrue(np.all(np.isfinite(res[key])))
            self.assertTrue(np.all(res[key] > 0))
        self.assertTrue(np.all(res["S1_ci_low"] <= res["S1_ci_high"]))
        self.assertTrue(np.all(res["ST_ci_low"] <= res["ST_ci_high"]))
        # The point estimate should typically fall within its own bootstrap interval for a smooth,
        # well-resolved statistic like this one -- checked across a few seeds with a loose floor since
        # a percentile bootstrap gives no strict per-seed guarantee, unlike the ordering checks above.
        contained = sum(
            np.all(r["S1_ci_low"] <= r["S1"]) and np.all(r["S1"] <= r["S1_ci_high"])
            for r in (sobol_indices(ishigami, BOUNDS, n=4096, seed=s) for s in range(5))
        )
        self.assertGreaterEqual(contained, 4)

    def test_confidence_intervals_are_calibrated(self):
        # Coverage check (mirrors calibrate_test.py's KOCalibrationTest.test_theta_confidence_
        # intervals_are_calibrated): an additive linear model on [0, 1]^3 has an exactly known
        # analytic first-order index, S1 = coef_i^2 / sum(coef_j^2) (uniform Var(x_i) cancels
        # identically): [1, 4, 9] / 14.
        true_s1 = np.array([1.0, 4.0, 9.0]) / 14.0
        trials = 20
        covered = np.zeros(3)
        for seed in range(trials):
            res = sobol_indices(
                lambda x: x[:, 0] + 2 * x[:, 1] + 3 * x[:, 2],
                [(0, 1)] * 3,
                n=1024,
                seed=seed + 500,
                n_bootstrap=150,
            )
            covered += (res["S1_ci_low"] <= true_s1) & (true_s1 <= res["S1_ci_high"])
        # Loose floor (nominal ~95% * 20 trials ~= 19): keeps this non-flaky while still catching a
        # badly miscalibrated interval (e.g. one that collapses to a point, or is centered elsewhere).
        self.assertTrue(np.all(covered >= trials * 0.5), f"coverage too low: {covered}/{trials}")

    def test_n_bootstrap_and_confidence_are_validated(self):
        with self.assertRaises(ValueError):
            sobol_indices(_linear, [(0, 1), (0, 1)], n=32, n_bootstrap=0)
        with self.assertRaises(ValueError):
            sobol_indices(_linear, [(0, 1), (0, 1)], n=32, confidence=1.5)
        with self.assertRaises(ValueError):
            sobol_indices(_linear, [(0, 1), (0, 1)], n=32, confidence=0.0)

    def test_constant_output_reports_exact_zero_with_zero_uncertainty(self):
        # Negative control for the var<=0 special case: no sampling noise to quantify, so every new
        # field must still be present with a sensible degenerate value, not missing or NaN.
        res = sobol_indices(lambda x: np.ones(len(x)), [(0, 1)] * 2, n=512)
        for key in (
            "S1",
            "ST",
            "S1_clipped",
            "ST_clipped",
            "S1_standard_error",
            "ST_standard_error",
            "S1_ci_low",
            "S1_ci_high",
            "ST_ci_low",
            "ST_ci_high",
        ):
            np.testing.assert_array_equal(res[key], [0.0, 0.0])

    def test_well_resolved_estimate_still_reports_sensible_values(self):
        # Negative control: a clearly-in-bounds, well-estimated index (large n) must not be disturbed
        # by the new machinery -- raw ~= clipped, tight standard error, matches the known analytic
        # value, same as the pre-existing SobolTest assertions.
        res = sobol_indices(ishigami, BOUNDS, n=16384, seed=0, names=["x1", "x2", "x3"])
        np.testing.assert_allclose(res["S1"][:2], res["S1_clipped"][:2], atol=1e-9)  # far from [0,1]'s edges
        np.testing.assert_allclose(res["S1"][:2], [0.314, 0.442], atol=0.03)
        self.assertTrue(np.all(res["S1_standard_error"][:2] < 0.02))  # n=16384 is well-resolved


class MorrisTest(unittest.TestCase):
    def test_ranks_influential_inputs(self):
        m = morris_screening(ishigami, BOUNDS, trajectories=60, seed=1, names=["x1", "x2", "x3"])
        self.assertEqual(m["mu_star"].shape, (3,))
        self.assertTrue(np.all(m["mu_star"] > 0))  # all three move the output (x3 via interaction)


class _PointRecorder:
    """Records every point morris_screening evaluates its model at, for direct trajectory inspection."""

    def __init__(self):
        self.points: list[np.ndarray] = []

    def __call__(self, x):
        self.points.append(x.copy())
        return np.sum(x, axis=1)


class MorrisTrajectoryTest(unittest.TestCase):
    """MXR-080-0194: Morris trajectories must be genuine randomized-direction paths on the
    prescribed grid -- every step exactly ``delta``, direction randomized per dimension, no
    systematic boundary bias, and a positive trajectory budget enforced.

    ``setUpClass`` records one large batch of real trajectories once; the individual tests below
    each inspect a different property of that same recorded data.
    """

    LEVELS = 4
    D = 4
    N_TRAJECTORIES = 400

    @classmethod
    def setUpClass(cls):
        cls.grid = np.linspace(0.0, 1.0, cls.LEVELS)
        cls.delta = cls.LEVELS / (2.0 * (cls.LEVELS - 1))  # = 2/3 for LEVELS=4, the audit's own case
        recorder = _PointRecorder()
        morris_screening(recorder, [(0.0, 1.0)] * cls.D, trajectories=cls.N_TRAJECTORIES, levels=cls.LEVELS, seed=3)
        cls.all_points = np.concatenate(recorder.points, axis=0)  # (N_TRAJECTORIES * (D+1), D)
        cls.pts_per_traj = cls.D + 1
        # (dimension touched, its value before the step, the signed step) for every one-at-a-time
        # move across every recorded trajectory.
        cls.steps: list[tuple[int, float, float]] = []
        for t in range(cls.N_TRAJECTORIES):
            traj = cls.all_points[t * cls.pts_per_traj : (t + 1) * cls.pts_per_traj]
            for k in range(1, cls.pts_per_traj):
                diff = traj[k] - traj[k - 1]
                touched = np.flatnonzero(np.abs(diff) > 1e-12)
                assert len(touched) == 1, f"expected exactly one coordinate to change, got {diff}"
                j = int(touched[0])
                cls.steps.append((j, traj[k - 1, j], diff[j]))

    def test_exactly_one_coordinate_changes_per_step(self):
        # every trajectory must contribute exactly D one-at-a-time steps (setUpClass already asserts
        # "exactly one coordinate changed" per step; this asserts none were silently dropped).
        self.assertEqual(len(self.steps), self.N_TRAJECTORIES * self.D)

    def test_four_level_start_at_two_thirds_takes_a_genuine_two_thirds_step(self):
        # The audit's own example: for levels=4, grid={0, 1/3, 2/3, 1}, delta=2/3. A step taken from
        # a dimension currently at 2/3 must never be silently shrunk to 1/3.
        two_thirds_steps = [abs(step) for _, before, step in self.steps if np.isclose(before, 2.0 / 3.0)]
        self.assertTrue(two_thirds_steps, "fixture should exercise the audit's 2/3-start case")
        for step_mag in two_thirds_steps:
            self.assertAlmostEqual(step_mag, self.delta, places=9)  # never shrunk to 1/3

    def test_every_step_is_exactly_delta_in_magnitude(self):
        for _, _before, step in self.steps:
            self.assertAlmostEqual(abs(step), self.delta, places=9)

    def test_every_point_stays_in_bounds_without_clipping(self):
        self.assertGreaterEqual(float(self.all_points.min()), 0.0)
        self.assertLessEqual(float(self.all_points.max()), 1.0)

    def test_step_direction_is_randomized_both_ways(self):
        signs = np.array([np.sign(step) for _, _before, step in self.steps])
        self.assertIn(1.0, signs)
        self.assertIn(-1.0, signs)
        up_frac = float(np.mean(signs > 0))
        self.assertTrue(0.35 < up_frac < 0.65, f"up/down should be roughly balanced, got {up_frac}")

    def test_visitation_is_approximately_uniform_no_boundary_bias(self):
        # Marginal visitation of each of the `levels` grid points, per dimension, across the whole
        # recorded trajectory set should be close to uniform (1/levels each) -- per this module's own
        # derivation (see morris_screening's docstring) of why the design is unbiased.
        idxs = np.argmin(np.abs(self.all_points[:, :, None] - self.grid[None, None, :]), axis=2)
        for j in range(self.D):
            with self.subTest(dim=j):
                counts = np.bincount(idxs[:, j], minlength=self.LEVELS)
                frac = counts / counts.sum()
                np.testing.assert_allclose(frac, 1.0 / self.LEVELS, atol=0.05)

    def test_zero_trajectories_is_rejected_not_fabricated_zero_importance(self):
        with self.assertRaises(ValueError):
            morris_screening(_linear, [(0, 1), (0, 1)], trajectories=0)

    def test_odd_levels_are_rejected(self):
        # levels=5 has no integer grid-index step matching delta = levels/(2*(levels-1)) exactly.
        with self.assertRaises(ValueError):
            morris_screening(_linear, [(0, 1), (0, 1)], levels=5, trajectories=3)

    def test_exact_elementary_effect_for_an_additive_linear_model(self):
        # Known-exact validation case: a purely additive linear model has no interactions and no
        # nonlinearity, so every one-at-a-time elementary effect for input i is *exactly* its
        # coefficient, regardless of starting point or step direction -- mu_star must equal the
        # coefficients exactly and sigma must be exactly 0 (to floating-point precision), for any
        # trajectory count / levels / seed.
        def linear3(x):
            return 2.0 * x[:, 0] + 5.0 * x[:, 1] - 3.0 * x[:, 2]

        for levels in (2, 4, 6, 8):
            with self.subTest(levels=levels):
                res = morris_screening(linear3, [(0.0, 1.0)] * 3, trajectories=15, levels=levels, seed=levels)
                np.testing.assert_allclose(res["mu_star"], [2.0, 5.0, 3.0], atol=1e-9)
                np.testing.assert_allclose(res["sigma"], [0.0, 0.0, 0.0], atol=1e-9)

    def test_well_configured_run_still_produces_sensible_importance(self):
        # Negative control: a normal Morris run on Ishigami still ranks all three inputs as
        # influential (no fabricated all-zero result, no crash) -- same case as MorrisTest above,
        # re-checked alongside the new algorithmic guarantees.
        m = morris_screening(ishigami, BOUNDS, trajectories=200, levels=4, seed=1, names=["x1", "x2", "x3"])
        self.assertEqual(m["mu_star"].shape, (3,))
        self.assertTrue(np.all(m["mu_star"] > 0))
        self.assertTrue(np.all(np.isfinite(m["sigma"])))


def _linear(x):
    return x[:, 0] + x[:, 1]


def _wrong_cardinality(_x):
    return np.array([1.0, 2.0, 3.0])  # always 3 outputs, regardless of how many input rows were given


def _non_finite_output(x):
    return x[:, 0] + np.nan


class SharedValidationContractTest(unittest.TestCase):
    """MXR-080-0192: bounds/count/name/model-output validation is centralized (``_as_bounds``,
    ``_require_exact_positive_int`` shared with :mod:`mixle.doe.designs`, plus this module's own
    ``_validate_names``/``_eval_model``) and applied identically by every estimator here -- not
    reimplemented (and silently diverging) per function.
    """

    # (estimator, kwargs that make it a cheap-but-otherwise-well-posed call) for a 2D box.
    ESTIMATORS = (
        (sobol_indices, {"n": 32}),
        (morris_screening, {"trajectories": 4}),
        (fast_indices, {"n": 32, "harmonics": 2}),
        (dgsm, {"n": 32}),
    )

    def test_bad_bounds_shape_is_rejected(self):
        for func, kwargs in self.ESTIMATORS:
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func(_linear, [(0, 1, 2), (0, 1, 2)], **kwargs)  # (d, 3), not (d, 2)

    def test_non_finite_bounds_are_rejected(self):
        for func, kwargs in self.ESTIMATORS:
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func(_linear, [(0, np.inf), (0, 1)], **kwargs)

    def test_reversed_bounds_are_rejected(self):
        for func, kwargs in self.ESTIMATORS:
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func(_linear, [(1, 0), (0, 1)], **kwargs)  # lower > upper

    def test_fractional_count_is_rejected(self):
        cases = (
            (sobol_indices, {"n": 8.5}),
            (morris_screening, {"trajectories": 4.5}),
            (fast_indices, {"n": 32.5, "harmonics": 2}),
            (dgsm, {"n": 32.5}),
        )
        for func, kwargs in cases:
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func(_linear, [(0, 1), (0, 1)], **kwargs)

    def test_fractional_harmonics_and_levels_are_also_rejected(self):
        # the estimator-specific extra counts (not just the shared "n") route through the same check.
        with self.assertRaises(ValueError):
            fast_indices(_linear, [(0, 1), (0, 1)], n=32, harmonics=2.5)
        with self.assertRaises(ValueError):
            morris_screening(_linear, [(0, 1), (0, 1)], trajectories=4, levels=3.5)

    def test_wrong_length_names_are_rejected(self):
        for func, kwargs in self.ESTIMATORS:
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func(_linear, [(0, 1), (0, 1)], names=["only_one"], **kwargs)

    def test_wrong_cardinality_model_output_is_rejected(self):
        for func, kwargs in self.ESTIMATORS:
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func(_wrong_cardinality, [(0, 1), (0, 1)], **kwargs)

    def test_non_finite_model_output_is_rejected(self):
        for func, kwargs in self.ESTIMATORS:
            with self.subTest(func=func.__name__), self.assertRaises(ValueError):
                func(_non_finite_output, [(0, 1), (0, 1)], **kwargs)

    def test_zero_count_is_rejected_before_any_computation_not_nan(self):
        # must raise a clean ValueError *before* touching func/np.var -- not warn-and-NaN. Promoting
        # warnings to errors here means any leftover "Mean of empty slice"/"invalid value encountered"
        # RuntimeWarning (the old NaN-index symptom) would itself fail the test.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(ValueError):
                sobol_indices(_linear, [(0, 1), (0, 1)], n=0)
            with self.assertRaises(ValueError):
                morris_screening(_linear, [(0, 1), (0, 1)], trajectories=0)
            with self.assertRaises(ValueError):
                fast_indices(_linear, [(0, 1), (0, 1)], n=0, harmonics=2)
            with self.assertRaises(ValueError):
                dgsm(_linear, [(0, 1), (0, 1)], n=0)

    def test_negative_count_is_also_rejected(self):
        with self.assertRaises(ValueError):
            sobol_indices(_linear, [(0, 1), (0, 1)], n=-8)

    def test_well_formed_inputs_still_work_for_every_estimator(self):
        # Negative control: the new validation must not reject a normal, well-posed call.
        for func, kwargs in self.ESTIMATORS:
            with self.subTest(func=func.__name__):
                res = func(_linear, [(0, 1), (0, 1)], seed=0, **kwargs)
                self.assertEqual(len(res["names"]), 2)
                self.assertEqual(res["names"], ["x0", "x1"])

    def test_well_formed_named_inputs_still_work(self):
        res = sobol_indices(_linear, [(0, 1), (0, 1)], n=32, names=["alpha", "beta"])
        self.assertEqual(res["names"], ["alpha", "beta"])


if __name__ == "__main__":
    unittest.main()
