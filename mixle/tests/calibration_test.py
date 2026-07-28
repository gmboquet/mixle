"""Kennedy-O'Hagan calibration: recover simulator parameters despite model discrepancy (Phase 4)."""

import unittest
import unittest.mock

import numpy as np

from mixle.doe import CalibrationIdentifiabilityError, KOCalibration, calibrate
from mixle.doe.calibrate import _NOISE_VAR_FLOOR, _iid_gaussian_neg_ll  # white-box: MXR-080-0171


def _sim(x, theta):
    return theta[0] + theta[1] * x  # linear simulator eta(x, theta)


TRUE = np.array([2.0, 3.0])


def _fit(delta, noise=0.1, seed=0):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 4, 50)
    y = _sim(x, TRUE) + delta(x) + rng.randn(50) * noise
    return x, y


class KOCalibrationTest(unittest.TestCase):
    def test_recovers_theta_under_smooth_discrepancy(self):
        x, y = _fit(lambda x: 0.8 * np.sin(4 * x))
        ko = calibrate(_sim, x, y, theta0=[0.0, 0.0])
        np.testing.assert_allclose(ko.theta, TRUE, atol=0.25)  # discrepancy absorbed by the GP, not theta

    def test_recovers_theta_under_localized_discrepancy(self):
        x, y = _fit(lambda x: 0.6 * np.exp(-((x - 3) ** 2) / 0.3), noise=0.05)
        ko = calibrate(_sim, x, y, theta0=[0.0, 0.0])
        np.testing.assert_allclose(ko.theta, TRUE, atol=0.25)

    def test_beats_least_squares_when_discrepancy_biases_it(self):
        x, y = _fit(lambda x: 0.8 * np.sin(4 * x))
        ko = calibrate(_sim, x, y, theta0=[0.0, 0.0])
        lsq = calibrate(_sim, x, y, theta0=[0.0, 0.0], discrepancy=False)
        self.assertLessEqual(np.linalg.norm(ko.theta - TRUE), np.linalg.norm(lsq.theta - TRUE) + 1e-9)

    def test_calibrated_prediction_fits_the_data(self):
        x, y = _fit(lambda x: 0.6 * np.exp(-((x - 3) ** 2) / 0.3), noise=0.05)
        ko = calibrate(_sim, x, y, theta0=[0.0, 0.0])
        rmse = np.sqrt(np.mean((ko.predict(x) - y) ** 2))
        self.assertLess(rmse, 0.1)  # simulator + GP discrepancy fits to ~noise

    def test_no_discrepancy_yields_near_zero_amplitude(self):
        x, y = _fit(lambda x: np.zeros_like(x))
        ko = calibrate(_sim, x, y, theta0=[0.0, 0.0])
        self.assertLess(ko.amplitude, 0.1)  # nothing for the GP to explain
        np.testing.assert_allclose(ko.theta, TRUE, atol=0.2)

    def test_predict_without_discrepancy_is_the_pure_simulator(self):
        x, y = _fit(lambda x: 0.5 * np.cos(3 * x), noise=0.08)
        ko = calibrate(_sim, x, y, theta0=[0.0, 0.0])
        np.testing.assert_allclose(ko.predict(x, with_discrepancy=False), _sim(x, ko.theta))


class NoiseLikelihoodConsistencyTest(unittest.TestCase):
    """MXR-080-0171: the no-discrepancy branch's quadratic penalty and log normalizer must share one
    floored noise variance, so the optimizer cannot treat noise -> 0 as a free, unbounded improvement."""

    def test_gaussian_neg_ll_uses_one_floored_variance_in_both_terms(self):
        rng = np.random.RandomState(0)
        r = rng.randn(30) * 0.3  # genuinely nonzero residuals
        for noise in [0.0, 1e-6, 1e-3, 0.1, 0.3, 1.0, 10.0]:
            var = noise**2 + _NOISE_VAR_FLOOR
            expected = 0.5 * np.sum(r**2) / var + 0.5 * len(r) * np.log(var) + 0.5 * len(r) * np.log(2 * np.pi)
            self.assertAlmostEqual(_iid_gaussian_neg_ll(r, noise), expected, places=8)

    def test_zero_noise_is_not_a_spurious_optimum_for_nonzero_residuals(self):
        rng = np.random.RandomState(0)
        r = rng.randn(30) * 0.3
        nll_at_zero = _iid_gaussian_neg_ll(r, 0.0)
        self.assertTrue(np.isfinite(nll_at_zero))  # floored, not an unbounded -inf "win"
        nll_at_true_scale = _iid_gaussian_neg_ll(r, float(np.std(r)))
        self.assertLess(nll_at_true_scale, nll_at_zero)  # the honest noise estimate beats noise=0

    def test_no_discrepancy_does_not_collapse_noise_for_small_but_real_noise(self):
        # Before the fix: this exact scenario (theta0=[0,0], small nonzero true noise) drove the
        # fitted noise to exactly 0.0 via dozens of divide-by-zero/log(0) warnings during optimization.
        x, y = _fit(lambda x: np.zeros_like(x), noise=1e-4)
        ko = calibrate(_sim, x, y, theta0=[0.0, 0.0], discrepancy=False)
        self.assertGreater(ko.noise, 1e-6)  # not collapsed to (near) exactly zero
        self.assertLess(ko.noise, 1e-2)  # and still recognizably small, tracking the true noise scale

    def test_no_discrepancy_still_finds_genuinely_low_noise(self):
        # Negative control: legitimately clean data must still yield a low noise estimate -- the fix
        # must not force noise upward regardless of what the data actually supports.
        x, y = _fit(lambda x: np.zeros_like(x), noise=1e-6)
        ko = calibrate(_sim, x, y, theta0=[0.0, 0.0], discrepancy=False)
        self.assertLess(ko.noise, 1e-3)  # correctly tiny, not inflated by the fix


class CalibrationValidationTest(unittest.TestCase):
    """MXR-080-0172: calibrate() must validate its data/model contract, require real optimizer
    convergence, actually consume ``seed``, and expose the point estimate's asymptotic uncertainty."""

    def setUp(self):
        self.x, self.y = _fit(lambda x: np.zeros_like(x), noise=0.05)

    def test_rejects_x_y_length_mismatch(self):
        with self.assertRaises(ValueError):
            calibrate(_sim, self.x, self.y[:15], theta0=[0.0, 0.0])

    def test_rejects_simulator_output_shape_mismatch(self):
        def bad_shape_sim(x, theta):
            return np.array([theta[0]])  # wrong output shape regardless of x

        with self.assertRaises(ValueError):
            calibrate(bad_shape_sim, self.x, self.y, theta0=[0.0, 0.0])

    def test_rejects_non_finite_y(self):
        y_nan = self.y.copy()
        y_nan[3] = np.nan
        with self.assertRaises(ValueError):
            calibrate(_sim, self.x, y_nan, theta0=[0.0, 0.0])

    def test_rejects_non_finite_x(self):
        x_inf = self.x.copy()
        x_inf[3] = np.inf
        with self.assertRaises(ValueError):
            calibrate(_sim, x_inf, self.y, theta0=[0.0, 0.0])

    def test_rejects_non_finite_theta0(self):
        with self.assertRaises(ValueError):
            calibrate(_sim, self.x, self.y, theta0=[0.0, np.nan])

    def test_zero_lengthscale_is_rejected_not_silently_defaulted(self):
        # Before the fix: `discrepancy_lengthscale=0.0` is falsy, so `0` silently fell through to the
        # same default as `None` -- indistinguishable from "use the default" instead of caller error.
        # (The 0.0 case is rejected before optimization even starts, so a tiny max_iter is fine there;
        # the None case needs a real iteration budget to converge, so it uses the default.)
        with self.assertRaises(ValueError):
            calibrate(_sim, self.x, self.y, theta0=[0.0, 0.0], discrepancy_lengthscale=0.0, max_iter=5)
        ko_default = calibrate(_sim, self.x, self.y, theta0=[0.0, 0.0], discrepancy_lengthscale=None)
        self.assertTrue(np.isfinite(ko_default.lengthscale))  # None still means "use the default"

    def test_rejects_negative_lengthscale(self):
        with self.assertRaises(ValueError):
            calibrate(_sim, self.x, self.y, theta0=[0.0, 0.0], discrepancy_lengthscale=-1.0, max_iter=5)

    def test_rejects_non_positive_max_iter(self):
        for bad in [0, -5]:
            with self.assertRaises(ValueError):
                calibrate(_sim, self.x, self.y, theta0=[0.0, 0.0], max_iter=bad)

    def test_rejects_fractional_max_iter(self):
        with self.assertRaises(ValueError):
            calibrate(_sim, self.x, self.y, theta0=[0.0, 0.0], max_iter=5.5)

    def test_rejects_underidentified_theta(self):
        # 3 free theta parameters, 2 data points: theta cannot be identified regardless of fit quality.
        with self.assertRaises(ValueError):
            calibrate(_sim, np.array([0.0, 1.0]), np.array([1.0, 2.0]), theta0=[0.0, 0.0, 0.0])

    def test_rejects_non_convergent_optimization(self):
        # An absurd starting point with essentially no iteration budget cannot converge from any of
        # the multi-start candidates; before the fix this silently returned the unconverged garbage.
        with self.assertRaises(ValueError):
            calibrate(_sim, self.x, self.y, theta0=[1e6, 1e6], max_iter=1)

    def test_seed_reproducibility(self):
        ko_a = calibrate(_sim, self.x, self.y, theta0=[0.0, 0.0], seed=7)
        ko_b = calibrate(_sim, self.x, self.y, theta0=[0.0, 0.0], seed=7)
        np.testing.assert_array_equal(ko_a.theta, ko_b.theta)
        self.assertEqual(ko_a.noise, ko_b.noise)

    def test_seed_is_actually_consumed(self):
        # Before the fix, `seed` was accepted but never read anywhere in the function body, so
        # identical/different seeds were indistinguishable; np.random.default_rng was never called.
        with unittest.mock.patch("numpy.random.default_rng", wraps=np.random.default_rng) as spy:
            calibrate(_sim, self.x, self.y, theta0=[0.0, 0.0], seed=4321, discrepancy=False)
        seeds_used = [c.args[0] for c in spy.call_args_list]
        self.assertIn(4321, seeds_used)

    def test_theta_uncertainty_is_present_and_reasonable(self):
        ko = calibrate(_sim, self.x, self.y, theta0=[0.0, 0.0], discrepancy=False)
        self.assertTrue(np.all(np.isfinite(ko.theta_standard_error)))
        self.assertTrue(np.all(ko.theta_standard_error > 0))
        self.assertTrue(np.all(ko.theta_ci_low <= ko.theta) and np.all(ko.theta <= ko.theta_ci_high))

    def test_theta_confidence_intervals_are_calibrated(self):
        # The asymptotic 95% CI should cover the true theta in most (not necessarily all) independent
        # fits -- a coverage check, not just "the field exists". Loose floor keeps this non-flaky.
        covered = 0
        trials = 10
        for seed in range(trials):
            x = np.linspace(0, 4, 50)
            y = _sim(x, TRUE) + np.random.RandomState(seed + 100).randn(50) * 0.1
            ko = calibrate(_sim, x, y, theta0=[0.0, 0.0], discrepancy=False, seed=seed)
            if np.all(ko.theta_ci_low <= TRUE) and np.all(TRUE <= ko.theta_ci_high):
                covered += 1
        self.assertGreaterEqual(covered, 7)  # ~95% nominal; loose floor keeps this non-flaky

    def test_well_posed_calibration_still_returns_correct_point_estimate(self):
        # Negative control: all the new validation/convergence strictness must not break a normal fit.
        x, y = _fit(lambda x: 0.6 * np.exp(-((x - 3) ** 2) / 0.3), noise=0.05)
        ko = calibrate(_sim, x, y, theta0=[0.0, 0.0])
        np.testing.assert_allclose(ko.theta, TRUE, atol=0.25)


class CalibrationStateAndIdentifiabilityTest(unittest.TestCase):
    def test_fitted_arrays_are_detached_and_public_views_cannot_rewrite_state(self):
        theta = np.array([1.0])
        x = np.array([0.0, 1.0, 2.0])
        y = theta[0] + x + np.array([0.1, -0.1, 0.05])

        def simulator(points, parameters):
            return parameters[0] + np.asarray(points)

        result = KOCalibration(theta, 1.0, 0.5, 0.1, simulator, x, y, theta_standard_error=[0.2])
        before = result.predict(np.array([0.5, 1.5]))
        theta[:] = 99.0
        x[:] = 99.0
        y[:] = 99.0
        public_theta = result.theta
        public_theta[:] = -99.0
        public_se = result.theta_standard_error
        public_se[:] = -99.0
        np.testing.assert_allclose(result.predict(np.array([0.5, 1.5])), before)
        np.testing.assert_array_equal(result.theta, [1.0])
        np.testing.assert_array_equal(result.theta_standard_error, [0.2])

    def test_prediction_uses_the_same_floored_noise_covariance_as_fitting(self):
        x = np.array([0.0, 0.0])
        y = np.array([1.0, 2.0])

        def simulator(points, parameters):
            return np.full(np.asarray(points).shape[0], parameters[0])

        result = KOCalibration([1.0], 1.0, 1.0, 0.0, simulator, x, y)
        prediction = result.predict(np.array([0.0]))
        covariance = np.ones((2, 2)) + _NOISE_VAR_FLOOR * np.eye(2)
        expected = 1.0 + np.ones((1, 2)) @ np.linalg.solve(covariance, y - 1.0)
        self.assertTrue(np.all(np.isfinite(prediction)))
        np.testing.assert_allclose(prediction, expected, rtol=1e-7, atol=1e-7)
        self.assertEqual(result.effective_noise_variance, _NOISE_VAR_FLOOR)

    def test_constant_parameter_direction_is_rejected_with_exposed_null_space(self):
        x = np.linspace(0.0, 1.0, 12)
        y = 2.0 + 3.0 * x

        def partially_constant(points, parameters):
            return parameters[0] + 3.0 * np.asarray(points)  # parameters[1] has no effect

        with self.assertRaises(CalibrationIdentifiabilityError) as context:
            calibrate(partially_constant, x, y, theta0=[0.0, 0.0], discrepancy=False)
        error = context.exception
        self.assertEqual(error.rank, 1)
        self.assertEqual(error.n_parameters, 2)
        self.assertEqual(error.non_identifiable_directions.shape, (1, 2))
        self.assertGreater(abs(error.non_identifiable_directions[0, 1]), 0.99)

    def test_well_identified_result_publishes_sensitivity_evidence(self):
        x, y = _fit(lambda values: np.zeros_like(values), noise=0.05)
        result = calibrate(_sim, x, y, theta0=[0.0, 0.0], discrepancy=False)
        self.assertTrue(result.identifiable)
        self.assertEqual(result.sensitivity_rank, 2)
        self.assertEqual(result.sensitivity_singular_values.shape, (2,))
        self.assertTrue(np.all(np.isfinite(result.sensitivity_singular_values)))
        self.assertTrue(np.isfinite(result.sensitivity_condition_number))


if __name__ == "__main__":
    unittest.main()
