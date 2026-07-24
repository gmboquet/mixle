"""Kennedy-O'Hagan calibration: recover simulator parameters despite model discrepancy (Phase 4)."""

import unittest

import numpy as np

from mixle.doe import calibrate
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


if __name__ == "__main__":
    unittest.main()
