"""Kennedy-O'Hagan calibration: recover simulator parameters despite model discrepancy (Phase 4)."""

import unittest

import numpy as np

from mixle.doe import calibrate


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


if __name__ == "__main__":
    unittest.main()
