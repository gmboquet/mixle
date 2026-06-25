"""Proper scoring rules (pysp.inference.scoring)."""

import unittest

import numpy as np

from pysp.inference import (
    brier_decomposition,
    brier_score,
    crps_ensemble,
    crps_gaussian,
    energy_score,
    interval_score,
    log_score,
    pinball_loss,
    skill_score,
    winkler_score,
)


class LogScoreTest(unittest.TestCase):
    def test_matches_negative_log(self):
        p = np.array([0.5, 0.25, 1.0])
        self.assertAlmostEqual(log_score(p), float(np.mean(-np.log(p))))

    def test_zero_probability_is_finite(self):
        self.assertTrue(np.isfinite(log_score(np.array([0.0]))))

    def test_proper_minimised_at_truth(self):
        # expected log score over Bernoulli(0.7) is minimised by reporting 0.7
        rng = np.random.RandomState(0)
        y = (rng.rand(20000) < 0.7).astype(float)
        grid = np.linspace(0.05, 0.95, 19)
        losses = [log_score(np.where(y == 1, q, 1 - q)) for q in grid]
        self.assertAlmostEqual(grid[int(np.argmin(losses))], 0.7, delta=0.06)


class BrierScoreTest(unittest.TestCase):
    def test_binary_known_value(self):
        p = np.array([0.9, 0.2, 0.6])
        y = np.array([1, 0, 1])
        self.assertAlmostEqual(brier_score(p, y), float(np.mean((p - y) ** 2)))

    def test_perfect_forecast_is_zero(self):
        y = np.array([1, 0, 1, 0])
        self.assertEqual(brier_score(y.astype(float), y), 0.0)

    def test_multiclass_labels_and_onehot_agree(self):
        rng = np.random.RandomState(1)
        logits = rng.randn(50, 3)
        p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        labels = rng.randint(0, 3, size=50)
        onehot = np.eye(3)[labels]
        self.assertAlmostEqual(brier_score(p, labels), brier_score(p, onehot))

    def test_decomposition_sums_to_brier(self):
        rng = np.random.RandomState(2)
        p = rng.rand(5000)
        y = (rng.rand(5000) < p).astype(float)  # perfectly reliable forecaster
        d = brier_decomposition(p, y, bins=10)
        self.assertAlmostEqual(d["reliability"] - d["resolution"] + d["uncertainty"], d["brier"], places=12)
        # a calibrated forecaster has near-zero reliability
        self.assertLess(d["reliability"], 0.01)


class CRPSTest(unittest.TestCase):
    def test_point_forecast_reduces_to_absolute_error(self):
        # a degenerate ensemble (all members equal) gives |x - y|
        x = np.full((1, 50), 2.0)
        self.assertAlmostEqual(crps_ensemble(x, np.array([5.0])), 3.0)

    def test_gaussian_closed_form_matches_large_ensemble(self):
        rng = np.random.RandomState(3)
        mu, sigma, y = 1.0, 2.0, 1.5
        draws = rng.normal(mu, sigma, size=(1, 200000))
        approx = crps_ensemble(draws, np.array([y]))
        exact = crps_gaussian(np.array([mu]), np.array([sigma]), np.array([y]))
        self.assertAlmostEqual(approx, exact, delta=0.02)

    def test_gaussian_crps_at_mean(self):
        # CRPS(N(0,1), 0) = 1/sqrt(pi) * (2 phi(0) - ... ) closed form value
        val = crps_gaussian(np.array([0.0]), np.array([1.0]), np.array([0.0]))
        self.assertAlmostEqual(val, 2.0 / np.sqrt(2 * np.pi) - 1.0 / np.sqrt(np.pi), places=10)

    def test_sharper_calibrated_forecast_scores_lower(self):
        rng = np.random.RandomState(4)
        y = rng.normal(0.0, 1.0, size=2000)
        sharp = crps_gaussian(np.zeros_like(y), np.ones_like(y), y)
        diffuse = crps_gaussian(np.zeros_like(y), 3.0 * np.ones_like(y), y)
        self.assertLess(sharp, diffuse)


class IntervalScoreTest(unittest.TestCase):
    def test_inside_interval_is_just_width(self):
        s = interval_score(np.array([0.0]), np.array([2.0]), np.array([1.0]), alpha=0.1)
        self.assertAlmostEqual(s, 2.0)

    def test_miss_adds_scaled_penalty(self):
        # y above upper by 1, alpha=0.1 -> width 2 + (2/0.1)*1 = 22
        s = interval_score(np.array([0.0]), np.array([2.0]), np.array([3.0]), alpha=0.1)
        self.assertAlmostEqual(s, 2.0 + 20.0)

    def test_winkler_is_alias(self):
        args = (np.array([0.0]), np.array([2.0]), np.array([3.0]), 0.2)
        self.assertEqual(interval_score(*args), winkler_score(*args))

    def test_tighter_interval_at_matched_coverage_wins(self):
        rng = np.random.RandomState(5)
        y = rng.normal(0.0, 1.0, size=4000)
        # both ~90% central intervals; the calibrated-width one is tighter
        tight = interval_score(-1.645 * np.ones_like(y), 1.645 * np.ones_like(y), y, alpha=0.1)
        wide = interval_score(-3.0 * np.ones_like(y), 3.0 * np.ones_like(y), y, alpha=0.1)
        self.assertLess(tight, wide)


class PinballTest(unittest.TestCase):
    def test_median_minimises_tau_half(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        med = np.median(y)
        grid = np.linspace(1.0, 10.0, 91)
        losses = [pinball_loss(np.full_like(y, q), y, 0.5) for q in grid]
        self.assertAlmostEqual(grid[int(np.argmin(losses))], med, delta=0.2)

    def test_known_value(self):
        # tau=0.9, pred=0, y=1 -> 0.9*1 ; y=-1 -> 0.1*1
        self.assertAlmostEqual(pinball_loss(np.array([0.0]), np.array([1.0]), 0.9), 0.9)
        self.assertAlmostEqual(pinball_loss(np.array([0.0]), np.array([-1.0]), 0.9), 0.1)

    def test_multi_level(self):
        pred = np.array([[0.0, 0.0]])
        loss = pinball_loss(pred, np.array([1.0]), np.array([0.1, 0.9]), mean=False)
        np.testing.assert_allclose(loss, [[0.1, 0.9]])


class EnergyScoreTest(unittest.TestCase):
    def test_reduces_to_crps_in_1d(self):
        rng = np.random.RandomState(6)
        draws = rng.normal(0.0, 1.0, size=(2000,))
        es = energy_score(draws[:, None], np.array([0.5]))
        crps = crps_ensemble(draws[None, :], np.array([0.5]))
        self.assertAlmostEqual(es, crps, places=10)


class SkillScoreTest(unittest.TestCase):
    def test_perfect_and_reference(self):
        self.assertAlmostEqual(skill_score(0.0, 2.0), 1.0)
        self.assertAlmostEqual(skill_score(2.0, 2.0), 0.0)
        self.assertLess(skill_score(3.0, 2.0), 0.0)


if __name__ == "__main__":
    unittest.main()
