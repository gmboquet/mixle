"""Noise-robust incumbent selection for BO (mixle.doe.robust): under a noisy objective, reporting the
posterior-mean incumbent recovers the true optimum better than argmin-of-observed, which chases noise.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")

from mixle.doe.bayesopt import minimize  # noqa: E402
from mixle.doe.robust import noisy_minimize, posterior_incumbent  # noqa: E402

_TARGET = np.array([1.0, -0.5])
_BOUNDS = [(-3.0, 3.0), (-3.0, 3.0)]


def _noisy_bowl(noise_std, seed):
    rng = np.random.RandomState(seed)

    def obj(x):
        clean = float(np.sum((np.asarray(x, dtype=float) - _TARGET) ** 2))
        return clean + rng.normal(0.0, noise_std)

    return obj


class PosteriorIncumbentTest(unittest.TestCase):
    def test_deterministic_case_reduces_to_argmin(self):
        rng = np.random.RandomState(0)
        x = rng.uniform(-3, 3, size=(15, 2))
        y = np.sum((x - _TARGET) ** 2, axis=1)  # noiseless
        inc = posterior_incumbent(x, y, maximize=False)
        # with no noise the posterior-mean incumbent is (essentially) the observed argmin
        self.assertLess(float(np.linalg.norm(inc.best_x - x[int(np.argmin(y))])), 0.5)

    def test_validates_shapes(self):
        with self.assertRaises(ValueError):
            posterior_incumbent(np.zeros((3, 2)), np.zeros(2))
        with self.assertRaises(ValueError):
            posterior_incumbent(np.zeros((0, 2)), np.zeros(0))


class NoisyIncumbentBeatsArgminTest(unittest.TestCase):
    def test_posterior_incumbent_is_closer_to_the_true_optimum_under_noise(self):
        seeds = range(6)
        noise_std = 1.5  # comparable to the objective's spread -> argmin-of-observed gets fooled
        argmin_err, robust_err = [], []
        for s in seeds:
            res = minimize(_noisy_bowl(noise_std, seed=100 + s), _BOUNDS, n_init=6, n_iter=10, seed=s)
            argmin_x = res.x[int(np.argmin(res.y))]  # what plain minimize reports
            robust_x = posterior_incumbent(res.x, res.y, maximize=False).best_x  # the robust rule
            argmin_err.append(float(np.linalg.norm(argmin_x - _TARGET)))
            robust_err.append(float(np.linalg.norm(robust_x - _TARGET)))
        # averaged over seeds, the denoised incumbent lands closer to the true optimum
        self.assertLess(float(np.mean(robust_err)), float(np.mean(argmin_err)))

    def test_noisy_minimize_reports_the_robust_incumbent_end_to_end(self):
        res = noisy_minimize(_noisy_bowl(1.0, seed=42), _BOUNDS, n_init=6, n_iter=12, seed=1)
        # best_x is a point actually evaluated (the believed optimum among them), history preserved
        self.assertEqual(res.x.shape[1], 2)
        self.assertEqual(len(res.y), res.x.shape[0])
        self.assertTrue(np.any(np.all(np.isclose(res.x, res.best_x), axis=1)))

    def test_deterministic_given_seed(self):
        a = noisy_minimize(_noisy_bowl(1.0, seed=5), _BOUNDS, n_init=5, n_iter=8, seed=3)
        b = noisy_minimize(_noisy_bowl(1.0, seed=5), _BOUNDS, n_init=5, n_iter=8, seed=3)
        np.testing.assert_allclose(a.best_x, b.best_x)
        np.testing.assert_allclose(a.y, b.y)


if __name__ == "__main__":
    unittest.main()
