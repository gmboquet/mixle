"""WS-1/WS-7: reflective HMC for box constraints, checked vs the truncated normal."""

import unittest

import numpy as np
import scipy.stats as ss

from pysp.inference.mcmc import reflective_hmc


def _samples(res):
    return np.array([np.atleast_1d(v) for v in res.samples])


class ReflectiveHMCTest(unittest.TestCase):
    def test_truncated_normal_1d(self):
        a, b = -0.5, 1.5
        res = reflective_hmc(
            lambda x: -0.5 * np.dot(x, x),
            lambda x: -np.asarray(x),
            [0.5],
            [a],
            [b],
            num_samples=20000,
            step_size=0.3,
            num_steps=15,
            rng=np.random.RandomState(0),
        )
        s = _samples(res)[2000:, 0]
        tn = ss.truncnorm(a, b)
        self.assertGreaterEqual(s.min(), a - 1e-9)
        self.assertLessEqual(s.max(), b + 1e-9)
        self.assertAlmostEqual(s.mean(), tn.mean(), delta=0.02)
        self.assertAlmostEqual(s.std(), tn.std(), delta=0.02)

    def test_box_2d_stays_inside(self):
        res = reflective_hmc(
            lambda x: -0.5 * np.dot(x, x),
            lambda x: -np.asarray(x),
            [0.0, 0.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            num_samples=15000,
            step_size=0.3,
            num_steps=15,
            rng=np.random.RandomState(1),
        )
        s = _samples(res)[2000:]
        self.assertTrue(np.all(s >= -1.0 - 1e-9) and np.all(s <= 1.0 + 1e-9))
        self.assertTrue(np.allclose(s.mean(axis=0), 0.0, atol=0.03))  # symmetric box -> mean 0
        self.assertAlmostEqual(s[:, 0].std(), ss.truncnorm(-1, 1).std(), delta=0.03)

    def test_initial_outside_box_raises(self):
        with self.assertRaises(ValueError):
            reflective_hmc(
                lambda x: 0.0,
                lambda x: np.zeros_like(x),
                [2.0],
                [-1.0],
                [1.0],
                num_samples=1,
                step_size=0.1,
                num_steps=5,
            )


if __name__ == "__main__":
    unittest.main()
