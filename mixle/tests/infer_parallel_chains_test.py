"""Tests for parallel multi-chain NUTS in mixle.inference (WS-N / distributed MCMC)."""

import unittest

import numpy as np

from mixle.inference import nuts


def _std_normal_value_and_grad(theta):
    # log N(0, I): value and gradient (the numpy/numba contract is a fused value_and_grad).
    return -0.5 * float(np.dot(theta, theta)), -theta


class ParallelChainsTest(unittest.TestCase):
    def test_thread_parallel_matches_serial_shape_and_recovers_posterior(self):
        kw = dict(dim=3, num_samples=400, warmup=400, chains=4, backend="numpy")
        serial = nuts(_std_normal_value_and_grad, rng=0, **kw)
        threaded = nuts(_std_normal_value_and_grad, rng=0, parallel="thread", **kw)
        # Same pooled shape and per-chain structure.
        self.assertEqual(serial.samples.shape, threaded.samples.shape)
        self.assertEqual(threaded.chains.shape, (4, 400, 3))
        # Both recover the standard-normal posterior (mean ~ 0, var ~ 1).
        for res in (serial, threaded):
            self.assertTrue(np.all(np.abs(res.samples.mean(axis=0)) < 0.25))
            self.assertTrue(np.all(np.abs(res.samples.var(axis=0) - 1.0) < 0.4))
        # R-hat is computed across the (real, independent) chains.
        self.assertEqual(threaded.rhat.shape, (3,))
        self.assertTrue(np.all(np.abs(threaded.rhat - 1.0) < 0.2))

    def test_process_parallel_runs_with_picklable_target(self):
        # `_std_normal_value_and_grad` is module-level (picklable), so the process pool works.
        res = nuts(
            _std_normal_value_and_grad,
            dim=2,
            num_samples=300,
            warmup=300,
            chains=2,
            backend="numpy",
            parallel=True,
            rng=1,
        )
        self.assertEqual(res.chains.shape, (2, 300, 2))
        self.assertTrue(np.all(np.abs(res.samples.mean(axis=0)) < 0.3))

    def test_invalid_parallel_mode_raises(self):
        with self.assertRaises(ValueError):
            nuts(_std_normal_value_and_grad, dim=2, chains=2, parallel="banana", num_samples=10, warmup=10)


if __name__ == "__main__":
    unittest.main()
