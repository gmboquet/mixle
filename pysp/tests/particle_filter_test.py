"""WS-7: bootstrap particle filter (SMC), checked vs the exact Kalman filter on a linear-Gaussian model."""

import unittest

import numpy as np

from pysp.inference.mcmc import particle_filter


def _kalman(ys, A, Q, H, R, m0, P0):
    m, P = np.asarray(m0, float).copy(), np.asarray(P0, float).copy()
    means, loglik = [], 0.0
    for y in ys:
        m = A @ m
        P = A @ P @ A.T + Q
        S = H @ P @ H.T + R
        innov = np.atleast_1d(y) - H @ m
        loglik += -0.5 * (innov @ np.linalg.solve(S, innov) + np.log(np.linalg.det(2 * np.pi * S)))
        K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ innov
        P = P - K @ H @ P
        means.append(m.copy())
    return np.array(means), loglik


class ParticleFilterTest(unittest.TestCase):
    def test_matches_kalman(self):
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        Q = 0.05 * np.eye(2)
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])
        rng = np.random.RandomState(1)
        x = np.array([1.0, -0.5])
        ys = []
        Lq = np.linalg.cholesky(Q)
        for _ in range(30):
            x = A @ x + Lq @ rng.standard_normal(2)
            ys.append(H @ x + np.sqrt(0.1) * rng.standard_normal(1))
        ys = np.array(ys)
        km, kll = _kalman(ys, A, Q, H, R, [0, 0], np.eye(2))

        def propagate(p, r):
            return (A @ p.T).T + r.standard_normal(p.shape) @ Lq.T

        rinv = np.linalg.inv(R)

        def loglik(p, y):
            resid = np.atleast_1d(y) - (H @ p.T).T
            return -0.5 * np.einsum("ij,jk,ik->i", resid, rinv, resid)

        p0 = np.random.RandomState(0).multivariate_normal([0, 0], np.eye(2), 60000)
        pm, pll = particle_filter(ys, propagate, loglik, p0, rng=np.random.RandomState(0))
        # filtered means converge to the exact Kalman filter (Monte Carlo error ~ 1/sqrt(N))
        self.assertLess(np.mean(np.abs(pm - km)), 0.03)  # tight on the average over all steps
        self.assertLess(np.max(np.abs(pm - km)), 0.12)   # even the worst step is close
        self.assertEqual(pm.shape, km.shape)
        self.assertTrue(np.isfinite(pll))


if __name__ == "__main__":
    unittest.main()
