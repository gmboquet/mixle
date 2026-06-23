"""WS-1: adaptive Dormand-Prince RK45 ODE integrator, checked vs scipy + analytic solutions."""

import unittest

import numpy as np
from scipy.integrate import solve_ivp

from pysp.ppl.physics.dynamics import integrate_adaptive


class AdaptiveODETest(unittest.TestCase):
    def test_matches_analytic(self):
        te = np.linspace(0.0, 3.0, 8)[1:]
        # dy/dt = -2y -> exp(-2t)
        y = integrate_adaptive(lambda t, v: -2.0 * np.asarray(v), [1.0], te, rtol=1e-10, atol=1e-12)
        self.assertTrue(np.allclose(y[:, 0], np.exp(-2.0 * te), atol=1e-8))
        # harmonic oscillator y'' = -y -> [cos t, -sin t]
        y2 = integrate_adaptive(lambda t, v: [v[1], -v[0]], [1.0, 0.0], te, rtol=1e-10, atol=1e-12)
        self.assertTrue(np.allclose(y2[:, 0], np.cos(te), atol=1e-7))
        self.assertTrue(np.allclose(y2[:, 1], -np.sin(te), atol=1e-7))

    def test_matches_scipy_on_nonlinear(self):
        def vdp(t, v):
            return [v[1], 3.0 * (1 - v[0] ** 2) * v[1] - v[0]]

        te = np.linspace(0.0, 6.0, 15)[1:]
        mine = integrate_adaptive(vdp, [2.0, 0.0], te, rtol=1e-9, atol=1e-11)
        ref = solve_ivp(vdp, [0.0, 6.0], [2.0, 0.0], t_eval=te, rtol=1e-11, atol=1e-13, method="RK45").y.T
        self.assertTrue(np.allclose(mine, ref, atol=1e-7))

    def test_shape_and_endpoints(self):
        te = np.array([0.5, 1.0, 2.0])
        y = integrate_adaptive(lambda t, v: -np.asarray(v), [3.0], te)
        self.assertEqual(y.shape, (3, 1))
        self.assertAlmostEqual(y[-1, 0], 3.0 * np.exp(-2.0), places=5)


if __name__ == "__main__":
    unittest.main()
