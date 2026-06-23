"""WS-1: L-stable implicit stiff ODE integrator (SDIRK2), checked vs analytic + scipy."""

import unittest

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pysp.ppl.physics.dynamics import integrate_stiff


class StiffODETest(unittest.TestCase):
    def test_linear_stiff_matches_matrix_exponential(self):
        a = np.array([[-1000.0, 1.0], [0.0, -1.0]])  # eigenvalues -1000 (stiff) and -1
        y0 = [1.0, 1.0]
        te = np.linspace(0.0, 2.0, 6)[1:]
        mine = integrate_stiff(lambda t, y: a @ y, y0, te, jac=lambda t, y: a)
        exact = np.array([expm(a * tt) @ y0 for tt in te])
        self.assertTrue(np.allclose(mine, exact, atol=1e-3))

    def test_nonlinear_stiff_matches_scipy(self):
        def f(t, y):
            return [-50.0 * (y[0] - np.cos(t)), y[0] - y[1]]

        te = np.linspace(0.0, 3.0, 8)[1:]
        mine = integrate_stiff(f, [0.0, 0.0], te)  # finite-difference Jacobian
        ref = solve_ivp(f, [0.0, 3.0], [0.0, 0.0], t_eval=te, method="Radau", rtol=1e-11, atol=1e-13).y.T
        self.assertTrue(np.allclose(mine, ref, atol=1e-3))

    def test_L_stable_on_extreme_stiffness(self):
        # lambda = -1e6: an L-stable method damps the fast mode to ~0 without blowing up at moderate steps
        big = np.array([[-1.0e6, 0.0], [0.0, -1.0]])
        y = integrate_stiff(lambda t, v: big @ v, [1.0, 1.0], [2.0], jac=lambda t, v: big)[0]
        self.assertLess(abs(y[0]), 1e-6)                       # stiff mode damped
        self.assertAlmostEqual(y[1], np.exp(-2.0), places=4)   # slow mode accurate


if __name__ == "__main__":
    unittest.main()
