"""WS-1: semi-explicit index-1 DAE solver (mass-matrix SDIRK2), checked vs analytic + the ODE limit."""

import unittest

import numpy as np

from pysp.ppl.physics.dynamics import integrate_dae, integrate_stiff


class DAETest(unittest.TestCase):
    def test_index1_dae_matches_analytic(self):
        # y1' = -y1 + y2 ; 0 = y1 - y2 + sin(t)  => y2 = y1 + sin(t), y1' = sin(t) => y1 = 1 - cos t
        mass = np.array([[1.0, 0.0], [0.0, 0.0]])

        def rhs(t, y):
            return [-y[0] + y[1], y[0] - y[1] + np.sin(t)]

        te = np.array([0.5, 1.0, 2.0, 3.0])
        sol = integrate_dae(rhs, [0.0, 0.0], te, mass)  # consistent IC: y2(0)=y1(0)+sin 0=0
        exact = np.array([[1 - np.cos(tt), 1 - np.cos(tt) + np.sin(tt)] for tt in te])
        self.assertTrue(np.allclose(sol, exact, atol=1e-4))
        # the algebraic constraint holds along the solution
        self.assertTrue(np.allclose(sol[:, 0] - sol[:, 1] + np.sin(te), 0.0, atol=1e-4))

    def test_identity_mass_reduces_to_stiff_ode(self):
        a = np.array([[-100.0, 1.0], [0.0, -1.0]])
        te = np.array([0.5, 1.0])
        dae = integrate_dae(lambda t, y: a @ y, [1.0, 1.0], te, np.eye(2), jac=lambda t, y: a, h_max=0.02)
        ode = integrate_stiff(lambda t, y: a @ y, [1.0, 1.0], te, jac=lambda t, y: a, h_max=0.02)
        self.assertTrue(np.allclose(dae, ode, atol=1e-8))  # identity mass == the SDIRK2 stiff ODE


if __name__ == "__main__":
    unittest.main()
