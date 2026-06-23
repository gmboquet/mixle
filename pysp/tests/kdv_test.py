"""WS-1: KdV equation (dispersive method of lines), checked vs the analytic soliton."""

import unittest

import numpy as np

from pysp.ppl.physics.dynamics import integrate_adaptive, kdv_rhs


class KdVTest(unittest.TestCase):
    def test_soliton_propagates_at_speed_c(self):
        c, L, n, x0 = 4.0, 40.0, 256, 10.0
        x = np.linspace(0.0, L, n, endpoint=False)
        dx = x[1] - x[0]

        def soliton(t):
            xi = (x - c * t - x0 + L / 2) % L - L / 2  # periodic distance to the moving centre
            return (c / 2) / np.cosh(np.sqrt(c) / 2 * xi) ** 2

        rhs = kdv_rhs(dx, nonlinearity=6.0, dispersion=1.0)
        sol = integrate_adaptive(rhs, soliton(0.0), [1.0], rtol=1e-6, atol=1e-8)[-1]
        self.assertLess(np.sqrt(np.mean((sol - soliton(1.0)) ** 2)), 1e-2)   # tracks the moved soliton
        self.assertAlmostEqual(np.max(sol), c / 2, delta=0.05)               # amplitude preserved
        self.assertGreater(np.sqrt(np.mean((sol - soliton(0.0)) ** 2)), 0.1)  # it actually moved

    def test_conserves_mass_and_momentum(self):
        x = np.linspace(0.0, 40.0, 256, endpoint=False)
        dx = x[1] - x[0]
        u0 = 2.0 / np.cosh(x - 20.0) ** 2
        rhs = kdv_rhs(dx)
        sol = integrate_adaptive(rhs, u0, [0.5], rtol=1e-7, atol=1e-9)[-1]
        self.assertAlmostEqual(sol.sum() * dx, u0.sum() * dx, places=4)          # mass  int u
        self.assertAlmostEqual((sol**2).sum() * dx, (u0**2).sum() * dx, places=3)  # momentum int u^2


if __name__ == "__main__":
    unittest.main()
