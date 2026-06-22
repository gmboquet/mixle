"""WS-1: viscous Burgers' equation (method of lines), checked vs the analytic travelling wave."""

import unittest

import numpy as np

from pysp.ppl.dynamics import burgers_rhs, integrate_adaptive


class BurgersTest(unittest.TestCase):
    def test_travelling_wave(self):
        uL, uR, nu = 1.0, 0.0, 0.08
        s = 0.5 * (uL + uR)
        x = np.linspace(-6.0, 6.0, 240)
        dx = x[1] - x[0]

        def wave(t):
            return 0.5 * (uL + uR) - 0.5 * (uL - uR) * np.tanh((uL - uR) * (x - s * t) / (4 * nu))

        rhs = burgers_rhs(nu, dx, bc="dirichlet")
        sol = integrate_adaptive(rhs, wave(0.0), [3.0], rtol=1e-7, atol=1e-9)[-1]
        # matching the wave shifted by s*t (not the t=0 profile) certifies the front travelled correctly
        self.assertLess(np.sqrt(np.mean((sol - wave(3.0)) ** 2)), 5e-3)
        self.assertGreater(np.sqrt(np.mean((sol - wave(0.0)) ** 2)), 0.05)  # and it moved off the initial profile

    def test_periodic_conserves_mass(self):
        x = np.linspace(0.0, 2 * np.pi, 128, endpoint=False)
        dx = x[1] - x[0]
        u0 = np.sin(x)
        rhs = burgers_rhs(0.1, dx, bc="periodic")
        sol = integrate_adaptive(rhs, u0, [0.5, 1.0], rtol=1e-8, atol=1e-10)
        # periodic Burgers conserves the integral of u (no flux through the boundary)
        self.assertAlmostEqual(sol[-1].sum() * dx, u0.sum() * dx, places=4)


if __name__ == "__main__":
    unittest.main()
