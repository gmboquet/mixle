"""WS-1: generic scalar hyperbolic conservation law (Rusanov) -- shocks + advection."""

import unittest

import numpy as np

from pysp.ppl.physics.dynamics import conservation_law_rhs, integrate_adaptive


class ConservationLawTest(unittest.TestCase):
    def test_burgers_shock_rankine_hugoniot(self):
        # Riemann problem u_l=1, u_r=0 for inviscid Burgers -> shock at speed (u_l+u_r)/2 = 0.5
        x = np.linspace(0.0, 20.0, 800)
        dx = x[1] - x[0]
        u0 = np.where(x < 8.0, 1.0, 0.0)
        rhs = conservation_law_rhs(lambda u: 0.5 * u * u, np.abs, dx, bc="outflow")
        t = 4.0
        sol = integrate_adaptive(rhs, u0, [t], rtol=1e-6, atol=1e-8)[-1]
        shock_x = x[np.argmin(np.abs(sol - 0.5))]
        self.assertAlmostEqual(shock_x, 8.0 + 0.5 * t, delta=0.2)  # Rankine-Hugoniot shock speed
        self.assertLessEqual(sol.max(), 1.0 + 1e-6)                # no spurious overshoot
        self.assertGreaterEqual(sol.min(), -1e-6)

    def test_linear_advection_transports(self):
        c = 1.5
        x = np.linspace(0.0, 20.0, 800)
        dx = x[1] - x[0]
        u0 = np.exp(-((x - 5.0) / 0.7) ** 2)
        rhs = conservation_law_rhs(lambda u: c * u, lambda u: np.full_like(u, abs(c)), dx, bc="outflow")
        sol = integrate_adaptive(rhs, u0, [3.0], rtol=1e-7, atol=1e-9)[-1]
        self.assertAlmostEqual(x[np.argmax(sol)], 5.0 + c * 3.0, delta=0.2)  # peak moved at speed c

    def test_periodic_mass_conservation(self):
        x = np.linspace(0.0, 2 * np.pi, 200, endpoint=False)
        dx = x[1] - x[0]
        u0 = 1.0 + 0.5 * np.sin(x)
        rhs = conservation_law_rhs(lambda u: 0.5 * u * u, np.abs, dx, bc="periodic")
        sol = integrate_adaptive(rhs, u0, [0.3], rtol=1e-7, atol=1e-9)[-1]
        self.assertAlmostEqual(sol.sum() * dx, u0.sum() * dx, places=4)


if __name__ == "__main__":
    unittest.main()
