"""WS-14: Lane-Emden polytrope equation, validated against the analytic n=0,1,5 solutions."""

import unittest

import numpy as np

from pysp.ppl.physics.dynamics import integrate_adaptive, lane_emden_rhs

_ANALYTIC = {0: lambda x: 1 - x**2 / 6, 1: lambda x: np.sin(x) / x, 5: lambda x: 1.0 / np.sqrt(1 + x**2 / 3)}


class LaneEmdenTest(unittest.TestCase):
    def test_matches_analytic_solutions(self):
        xi0 = 1e-4
        for n, upper in [(0, 2.4), (1, 3.0), (5, 6.0)]:  # n=0 truncated just before its zero xi1=sqrt(6)
            y0 = np.array([1 - xi0**2 / 6, -xi0 / 3])  # series start off the regular singular point
            xs = np.linspace(xi0, upper, 60)
            sol = integrate_adaptive(lane_emden_rhs(n), y0, xs, t0=xi0, rtol=1e-9, atol=1e-12)
            err = max(abs(sol[i, 0] - _ANALYTIC[n](xs[i])) for i in range(len(xs)))
            with self.subTest(n=n):
                self.assertLess(err, 1e-6)

    def test_n1_first_zero_at_pi(self):
        # n=1 surface (theta=0) is the first zero of sin(xi)/xi, i.e. xi_1 = pi
        xi0 = 1e-4
        xs = np.linspace(xi0, np.pi, 400)
        sol = integrate_adaptive(lane_emden_rhs(1), np.array([1 - xi0**2 / 6, -xi0 / 3]), xs, t0=xi0,
                                 rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(sol[-1, 0], 0.0, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
