"""WS-1: 1-D shallow-water equations (Rusanov finite volume) -- conservation + gravity-wave speed."""

import unittest

import numpy as np

from pysp.ppl.dynamics import integrate_adaptive, shallow_water_rhs


class ShallowWaterTest(unittest.TestCase):
    def test_conservation_and_wave_speed(self):
        g, h0, L, n = 9.81, 1.0, 20.0, 400
        x = np.linspace(0.0, L, n, endpoint=False)
        dx = x[1] - x[0]
        h = h0 + 0.01 * np.exp(-((x - 10.0) / 0.5) ** 2)  # small bump, water at rest
        z0 = np.concatenate([h, np.zeros(n)])
        rhs = shallow_water_rhs(dx, g)
        sol = integrate_adaptive(rhs, z0, [0.5], rtol=1e-7, atol=1e-9)[-1]
        hf, huf = sol[:n], sol[n:]
        self.assertAlmostEqual(hf.sum() * dx, h.sum() * dx, places=8)   # mass conserved
        self.assertAlmostEqual(huf.sum() * dx, 0.0, places=6)           # momentum conserved (symmetric)
        # the bump splits into two waves at +/- sqrt(g h0); the right one is near 10 + c t
        c = np.sqrt(g * h0)
        right = x > 10.0
        peak_x = x[right][np.argmax(hf[right] - h0)]
        self.assertAlmostEqual(peak_x, 10.0 + c * 0.5, delta=0.3)

    def test_still_water_stays_still(self):
        n = 100
        x = np.linspace(0.0, 10.0, n, endpoint=False)
        dx = x[1] - x[0]
        z0 = np.concatenate([2.0 * np.ones(n), np.zeros(n)])  # flat lake at rest
        sol = integrate_adaptive(shallow_water_rhs(dx), z0, [1.0], rtol=1e-8, atol=1e-10)[-1]
        self.assertTrue(np.allclose(sol[:n], 2.0, atol=1e-9))   # depth unchanged
        self.assertTrue(np.allclose(sol[n:], 0.0, atol=1e-9))   # still at rest


if __name__ == "__main__":
    unittest.main()
