"""WS-1: forward ODE parameter sensitivity (variational equation), checked vs finite differences."""

import unittest

import numpy as np

from pysp.ppl.dynamics import integrate_adaptive, integrate_sensitivity


class ODESensitivityTest(unittest.TestCase):
    def _fd(self, rhs, y0, te, p, j, h=1e-6):
        pe, pm = p.copy(), p.copy()
        pe[j] += h
        pm[j] -= h
        ye = integrate_adaptive(lambda t, y: rhs(t, y, pe), y0, te, rtol=1e-11, atol=1e-13)
        ym = integrate_adaptive(lambda t, y: rhs(t, y, pm), y0, te, rtol=1e-11, atol=1e-13)
        return (ye[-1] - ym[-1]) / (2 * h)

    def test_linear_ode(self):
        rhs = lambda t, y, p: [-p[0] * y[0] + p[1]]  # noqa: E731
        p = np.array([2.0, 0.5])
        te = np.array([1.0, 2.0])
        _, s = integrate_sensitivity(rhs, [1.0], te, p)
        self.assertEqual(s.shape, (2, 1, 2))
        for j in range(2):
            self.assertAlmostEqual(s[-1, 0, j], self._fd(rhs, [1.0], te, p, j)[0], places=5)

    def test_lotka_volterra(self):
        def rhs(t, y, p):
            return [p[0] * y[0] - p[1] * y[0] * y[1], -p[2] * y[1] + p[3] * y[0] * y[1]]

        p = np.array([1.5, 1.0, 3.0, 1.0])
        te = np.array([2.0])
        _, s = integrate_sensitivity(rhs, [1.0, 1.0], te, p)
        for j in range(4):
            self.assertTrue(np.allclose(s[-1, :, j], self._fd(rhs, [1.0, 1.0], te, p, j), atol=1e-3))


if __name__ == "__main__":
    unittest.main()
