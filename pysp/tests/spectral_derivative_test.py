"""WS-1: Fourier spectral differentiation (spectral/Galerkin discretization option)."""

import unittest

import numpy as np

from pysp.ppl.physics.dynamics import spectral_derivative


class SpectralDerivativeTest(unittest.TestCase):
    def test_trig_derivatives_machine_precision(self):
        L, n = 2 * np.pi, 64
        x = np.linspace(0.0, L, n, endpoint=False)
        for m in (1, 3, 5):
            u = np.sin(m * x)
            self.assertTrue(np.allclose(spectral_derivative(u, L, 1), m * np.cos(m * x), atol=1e-10))
            self.assertTrue(np.allclose(spectral_derivative(u, L, 2), -(m**2) * np.sin(m * x), atol=1e-9))
            self.assertTrue(np.allclose(spectral_derivative(u, L, 3), -(m**3) * np.cos(m * x), atol=1e-8))

    def test_far_more_accurate_than_finite_difference(self):
        L, n = 2 * np.pi, 64
        x = np.linspace(0.0, L, n, endpoint=False)
        u = np.exp(np.sin(x))
        exact = np.cos(x) * np.exp(np.sin(x))
        dx = x[1] - x[0]
        fd = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
        spec = spectral_derivative(u, L, 1)
        self.assertLess(np.max(np.abs(spec - exact)), 1e-10)
        self.assertGreater(np.max(np.abs(fd - exact)), 1e-4)  # finite difference is far coarser

    def test_arbitrary_length(self):
        L, n = 5.0, 100
        x = np.linspace(0.0, L, n, endpoint=False)
        u = np.cos(2 * np.pi * x / L)
        self.assertTrue(np.allclose(spectral_derivative(u, L, 1), -(2 * np.pi / L) * np.sin(2 * np.pi * x / L), atol=1e-10))


if __name__ == "__main__":
    unittest.main()
