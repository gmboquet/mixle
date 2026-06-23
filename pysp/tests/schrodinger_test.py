"""WS-14: time-dependent Schrodinger equation (split-step Fourier), validated vs exact quantum solutions."""

import unittest

import numpy as np

from pysp.ppl.physics.schrodinger import norm, probability_density, schrodinger_split_step


class SchrodingerTest(unittest.TestCase):
    def setUp(self):
        self.L, self.N = 40.0, 1024
        self.x = np.linspace(-self.L / 2, self.L / 2, self.N, endpoint=False)
        self.dx = self.x[1] - self.x[0]

    def test_harmonic_oscillator_ground_state_stationary(self):
        # the n=0 eigenstate exp(-x^2/2) is stationary in |psi|^2 and conserves the norm
        v = 0.5 * self.x**2
        psi = np.exp(-self.x**2 / 2) / np.pi**0.25
        psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * self.dx)
        p0 = probability_density(psi)
        out = schrodinger_split_step(psi, v, 0.005, 400, length=self.L)
        self.assertLess(np.abs(probability_density(out) - p0).max(), 1e-4)
        self.assertAlmostEqual(norm(out, self.L), norm(psi, self.L), places=10)

    def test_free_particle_gaussian_spreading(self):
        # a free Gaussian packet spreads at sigma(t) = sigma0 sqrt(1 + (t/2 sigma0^2)^2) (hbar=m=1)
        s0, t = 1.0, 2.0
        psi = (np.exp(-self.x**2 / (4 * s0**2)) * (2 * np.pi * s0**2) ** -0.25).astype(complex)
        out = schrodinger_split_step(psi, np.zeros(self.N), 0.005, int(t / 0.005), length=self.L)
        width = np.sqrt(np.sum(self.x**2 * probability_density(out)) * self.dx)
        self.assertAlmostEqual(width, s0 * np.sqrt(1 + (t / (2 * s0**2)) ** 2), places=4)

    def test_norm_conserved_with_potential(self):
        # unitary evolution: the norm is preserved even in a non-trivial potential
        psi = np.exp(-(self.x - 2.0) ** 2 / 2).astype(complex)
        psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * self.dx)
        out = schrodinger_split_step(psi, 0.2 * self.x**2, 0.002, 500, length=self.L)
        self.assertAlmostEqual(norm(out, self.L), 1.0, places=8)

    def test_2d_norm_conserved(self):
        n = 64
        x = np.linspace(-10, 10, n, endpoint=False)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        psi = np.exp(-(xx**2 + yy**2) / 2).astype(complex)
        out = schrodinger_split_step(psi, 0.5 * (xx**2 + yy**2), 0.005, 50, length=20.0)
        self.assertAlmostEqual(norm(out, 20.0), norm(psi, 20.0), places=8)


if __name__ == "__main__":
    unittest.main()
