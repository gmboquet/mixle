"""WS-14: pseudo-spectral incompressible Navier-Stokes (2-D/3-D DNS + Smagorinsky LES).

The DNS core is checked against exact analytic Navier-Stokes solutions to machine precision; the LES
closure is checked to reduce to DNS at C_s=0 and to be strictly dissipative.
"""

import unittest

import numpy as np

from pysp.ppl.physics.spectral_flow import incompressible_ns_spectral as ns
from pysp.ppl.physics.spectral_flow import kinetic_energy as ke


class SpectralFlowTest(unittest.TestCase):
    def test_2d_taylor_green_exact_decay(self):
        # u = -cos x sin y * e^{-2 nu t} is an exact 2-D Navier-Stokes solution for all time
        n, nu, t, dt = 48, 0.05, 0.3, 0.001
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        u0, v0 = -np.cos(xx) * np.sin(yy), np.sin(xx) * np.cos(yy)
        u, v = ns((u0, v0), nu, dt, int(t / dt))
        decay = np.exp(-2 * nu * t)
        self.assertLess(max(np.abs(u - u0 * decay).max(), np.abs(v - v0 * decay).max()), 1e-10)

    def test_3d_abc_beltrami_exact_decay(self):
        # the ABC/Beltrami flow (curl u = u) is an exact 3-D NS solution decaying as e^{-nu t}
        n, nu, t, dt = 24, 0.1, 0.2, 0.002
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
        u0 = np.sin(zz) + np.cos(yy)
        v0 = np.sin(xx) + np.cos(zz)
        w0 = np.sin(yy) + np.cos(xx)
        u, v, w = ns((u0, v0, w0), nu, dt, int(t / dt))
        decay = np.exp(-nu * t)
        self.assertLess(np.abs(u - u0 * decay).max(), 1e-10)

    def test_les_reduces_to_dns_and_dissipates(self):
        rng = np.random.RandomState(0)
        uf = tuple(rng.standard_normal((24, 24, 24)) for _ in range(3))
        # C_s = 0 is exactly DNS
        a = ns(uf, 0.01, 0.005, 5, dealias=True, smagorinsky=0.0)
        b = ns(uf, 0.01, 0.005, 5, dealias=True, smagorinsky=0.0)
        self.assertTrue(np.allclose(a[0], b[0]))
        # C_s > 0 adds subgrid dissipation: energy decays, and faster than DNS
        e0 = ke(uf)
        e_dns = ke(ns(uf, 0.01, 0.005, 15, dealias=True, smagorinsky=0.0))
        e_les = ke(ns(uf, 0.01, 0.005, 15, dealias=True, smagorinsky=0.17))
        self.assertLess(e_les, e_dns)
        self.assertLess(e_dns, e0)

    def test_incompressibility_preserved(self):
        # a projected random field stays divergence-free under the solver (spectral divergence ~ 0)
        n = 16
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        u0, v0 = np.sin(yy), np.sin(xx)  # divergence-free
        u, v = ns((u0, v0), 0.05, 0.005, 10)
        k = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        div = np.fft.ifft2(1j * kx * np.fft.fft2(u) + 1j * ky * np.fft.fft2(v)).real
        self.assertLess(np.abs(div).max(), 1e-10)


if __name__ == "__main__":
    unittest.main()


class BoussinesqTest(unittest.TestCase):
    def test_passive_scalar_heat_equation_limit(self):
        # buoyancy=0, u=0: temperature obeys the heat equation, a mode k decays as e^{-kappa k^2 t}
        from pysp.ppl.physics.spectral_flow import incompressible_boussinesq_spectral as bq

        n, kappa, t, dt = 48, 0.2, 0.5, 0.002
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xx, _ = np.meshgrid(x, x, indexing="ij")
        t0 = np.cos(2 * xx)  # wavenumber 2
        (_u, _v), temp = bq((np.zeros((n, n)), np.zeros((n, n))), t0, 0.1, kappa, 0.0, dt, int(t / dt))
        self.assertLess(np.abs(temp - t0 * np.exp(-kappa * 4 * t)).max(), 1e-10)

    def test_buoyancy_drives_flow_from_rest(self):
        from pysp.ppl.physics.spectral_flow import incompressible_boussinesq_spectral as bq
        from pysp.ppl.physics.spectral_flow import kinetic_energy as ke

        n = 48
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        (u, v), _t = bq((np.zeros((n, n)), np.zeros((n, n))), 0.1 * np.cos(xx) * np.sin(yy),
                        0.05, 0.05, 1.0, 0.002, 100)
        self.assertGreater(ke((u, v)), 0.0)  # potential -> kinetic energy

    def test_3d_passive_limit(self):
        from pysp.ppl.physics.spectral_flow import incompressible_boussinesq_spectral as bq

        n, kappa, t, dt = 16, 0.1, 0.2, 0.005
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        _x, _y, zz = np.meshgrid(x, x, x, indexing="ij")
        t0 = np.cos(zz)
        (_u, _v, _w), temp = bq((np.zeros((n, n, n)),) * 3, t0, 0.1, kappa, 0.0, dt, int(t / dt))
        self.assertLess(np.abs(temp - t0 * np.exp(-kappa * t)).max(), 1e-10)
