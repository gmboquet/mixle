"""2-D linear elasticity (Navier-Cauchy) forward solver (Phase 5)."""

import unittest

import numpy as np

from pysp.ppl.multiphysics import solve_elasticity


class ElasticityTest(unittest.TestCase):
    def test_manufactured_solution(self):
        n, h = 41, 1.0 / 40
        x = np.linspace(0, 1, n)
        gx, gy = np.meshgrid(x, x, indexing="ij")
        mu = 1.5
        ux = np.sin(np.pi * gx) * np.sin(np.pi * gy)
        uy = np.cos(np.pi * gx) * np.cos(np.pi * gy)
        u_true = np.stack([ux, uy], axis=2)
        f = np.stack([2 * mu * np.pi**2 * ux, 2 * mu * np.pi**2 * uy], axis=2)  # body force for this u
        u = solve_elasticity((n, n), f, lame_lambda=1.0, lame_mu=mu, spacing=h, dirichlet=u_true)
        self.assertLess(np.max(np.abs(u - u_true)), 0.005)  # O(h^2)

    def test_patch_test_rigid_translation(self):
        u = solve_elasticity((20, 20), np.zeros(2), dirichlet=np.array([0.3, -0.2]))
        np.testing.assert_allclose(u, np.broadcast_to([0.3, -0.2], (20, 20, 2)), atol=1e-6)  # constant field

    def test_stiffer_material_deflects_less(self):
        load = np.array([0.0, -1.0])
        soft = solve_elasticity((25, 25), load, lame_mu=0.5, lame_lambda=0.5)
        stiff = solve_elasticity((25, 25), load, lame_mu=5.0, lame_lambda=5.0)
        self.assertGreater(np.abs(soft).max(), np.abs(stiff).max())

    def test_output_shape(self):
        u = solve_elasticity((12, 15), np.array([0.0, -0.5]))
        self.assertEqual(u.shape, (12, 15, 2))


if __name__ == "__main__":
    unittest.main()
