"""Multiphysics forward solvers: nD Poisson (manufactured solution) + coupled PDE system (Phase 5)."""

import unittest

import numpy as np

from pysp.ppl.multiphysics import CoupledPDESystem, _diffusion_blocks, solve_poisson


def _manufactured(d, n=21):
    """u = prod sin(pi x_i) on [0,1]^d solves -lap u = d pi^2 u with u=0 on the boundary."""
    h = 1.0 / (n - 1)
    shape = (n,) * d
    grids = np.meshgrid(*([np.linspace(0, 1, n)] * d), indexing="ij")
    u_true = np.ones(shape)
    for g in grids:
        u_true = u_true * np.sin(np.pi * g)
    return shape, h, u_true, d * np.pi**2 * u_true


class PoissonTest(unittest.TestCase):
    def test_manufactured_solution_1d_2d_3d(self):
        for d in (1, 2, 3):
            shape, h, u_true, f = _manufactured(d)
            u = solve_poisson(shape, f, conductivity=1.0, dirichlet=0.0, spacing=h)
            self.assertLess(np.max(np.abs(u - u_true)), 0.01)  # O(h^2) discretization error

    def test_convergence_order(self):
        errs = []
        for n in (11, 21, 41):
            shape, h, u_true, f = _manufactured(2, n)
            errs.append(np.max(np.abs(solve_poisson(shape, f, spacing=h) - u_true)))
        self.assertLess(errs[1], errs[0])  # refining the grid reduces the error
        self.assertLess(errs[2], errs[1])

    def test_heterogeneous_conductivity_runs(self):
        u = solve_poisson((15, 15), 1.0, conductivity=np.linspace(0.5, 2, 225), spacing=1 / 14)
        self.assertTrue(u[1:-1, 1:-1].min() > 0)  # positive source, zero BC -> positive interior

    def test_nonzero_dirichlet(self):
        u = solve_poisson((10, 10), 0.0, conductivity=1.0, dirichlet=3.0, spacing=1 / 9)
        np.testing.assert_allclose(u, 3.0, atol=1e-8)  # Laplace with constant BC -> constant field


class CoupledPDETest(unittest.TestCase):
    def test_zero_coupling_decouples(self):
        shape, h = (12, 12), 1 / 11
        sys = CoupledPDESystem(shape, [1.0, 1.0], np.zeros((2, 2)), spacing=h).solve([1.0, 0.5])
        np.testing.assert_allclose(sys[0], solve_poisson(shape, 1.0, 1.0, spacing=h))
        np.testing.assert_allclose(sys[1], solve_poisson(shape, 0.5, 1.0, spacing=h))

    def test_coupling_pulls_source_free_field(self):
        shape, h = (12, 12), 1 / 11
        c = np.array([[0.0, 0.0], [-5.0, 5.0]])  # field2 exchanges toward field1
        sys = CoupledPDESystem(shape, [1.0, 1.0], c, spacing=h).solve([1.0, 0.0])
        self.assertGreater(sys[1].max(), 0.0)  # field2 has no source but is pulled up by the coupling

    def test_matches_dense_reference(self):
        shape, h = (8, 8), 1 / 7
        c = np.array([[2.0, -2.0], [-2.0, 2.0]])
        kappas, sources = [1.0, 1.5], [1.0, 0.3]
        got = CoupledPDESystem(shape, kappas, c, spacing=h).solve(sources)
        # dense block assembly as an independent reference
        blocks = [_diffusion_blocks(shape, kp, h) for kp in kappas]
        n, interior = blocks[0][3], blocks[0][5]
        a = np.zeros((2 * n, 2 * n))
        rhs = np.zeros(2 * n)
        for i in range(2):
            r, col, v, _, bnd, _ = blocks[i]
            blk = np.zeros((n, n))
            np.add.at(blk, (r, col), v)
            a[i * n : (i + 1) * n, i * n : (i + 1) * n] = blk
            bi = np.full(n, float(sources[i]))
            bi[bnd] = 0.0
            rhs[i * n : (i + 1) * n] = bi
            for j in range(2):
                if c[i, j]:
                    a[i * n + interior, j * n + interior] += c[i, j]
        u = np.linalg.solve(a, rhs)
        np.testing.assert_allclose(got[0], u[:n].reshape(shape), atol=1e-8)
        np.testing.assert_allclose(got[1], u[n:].reshape(shape), atol=1e-8)

    def test_coupling_shape_validation(self):
        with self.assertRaises(ValueError):
            CoupledPDESystem((5, 5), [1.0, 1.0], np.eye(3))


if __name__ == "__main__":
    unittest.main()
