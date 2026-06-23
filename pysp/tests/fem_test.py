"""Unstructured P1 finite-element Poisson solver (Phase 5)."""

import unittest

import numpy as np
from scipy.spatial import Delaunay

from pysp.ppl.physics.fem import boundary_nodes, fem_poisson


def _mesh_square(n):
    xs = np.linspace(0, 1, n)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            tris.append([a, a + 1, a + n + 1])
            tris.append([a, a + n + 1, a + n])
    return nodes, np.array(tris)


class FEMPoissonTest(unittest.TestCase):
    def test_manufactured_solution_converges(self):
        errs = []
        for n in (11, 21, 41):
            nodes, tris = _mesh_square(n)
            u_true = np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
            u = fem_poisson(nodes, tris, 2 * np.pi**2 * u_true)  # boundary pinned to 0 (= u_true there)
            errs.append(np.max(np.abs(u - u_true)))
        self.assertLess(errs[-1], 0.005)
        self.assertGreater(errs[0] / errs[1], 2.5)  # ~O(h^2): error drops ~4x per halving
        self.assertGreater(errs[1] / errs[2], 2.5)

    def test_unstructured_delaunay_mesh(self):
        rng = np.random.RandomState(1)
        pts = np.vstack([rng.uniform(0, 1, (400, 2)), [[0, 0], [1, 0], [0, 1], [1, 1]]])
        tri = Delaunay(pts)
        u_true = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
        bnd = boundary_nodes(tri.simplices)
        u = fem_poisson(pts, tri.simplices, 2 * np.pi**2 * u_true, dirichlet={int(i): float(u_true[i]) for i in bnd})
        interior = np.setdiff1d(np.arange(len(pts)), bnd)
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertLess(np.max(np.abs(u[interior] - u_true[interior])), 0.05)

    def test_boundary_detection(self):
        nodes, tris = _mesh_square(11)
        bnd = set(boundary_nodes(tris))
        expected = set(np.where((nodes[:, 0] == 0) | (nodes[:, 0] == 1) | (nodes[:, 1] == 0) | (nodes[:, 1] == 1))[0])
        self.assertEqual(bnd, expected)

    def test_heterogeneous_conductivity_runs(self):
        nodes, tris = _mesh_square(15)
        u = fem_poisson(nodes, tris, 1.0, conductivity=np.linspace(0.5, 2.0, len(tris)))
        self.assertTrue(np.all(np.isfinite(u)))


if __name__ == "__main__":
    unittest.main()
