"""Tests for the adjoint-capable sparse PDE solve and differentiable operator assembly."""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pysp.ppl import GP, RBF, Differential, GaussianField, RandomWalk, joint
    from pysp.ppl.pde_solve import divergence_form, helmholtz_operator, laplacian, sparse_solve


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class SparseSolveTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_solves_correctly(self):
        torch.manual_seed(0)
        n = 15
        M = torch.randn(n, n)
        A = M @ M.T + n * torch.eye(n)
        b = torch.randn(n)
        r, c = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        u = sparse_solve(A.reshape(-1), r.reshape(-1), c.reshape(-1), n, b)
        self.assertLess(float((u - torch.linalg.solve(A, b)).abs().max()), 1e-9)

    def test_adjoint_gradient_matches_dense(self):
        torch.manual_seed(1)
        n = 12
        M = torch.randn(n, n)
        A = M @ M.T + n * torch.eye(n)
        b = torch.randn(n)
        r, c = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        r, c = r.reshape(-1), c.reshape(-1)

        vals = A.reshape(-1).clone().requires_grad_(True)
        b1 = b.clone().requires_grad_(True)
        sparse_solve(vals, r, c, n, b1).pow(2).sum().backward()

        av = A.reshape(-1).clone().requires_grad_(True)
        b2 = b.clone().requires_grad_(True)
        torch.linalg.solve(av.reshape(n, n), b2).pow(2).sum().backward()

        self.assertLess(float((vals.grad - av.grad).abs().max()), 1e-8)
        self.assertLess(float((b1.grad - b2.grad).abs().max()), 1e-8)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class AssemblyTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_divergence_form_matches_dense_reference(self):
        m = 6
        shape = (m, m)
        kap = torch.rand(m * m) + 0.5
        r, c, v, n = divergence_form(kap, shape)
        A = torch.zeros(n, n)
        A.index_put_((r, c), v, accumulate=True)

        idx = np.arange(m * m).reshape(m, m)
        bnd = np.zeros((m, m), bool)
        bnd[0] = bnd[-1] = bnd[:, 0] = bnd[:, -1] = True
        ref = np.zeros((m * m, m * m))
        kn = kap.numpy()
        for i in range(m):
            for j in range(m):
                p = idx[i, j]
                if bnd[i, j]:
                    ref[p, p] = 1.0
                    continue
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    q = idx[i + di, j + dj]
                    w = 0.5 * (kn[p] + kn[q])
                    ref[p, q] -= w
                    ref[p, p] += w
        self.assertLess(np.abs(A.numpy() - ref).max(), 1e-10)

    def test_helmholtz_is_laplacian_minus_shift(self):
        m = 8
        shape = (m, m)
        s2 = torch.rand(m * m) + 0.1
        r0, c0, v0, n = divergence_form(torch.ones(m * m), shape)
        r1, c1, v1, _ = helmholtz_operator(s2, shape, omega=2.0)
        A0 = torch.zeros(n, n).index_put_((r0, c0), v0, accumulate=True)
        A1 = torch.zeros(n, n).index_put_((r1, c1), v1, accumulate=True)
        idx = np.arange(m * m).reshape(m, m)
        interior = [idx[i, j] for i in range(1, m - 1) for j in range(1, m - 1)]
        diff = (A0 - A1).numpy()
        # on interior nodes the difference is +omega^2 * s2 on the diagonal, zero elsewhere
        for p in interior:
            self.assertAlmostEqual(diff[p, p], 4.0 * float(s2[p]), places=9)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class SparseSolverScalingTestCase(unittest.TestCase):
    def test_large_grid_runs(self):
        torch.set_default_dtype(torch.float64)
        m = 120
        shape = (m, m)
        n = m * m
        r, c, v, _ = laplacian(shape)
        b = torch.zeros(n)
        b[n // 2 + m // 3] = 1.0
        u = sparse_solve(v, r, c, n, b)  # dense would need ~1.4 GB just for A
        self.assertEqual(u.shape[0], n)
        self.assertTrue(torch.isfinite(u).all())


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class DifferentialProxySparseTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_sparse_matches_dense_solver(self):
        # the sparse path and a dense linear_steady solve must agree at the MAP on a small problem
        n = 25
        shape = (5, 5)
        kap_t = torch.rand(n) + 0.5
        r, c, v, _ = divergence_form(kap_t, shape)
        A = torch.zeros(n, n).index_put_((r, c), v, accumulate=True)
        b = torch.zeros(n)
        b[12] = 1.0
        u_sparse = sparse_solve(v, r, c, n, b)
        u_dense = torch.linalg.solve(A, b)
        self.assertLess(float((u_sparse - u_dense).abs().max()), 1e-9)

    def test_recovers_1d_material_field(self):
        # the clean, well-posed 1D case: recover a variable conductivity from sensors
        n = 50
        x = np.linspace(0, 1, n)
        logk_true = 0.8 * np.sin(3 * np.pi * x) - 0.3 * x
        kap_true = torch.as_tensor(np.exp(logk_true))
        f = torch.ones(n)
        r, c, v, _ = divergence_form(kap_true, (n,))
        A = torch.zeros(n, n).index_put_((r, c), v, accumulate=True)
        u_true = torch.linalg.solve(A, f).numpy()
        sens = np.arange(3, n - 3, 3)
        sig = 0.002
        u_obs = u_true[sens] + sig * np.random.RandomState(0).randn(len(sens))

        logk = GaussianField(np.arange(n), RandomWalk(scale=0.4, ridge=3.0), name="logk")
        ft = torch.ones(n)
        sens_t = torch.as_tensor(sens)
        obs = Differential(
            u_obs,
            over=logk,
            scale=sig,
            forward=lambda p, ops: ops.sparse_solve(*ops.divergence_form(ops.exp(p.field), (n,)), ft),
            observe=lambda u, p, ops: u[sens_t],
        )
        post = joint([obs]).fit(how="map")
        kap_m = np.exp(post.mean("logk"))
        self.assertGreater(np.corrcoef(kap_m, np.exp(logk_true))[0, 1], 0.9)


def _dirichlet_2d_field_problem(m=18, n_sens=80, seed=0):
    shape = (m, m)
    N = m * m
    xx, yy = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m), indexing="ij")
    logk_true = (0.9 * np.sin(3 * np.pi * xx) * np.cos(2 * np.pi * yy)).ravel()
    f = np.zeros(N)
    f[8 * m + 8] = 1.0
    r, c, v, _ = divergence_form(torch.as_tensor(np.exp(logk_true)), shape)
    A = torch.zeros(N, N).index_put_((r, c), v, accumulate=True)
    u_true = torch.linalg.solve(A, torch.as_tensor(f)).numpy()
    rng = np.random.RandomState(seed)
    interior = [i for i in range(N) if not (i < m or i >= N - m or i % m == 0 or i % m == m - 1)]
    sens = np.array(sorted(rng.choice(interior, n_sens, replace=False)))
    sig = 0.002
    u_obs = u_true[sens] + sig * rng.randn(len(sens))
    coords = np.stack([xx.ravel(), yy.ravel()], 1)
    return shape, logk_true, f, sens, sig, u_obs, coords


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class GaussNewtonPosteriorTestCase(unittest.TestCase):
    """Scalable posteriors for the sparse path (phase 1b): Gauss-Newton, exact for a linear forward."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_gn_matches_exact_laplace_linear_gaussian(self):
        n = 30
        x = np.linspace(0, 1, n)
        h = x[1] - x[0]
        sig = 0.003
        sens = np.arange(2, n - 2, 3)
        L = np.zeros((n, n))
        for i in range(1, n - 1):
            L[i, i - 1] = -1 / h**2
            L[i, i] = 2 / h**2
            L[i, i + 1] = -1 / h**2
        L[0, 0] = L[-1, -1] = 1.0
        q_true = np.sin(2 * np.pi * x) * x
        q_true[0] = q_true[-1] = 0.0
        u_obs = np.linalg.solve(L, q_true)[sens] + sig * np.random.RandomState(2).randn(len(sens))
        q = GaussianField(np.arange(n), RandomWalk(scale=0.3, ridge=5.0), name="q")
        Lt = torch.as_tensor(L)
        st = torch.as_tensor(sens)

        def mk():
            return Differential(
                u_obs,
                over=q,
                scale=sig,
                forward=lambda p, ops: ops.solve(Lt, p.field),
                observe=lambda u, p, ops: u[st],
            )

        _, sd_lap = joint([mk()]).fit(how="laplace").posterior("q")
        _, sd_gn = joint([mk()]).fit(how="gauss_newton").posterior("q")
        self.assertLess(np.max(np.abs(sd_gn - sd_lap)), 1e-9)  # linear forward -> GN is exact

    def test_gn_posterior_on_sparse_path(self):
        shape, logk_true, f, sens, sig, u_obs, coords = _dirichlet_2d_field_problem()
        field = GP("logk", index=coords, kernel=RBF(lengthscale=0.2))
        ft = torch.as_tensor(f)
        st = torch.as_tensor(sens)
        obs = Differential(
            u_obs,
            over=field,
            scale=sig,
            forward=lambda p, ops: ops.sparse_solve(*ops.divergence_form(ops.exp(p.field), shape), ft),
            observe=lambda u, p, ops: u[st],
        )
        post = joint([obs]).fit(how="gauss_newton")
        lm, ls = post.posterior("logk")
        self.assertGreater(np.corrcoef(lm, logk_true)[0, 1], 0.6)
        self.assertTrue(np.all(ls > 0))

    def test_laplace_blocked_on_sparse_path(self):
        shape, logk_true, f, sens, sig, u_obs, coords = _dirichlet_2d_field_problem(m=14, n_sens=40)
        field = GP("logk", index=coords, kernel=RBF(lengthscale=0.25))
        ft = torch.as_tensor(f)
        st = torch.as_tensor(sens)
        obs = Differential(
            u_obs,
            over=field,
            scale=sig,
            forward=lambda p, ops: ops.sparse_solve(*ops.divergence_form(ops.exp(p.field), shape), ft),
            observe=lambda u, p, ops: u[st],
        )
        with self.assertRaises(ValueError):
            joint([obs]).fit(how="laplace")


if __name__ == "__main__":
    unittest.main()
