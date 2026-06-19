"""Tests for the low-rank (Woodbury) Gauss-Newton field marginals (phase 1b completion)."""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pysp.ppl import GP, RBF, Differential, joint


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class WoodburyIdentityTestCase(unittest.TestCase):
    def test_marginal_variances_match_dense(self):
        rng = np.random.RandomState(0)
        nf, nobs = 40, 12
        b = rng.randn(nf, nf)
        K = b @ b.T + nf * np.eye(nf)
        J = rng.randn(nobs, nf)
        M = K @ J.T
        C = np.linalg.inv(np.eye(nobs) + J @ M)
        woodbury = np.diag(K) - np.einsum("ij,jk,ik->i", M, C, M)
        dense = np.diag(np.linalg.inv(np.linalg.inv(K) + J.T @ J))
        self.assertLess(np.max(np.abs(woodbury - dense)), 1e-9)


def _rbf_source_problem(n=20, lengthscale=0.25, amplitude=2.0):
    x = np.linspace(0, 1, n)
    h = x[1] - x[0]
    sig = 0.003
    sens = np.arange(2, n - 2, 2)
    L = np.zeros((n, n))
    for i in range(1, n - 1):
        L[i, i - 1] = -1 / h**2
        L[i, i] = 2 / h**2
        L[i, i + 1] = -1 / h**2
    L[0, 0] = L[-1, -1] = 1.0
    q_true = np.sin(2 * np.pi * x) * x
    q_true[0] = q_true[-1] = 0.0
    u_obs = np.linalg.solve(L, q_true)[sens] + sig * np.random.RandomState(1).randn(len(sens))
    Lt = torch.as_tensor(L)
    st = torch.as_tensor(sens)

    def make(field):
        return Differential(
            u_obs, over=field, scale=sig, forward=lambda p, o: o.solve(Lt, p.field), observe=lambda u, p, o: u[st]
        )

    return make, lengthscale, amplitude


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class WoodburyEndToEndTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_low_rank_matches_dense_gauss_newton(self):
        make, ls, amp = _rbf_source_problem()
        n = 20
        q = GP("q", index=np.arange(n), kernel=RBF(lengthscale=ls, amplitude=amp))
        _, sd_low = joint([make(q)]).fit(how="gauss_newton").posterior("q")

        q2 = GP("q", index=np.arange(n), kernel=RBF(lengthscale=ls, amplitude=amp))
        q2.field.covariance = None  # force the dense Gauss-Newton path
        _, sd_dense = joint([make(q2)]).fit(how="gauss_newton").posterior("q")
        self.assertLess(np.max(np.abs(sd_low - sd_dense)), 1e-7)

    def test_scales_to_large_field(self):
        m = 46
        N = m * m
        xx, yy = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m), indexing="ij")
        field = GP("g", index=np.stack([xx.ravel(), yy.ravel()], 1), kernel=RBF(lengthscale=0.2))
        rng = np.random.RandomState(2)
        idx = rng.choice(N, 150, replace=False)
        st = torch.as_tensor(idx)
        y = np.sin(3 * xx).ravel()[idx] + 0.01 * rng.randn(len(idx))
        obs = Differential(y, over=field, scale=0.01, forward=lambda p, o: p.field, observe=lambda f, p, o: f[st])
        gm, gs = joint([obs]).fit(how="gauss_newton").posterior("g")  # no dense N x N inverse
        self.assertEqual(gm.shape[0], N)
        self.assertTrue(np.all(gs > 0))


if __name__ == "__main__":
    unittest.main()
