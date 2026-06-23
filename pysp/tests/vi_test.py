"""Tests for mean-field variational inference on latent fields (how='vi', phase 4 completion)."""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pysp.ppl import GP, Differential, Gaussian, GaussianField, RandomWalk, TotalVariation, joint


def _source_problem(n=25, sig=0.01):
    x = np.linspace(0, 1, n)
    h = x[1] - x[0]
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

    def make():
        return Differential(
            u_obs,
            over=GaussianField(np.arange(n), RandomWalk(scale=0.3, ridge=5.0), name="q"),
            scale=sig,
            forward=lambda p, o: o.solve(Lt, p.field),
            observe=lambda u, p, o: u[st],
        )

    return make


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class VariationalCalibrationTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)

    def test_mean_field_matches_exact_mean(self):
        make = _source_problem()
        m_lap, sd_lap = joint([make()]).fit(how="laplace").posterior("q")
        m_vi, sd_vi = joint([make()]).fit(how="vi", vi_steps=600, vi_lr=0.05).posterior("q")
        # VI recovers the posterior mean; mean-field underestimates correlated variances (sd ratio < 1)
        self.assertGreater(np.corrcoef(m_vi, m_lap)[0, 1], 0.9)
        ratio = np.median(sd_vi / sd_lap)
        self.assertTrue(0.3 < ratio <= 1.2)
        self.assertTrue(np.all(sd_vi > 0))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class VariationalSparsePathTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)

    def test_vi_on_sparse_forward(self):
        n = 40
        x = np.linspace(0, 1, n)
        h = x[1] - x[0]
        from pysp.ppl.physics.pde_solve import divergence_form

        logk_true = 0.8 * np.sin(3 * np.pi * x) - 0.3 * x
        kt = torch.as_tensor(np.exp(logk_true))
        r, c, v, _ = divergence_form(kt, (n,))
        A = torch.zeros(n, n).index_put_((r, c), v, accumulate=True)
        u_true = torch.linalg.solve(A, torch.ones(n)).numpy()
        sens = np.arange(3, n - 3, 3)
        u_obs = u_true[sens] + 0.002 * np.random.RandomState(0).randn(len(sens))
        fld = GP("logk", index=np.arange(n), kernel=RandomWalk(scale=0.4, ridge=3.0))
        ft = torch.ones(n)
        st = torch.as_tensor(sens)
        obs = Differential(
            u_obs,
            over=fld,
            scale=0.002,
            forward=lambda p, o: o.sparse_solve(*o.divergence_form(o.exp(p.field), (n,)), ft),
            observe=lambda u, p, o: u[st],
        )
        m, sd = joint([obs]).fit(how="vi", vi_steps=500).posterior("logk")
        self.assertGreater(np.corrcoef(m, logk_true)[0, 1], 0.7)  # VI runs where how='laplace' cannot
        self.assertTrue(np.all(sd > 0))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class VariationalNonGaussianPriorTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)

    def test_vi_with_total_variation_prior(self):
        n = 50
        x = np.linspace(0, 1, n)
        f_true = (x > 0.5).astype(float) * 2.0
        y = f_true + 0.2 * np.random.RandomState(0).randn(n)
        fld = GP("f", index=np.arange(n), kernel=RandomWalk(scale=8.0, ridge=10.0))
        post = joint([Gaussian(y, mean=1.0 * fld, sd=0.2), TotalVariation(over=fld, shape=(n,), weight=4.0)]).fit(
            how="vi", vi_steps=400
        )
        m, sd = post.posterior("f")
        self.assertGreater(np.max(np.abs(np.diff(m))), 0.8)  # edge preserved under the non-Gaussian prior
        self.assertTrue(np.all(sd > 0))


if __name__ == "__main__":
    unittest.main()
