"""Tests for adjoint-state time integration with checkpointing and the differentiable sparse matvec (phase 2)."""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pysp.ppl import Differential, free, joint
    from pysp.ppl.ops import make_ops
    from pysp.ppl.pde_solve import laplacian


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class MatvecTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_matvec_matches_dense(self):
        ops = make_ops()
        n = 14
        r, c, v, N = laplacian((n,))
        x = torch.randn(n)
        dense = torch.zeros(n, n).index_put_((r, c), v, accumulate=True) @ x
        self.assertLess(float((ops.matvec(r, c, v, N, x) - dense).abs().max()), 1e-12)

    def test_matvec_gradient(self):
        ops = make_ops()
        n = 10
        r, c, v, N = laplacian((n,))
        x = torch.randn(n, requires_grad=True)
        ops.matvec(r, c, v, N, x).pow(2).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


def _diffusion_problem(m=40, nt=120, D_true=0.6, seed=0):
    dx = 1.0 / (m - 1)
    dt = 0.15 * dx**2
    xs = np.linspace(0, 1, m)
    u0 = torch.as_tensor(np.exp(-(((xs - 0.5) / 0.08) ** 2)))
    recv = torch.as_tensor(np.array([10, 20, 30]))

    def lap1d(u):
        out = torch.zeros_like(u)
        out[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        return out

    def make_step(D):
        return lambda y, i: y + dt * D * lap1d(y)

    def record(y, i):
        return y[recv]

    ops = make_ops()
    rec_true = ops.integrate_record(make_step(torch.tensor(D_true)), u0, nt, record, checkpoint=None).detach().numpy()
    y_obs = rec_true.reshape(-1) + 0.001 * np.random.RandomState(seed).randn(rec_true.size)
    return u0, nt, make_step, record, rec_true, y_obs


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class CheckpointTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_checkpointed_gradient_equals_full(self):
        ops = make_ops()
        u0, nt, make_step, record, rec_true, _ = _diffusion_problem()
        target = torch.as_tensor(rec_true)

        def grad_at(checkpoint):
            d = torch.tensor(0.5, requires_grad=True)
            loss = ((ops.integrate_record(make_step(d), u0, nt, record, checkpoint=checkpoint) - target) ** 2).sum()
            return torch.autograd.grad(loss, d)[0].item()

        self.assertLess(abs(grad_at(None) - grad_at(11)), 1e-9)

    def test_record_shape(self):
        ops = make_ops()
        u0, nt, make_step, record, rec_true, _ = _diffusion_problem()
        rec = ops.integrate_record(make_step(torch.tensor(0.6)), u0, nt, record, checkpoint=11)
        self.assertEqual(rec.shape, (nt + 1, 3))  # one record per step plus the final state


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class TransientInverseTestCase(unittest.TestCase):
    """Recover a diffusivity from a recorded receiver time series, with the checkpointed adjoint."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_recovers_diffusivity(self):
        u0, nt, make_step, record, rec_true, y_obs = _diffusion_problem()
        D = free(1, name="D", support="positive")

        def forward(p, ops):
            return ops.integrate_record(make_step(p.D), u0, nt, record, checkpoint=11).reshape(-1)

        obs = Differential(y_obs, drivers=[D], scale=0.001, forward=forward)
        Dm, Ds = joint([obs]).fit(how="laplace").posterior("D")
        self.assertLess(abs(Dm - 0.6), 2 * Ds)
        self.assertGreater(Ds, 0.0)


if __name__ == "__main__":
    unittest.main()
