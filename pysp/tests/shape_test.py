"""Tests for level-set geometry: shape optimization and inverse shape inference (phase 3)."""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pysp.ppl import GP, Differential, RandomWalk, joint, level_set_material, shape_optimize


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ShapeOptimizeTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_recovers_target_region(self):
        m = 24
        xx, yy = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m), indexing="ij")
        target = ((xx - 0.5) ** 2 + (yy - 0.4) ** 2 < 0.18**2).astype(float).ravel()
        tgt = torch.as_tensor(target)

        def objective(phi, ops):
            return ((ops.heaviside(phi, eps=0.05) - tgt) ** 2).sum()

        phi_opt = shape_optimize(np.full(m * m, -0.1), objective, steps=300)
        acc = ((phi_opt > 0).astype(float) == target).mean()
        self.assertGreater(acc, 0.95)

    def test_smoothed_heaviside_is_monotone_and_bounded(self):
        from pysp.ppl.physics.ops import make_ops

        ops = make_ops()
        phi = torch.linspace(-2, 2, 50)
        h = ops.heaviside(phi, eps=0.2)
        self.assertTrue(torch.all((h >= 0) & (h <= 1)))
        self.assertTrue(torch.all(torch.diff(h) >= -1e-12))  # nondecreasing in phi


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class InverseShapeInferenceTestCase(unittest.TestCase):
    """Recover a high-conductivity inclusion (a shape) from steady-diffusion sensors, with uncertainty."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_recovers_inclusion(self):
        n = 50
        x = np.linspace(0, 1, n)
        h = x[1] - x[0]
        inside, outside = 5.0, 1.0
        true_phi = -np.abs(x - 0.5) + 0.18  # the slab |x-0.5| < 0.18
        kap_true = level_set_material(true_phi, inside, outside, eps=0.03)

        def assemble_np(kap):
            M = np.zeros((n, n))
            for i in range(1, n - 1):
                kl = 0.5 * (kap[i] + kap[i - 1])
                kr = 0.5 * (kap[i] + kap[i + 1])
                M[i, i - 1] = -kl / h**2
                M[i, i] = (kl + kr) / h**2
                M[i, i + 1] = -kr / h**2
            M[0, 0] = M[-1, -1] = 1.0
            return M

        f = np.ones(n)
        u_true = np.linalg.solve(assemble_np(kap_true), f)
        sens = np.arange(2, n - 2, 2)
        sig = 0.002
        u_obs = u_true[sens] + sig * np.random.RandomState(0).randn(len(sens))

        phi = GP("phi", index=np.arange(n), kernel=RandomWalk(scale=0.5, ridge=2.0))
        ft = torch.as_tensor(f)
        st = torch.as_tensor(sens)

        def forward(p, ops):
            kap = ops.level_set(p.field, inside, outside, eps=0.03)
            A = ops.zeros(n, n).clone()
            for i in range(1, n - 1):
                kl = 0.5 * (kap[i] + kap[i - 1])
                kr = 0.5 * (kap[i] + kap[i + 1])
                A[i, i - 1] = -kl / h**2
                A[i, i] = (kl + kr) / h**2
                A[i, i + 1] = -kr / h**2
            A[0, 0] = 1.0
            A[-1, -1] = 1.0
            return ops.solve(A, ft)

        obs = Differential(u_obs, over=phi, scale=sig, forward=forward, observe=lambda u, p, ops: u[st])
        post = joint([obs]).fit(how="laplace", max_iter=400)
        phi_m, phi_s = post.posterior("phi")
        region_acc = ((phi_m > 0) == (true_phi > 0)).mean()
        self.assertGreater(region_acc, 0.7)  # recovered the inclusion region
        self.assertTrue(np.all(phi_s > 0))  # with uncertainty everywhere


if __name__ == "__main__":
    unittest.main()
