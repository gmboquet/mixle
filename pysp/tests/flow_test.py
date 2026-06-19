"""Tests for 2D Navier-Stokes flow inverse problems and the finite-difference gradient op (phase 5)."""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pysp.ppl import Differential, NavierStokes2D, free, joint
    from pysp.ppl.ops import make_ops


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class GradTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_central_difference_matches_analytic(self):
        ops = make_ops()
        m = 30
        x = np.linspace(0, 1, m)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        f = torch.as_tensor(np.sin(2 * np.pi * xx) * yy)
        dfdx = ops.grad(f.reshape(-1), (m, m), 0, spacing=x[1] - x[0]).reshape(m, m).numpy()
        truth = 2 * np.pi * np.cos(2 * np.pi * xx) * yy
        self.assertLess(np.abs(dfdx[2:-2, 2:-2] - truth[2:-2, 2:-2]).max(), 0.05)


def _ns_problem(n=24, nt=60, amp_true=1.0):
    nu = 0.1
    h = 1.0 / (n - 1)
    dt = 0.1 * h
    ns = NavierStokes2D(n, viscosity=nu, dt=dt)
    xx, yy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    shape0 = torch.as_tensor(np.exp(-((xx - 0.5) ** 2 + (yy - 0.45) ** 2) / 0.03).ravel())
    recv = torch.as_tensor(np.array([n * 8 + 10, n * 8 + 14, n * 16 + 10, n * 16 + 14, n * 12 + 12]))
    maskt = torch.as_tensor(ns._mask)

    def make_forward(checkpoint):
        def forward(p, ops):
            def step(om, i):
                return ns.step(om, ops)

            def record(om, i):
                u, _ = ns.velocity(ns.streamfunction(om, ops), ops)
                return u[recv]

            return ops.integrate_record(step, (p.amp * shape0) * maskt, nt, record, checkpoint=checkpoint).reshape(-1)

        return forward

    return ns, shape0, recv, maskt, make_forward


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class NavierStokesForwardTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_forward_is_stable(self):
        ops = make_ops()
        ns, shape0, recv, maskt, make_forward = _ns_problem()

        def step(om, i):
            return ns.step(om, ops)

        def record(om, i):
            u, _ = ns.velocity(ns.streamfunction(om, ops), ops)
            return u[recv]

        rec = ops.integrate_record(step, shape0 * maskt, 60, record, checkpoint=8)
        self.assertTrue(torch.isfinite(rec).all())
        self.assertLess(float(rec.abs().max()), 10.0)  # bounded flow (no blow-up)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class NavierStokesInverseTestCase(unittest.TestCase):
    """Recover the upstream/initial flow configuration from downstream velocity observations."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_recovers_upstream_amplitude(self):
        ops = make_ops()
        ns, shape0, recv, maskt, make_forward = _ns_problem(amp_true=1.0)
        forward = make_forward(checkpoint=8)

        class _P:
            amp = torch.tensor(1.0)

        rec_true = forward(_P(), ops).detach().numpy()
        y_obs = rec_true + 1e-4 * np.random.RandomState(0).randn(rec_true.size)

        amp = free(1, name="amp", support="positive")
        obs = Differential(y_obs, drivers=[amp], scale=1e-4, forward=make_forward(checkpoint=8))
        am, asd = joint([obs]).fit(how="gauss_newton").posterior("amp")
        self.assertLess(abs(am - 1.0), 2 * asd)
        self.assertGreater(asd, 0.0)


if __name__ == "__main__":
    unittest.main()
