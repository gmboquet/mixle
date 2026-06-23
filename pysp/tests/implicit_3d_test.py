"""Tests for implicit-diffusion stepping and 3D PDE inverse problems (phase 5 completion)."""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pysp.ppl import GP, RBF, Differential, NavierStokes2D, free, joint
    from pysp.ppl.physics.ops import make_ops
    from pysp.ppl.physics.pde_solve import laplacian


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ImplicitDiffusionTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_implicit_stable_where_explicit_blows_up(self):
        ops = make_ops()
        n = 24
        h = 1.0 / (n - 1)
        nu = 1.0
        dt = 2.0 * h**2 / nu  # dt*nu/h^2 = 2 >> 0.25: explicit diffusion is unstable
        xx, yy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
        om0 = torch.as_tensor(np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.02).ravel())
        expl = NavierStokes2D(n, viscosity=nu, dt=dt)
        impl = NavierStokes2D(n, viscosity=nu, dt=dt, implicit_diffusion=True)
        oe, oi = om0.clone(), om0.clone()
        for _ in range(30):
            oe = expl.step(oe, ops)
            oi = impl.step(oi, ops)
        self.assertFalse(torch.isfinite(oe).all())  # explicit diverges
        self.assertTrue(torch.isfinite(oi).all())  # implicit stays bounded
        self.assertLess(float(oi.abs().max()), 1.0)

    def test_implicit_navier_stokes_inverse(self):
        ops = make_ops()
        n = 24
        h = 1.0 / (n - 1)
        impl = NavierStokes2D(n, viscosity=0.1, dt=0.1 * h, implicit_diffusion=True)
        xx, yy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
        shape0 = torch.as_tensor(np.exp(-((xx - 0.5) ** 2 + (yy - 0.45) ** 2) / 0.03).ravel())
        maskt = torch.as_tensor(impl._mask)
        recv = torch.as_tensor(np.array([n * 8 + 10, n * 8 + 14, n * 16 + 10, n * 16 + 14]))
        nt = 40

        def record(om, i):
            u, _ = impl.velocity(impl.streamfunction(om, ops), ops)
            return u[recv]

        rec = ops.integrate_record(lambda om, i: impl.step(om, ops), shape0 * maskt, nt, record, checkpoint=7)
        y_obs = rec.detach().numpy().reshape(-1) + 1e-4 * np.random.RandomState(0).randn(rec.numel())
        amp = free(1, name="amp", support="positive")

        def forward(p, o):
            return o.integrate_record(
                lambda om, i: impl.step(om, o), (p.amp * shape0) * maskt, nt, record, checkpoint=7
            ).reshape(-1)

        am, asd = (
            joint([Differential(y_obs, drivers=[amp], scale=1e-4, forward=forward)])
            .fit(how="gauss_newton")
            .posterior("amp")
        )
        self.assertLess(abs(am - 1.0), 2 * asd)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ThreeDimensionalInverseTestCase(unittest.TestCase):
    """The whole pipeline (n-D assembly + adjoint sparse solve + posterior) works in 3D."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_recovers_3d_coefficient(self):
        m = 10
        M = m**3
        gx = np.linspace(0, 1, m)
        h = gx[1] - gx[0]
        xx, yy, zz = np.meshgrid(gx, gx, gx, indexing="ij")
        Lr, Lc, Lv, N = laplacian((m, m, m), spacing=h)
        bnd = np.zeros((m, m, m), bool)
        bnd[0] = bnd[-1] = bnd[:, 0] = bnd[:, -1] = bnd[:, :, 0] = bnd[:, :, -1] = True
        q = np.sin(np.pi * xx) * np.sin(np.pi * yy) * np.sin(np.pi * zz)
        q[bnd] = 0.0
        qt = torch.as_tensor(q.ravel())
        u1 = torch.linalg.solve(torch.zeros(M, M).index_put_((Lr, Lc), Lv, accumulate=True), qt).numpy()
        u_true = u1 / 2.5
        rng = np.random.RandomState(0)
        sens = np.sort(rng.choice(np.where(~bnd.ravel())[0], 80, replace=False))
        sig = 1e-4
        u_obs = u_true[sens] + sig * rng.randn(len(sens))
        st = torch.as_tensor(sens)
        D = free(1, name="D", support="positive")
        obs = Differential(
            u_obs,
            drivers=[D],
            scale=sig,
            forward=lambda p, o: o.sparse_solve(Lr, Lc, Lv, N, qt / p.D),
            observe=lambda u, p, o: u[st],
        )
        Dm, Ds = joint([obs]).fit(how="gauss_newton").posterior("D")
        self.assertLess(abs(Dm - 2.5), 2 * Ds)

    def test_3d_field_posterior_runs(self):
        m = 8
        xx, yy, zz = np.meshgrid(*[np.linspace(0, 1, m)] * 3, indexing="ij")
        coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1)
        field = GP("q", index=coords, kernel=RBF(lengthscale=0.4))
        rng = np.random.RandomState(0)
        idx = rng.choice(m**3, 60, replace=False)
        st = torch.as_tensor(idx)
        y = np.sin(2 * xx).ravel()[idx] + 0.01 * rng.randn(len(idx))
        obs = Differential(y, over=field, scale=0.01, forward=lambda p, o: p.field, observe=lambda f, p, o: f[st])
        qm, qs = joint([obs]).fit(how="gauss_newton").posterior("q")
        self.assertEqual(qm.shape[0], m**3)
        self.assertTrue(np.all(qs > 0))


if __name__ == "__main__":
    unittest.main()
