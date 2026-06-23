"""Tests for pysp.ppl differential-equation inverse problems (ODE/PDE forward models, posteriors over drivers)."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pysp.ppl import Differential, GaussianField, RandomWalk, free, joint
    from pysp.ppl.physics.ops import make_ops


def _sim(rhs, y0, t, ops=None):
    """Simulate a trajectory with the same integrator the model uses (no discretization mismatch)."""
    ops = ops or make_ops()
    return ops.integrate(rhs, ops.tensor(float(y0)), ops.tensor(t)).detach().numpy()


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class IntegratorTestCase(unittest.TestCase):
    def test_rk4_matches_analytic_decay(self):
        ops = make_ops()
        t = np.linspace(0, 4, 200)
        traj = ops.integrate(lambda u, tt: -0.7 * u, ops.tensor(1.0), ops.tensor(t))
        self.assertLess(np.max(np.abs(traj.detach().numpy() - np.exp(-0.7 * t))), 1e-5)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class OdeParameterInferenceTestCase(unittest.TestCase):
    """Recover ODE coefficients (the unobservable drivers) from a noisy trajectory; no shared field."""

    def test_linear_decay_rate(self):
        rng = np.random.RandomState(0)
        t = np.linspace(0, 5, 40)
        y = _sim(lambda u, tt: -0.8 * u, 1.0, t) + 0.03 * rng.randn(40)
        k = free(1, name="k", support="positive")
        obs = Differential(y, drivers=[k], y0=1.0, t_grid=t, scale=0.03, rhs=lambda u, t, p, ops: -p.k * u)
        post = joint([obs]).fit(how="laplace")
        km, ks = post.posterior("k")
        self.assertLess(abs(km - 0.8), 2 * ks)
        self.assertGreater(ks, 0.0)
        self.assertIn("k", post.summary())

    def test_nonlinear_logistic(self):
        rng = np.random.RandomState(1)
        t = np.linspace(0, 10, 60)
        y = _sim(lambda u, tt: 0.9 * u * (1 - u / 5.0), 0.5, t) + 0.06 * rng.randn(60)
        r = free(1, name="r", support="positive")
        K = free(1, name="K", support="positive")
        obs = Differential(
            y, drivers=[r, K], y0=0.5, t_grid=t, scale=0.06, rhs=lambda u, t, p, ops: p.r * u * (1 - u / p.K)
        )
        post = joint([obs]).fit(how="laplace")
        rm, rs = post.posterior("r")
        Km, Ks = post.posterior("K")
        self.assertLess(abs(rm - 0.9), 2 * rs)
        self.assertLess(abs(Km - 5.0), 2 * Ks)

    def test_coverage_over_seeds(self):
        t = np.linspace(0, 5, 30)
        inside = 0
        for s in range(20):
            rng = np.random.RandomState(s)
            y = _sim(lambda u, tt: -0.6 * u, 1.0, t) + 0.04 * rng.randn(30)
            k = free(1, name="k", support="positive")
            obs = Differential(y, drivers=[k], y0=1.0, t_grid=t, scale=0.04, rhs=lambda u, t, p, ops: -p.k * u)
            km, ks = joint([obs]).fit(how="laplace").posterior("k")
            inside += abs(km - 0.6) < 1.96 * ks
        self.assertGreaterEqual(inside, 16)


def _laplacian_dirichlet(n, h, D=1.0):
    L = np.zeros((n, n))
    for i in range(1, n - 1):
        L[i, i - 1] = -D / h**2
        L[i, i] = 2 * D / h**2
        L[i, i + 1] = -D / h**2
    L[0, 0] = L[-1, -1] = 1.0
    return L


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class PdeSteadyInversionTestCase(unittest.TestCase):
    """Steady diffusion -D u'' = q via a dense solve in the forward callback (Laplace exact here)."""

    def test_recovers_source_field_and_matches_closed_form(self):
        n = 30
        x = np.linspace(0, 1, n)
        h = x[1] - x[0]
        sig = 0.003
        sens = np.arange(2, n - 2, 3)
        L = _laplacian_dirichlet(n, h, 1.0)
        q_true = np.sin(2 * np.pi * x) * x
        q_true[0] = q_true[-1] = 0.0
        u_obs = np.linalg.solve(L, q_true)[sens] + sig * np.random.RandomState(2).randn(len(sens))

        q = GaussianField(np.arange(n), RandomWalk(scale=0.3, ridge=5.0), name="q")
        Lt = torch.as_tensor(L)
        sens_t = torch.as_tensor(sens)
        obs = Differential(
            u_obs,
            over=q,
            scale=sig,
            forward=lambda p, ops: ops.solve(Lt, p.field),
            observe=lambda u, p, ops: u[sens_t],
        )
        post = joint([obs]).fit(how="laplace")
        qm, qs = post.posterior("q")
        self.assertGreater(np.corrcoef(qm, q_true)[0, 1], 0.9)
        G = np.linalg.inv(L)[sens, :]
        sd_cf = np.sqrt(np.diag(np.linalg.inv(q.precision + G.T @ G / sig**2)))
        self.assertLess(np.max(np.abs(qs - sd_cf)), 1e-6)

    def test_recovers_coefficient(self):
        n = 30
        x = np.linspace(0, 1, n)
        h = x[1] - x[0]
        sig = 0.002
        sens = np.arange(2, n - 2, 3)
        q_true = np.sin(2 * np.pi * x) * x
        q_true[0] = q_true[-1] = 0.0
        u_obs = np.linalg.solve(_laplacian_dirichlet(n, h, 2.3), q_true)[sens]
        u_obs = u_obs + sig * np.random.RandomState(3).randn(len(sens))
        qt = torch.as_tensor(q_true)
        sens_t = torch.as_tensor(sens)

        def forward(p, ops):
            A = ops.zeros(n, n).clone()
            idx = ops.arange(n - 2) + 1
            A[idx, idx - 1] = -p.D / h**2
            A[idx, idx] = 2 * p.D / h**2
            A[idx, idx + 1] = -p.D / h**2
            A[0, 0] = 1.0
            A[-1, -1] = 1.0
            return ops.solve(A, qt)

        D = free(1, name="D", support="positive")
        obs = Differential(u_obs, drivers=[D], scale=sig, forward=forward, observe=lambda u, p, ops: u[sens_t])
        Dm, Ds = joint([obs]).fit(how="laplace").posterior("D")
        self.assertLess(abs(Dm - 2.3), 2 * Ds)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class PoissonObservationTestCase(unittest.TestCase):
    def test_counts_from_growth(self):
        rng = np.random.RandomState(4)
        t = np.linspace(0, 4, 30)
        counts = rng.poisson(_sim(lambda u, tt: 0.5 * u, 2.0, t))
        r = free(1, name="r", support="positive")
        obs = Differential(counts, drivers=[r], y0=2.0, t_grid=t, family="poisson", rhs=lambda u, t, p, ops: p.r * u)
        rm, rs = joint([obs]).fit(how="laplace").posterior("r")
        self.assertLess(abs(rm - 0.5), 3 * rs)


if __name__ == "__main__":
    unittest.main()
