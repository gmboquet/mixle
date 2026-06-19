"""Tests for pysp.ppl differential-equation inverse problems (ODE/PDE forward models, posteriors over drivers)."""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from pysp.ppl import DifferentialProxy, GaussianField, RandomWalk, fit_field


@unittest.skipUnless(HAS_TORCH, "DifferentialProxy requires PyTorch")
class IntegratorTestCase(unittest.TestCase):
    def test_rk4_matches_analytic_decay(self):
        from pysp.ppl import integrate_ode

        t = np.linspace(0, 4, 200)
        theta = {"k": torch.tensor(0.7)}
        traj = integrate_ode(lambda s, t, th, T: -th["k"] * s, torch.tensor(1.0), torch.as_tensor(t), theta, torch)
        analytic = np.exp(-0.7 * t)
        self.assertLess(np.max(np.abs(traj.detach().numpy() - analytic)), 1e-5)


@unittest.skipUnless(HAS_TORCH, "DifferentialProxy requires PyTorch")
class OdeParameterInferenceTestCase(unittest.TestCase):
    """Recover ODE coefficients (the unobservable drivers) from a noisy trajectory; field=None."""

    def _sim(self, rhs, y0, t, theta):
        from pysp.ppl import integrate_ode

        return integrate_ode(rhs, torch.tensor(float(y0)), torch.as_tensor(t), theta, torch).detach().numpy()

    def test_linear_decay_rate(self):
        rng = np.random.RandomState(0)
        t = np.linspace(0, 5, 40)
        rhs = lambda s, t, th, T: -th["k"] * s
        y = self._sim(rhs, 1.0, t, {"k": torch.tensor(0.8)}) + 0.03 * rng.randn(40)
        post = fit_field(
            None,
            [DifferentialProxy(y, t_grid=t, y0=1.0, scale=0.03, rhs=rhs, params=[("k", "positive", 0.5)])],
            how="laplace",
            max_iter=200,
        )
        km, ks = post.posterior("ode.k")
        self.assertLess(abs(km - 0.8), 2 * ks)
        self.assertGreater(ks, 0.0)
        self.assertIn("ode.k", post.summary())

    def test_nonlinear_logistic(self):
        rng = np.random.RandomState(1)
        t = np.linspace(0, 10, 60)
        rhs = lambda s, t, th, T: th["r"] * s * (1 - s / th["K"])
        y = self._sim(rhs, 0.5, t, {"r": torch.tensor(0.9), "K": torch.tensor(5.0)}) + 0.06 * rng.randn(60)
        post = fit_field(
            None,
            [
                DifferentialProxy(
                    y, t_grid=t, y0=0.5, scale=0.06, rhs=rhs, params=[("r", "positive", 0.5), ("K", "positive", 3.0)]
                )
            ],
            how="laplace",
            max_iter=400,
        )
        rm, rs = post.posterior("ode.r")
        Km, Ks = post.posterior("ode.K")
        self.assertLess(abs(rm - 0.9), 2 * rs)
        self.assertLess(abs(Km - 5.0), 2 * Ks)

    def test_coverage_over_seeds(self):
        t = np.linspace(0, 5, 30)
        rhs = lambda s, t, th, T: -th["k"] * s
        inside = 0
        for s in range(20):
            rng = np.random.RandomState(s)
            y = self._sim(rhs, 1.0, t, {"k": torch.tensor(0.6)}) + 0.04 * rng.randn(30)
            post = fit_field(
                None,
                [DifferentialProxy(y, t_grid=t, y0=1.0, scale=0.04, rhs=rhs, params=[("k", "positive", 0.4)])],
                how="laplace",
                max_iter=200,
            )
            km, ks = post.posterior("ode.k")
            inside += abs(km - 0.6) < 1.96 * ks
        self.assertGreaterEqual(inside, 16)  # nominal ~19/20


def _laplacian_dirichlet(n, h, D=1.0):
    L = np.zeros((n, n))
    for i in range(1, n - 1):
        L[i, i - 1] = -D / h**2
        L[i, i] = 2 * D / h**2
        L[i, i + 1] = -D / h**2
    L[0, 0] = L[-1, -1] = 1.0  # Dirichlet u=0 at the ends
    return L


@unittest.skipUnless(HAS_TORCH, "DifferentialProxy requires PyTorch")
class PdeSourceInversionTestCase(unittest.TestCase):
    """Recover a latent source field q from a steady diffusion equation -D u'' = q via sensors."""

    def test_recovers_source_and_matches_closed_form(self):
        n = 30
        x = np.linspace(0, 1, n)
        h = x[1] - x[0]
        sig = 0.003
        sens = np.arange(2, n - 2, 3)
        L = _laplacian_dirichlet(n, h, D=1.0)
        q_true = np.sin(2 * np.pi * x) * x
        q_true[0] = q_true[-1] = 0.0
        u_obs = np.linalg.solve(L, q_true)[sens] + sig * np.random.RandomState(2).randn(len(sens))

        field = GaussianField(np.arange(n), RandomWalk(scale=0.3, ridge=5.0), name="q")
        Lt = torch.as_tensor(L)
        proxy = DifferentialProxy(
            u_obs,
            solver="linear_steady",
            uses_field=True,
            scale=sig,
            operator=lambda th, T: Lt,
            source=lambda th, T: th["field"],
            observe=lambda u, th, T: u[torch.as_tensor(sens)],
        )
        post = fit_field(field, [proxy], how="laplace", max_iter=500)
        qm, qs = post.posterior("q")
        self.assertGreater(np.corrcoef(qm, q_true)[0, 1], 0.9)

        # closed form: predicted = (H L^{-1}) q is linear in q, so the posterior is exactly Gaussian
        G = np.linalg.inv(L)[sens, :]
        prec_cf = field.precision + G.T @ G / sig**2
        sd_cf = np.sqrt(np.diag(np.linalg.inv(prec_cf)))
        self.assertLess(np.max(np.abs(qs - sd_cf)), 1e-6)

    def test_recovers_coefficient(self):
        n = 30
        x = np.linspace(0, 1, n)
        h = x[1] - x[0]
        sig = 0.002
        sens = np.arange(2, n - 2, 3)
        q_true = np.sin(2 * np.pi * x) * x
        q_true[0] = q_true[-1] = 0.0
        u_obs = np.linalg.solve(_laplacian_dirichlet(n, h, D=2.3), q_true)[sens]
        u_obs = u_obs + sig * np.random.RandomState(3).randn(len(sens))
        qt = torch.as_tensor(q_true)

        def op(th, T):
            D = th["D"]
            M = T.zeros((n, n), dtype=T.float64).clone()
            idx = T.arange(1, n - 1)
            M[idx, idx - 1] = -D / h**2
            M[idx, idx] = 2 * D / h**2
            M[idx, idx + 1] = -D / h**2
            M[0, 0] = 1.0
            M[-1, -1] = 1.0
            return M

        proxy = DifferentialProxy(
            u_obs,
            solver="linear_steady",
            scale=sig,
            operator=op,
            source=lambda th, T: qt,
            observe=lambda u, th, T: u[torch.as_tensor(sens)],
            params=[("D", "positive", 1.0)],
        )
        post = fit_field(None, [proxy], how="laplace", max_iter=300)
        Dm, Ds = post.posterior("ode.D")
        self.assertLess(abs(Dm - 2.3), 2 * Ds)


@unittest.skipUnless(HAS_TORCH, "DifferentialProxy requires PyTorch")
class PoissonObservationTestCase(unittest.TestCase):
    """An ODE forward model with count observations (Poisson family)."""

    def test_counts_from_growth(self):
        from pysp.ppl import integrate_ode

        rng = np.random.RandomState(4)
        t = np.linspace(0, 4, 30)
        rhs = lambda s, t, th, T: th["r"] * s
        traj = integrate_ode(rhs, torch.tensor(2.0), torch.as_tensor(t), {"r": torch.tensor(0.5)}, torch)
        counts = rng.poisson(traj.detach().numpy())
        proxy = DifferentialProxy(counts, t_grid=t, y0=2.0, family="poisson", rhs=rhs, params=[("r", "positive", 0.2)])
        post = fit_field(None, [proxy], how="laplace", max_iter=200)
        rm, rs = post.posterior("ode.r")
        self.assertLess(abs(rm - 0.5), 3 * rs)


if __name__ == "__main__":
    unittest.main()
