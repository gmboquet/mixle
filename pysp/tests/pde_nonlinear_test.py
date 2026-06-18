"""Tests for nonlinear / multi-parameter PDE inference via the autograd adjoint (WS-D)."""

import importlib.util
import unittest

import numpy as np

from pysp.ppl.dynamics import laplacian_matrix
from pysp.ppl.pde import fit_pde_parameters, fit_reaction_diffusion

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _simulate_fisher_kpp(d_true, r_true, *, n=24, steps=60, dt=0.002, noise=1e-4, seed=0):
    """Explicit-Euler Fisher-KPP rollout: u_{t+1} = u + dt (D L u + r u (1-u))."""
    lap = laplacian_matrix(n, 1.0 / (n - 1), "neumann")
    rng = np.random.RandomState(seed)
    grid = np.linspace(0.0, 1.0, n)
    u = 0.3 + 0.2 * np.sin(np.pi * grid)  # interior initial profile (so the nonlinear term is active)
    rows = [u.copy()]
    for _ in range(steps):
        u = u + dt * (d_true * (lap @ u) + r_true * u * (1.0 - u))
        rows.append(u.copy())
    y = np.asarray(rows)
    return y + rng.normal(0.0, noise, y.shape)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class NonlinearPdeTest(unittest.TestCase):
    def test_recovers_diffusion_and_growth(self):
        d_true, r_true = 0.4, 2.5
        y = _simulate_fisher_kpp(d_true, r_true)
        fit = fit_reaction_diffusion(y, dt=0.002, init_diffusivity=1.0, init_growth=1.0, max_its=1500)
        self.assertAlmostEqual(fit["diffusivity"], d_true, delta=0.12)
        self.assertAlmostEqual(fit["growth"], r_true, delta=0.4)
        self.assertLess(fit["obs_sd"], 0.02)

    def test_general_fit_matches_linear_diffusion(self):
        # The general multi-parameter fit reduces to pure diffusion when the reaction rate -> 0.
        import torch

        n = 20
        lap = torch.tensor(laplacian_matrix(n, 1.0 / (n - 1), "neumann"), dtype=torch.float64)
        d_true = 0.8
        lap_np = laplacian_matrix(n, 1.0 / (n - 1), "neumann")
        rng = np.random.RandomState(1)
        u = np.sin(np.pi * np.linspace(0, 1, n))
        rows = [u.copy()]
        for _ in range(50):
            u = u + 0.002 * (d_true * (lap_np @ u))
            rows.append(u.copy())
        y = np.asarray(rows) + rng.normal(0, 1e-5, (51, n))

        def transition(u_prev, params):
            return u_prev + 0.002 * (u_prev @ (params["diffusivity"] * lap).T)

        fit = fit_pde_parameters(y, transition, {"diffusivity": 0.2}, max_its=1200)
        self.assertAlmostEqual(fit["diffusivity"], d_true, delta=0.1)

    def test_requires_timeseries(self):
        with self.assertRaises(ValueError):
            fit_reaction_diffusion(np.zeros((1, 10)))


if __name__ == "__main__":
    unittest.main()
