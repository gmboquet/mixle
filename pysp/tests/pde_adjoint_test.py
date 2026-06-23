"""Tests for autograd-adjoint PDE-parameter inference (WS-D).

Recovers a known diffusion coefficient from simulated noisy snapshots; gradients flow through the
differentiable transition (reverse-mode autodiff = the discrete adjoint of the forward solve).
"""

import importlib.util
import unittest

import numpy as np

from pysp.ppl.physics.dynamics import DiffusionOperator
from pysp.ppl.physics.pde import fit_diffusivity

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _simulate(diffusivity, n=24, steps=40, dt=0.02, noise=1e-3, seed=0, scheme="exact"):
    op = DiffusionOperator(diffusivity, n=n, length=1.0, bc="neumann", scheme=scheme)
    a = op.transition_matrix(dt)
    rng = np.random.RandomState(seed)
    grid = np.linspace(0.0, 1.0, n)
    u = np.sin(np.pi * grid) + 0.5 * np.sin(3.0 * np.pi * grid)  # multi-mode initial field
    rows = [u.copy()]
    for _ in range(steps):
        u = a @ u
        rows.append(u.copy())
    y = np.asarray(rows)
    return y + rng.normal(0.0, noise, y.shape)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class PdeAdjointTest(unittest.TestCase):
    def test_recovers_diffusivity(self):
        for d_true in (0.5, 2.0):
            with self.subTest(d_true=d_true):
                y = _simulate(d_true, scheme="exact")
                fit = fit_diffusivity(y, scheme="exact", dt=0.02, init_diffusivity=1.0, max_its=600)
                self.assertAlmostEqual(fit["diffusivity"], d_true, delta=0.15 * d_true)
                self.assertLess(fit["obs_sd"], 0.05)

    def test_recovers_under_implicit_scheme(self):
        y = _simulate(1.0, scheme="implicit")
        fit = fit_diffusivity(y, scheme="implicit", dt=0.02, init_diffusivity=0.3, max_its=600)
        self.assertAlmostEqual(fit["diffusivity"], 1.0, delta=0.2)

    def test_requires_timeseries(self):
        with self.assertRaises(ValueError):
            fit_diffusivity(np.zeros((1, 10)))


if __name__ == "__main__":
    unittest.main()
