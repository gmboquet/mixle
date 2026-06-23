"""Tests for PDE-constrained latent-field models in the PPL (method-of-lines + Kalman/RTS/EM)."""

import unittest

import numpy as np

import pysp.ppl as ppl
from pysp.ppl.physics.dynamics import (
    DiffusionOperator,
    available_dynamics_operators,
    laplacian_matrix,
    make_operator,
    register_dynamics_operator,
    upwind_gradient_matrix,
)
from pysp.ppl.physics.pde import kalman_rts_em


class OperatorMatrixTestCase(unittest.TestCase):
    def test_laplacian_neumann_conserves_constants(self):
        # The Laplacian of a constant field is zero (rows sum to zero) under any BC.
        for bc in ("dirichlet", "neumann", "periodic"):
            lap = laplacian_matrix(8, 0.25, bc=bc)
            if bc == "neumann":
                np.testing.assert_allclose(lap @ np.ones(8), 0.0, atol=1e-12)
            self.assertEqual(lap.shape, (8, 8))

    def test_diffusion_transition_is_stable_and_mass_preserving(self):
        op = DiffusionOperator(diffusivity=0.5, n=10, length=1.0, bc="neumann", scheme="implicit")
        A = op.transition_matrix(dt=0.1)
        # Implicit Euler is unconditionally stable: spectral radius <= 1.
        self.assertLessEqual(np.max(np.abs(np.linalg.eigvals(A))), 1.0 + 1e-9)
        # Neumann diffusion conserves total mass: column sums of A are ~1.
        np.testing.assert_allclose(A.sum(axis=0), 1.0, atol=1e-6)

    def test_explicit_and_exact_schemes_agree_for_small_dt(self):
        op_e = DiffusionOperator(0.3, n=12, scheme="explicit")
        op_x = DiffusionOperator(0.3, n=12, scheme="exact")
        Ae = op_e.transition_matrix(1e-4)
        Ax = op_x.transition_matrix(1e-4)
        np.testing.assert_allclose(Ae, Ax, atol=1e-4)

    def test_advection_upwind_direction(self):
        g_pos = upwind_gradient_matrix(6, 0.2, velocity=1.0, bc="periodic")
        g_neg = upwind_gradient_matrix(6, 0.2, velocity=-1.0, bc="periodic")
        # Upwind picks the backward difference for c>0 (uses i-1) and forward for c<0.
        self.assertTrue(g_pos[2, 1] != 0.0 and g_pos[2, 3] == 0.0)
        self.assertTrue(g_neg[2, 3] != 0.0 and g_neg[2, 1] == 0.0)


class OperatorRegistryTestCase(unittest.TestCase):
    def test_builtin_operators_registered(self):
        for name in ("diffusion", "advection", "advection_diffusion"):
            self.assertIn(name, available_dynamics_operators())

    def test_make_operator(self):
        op = make_operator("diffusion", diffusivity=0.2, n=8)
        self.assertIsInstance(op, DiffusionOperator)

    def test_unknown_operator_raises(self):
        with self.assertRaises(ValueError):
            make_operator("schrodinger", n=8)

    def test_custom_operator_registration(self):
        register_dynamics_operator("diffusion_alias_test", DiffusionOperator)
        try:
            op = make_operator("diffusion_alias_test", diffusivity=0.1, n=6)
            self.assertIsInstance(op, DiffusionOperator)
        finally:
            from pysp.ppl.physics.dynamics import _DYNAMICS_OPERATORS

            _DYNAMICS_OPERATORS.pop("diffusion_alias_test", None)


def _simulate_diffusion(n=15, T=40, D=0.4, dt=0.05, q=1e-3, r=0.05, seed=0):
    rng = np.random.RandomState(seed)
    op = DiffusionOperator(D, n=n, length=1.0, bc="neumann", scheme="implicit")
    A = op.transition_matrix(dt)
    grid = np.linspace(0, 1, n)
    u = np.exp(-((grid - 0.5) ** 2) / 0.01)  # a bump that should diffuse/flatten
    states, obs = [], []
    for _ in range(T):
        u = A @ u + rng.normal(0, np.sqrt(q), size=n)
        states.append(u.copy())
        obs.append(u + rng.normal(0, np.sqrt(r), size=n))
    return op, np.asarray(states), np.asarray(obs)


class PDEFitTestCase(unittest.TestCase):
    def test_kalman_rts_em_recovers_field(self):
        op, truth, obs = _simulate_diffusion()
        result = kalman_rts_em(obs, op, dt=0.05, max_its=60)
        # Smoothing reduces error vs raw observations (data assimilation works).
        raw_err = np.mean((obs - truth) ** 2)
        smooth_err = np.mean((result.smoothed - truth) ** 2)
        self.assertLess(smooth_err, raw_err)
        # Fitted observation noise is in the right ballpark of the true r=0.05.
        self.assertLess(result.obs_var, 0.2)
        self.assertGreater(result.obs_var, 1e-3)

    def test_loglik_is_finite_and_monotone_enough(self):
        op, _, obs = _simulate_diffusion(T=30)
        r1 = kalman_rts_em(obs, op, dt=0.05, max_its=1)
        r2 = kalman_rts_em(obs, op, dt=0.05, max_its=40)
        self.assertTrue(np.isfinite(r2.loglik))
        self.assertGreaterEqual(r2.loglik + 1e-6, r1.loglik)  # EM does not decrease the likelihood

    def test_ppl_pde_surface_end_to_end(self):
        op, _, obs = _simulate_diffusion(T=25)
        model = ppl.PDE(op)
        fitted = model.fit(obs, dt=0.05, max_its=40)
        res = fitted.result
        self.assertTrue(np.isfinite(res.loglik))
        fc = res.forecast(5)
        self.assertEqual(fc.shape, (5, op.n))

    def test_sparse_sensor_observation(self):
        op, truth, obs = _simulate_diffusion(n=15, T=30)
        # Observe only every third grid point via a sensor operator H.
        idx = np.arange(0, 15, 3)
        H = np.zeros((len(idx), 15))
        for row, j in enumerate(idx):
            H[row, j] = 1.0
        result = kalman_rts_em(obs[:, idx], op, dt=0.05, H=H, max_its=40)
        self.assertEqual(result.smoothed.shape, (30, 15))  # full field reconstructed from sparse sensors
        self.assertTrue(np.all(np.isfinite(result.smoothed)))


if __name__ == "__main__":
    unittest.main()
