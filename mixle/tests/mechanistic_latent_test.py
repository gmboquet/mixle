"""Tests for the mechanistic (linear state-space) latent prior (mixle.reason Latent.mechanistic)."""

import unittest

import numpy as np

from mixle.reason import Evidence, Latent, block_selector, reason


class MechanisticPriorTest(unittest.TestCase):
    def test_marginal_blocks_and_mean_follow_dynamics(self):
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        P0 = np.eye(2) * 2.0
        Q = np.eye(2) * 0.1
        prior = Latent.mechanistic(A, steps=5, x0_mean=[1.0, -1.0], x0_cov=P0, process_cov=Q)
        self.assertEqual(np.size(prior.mean()), 10)  # 5 steps x 2-dim
        m = prior.mean().reshape(5, 2)
        # mean propagates by A
        np.testing.assert_allclose(m[1], A @ m[0], atol=1e-10)
        np.testing.assert_allclose(m[4], A @ m[3], atol=1e-10)
        # diagonal blocks are the forward marginal covariances P_{t+1} = A P_t Aᵀ + Q
        cov = prior.cov()
        P1 = A @ P0 @ A.T + Q
        np.testing.assert_allclose(cov[2:4, 2:4], P1, atol=1e-10)

    def test_prior_is_valid_psd(self):
        A = np.array([[0.95]])
        prior = Latent.mechanistic(A, steps=8, x0_cov=[[1.0]], process_cov=[[0.05]])
        evals = np.linalg.eigvalsh(prior.cov())
        self.assertGreaterEqual(evals.min(), -1e-10)

    def test_sampled_trajectories_obey_dynamics_when_noise_small(self):
        A = np.array([[0.8]])
        prior = Latent.mechanistic(A, steps=6, x0_cov=[[1.0]], process_cov=[[1e-8]])
        traj = prior.sample(1, rng=0).reshape(6)
        # near-deterministic: z_{t+1} ~ 0.8 z_t
        for t in range(5):
            self.assertAlmostEqual(traj[t + 1], 0.8 * traj[t], delta=1e-2)


class MechanisticReasoningTest(unittest.TestCase):
    def test_evidence_at_one_time_informs_all_times(self):
        # The point of a mechanistic prior: observing the state at t=0 sharpens the belief at LATER
        # times through the dynamics -- an isotropic prior cannot do this.
        A = np.array([[0.9]])
        prior = Latent.mechanistic(A, steps=6, x0_cov=[[3.0]], process_cov=[[0.02]])
        H0 = block_selector(0, n_blocks=6, block_dim=1)
        ans = reason(prior, [Evidence(H0, [2.0], [[0.01]], "obs@0")])
        post = ans.belief
        # uncertainty at t=5 drops relative to the prior, purely from an observation at t=0
        self.assertLess(post.marginal([5]).sd()[0], prior.marginal([5]).sd()[0])
        # and the posterior mean at t=5 is the dynamics-propagated observation (~0.9^5 * 2)
        self.assertAlmostEqual(float(post.mean()[5]), 0.9**5 * 2.0, delta=0.1)

    def test_sparse_observations_are_smoothed_by_physics(self):
        # Observe only t=0 and t=5; the unobserved middle states are filled in following A, and the
        # smoothed trajectory tracks the true one.
        A = np.array([[0.85, 0.0], [0.15, 0.9]])
        d, T = 2, 6
        rng = np.random.RandomState(0)
        z = [np.array([2.0, 0.0])]
        for _ in range(1, T):
            z.append(A @ z[-1])
        z_true = np.array(z)

        prior = Latent.mechanistic(A, steps=T, x0_cov=np.eye(d) * 5.0, process_cov=np.eye(d) * 1e-4)
        ev = []
        for t in (0, 5):
            H = block_selector(t, T, d)
            ev.append(Evidence(H, z_true[t] + rng.normal(0, 0.01, d), np.eye(d) * 1e-3, f"obs@{t}"))
        post = reason(prior, ev).belief
        m = post.mean().reshape(T, d)
        # the unobserved middle states (t=2,3) are recovered by the dynamics
        self.assertLess(np.linalg.norm(m[2] - z_true[2]), 0.2)
        self.assertLess(np.linalg.norm(m[3] - z_true[3]), 0.2)

    def test_block_selector_reads_the_right_block(self):
        H = block_selector(2, n_blocks=4, block_dim=3)
        self.assertEqual(H.shape, (3, 12))
        # only block-2 columns are the identity
        self.assertTrue(np.allclose(H[:, 6:9], np.eye(3)))
        self.assertEqual(H[:, :6].sum(), 0.0)
        self.assertEqual(H[:, 9:].sum(), 0.0)
        # a local within-block readout
        Hloc = block_selector(1, 3, 2, within=[[1.0, 1.0]])
        self.assertEqual(Hloc.shape, (1, 6))
        np.testing.assert_allclose(Hloc[0], [0, 0, 1, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
