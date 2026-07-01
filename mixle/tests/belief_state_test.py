"""Tests for belief states (mixle.inference.belief) — exactness and the assimilation loop."""

import unittest

import numpy as np

from mixle.inference.belief import GaussianBelief, as_belief


class GaussianBeliefBasicsTest(unittest.TestCase):
    def test_moments_entropy_interval(self):
        b = GaussianBelief([1.0, -2.0], [[4.0, 0.0], [0.0, 9.0]])
        np.testing.assert_allclose(b.mean(), [1.0, -2.0])
        np.testing.assert_allclose(b.var(), [4.0, 9.0])
        np.testing.assert_allclose(b.sd(), [2.0, 3.0])
        # H[N] = 0.5 (d log(2 pi e) + log|P|)
        expected_h = 0.5 * (2 * np.log(2 * np.pi * np.e) + np.log(4.0 * 9.0))
        self.assertAlmostEqual(b.entropy(), expected_h, places=10)
        # 90% interval half-width = 1.6449 * sd
        lo_hi = b.interval(0.9)
        z = 1.6448536269514722
        np.testing.assert_allclose(lo_hi[:, 1] - b.mean(), z * b.sd(), rtol=1e-6)

    def test_sample_recovers_moments(self):
        b = GaussianBelief([0.0, 5.0], [[1.0, 0.5], [0.5, 2.0]])
        s = b.sample(200_000, rng=0)
        np.testing.assert_allclose(s.mean(axis=0), b.mean(), atol=0.02)
        np.testing.assert_allclose(np.cov(s.T), b.cov(), atol=0.03)


class KalmanUpdateTest(unittest.TestCase):
    def test_scalar_update_matches_closed_form(self):
        # prior N(0, 4), observe y=3 with noise var 1 -> posterior mean 4/5*3, var 4/5.
        b = GaussianBelief([0.0], [[4.0]])
        post = b.update(H=[[1.0]], y=[3.0], R=[[1.0]])
        self.assertAlmostEqual(float(post.mean()[0]), 4.0 / 5.0 * 3.0, places=10)
        self.assertAlmostEqual(float(post.var()[0]), 4.0 / 5.0, places=10)
        # evidence reduces uncertainty
        self.assertLess(post.entropy(), b.entropy())

    def test_sequential_equals_batch(self):
        # Folding evidence in one datum at a time == conditioning on all of it at once (exactness).
        rng = np.random.RandomState(0)
        d, k = 3, 2
        m0 = rng.normal(size=d)
        A = rng.normal(size=(d, d))
        P0 = A @ A.T + np.eye(d)
        prior = GaussianBelief(m0, P0)

        Hs = [rng.normal(size=(k, d)) for _ in range(5)]
        ys = [rng.normal(size=k) for _ in range(5)]
        Rs = [np.diag(rng.random(k) + 0.1) for _ in range(5)]

        seq = prior
        for H, y, R in zip(Hs, ys, Rs):
            seq = seq.update(H, y, R)

        Hstack = np.vstack(Hs)
        ystack = np.concatenate(ys)
        from scipy.linalg import block_diag

        Rstack = block_diag(*Rs)
        batch = prior.update(Hstack, ystack, Rstack)

        np.testing.assert_allclose(seq.mean(), batch.mean(), atol=1e-9)
        np.testing.assert_allclose(seq.cov(), batch.cov(), atol=1e-9)

    def test_order_independence(self):
        prior = GaussianBelief([0.0, 0.0], np.eye(2) * 3.0)
        u1 = (np.array([[1.0, 0.0]]), np.array([2.0]), np.array([[0.5]]))
        u2 = (np.array([[0.0, 1.0]]), np.array([-1.0]), np.array([[0.7]]))
        ab = prior.update(*u1).update(*u2)
        ba = prior.update(*u2).update(*u1)
        np.testing.assert_allclose(ab.mean(), ba.mean(), atol=1e-10)
        np.testing.assert_allclose(ab.cov(), ba.cov(), atol=1e-10)

    def test_covariance_stays_symmetric_psd(self):
        prior = GaussianBelief(np.zeros(4), np.eye(4) * 2.0)
        rng = np.random.RandomState(3)
        b = prior
        for _ in range(10):
            H = rng.normal(size=(2, 4))
            b = b.update(H, rng.normal(size=2), np.diag(rng.random(2) + 0.05))
        P = b.cov()
        np.testing.assert_allclose(P, P.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(P).min(), -1e-10)


class FusionAndConditioningTest(unittest.TestCase):
    def test_product_of_experts(self):
        # Two Gaussian experts about the same scalar latent: exact precision-weighted combination.
        e1 = GaussianBelief([2.0], [[1.0]])  # precision 1
        e2 = GaussianBelief([6.0], [[3.0]])  # precision 1/3
        fused = e1.fuse(e2)
        prec = 1.0 + 1.0 / 3.0
        exp_var = 1.0 / prec
        exp_mean = exp_var * (2.0 * 1.0 + 6.0 * (1.0 / 3.0))
        self.assertAlmostEqual(float(fused.var()[0]), exp_var, places=10)
        self.assertAlmostEqual(float(fused.mean()[0]), exp_mean, places=10)
        # fusing evidence never widens the belief
        self.assertLessEqual(fused.var()[0], e1.var()[0] + 1e-12)

    def test_fusion_symmetric(self):
        e1 = GaussianBelief([1.0, 2.0], [[2.0, 0.3], [0.3, 1.0]])
        e2 = GaussianBelief([0.0, 1.0], [[1.0, -0.2], [-0.2, 2.0]])
        np.testing.assert_allclose(e1.fuse(e2).mean(), e2.fuse(e1).mean(), atol=1e-10)
        np.testing.assert_allclose(e1.fuse(e2).cov(), e2.fuse(e1).cov(), atol=1e-10)

    def test_gaussian_conditioning(self):
        # Condition z=[a,b] on b: closed-form Schur complement.
        m = np.array([1.0, 2.0])
        P = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = GaussianBelief(m, P)
        cond = b.condition(indices=[1], values=[5.0])
        exp_mean = m[0] + P[0, 1] / P[1, 1] * (5.0 - m[1])
        exp_var = P[0, 0] - P[0, 1] ** 2 / P[1, 1]
        self.assertAlmostEqual(float(cond.mean()[0]), exp_mean, places=10)
        self.assertAlmostEqual(float(cond.var()[0]), exp_var, places=10)

    def test_marginal(self):
        b = GaussianBelief([1.0, 2.0, 3.0], np.diag([4.0, 5.0, 6.0]))
        mb = b.marginal([0, 2])
        np.testing.assert_allclose(mb.mean(), [1.0, 3.0])
        np.testing.assert_allclose(mb.var(), [4.0, 6.0])


class AssimilationLoopTest(unittest.TestCase):
    def test_entropy_monotonically_shrinks(self):
        # The core promise: each independent observation reduces the belief's entropy.
        rng = np.random.RandomState(7)
        b = GaussianBelief(np.zeros(2), np.eye(2) * 5.0)
        entropies = [b.entropy()]
        for _ in range(6):
            H = rng.normal(size=(1, 2))
            b = b.update(H, rng.normal(size=1), np.array([[0.4]]))
            entropies.append(b.entropy())
        self.assertTrue(all(a >= c - 1e-9 for a, c in zip(entropies, entropies[1:])))
        self.assertLess(entropies[-1], entropies[0])


class AsBeliefAdapterTest(unittest.TestCase):
    def test_adapts_object_with_mean_cov(self):
        class FakeNode:
            def mean(self, node=None):
                return np.array([1.0, 2.0])

            def cov(self, node=None):
                return np.array([[1.0, 0.0], [0.0, 4.0]])

        b = as_belief(FakeNode(), node="temperature")
        np.testing.assert_allclose(b.mean(), [1.0, 2.0])
        np.testing.assert_allclose(b.var(), [1.0, 4.0])


if __name__ == "__main__":
    unittest.main()
