"""Tests for belief states (mixle.inference.belief) — exactness and the assimilation loop."""

import unittest

import numpy as np

from mixle.inference.belief import CategoricalBelief, GaussianBelief, as_belief


class CategoricalBeliefBasicsTest(unittest.TestCase):
    def test_construct_update_map_entropy(self):
        b = CategoricalBelief([0.5, 0.5], labels=["a", "b"])
        post = b.update([0.0, np.log(3.0)])  # evidence favors "b" 3:1
        np.testing.assert_allclose(post.probs, [0.25, 0.75])
        self.assertEqual(post.map(), "b")
        self.assertLess(post.entropy(), b.entropy())

    def test_rejects_mismatched_labels(self):
        with self.assertRaises(ValueError):
            CategoricalBelief([0.5, 0.3, 0.2], labels=["a", "b"])  # 3 probs, 2 labels

    def test_rejects_too_many_labels(self):
        with self.assertRaises(ValueError):
            CategoricalBelief([0.5, 0.5], labels=["a", "b", "c"])  # 2 probs, 3 labels

    def test_sample_accepts_an_integer_seed(self):
        # GaussianBelief.sample(rng=<int>) has always worked via _as_rng; CategoricalBelief.sample
        # used `rng if rng is not None else np.random.RandomState()`, which does not convert an int
        # seed and crashed with AttributeError.
        b = CategoricalBelief([0.5, 0.5], labels=["a", "b"])
        draws = b.sample(5, rng=42)
        self.assertEqual(len(draws), 5)

    def test_uniform_rejects_zero_labels(self):
        with self.assertRaises(ValueError):
            CategoricalBelief.uniform(0)


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

    def test_rejects_indefinite_covariance(self):
        # eigenvalues [-1, 3]: not a valid covariance matrix.
        bad = np.array([[1.0, 2.0], [2.0, 1.0]])
        self.assertLess(np.linalg.eigvalsh(bad).min(), 0.0)
        with self.assertRaises(ValueError):
            GaussianBelief([0.0, 0.0], bad)

    def test_allows_exactly_singular_covariance(self):
        # a valid degenerate (rank-deficient) covariance -- a deterministic relationship between
        # coordinates -- must still construct; only genuinely negative eigenvalues are rejected.
        singular = np.array([[1.0, 1.0], [1.0, 1.0]])  # eigenvalues [0, 2]
        self.assertAlmostEqual(np.linalg.eigvalsh(singular).min(), 0.0, places=10)
        b = GaussianBelief([0.0, 0.0], singular)
        np.testing.assert_allclose(b.cov(), singular)

    def test_accepts_large_scale_near_singular_covariance(self):
        # eigenvalues [1e9, -1e-3]: the negative eigenvalue is a tiny fraction (1e-12) of the
        # matrix's own scale -- exactly the shape of float noise a large-scale PSD matrix produces
        # under eigvalsh. A fixed absolute tolerance (the old -1e-9 constant) incorrectly rejects
        # this; the tolerance must scale with the matrix's own eigenvalue magnitude.
        P = np.diag([1e9, -1e-3])
        b = GaussianBelief([0.0, 0.0], P)
        self.assertAlmostEqual(float(b.var()[0]), 1e9, places=0)

    def test_rejects_small_scale_proportionally_indefinite_covariance(self):
        # eigenvalues [1e-9, -1e-9]: the negative eigenvalue is 100% as large as the positive one
        # -- genuinely, proportionally indefinite, not float noise -- even though both are tiny in
        # absolute terms. The old fixed absolute tolerance (-1e-9) incorrectly accepted this exact
        # case (-1e-9 is not strictly less than -1e-9); a relative tolerance correctly rejects it.
        P = np.diag([1e-9, -1e-9])
        with self.assertRaises(ValueError):
            GaussianBelief([0.0, 0.0], P)

    def test_var_sd_interval_clip_a_boundary_negative_diagonal_instead_of_nan(self):
        # a covariance accepted by the PSD tolerance can still carry a tiny negative diagonal entry
        # from float noise (never a real negative variance); var()/sd()/interval() must clip it to
        # 0 rather than silently propagate NaN through sqrt.
        P = np.diag([1.0, -1e-15])
        b = GaussianBelief([0.0, 0.0], P)
        self.assertTrue(np.isfinite(b.var()).all())
        self.assertTrue(np.isfinite(b.sd()).all())
        self.assertTrue(np.isfinite(b.interval()).all())
        self.assertEqual(float(b.var()[1]), 0.0)

    def test_rejects_nan_covariance(self):
        # NaN does not reliably surface as a NaN eigenvalue (eigvalsh can return finite, even zero,
        # eigenvalues for a NaN-containing matrix), and "nan < threshold" is always False regardless
        # -- so the PSD check alone cannot be trusted to catch it; NaN must be rejected explicitly.
        nan_cov = np.array([[np.nan, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            GaussianBelief([0.0, 0.0], nan_cov)

    def test_rejects_nan_covariance_scalar(self):
        with self.assertRaises(ValueError):
            GaussianBelief([0.0], [[np.nan]])


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

    def test_rejects_nan_observation_noise_covariance(self):
        # Same NaN-defeats-the-eigenvalue-check hazard as __init__'s cov (see GaussianBeliefBasicsTest.
        # test_rejects_nan_covariance): R has no other check standing in for this, since it is used
        # directly in the Joseph-form computation before ever reaching a constructor.
        b = GaussianBelief([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            b.update([1.0, 0.0], [0.5], np.nan)
        with self.assertRaises(ValueError):
            b.update([1.0, 0.0], [0.5], [np.nan])

    def test_rejects_negative_observation_noise_covariance(self):
        # A negative R is not reliably caught downstream by the returned belief's own PSD check on
        # P_new: the Joseph form's K @ R @ K.T term can still land P_new inside the PSD cone for a
        # substantially negative R, depending on how H relates to P (e.g. R=-10 with H=[1,0] against
        # P=I leaves P_new's eigenvalues both positive) -- so R itself must be checked directly.
        b = GaussianBelief([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            b.update([1.0, 0.0], [0.5], -10.0)
        with self.assertRaises(ValueError):
            b.update([1.0, 0.0], [0.5], [-10.0])
        with self.assertRaises(ValueError):
            b.update(np.eye(2), [0.5, 0.5], np.diag([-1.0, -1.0]))

    def test_allows_zero_observation_noise_covariance(self):
        # R=0 (a noiseless observation) is the documented R -> 0 limit and must still be allowed --
        # the new PSD check on R must not overreach into rejecting this legitimate boundary.
        b = GaussianBelief([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        post = b.update([1.0, 0.0], [0.5], 0.0)
        self.assertAlmostEqual(float(post.mean()[0]), 0.5, places=10)
        self.assertAlmostEqual(float(post.var()[0]), 0.0, places=10)


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

    def test_a_bug_inside_mean_is_not_masked_by_a_retry_without_node(self):
        # mean(self, node=None) accepts node, so it must be called with it exactly once; a TypeError
        # from inside its own body must propagate, not be swallowed and silently retried as mean().
        calls = []

        class BuggyNode:
            def mean(self, node=None):
                calls.append(node)
                return None + 1  # an internal bug unrelated to whether node is accepted

            def cov(self, node=None):
                return np.eye(2)

        with self.assertRaises(TypeError):
            as_belief(BuggyNode(), node="temperature")
        self.assertEqual(calls, ["temperature"])  # called once, with node -- never retried as mean()


if __name__ == "__main__":
    unittest.main()
