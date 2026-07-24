"""Inverse-Wishart over SPD matrices: density vs scipy, sampling (invert Wishart), scale MLE."""

import unittest

import numpy as np
from scipy.stats import invwishart as siw

from mixle.inference import estimate
from mixle.stats import InverseWishartDistribution


class InverseWishartTest(unittest.TestCase):
    def setUp(self):
        self.P = np.array([[2.0, 0.3], [0.3, 1.0]])
        self.df = 8
        self.p = 2
        self.d = InverseWishartDistribution(self.df, self.P)

    def test_log_density_matches_scipy(self):
        xs = np.array([self.d.sampler(seed=k).sample() for k in range(4)])
        mine = self.d.seq_log_density(xs)
        ref = np.array([siw.logpdf(x, self.df, self.P) for x in xs])
        np.testing.assert_allclose(mine, ref, atol=1e-9)
        np.testing.assert_allclose(mine, [self.d.log_density(x) for x in xs], atol=1e-10)

    def test_non_pd_is_minus_inf(self):
        self.assertEqual(self.d.log_density(np.array([[1.0, 2.0], [2.0, 1.0]])), -np.inf)

    def test_negative_definite_with_positive_determinant_is_minus_inf(self):
        # -I in an even dimension has determinant (-1)^2 = +1 while every eigenvalue is negative;
        # a determinant-sign check alone would wrongly accept this as positive definite.
        neg_i = -np.eye(2)
        self.assertGreater(np.linalg.det(neg_i), 0.0)
        self.assertEqual(self.d.log_density(neg_i), -np.inf)
        seq = self.d.seq_log_density(np.array([neg_i, self.P]))
        self.assertEqual(seq[0], -np.inf)
        self.assertAlmostEqual(seq[1], self.d.log_density(self.P), places=9)

    def test_negative_definite_scale_with_positive_determinant_raises(self):
        with self.assertRaises(ValueError):
            InverseWishartDistribution(self.df, -np.eye(2))

    def test_sampler_is_spd_with_correct_mean(self):
        s = self.d.sampler(seed=0).sample(40000)
        self.assertTrue(np.all(np.linalg.eigvalsh(s[:300]) > 0))
        np.testing.assert_allclose(s.mean(axis=0), self.P / (self.df - self.p - 1), atol=0.04)  # E[X]=Psi/(df-p-1)

    def test_scale_estimator_recovers_psi(self):
        est = estimate(list(self.d.sampler(seed=1).sample(30000)), self.d.estimator())
        np.testing.assert_allclose(est.scale, self.P, atol=0.12)

    def test_df_too_small_raises(self):
        with self.assertRaises(ValueError):
            InverseWishartDistribution(2, np.eye(3))  # df must be > p-1 = 2

    def test_df_nan_raises(self):
        with self.assertRaises(ValueError):
            InverseWishartDistribution(np.nan, self.P)

    def test_df_pos_inf_raises(self):
        with self.assertRaises(ValueError):
            InverseWishartDistribution(np.inf, self.P)

    def test_df_neg_inf_raises(self):
        with self.assertRaises(ValueError):
            InverseWishartDistribution(-np.inf, self.P)


class InverseWishartScalarVectorizedAgreementTest(unittest.TestCase):
    """External review: log_density (scalar) and seq_log_density (vectorized) must agree on every
    input, not just well-conditioned ones. Before the fix, log_density checked positive-definiteness
    and its log-determinant via cholesky_logdet (Cholesky factorization) while seq_log_density used
    batched_pd_logdet (eigendecomposition), and the two trace computations (``trace(scale @ X^-1)``
    vs an ``einsum`` contraction) also accumulated in different orders -- the latter is especially
    visible here since X^-1 can be numerically huge for a near-singular X. Near the
    positive-definiteness boundary a Cholesky factor and an eigendecomposition round differently and
    can even disagree on whether a matrix is PD at all, so log_density and seq_log_density could
    return substantially different values -- or one -inf and the other finite -- for the identical
    input matrix.
    """

    def test_agreement_near_positive_definiteness_boundary(self):
        # Deliberately construct matrices with one eigenvalue right at the edge of positive
        # definiteness -- this is exactly where Cholesky and eigh used to disagree.
        p = 8
        rng = np.random.RandomState(0)
        q, _ = np.linalg.qr(rng.randn(p, p))
        d = InverseWishartDistribution(p + 3, np.eye(p))
        for tiny in (1e-10, 1e-12, 1e-14, 1e-16, 0.0, -1e-14):
            eigs = np.concatenate([[tiny], np.linspace(1.0, 10.0, p - 1)])
            m = (q * eigs) @ q.T
            m = 0.5 * (m + m.T)
            scalar = d.log_density(m)
            vec = float(d.seq_log_density(m[None, ...])[0])
            self.assertEqual(scalar, vec, msg=f"tiny_eig={tiny}: scalar={scalar!r} vec={vec!r}")

    def test_agreement_on_sampled_batch(self):
        p_mat = np.array([[2.0, 0.3], [0.3, 1.0]])
        d = InverseWishartDistribution(8, p_mat)
        data = list(d.sampler(seed=7).sample(200))
        enc = d.dist_to_encoder().seq_encode(data)
        seq = np.asarray(d.seq_log_density(enc), dtype=float)
        scalar = np.array([float(d.log_density(x)) for x in data])
        np.testing.assert_allclose(seq, scalar, atol=1e-8, err_msg="scalar != vectorized")


class InverseWishartEstimatorUsesDataTest(unittest.TestCase):
    """External review (finding c): checks whether InverseWishartEstimator.estimate() actually uses
    the observed data's sufficient statistics, or silently returns something data-independent (e.g.
    always the prior/a fixed default) regardless of what was observed. Investigated and NOT
    reproduced -- estimate() computes ``scale = (df - p - 1) * (sum_x / count)`` directly from the
    accumulator's weighted sufficient statistics, matching the class's own documented closed-form MOM
    estimator exactly. These tests lock that in: fitting to two very different datasets must produce
    two substantially different, and individually correct, fitted scales -- not the same
    (data-independent) answer both times.
    """

    def _fit_direct(self, est, data):
        # deterministic direct M-step (no fit()/global-state dependence), mirroring the pattern
        # already used for WishartEstimator in wishart_test.py's WishartEstimatedDFTest.
        acc = est.accumulator_factory().make()
        acc.seq_update(np.asarray(data), np.ones(len(data), dtype=np.float64), None)
        return est.estimate(None, acc.value())

    def test_two_different_datasets_yield_different_fits(self):
        from mixle.stats.matrix.inverse_wishart import InverseWishartEstimator

        df = 10.0
        dim = 3
        p_small = np.array([[1.0, 0.1, 0.0], [0.1, 1.5, 0.05], [0.0, 0.05, 0.8]])
        p_large = np.array([[50.0, -10.0, 5.0], [-10.0, 40.0, -2.0], [5.0, -2.0, 30.0]])
        data_small = InverseWishartDistribution(df, p_small).sampler(seed=1).sample(4000)
        data_large = InverseWishartDistribution(df, p_large).sampler(seed=2).sample(4000)

        fit_small = self._fit_direct(InverseWishartEstimator(dim=dim, df=df), data_small)
        fit_large = self._fit_direct(InverseWishartEstimator(dim=dim, df=df), data_large)

        # each fit recovers its OWN generating scale, not some shared data-independent answer
        np.testing.assert_allclose(fit_small.scale, p_small, atol=0.3)
        np.testing.assert_allclose(fit_large.scale, p_large, atol=3.0)
        self.assertGreater(np.abs(fit_small.scale - fit_large.scale).max(), 10.0)

    def test_fit_matches_independent_closed_form_mom(self):
        from mixle.stats.matrix.inverse_wishart import InverseWishartEstimator

        df = 9.0
        dim = 2
        data = InverseWishartDistribution(df, np.array([[2.0, 0.3], [0.3, 1.0]])).sampler(seed=3).sample(500)
        fitted = self._fit_direct(InverseWishartEstimator(dim=dim, df=df), data)

        # independently-written reference: Psi_hat = (df - p - 1) * mean(X)
        data_arr = np.asarray(data)
        mean_x = data_arr.mean(axis=0)
        expected = (df - dim - 1.0) * mean_x
        expected = 0.5 * (expected + expected.T)
        np.testing.assert_allclose(fitted.scale, expected, atol=1e-8)

    def test_changing_weights_changes_the_fit(self):
        from mixle.stats.matrix.inverse_wishart import InverseWishartEstimator

        df = 10.0
        dim = 2
        data = InverseWishartDistribution(df, np.array([[2.0, 0.3], [0.3, 1.0]])).sampler(seed=4).sample(400)
        est = InverseWishartEstimator(dim=dim, df=df)

        acc_uniform = est.accumulator_factory().make()
        acc_uniform.seq_update(np.asarray(data), np.ones(len(data)), None)
        fit_uniform = est.estimate(None, acc_uniform.value())

        weights_skewed = np.concatenate([np.full(len(data) // 2, 5.0), np.full(len(data) - len(data) // 2, 0.1)])
        acc_skewed = est.accumulator_factory().make()
        acc_skewed.seq_update(np.asarray(data), weights_skewed, None)
        fit_skewed = est.estimate(None, acc_skewed.value())

        self.assertGreater(np.abs(fit_uniform.scale - fit_skewed.scale).max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
