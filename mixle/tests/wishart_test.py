"""Wishart distribution over SPD matrices: density vs scipy, Bartlett sampling, closed-form scale MLE."""

import unittest

import numpy as np
from scipy.stats import wishart as sw

from mixle.inference import estimate
from mixle.stats import WishartDistribution


class WishartTest(unittest.TestCase):
    def setUp(self):
        self.V = np.array([[2.0, 0.3], [0.3, 1.0]])
        self.df = 6
        self.d = WishartDistribution(self.df, self.V)

    def test_log_density_matches_scipy(self):
        xs = np.array([self.d.sampler(seed=k).sample() for k in range(4)])
        mine = self.d.seq_log_density(xs)
        ref = np.array([sw.logpdf(x, self.df, self.V) for x in xs])
        np.testing.assert_allclose(mine, ref, atol=1e-9)
        np.testing.assert_allclose(mine, [self.d.log_density(x) for x in xs], atol=1e-10)

    def test_non_pd_is_minus_inf(self):
        self.assertEqual(self.d.log_density(np.array([[1.0, 2.0], [2.0, 1.0]])), -np.inf)  # indefinite

    def test_negative_definite_with_positive_determinant_is_minus_inf(self):
        # -I in an even dimension has determinant (-1)^2 = +1 while every eigenvalue is negative;
        # a determinant-sign check alone would wrongly accept this as positive definite.
        neg_i = -np.eye(2)
        self.assertGreater(np.linalg.det(neg_i), 0.0)
        self.assertEqual(self.d.log_density(neg_i), -np.inf)
        seq = self.d.seq_log_density(np.array([neg_i, self.V]))
        self.assertEqual(seq[0], -np.inf)
        self.assertAlmostEqual(seq[1], self.d.log_density(self.V), places=9)

    def test_asymmetric_observation_is_minus_inf(self):
        # a Wishart RV is by definition a symmetric matrix, so an asymmetric observation is not a
        # member of the support at all. batched_pd_logdet reads one triangle only (like
        # np.linalg.eigvalsh, which it wraps, defaults to UPLO='L'), so this used to be silently
        # read as np.eye(2) (is_pd=True, logdet=0) and scored a finite density instead of -inf.
        asym = np.array([[1.0, 100.0], [0.0, 1.0]])
        self.assertFalse(np.allclose(asym, asym.T))
        self.assertEqual(self.d.log_density(asym), -np.inf)
        seq = self.d.seq_log_density(np.array([asym, self.V]))
        self.assertEqual(seq[0], -np.inf)
        self.assertAlmostEqual(seq[1], self.d.log_density(self.V), places=9)

    def test_negative_definite_scale_with_positive_determinant_raises(self):
        # the constructor's own explicit check must reject this, with its own message -- not rely
        # on np.linalg.cholesky() happening to raise (a LinAlgError, which subclasses ValueError,
        # with numpy/LAPACK's own message) a few lines later as an accidental safety net.
        with self.assertRaises(ValueError) as cm:
            WishartDistribution(self.df, -np.eye(2))
        self.assertEqual(str(cm.exception), "scale must be positive definite")

    def test_sampler_is_spd_with_correct_mean(self):
        s = self.d.sampler(seed=0).sample(40000)
        self.assertTrue(np.all(np.linalg.eigvalsh(s[:300]) > 0))  # all SPD
        np.testing.assert_allclose(s.mean(axis=0), self.df * self.V, atol=0.12)  # E[X] = df V

    def test_scale_estimator_recovers_V(self):
        est = estimate(list(self.d.sampler(seed=1).sample(20000)), self.d.estimator())
        np.testing.assert_allclose(est.scale, self.V, atol=0.06)

    def test_df_below_dim_raises(self):
        with self.assertRaises(ValueError):
            WishartDistribution(1, np.eye(3))


if __name__ == "__main__":
    unittest.main()


class WishartEstimatedDFTest(unittest.TestCase):
    """WS-2: WishartEstimator(df=None) estimates the degrees of freedom by maximum likelihood."""

    def _V(self):
        return np.array([[1.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 2.0]])

    def _fit_direct(self, est, data):
        # deterministic direct M-step (no fit()/global-state dependence)
        acc = est.accumulator_factory().make()
        acc.seq_update(np.asarray(data), np.ones(len(data), dtype=np.float64), None)
        return est.estimate(None, acc.value())

    def test_recovers_degrees_of_freedom(self):
        from mixle.stats.matrix.wishart import WishartDistribution, WishartEstimator

        for true_df in (8.0, 15.0):
            data = WishartDistribution(df=true_df, scale=self._V()).sampler(seed=1).sample(4000)
            m = self._fit_direct(WishartEstimator(dim=3, df=None), data)
            self.assertAlmostEqual(m.df, true_df, delta=0.7)  # consistent df MLE
            self.assertAlmostEqual(m.scale[0, 0], 1.0, delta=0.1)  # scale recovered too

    def test_fixed_df_is_unchanged(self):
        from mixle.stats.matrix.wishart import WishartDistribution, WishartEstimator

        data = WishartDistribution(df=8.0, scale=self._V()).sampler(seed=2).sample(500)
        m = self._fit_direct(WishartEstimator(dim=3, df=8.0), data)
        self.assertEqual(m.df, 8.0)  # fixed df is honored exactly


class WishartAccumulatorLogdetTest(unittest.TestCase):
    """The sum_logdet sufficient statistic (used for df MLE) must reject a positive-determinant,
    negative-definite matrix the same way log_density does, not silently include a wrong value."""

    def test_negative_definite_with_positive_determinant_contributes_minus_inf(self):
        from mixle.stats.matrix.wishart import WishartAccumulator

        acc = WishartAccumulator(dim=2)
        acc.update(-np.eye(2), 1.0, None)  # det = +1, but negative definite
        _, count, sum_logdet = acc.value()
        self.assertEqual(count, 1.0)
        self.assertEqual(sum_logdet, -np.inf)

    def test_seq_update_matches_update_for_negative_definite(self):
        from mixle.stats.matrix.wishart import WishartAccumulator

        acc = WishartAccumulator(dim=2)
        acc.seq_update(np.array([-np.eye(2), np.eye(2)]), np.array([1.0, 1.0]), None)
        _, count, sum_logdet = acc.value()
        self.assertEqual(count, 2.0)
        self.assertEqual(sum_logdet, -np.inf)  # one -inf term poisons the weighted sum, as intended


class WishartValidationTest(unittest.TestCase):
    """df must be a finite real satisfying df >= p; nan and +/-inf all satisfy no useful ordering
    (a nan comparison is always False, and +inf passes a `< dim` check meant to exclude it) and must
    be rejected explicitly rather than silently accepted into a nan-poisoned distribution."""

    def setUp(self):
        self.V = np.array([[2.0, 0.3], [0.3, 1.0]])
        self.df = 6
        self.d = WishartDistribution(self.df, self.V)

    def test_df_nan_raises(self):
        with self.assertRaises(ValueError):
            WishartDistribution(np.nan, self.V)

    def test_df_pos_inf_raises(self):
        with self.assertRaises(ValueError):
            WishartDistribution(np.inf, self.V)

    def test_df_neg_inf_raises(self):
        with self.assertRaises(ValueError):
            WishartDistribution(-np.inf, self.V)

    def test_asymmetric_scale_raises(self):
        # cholesky_logdet reads one triangle only, so an asymmetric matrix with a
        # positive-definite-looking triangle would otherwise pass straight through.
        with self.assertRaises(ValueError):
            WishartDistribution(self.df, np.array([[2.0, 1.0], [0.0, 1.0]]))

    def test_symmetric_scale_still_valid(self):
        WishartDistribution(self.df, self.V)  # must not raise


class WishartScalarVectorizedAgreementTest(unittest.TestCase):
    """External review: log_density (scalar) and seq_log_density (vectorized) must agree on every
    input, not just well-conditioned ones. Before the fix, log_density checked positive-definiteness
    and its log-determinant via cholesky_logdet (Cholesky factorization) while seq_log_density used
    batched_pd_logdet (eigendecomposition), and the two trace computations (``trace(A @ B)`` vs an
    ``einsum`` contraction) also accumulated in different orders. Near the positive-definiteness
    boundary a Cholesky factor and an eigendecomposition round differently and can even disagree on
    whether a matrix is PD at all, so log_density and seq_log_density could return substantially
    different values -- or one -inf and the other finite -- for the identical input matrix.
    """

    def test_agreement_near_positive_definiteness_boundary(self):
        # Deliberately construct matrices with one eigenvalue right at the edge of positive
        # definiteness -- this is exactly where Cholesky and eigh used to disagree.
        p = 8
        rng = np.random.RandomState(0)
        q, _ = np.linalg.qr(rng.randn(p, p))
        d = WishartDistribution(p + 3, np.eye(p))
        for tiny in (1e-10, 1e-12, 1e-14, 1e-16, 0.0, -1e-14):
            eigs = np.concatenate([[tiny], np.linspace(1.0, 10.0, p - 1)])
            m = (q * eigs) @ q.T
            m = 0.5 * (m + m.T)
            scalar = d.log_density(m)
            vec = float(d.seq_log_density(m[None, ...])[0])
            self.assertEqual(scalar, vec, msg=f"tiny_eig={tiny}: scalar={scalar!r} vec={vec!r}")

    def test_agreement_on_sampled_batch(self):
        v = np.array([[2.0, 0.3], [0.3, 1.0]])
        d = WishartDistribution(6, v)
        data = list(d.sampler(seed=7).sample(200))
        enc = d.dist_to_encoder().seq_encode(data)
        seq = np.asarray(d.seq_log_density(enc), dtype=float)
        scalar = np.array([float(d.log_density(x)) for x in data])
        np.testing.assert_allclose(seq, scalar, atol=1e-8, err_msg="scalar != vectorized")
