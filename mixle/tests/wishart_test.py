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
