"""Sandwich / robust covariance estimators (pysp.inference.robust)."""

import unittest

import numpy as np

from pysp.inference import (
    cluster_robust_covariance,
    newey_west_covariance,
    ols_robust_covariance,
    robust_standard_errors,
    sandwich_covariance,
)


def _ols(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta, y - X @ beta


class HCTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.n = 500
        self.X = np.column_stack([np.ones(self.n), rng.normal(0, 1, self.n), rng.normal(0, 1, self.n)])
        self.beta = np.array([1.0, 2.0, -1.0])
        self.rng = rng

    def test_hc0_matches_classical_under_homoscedasticity(self):
        y = self.X @ self.beta + self.rng.normal(0, 1.0, self.n)
        _, e = _ols(self.X, y)
        sigma2 = (e @ e) / (self.n - 3)
        classical = sigma2 * np.linalg.inv(self.X.T @ self.X)
        hc0 = ols_robust_covariance(self.X, e, hc="hc0")
        np.testing.assert_allclose(robust_standard_errors(hc0), robust_standard_errors(classical), rtol=0.15)

    def test_robust_exceeds_classical_under_heteroscedasticity(self):
        # variance grows with |x1|, so the x1 coefficient SE must inflate
        y = self.X @ self.beta + self.rng.normal(0, 1, self.n) * (0.5 + np.abs(self.X[:, 1]))
        _, e = _ols(self.X, y)
        sigma2 = (e @ e) / (self.n - 3)
        classical_se = robust_standard_errors(sigma2 * np.linalg.inv(self.X.T @ self.X))
        hc3_se = robust_standard_errors(ols_robust_covariance(self.X, e, hc="hc3"))
        self.assertGreater(hc3_se[1], 1.3 * classical_se[1])

    def test_hc_ordering(self):
        y = self.X @ self.beta + self.rng.normal(0, 1, self.n) * (0.5 + np.abs(self.X[:, 1]))
        _, e = _ols(self.X, y)
        se = {
            hc: robust_standard_errors(ols_robust_covariance(self.X, e, hc=hc)) for hc in ("hc0", "hc1", "hc2", "hc3")
        }
        # HC0 <= HC1 <= HC2 <= HC3 componentwise
        self.assertTrue(np.all(se["hc0"] <= se["hc1"] + 1e-12))
        self.assertTrue(np.all(se["hc1"] <= se["hc2"] + 1e-9))
        self.assertTrue(np.all(se["hc2"] <= se["hc3"] + 1e-9))

    def test_invalid_hc(self):
        with self.assertRaises(ValueError):
            ols_robust_covariance(self.X, np.zeros(self.n), hc="hc9")


class ClusterTest(unittest.TestCase):
    def test_singleton_clusters_match_hc0(self):
        rng = np.random.RandomState(1)
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0, 1, n)
        _, e = _ols(X, y)
        singleton = cluster_robust_covariance(X, e, np.arange(n), small_sample=False)
        hc0 = ols_robust_covariance(X, e, hc="hc0")
        np.testing.assert_allclose(singleton, hc0)

    def test_cluster_se_inflates_with_intracluster_correlation(self):
        rng = np.random.RandomState(2)
        n_clusters, per = 40, 15
        n = n_clusters * per
        clusters = np.repeat(np.arange(n_clusters), per)
        x = rng.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x])
        # shared cluster-level shock -> strong within-cluster error correlation
        u = np.repeat(rng.normal(0, 1.0, n_clusters), per) + rng.normal(0, 0.3, n)
        y = X @ np.array([0.0, 1.0]) + u
        _, e = _ols(X, y)
        hc1_se = robust_standard_errors(ols_robust_covariance(X, e, hc="hc1"))
        clust_se = robust_standard_errors(cluster_robust_covariance(X, e, clusters))
        # ignoring clustering badly understates the intercept SE
        self.assertGreater(clust_se[0], 2.0 * hc1_se[0])

    def test_two_way_clustering_runs_and_is_symmetric(self):
        rng = np.random.RandomState(3)
        n = 300
        a = np.repeat(np.arange(20), 15)
        b = np.tile(np.repeat(np.arange(5), 3), 20)
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ np.array([1.0, -0.5]) + rng.normal(0, 1, n)
        _, e = _ols(X, y)
        cov = cluster_robust_covariance(X, e, [a, b])
        np.testing.assert_allclose(cov, cov.T)
        self.assertTrue(np.all(np.diag(cov) > 0))


class HACTest(unittest.TestCase):
    def test_lag0_matches_hc0(self):
        rng = np.random.RandomState(0)
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0, 1, n)
        _, e = _ols(X, y)
        nw0 = newey_west_covariance(X, e, lags=0, small_sample=False)
        hc0 = ols_robust_covariance(X, e, hc="hc0")
        np.testing.assert_allclose(nw0, hc0)

    def test_hac_inflates_se_under_serial_correlation(self):
        rng = np.random.RandomState(4)
        n = 600
        t = np.arange(n)
        X = np.column_stack([np.ones(n), t / n])
        # AR(1) errors -> OLS SE understated; HAC should be larger
        e_proc = np.zeros(n)
        for i in range(1, n):
            e_proc[i] = 0.7 * e_proc[i - 1] + rng.randn()
        y = X @ np.array([1.0, 2.0]) + e_proc
        _, resid = _ols(X, y)
        hc1_se = robust_standard_errors(ols_robust_covariance(X, resid, hc="hc1"))
        hac_se = robust_standard_errors(newey_west_covariance(X, resid, lags=12))
        self.assertGreater(hac_se[0], 1.3 * hc1_se[0])

    def test_default_lag_rule(self):
        rng = np.random.RandomState(5)
        n = 250
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0, 1, n)
        _, e = _ols(X, y)
        cov = newey_west_covariance(X, e)  # default lags
        self.assertEqual(cov.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(cov)))


class GenericSandwichTest(unittest.TestCase):
    def test_generic_matches_ols_builder(self):
        rng = np.random.RandomState(6)
        n = 150
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0, 1, n)
        _, e = _ols(X, y)
        bread = np.linalg.inv(X.T @ X)
        scores = X * e[:, None]
        generic = sandwich_covariance(scores, bread, small_sample=False)
        hc0 = ols_robust_covariance(X, e, hc="hc0")
        np.testing.assert_allclose(generic, hc0)


if __name__ == "__main__":
    unittest.main()
