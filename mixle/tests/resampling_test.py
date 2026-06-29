"""Bootstrap and permutation inference (mixle.inference.resampling)."""

import unittest

import numpy as np

from mixle.inference import (
    block_bootstrap,
    bootstrap,
    permutation_test,
    wild_bootstrap,
)


class BootstrapTest(unittest.TestCase):
    def test_mean_ci_brackets_truth(self):
        rng = np.random.RandomState(0)
        for method in ("percentile", "basic", "bca"):
            x = rng.normal(5.0, 1.0, 600)
            r = bootstrap(x, lambda d: float(np.mean(d)), n_boot=2000, method=method, seed=1)
            self.assertEqual(r.method, method)
            self.assertLess(float(r.ci_low), 5.0)
            self.assertLess(5.0, float(r.ci_high))

    def test_ci_width_matches_analytic_se(self):
        rng = np.random.RandomState(1)
        x = rng.normal(0.0, 2.0, 1000)
        r = bootstrap(x, lambda d: float(np.mean(d)), n_boot=3000, method="percentile", seed=2)
        analytic_se = 2.0 / np.sqrt(1000)
        self.assertAlmostEqual(float(r.standard_error), analytic_se, delta=0.01)
        width = float(r.ci_high) - float(r.ci_low)
        self.assertAlmostEqual(width, 2 * 1.96 * analytic_se, delta=0.02)

    def test_vector_statistic_regression_coefficients(self):
        rng = np.random.RandomState(2)
        X = rng.normal(0, 1, (400, 2))
        beta = np.array([1.5, -2.0])
        y = X @ beta + rng.normal(0, 0.5, 400)

        def coef(X, y):
            return np.linalg.lstsq(X, y, rcond=None)[0]

        r = bootstrap((X, y), coef, n_boot=1000, method="bca", seed=3)
        self.assertEqual(r.estimate.shape, (2,))
        self.assertTrue(np.all(r.ci_low < beta))
        self.assertTrue(np.all(beta < r.ci_high))

    def test_bca_falls_back_to_percentile_for_clustered(self):
        rng = np.random.RandomState(3)
        x = rng.normal(0, 1, 100)
        clusters = np.repeat(np.arange(20), 5)
        r = bootstrap(x, lambda d: float(np.mean(d)), n_boot=500, method="bca", clusters=clusters, seed=4)
        self.assertEqual(r.method, "percentile")

    def test_cluster_bootstrap_widens_with_intracluster_correlation(self):
        # strongly correlated within clusters -> effective n is the #clusters, so CI is wider
        rng = np.random.RandomState(4)
        cluster_means = rng.normal(0, 1, 30)
        x = np.repeat(cluster_means, 10) + rng.normal(0, 0.01, 300)
        clusters = np.repeat(np.arange(30), 10)
        naive = bootstrap(x, lambda d: float(np.mean(d)), n_boot=800, method="percentile", seed=5)
        clustered = bootstrap(
            x, lambda d: float(np.mean(d)), n_boot=800, method="percentile", clusters=clusters, seed=5
        )
        naive_w = float(naive.ci_high) - float(naive.ci_low)
        clust_w = float(clustered.ci_high) - float(clustered.ci_low)
        # effective n is the number of clusters, so the honest CI is ~sqrt(300/30) wider
        self.assertGreater(clust_w, 2.5 * naive_w)

    def test_stratified_resampling_preserves_group_sizes(self):
        rng = np.random.RandomState(6)
        x = rng.normal(0, 1, 50)
        groups = np.repeat([0, 1], 25)
        # should run without error and produce a finite interval
        r = bootstrap(x, lambda d: float(np.mean(d)), n_boot=300, method="percentile", groups=groups, seed=7)
        self.assertTrue(np.isfinite(r.ci_low) and np.isfinite(r.ci_high))

    def test_subsampling_runs(self):
        rng = np.random.RandomState(7)
        x = rng.normal(0, 1, 500)
        r = bootstrap(x, lambda d: float(np.mean(d)), n_boot=400, method="percentile", m=100, seed=8)
        self.assertTrue(np.isfinite(r.ci_low))


class BlockBootstrapTest(unittest.TestCase):
    def test_block_bootstrap_wider_than_iid_for_ar1(self):
        rng = np.random.RandomState(0)
        n = 1000
        ar = np.zeros(n)
        for t in range(1, n):
            ar[t] = 0.8 * ar[t - 1] + rng.randn()
        iid = bootstrap(ar, lambda d: float(np.mean(d)), n_boot=800, method="percentile", seed=1)
        blk = block_bootstrap(ar, lambda d: float(np.mean(d)), block_length=40, n_boot=800, seed=1)
        iid_w = float(iid.ci_high) - float(iid.ci_low)
        blk_w = float(blk.ci_high) - float(blk.ci_low)
        # ignoring autocorrelation badly understates uncertainty
        self.assertGreater(blk_w, 1.5 * iid_w)


class WildBootstrapTest(unittest.TestCase):
    def test_wild_bootstrap_coefficient_ci(self):
        rng = np.random.RandomState(0)
        n = 400
        x = rng.normal(0, 1, n)
        X = np.column_stack([np.ones(n), x])
        beta = np.array([0.5, 2.0])
        # heteroscedastic noise
        y = X @ beta + rng.normal(0, 0.2 + 0.5 * np.abs(x))
        fitted = X @ np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - fitted

        def slope(ystar):
            return np.linalg.lstsq(X, ystar, rcond=None)[0]

        for kind in ("rademacher", "mammen"):
            r = wild_bootstrap(fitted, resid, slope, n_boot=1000, kind=kind, seed=1)
            self.assertEqual(r.estimate.shape, (2,))
            self.assertLess(r.ci_low[1], 2.0)
            self.assertLess(2.0, r.ci_high[1])


class PermutationTest(unittest.TestCase):
    def test_detects_real_shift(self):
        rng = np.random.RandomState(0)
        a = rng.normal(0.0, 1.0, 60)
        b = rng.normal(1.0, 1.0, 60)
        r = permutation_test(a, b, n_perm=5000, seed=1)
        self.assertLess(r.pvalue, 0.01)

    def test_null_is_uniformish(self):
        # under the null the p-value should rarely be tiny
        rng = np.random.RandomState(1)
        small = 0
        for _ in range(200):
            a = rng.normal(0, 1, 30)
            b = rng.normal(0, 1, 30)
            if permutation_test(a, b, n_perm=500, seed=int(rng.randint(1 << 30))).pvalue < 0.05:
                small += 1
        self.assertLess(small / 200, 0.1)

    def test_exact_enumeration_for_small_samples(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0])
        r = permutation_test(a, b, exact_max=100)
        self.assertTrue(r.exact)
        # C(5,3) = 10 distinct splits
        self.assertEqual(r.n_perm, 10)

    def test_paired_signflip(self):
        rng = np.random.RandomState(2)
        d1 = rng.normal(0, 1, 9)
        d2 = d1 + 0.8  # consistent positive shift
        r = permutation_test(d1, d2, paired=True)
        self.assertTrue(r.exact)
        self.assertEqual(r.n_perm, 2**9)
        self.assertLess(r.pvalue, 0.05)

    def test_stratified_restricted_permutation(self):
        rng = np.random.RandomState(3)
        # two strata with different baselines but the same a-vs-b effect within each
        a = np.concatenate([rng.normal(0, 1, 30), rng.normal(5, 1, 30)])
        b = np.concatenate([rng.normal(0.8, 1, 30), rng.normal(5.8, 1, 30)])
        strat = np.concatenate([np.repeat([0, 1], 30), np.repeat([0, 1], 30)])
        r = permutation_test(a, b, n_perm=2000, stratify=strat, seed=4)
        self.assertFalse(r.exact)
        self.assertLess(r.pvalue, 0.05)

    def test_custom_statistic_and_alternative(self):
        rng = np.random.RandomState(5)
        a = rng.normal(0, 1, 50)
        b = rng.normal(0, 2, 50)  # same mean, larger spread

        def var_ratio(x, y):
            return float(np.var(x) - np.var(y))

        r = permutation_test(a, b, statistic=var_ratio, alternative="less", n_perm=3000, seed=6)
        self.assertLess(r.pvalue, 0.05)


if __name__ == "__main__":
    unittest.main()
