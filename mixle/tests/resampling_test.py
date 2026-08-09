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

    def test_bca_single_observation_raises_a_clear_error(self):
        # the jackknife loop leaves one observation out per iteration; at n=1 that used to compute
        # the statistic on an EMPTY sample (NaN), which `den != 0` (NaN != 0 is True in numpy) did
        # not catch either -- the NaN silently propagated and surfaced downstream as an unrelated
        # "Quantiles must be in the range [0, 1]" ValueError instead of this clear one. Checking the
        # message (not just the exception type) matters here: the old code eventually raised A
        # ValueError too, just the wrong, confusing one -- asserting only the type would not have
        # distinguished the fix from the bug.
        with self.assertRaisesRegex(ValueError, "at least 2 observations"):
            bootstrap(np.array([1.0]), lambda d: float(np.mean(d)), n_boot=50, method="bca", seed=5)

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

    def test_rejects_conflicting_resampling_schemes(self):
        # _resample_indices honors at most one of groups/clusters/block_length/m by a fixed
        # priority order (clusters, then groups, then block_length, then m); passing more than one
        # used to silently resolve to that priority instead of raising, so a caller who meant to
        # combine schemes (or passed both by mistake) got the OTHER one dropped with no signal.
        rng = np.random.RandomState(8)
        x = rng.normal(0, 1, 100)
        clusters = np.repeat(np.arange(20), 5)
        groups = np.repeat([0, 1], 50)
        with self.assertRaisesRegex(ValueError, "at most one"):
            bootstrap(x, lambda d: float(np.mean(d)), n_boot=50, groups=groups, clusters=clusters, seed=9)


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

    def test_rejects_empty_groups_instead_of_a_fake_zero_pvalue(self):
        # exact enumeration with an empty group falls back to comb(n, 0) == 1: it "enumerates" one
        # degenerate combination that recomputes stat() on the same empty-vs-everything split,
        # producing a null distribution of a single NaN. NaN >= NaN is False in numpy, so the
        # exceedance count was 0 out of 1 and the exact p-value formula (count / n) silently
        # returned 0.0 -- maximal significance from zero evidence -- instead of raising.
        empty = np.array([])
        nonempty = np.array([1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            permutation_test(empty, empty, exact_max=100)
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            permutation_test(empty, nonempty, exact_max=100)
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            permutation_test(nonempty, empty, exact_max=100)

    def test_rejects_empty_paired_input(self):
        empty = np.array([])
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            permutation_test(empty, empty, paired=True, exact_max=100)

    def test_rejects_empty_group_under_stratify(self):
        # the stratify branch has its own Monte-Carlo loop and bypasses the exact-enumeration
        # arithmetic entirely, but an empty group is just as degenerate there (every stratum
        # permutation is a no-op on a single-label array) -- the entry guard must catch this too,
        # not just the exact-enumeration path.
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            permutation_test(np.array([1.0, 2.0]), np.array([]), stratify=np.array([0, 1]), n_perm=50, exact_max=100)

    def test_single_observation_per_group_is_not_swept_into_the_empty_guard(self):
        # a single observation per group is a legitimately different case from empty: there are 2
        # distinct label-swap rearrangements (comb(2, 1) == 2), so it's a real, if maximally weak,
        # exact test -- it should run and correctly report no evidence (pvalue == 1.0), not raise.
        r = permutation_test(np.array([5.0]), np.array([1.0]), exact_max=100)
        self.assertTrue(r.exact)
        self.assertEqual(r.n_perm, 2)
        self.assertEqual(r.pvalue, 1.0)

    def test_single_pair_is_not_swept_into_the_empty_guard(self):
        # same reasoning for paired mode: n=1 still has 2 sign-flip rearrangements (2**1 == 2).
        r = permutation_test(np.array([5.0]), np.array([1.0]), paired=True, exact_max=100)
        self.assertTrue(r.exact)
        self.assertEqual(r.n_perm, 2)
        self.assertEqual(r.pvalue, 1.0)


if __name__ == "__main__":
    unittest.main()


class SubsamplingOrientationTest(unittest.TestCase):
    """Audit RS-1/RS-9: m-out-of-n takes the pivotal orientation and refuses m == n."""

    def test_m_equal_n_is_refused(self):
        # m == n is a permutation: every replicate equals the estimate and the "interval" has
        # width zero -- a silent 0%-coverage CI (measured width 5.6e-17 before the refusal)
        with self.assertRaisesRegex(ValueError, "m < n"):
            bootstrap(np.random.RandomState(0).randn(50), np.mean, m=50)

    def test_subsampling_interval_is_pivotal_not_percentile(self):
        # canonical U(0, theta) sample-max: every subsample max sits at or below the full-sample
        # max, so the PERCENTILE upper endpoint equals the sample max < theta almost surely --
        # coverage exactly zero. The basic (pivotal) orientation reflects through the estimate,
        # so the upper endpoint must exceed the sample max and cover theta here.
        x = np.random.RandomState(1).uniform(0.0, 1.0, 400)
        res = bootstrap(x, np.max, m=60, n_boot=400, seed=7)
        self.assertGreater(res.ci_high, float(np.max(x)))
        self.assertLessEqual(res.ci_low, 1.0)
        self.assertGreaterEqual(res.ci_high, 1.0)
        # the caller cannot opt back into the reflected orientation in m-mode
        again = bootstrap(x, np.max, m=60, n_boot=400, seed=7, method="percentile")
        self.assertEqual((res.ci_low, res.ci_high), (again.ci_low, again.ci_high))


class ResamplingHonestyTest(unittest.TestCase):
    """Audit RS-2/3/4/6: degenerate-jackknife fallback, mid-p z0, circular blocks, leverage."""

    def test_bca_falls_back_labelled_for_quantile_statistics(self):
        x = np.random.RandomState(1).standard_normal(60)
        med = bootstrap(x, lambda d: float(np.median(d)), n_boot=300, method="bca", seed=2)
        self.assertEqual(med.method, "percentile (bca-degenerate-jackknife)")
        mean = bootstrap(x, lambda d: float(np.mean(d)), n_boot=300, method="bca", seed=2)
        self.assertEqual(mean.method, "bca")

    def test_bca_z0_uses_the_mid_p_convention_for_atoms_at_the_estimate(self):
        # A balanced replicate set with a big atom AT the estimate: strict `<` reads prop = 0.1
        # (z0 = -1.28, a strong spurious shift); mid-p reads 0.1 + 0.8/2 = 0.5, i.e. z0 = 0, and
        # the BCa quantiles collapse to the plain percentile ones (up to tiny acceleration).
        from mixle.inference.resampling import _bca_interval

        rng = np.random.RandomState(3)
        data = rng.standard_normal(40)
        est = float(np.mean(data))
        reps = np.concatenate([np.full(100, est - 1.0), np.full(800, est), np.full(100, est + 1.0)])
        lo, hi = _bca_interval(data, lambda d: float(np.mean(d)), np.asarray(est), reps, alpha=0.05)
        self.assertAlmostEqual(float(lo), float(np.quantile(reps, 0.025)), delta=0.05)
        self.assertAlmostEqual(float(hi), float(np.quantile(reps, 0.975)), delta=0.05)

    def test_circular_blocks_remove_the_centring_bias(self):
        rng = np.random.RandomState(4)
        series = np.cumsum(rng.standard_normal(300)) * 0.1 + rng.standard_normal(300)
        blk = block_bootstrap(series, lambda d: float(np.mean(d)), block_length=25, n_boot=3000, seed=5)
        gap = abs(float(blk.distribution.mean()) - float(series.mean()))
        self.assertLess(gap, 0.02 * float(blk.distribution.std()) + 0.005)

    def test_wild_leverage_adjustment_widens_high_leverage_intervals(self):
        rng = np.random.RandomState(6)
        n = 60
        covariate = np.concatenate([rng.uniform(-1, 1, n - 3), np.array([6.0, 6.5, 7.0])])
        design = np.column_stack([np.ones(n), covariate])
        hat = np.diag(design @ np.linalg.inv(design.T @ design) @ design.T)
        y = 1.0 + 2.0 * covariate + rng.standard_normal(n) * (0.3 + 0.5 * np.abs(covariate))
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        fitted = design @ beta
        residuals = y - fitted

        def slope(ystar):
            return float(np.linalg.lstsq(design, ystar, rcond=None)[0][1])

        raw = wild_bootstrap(fitted, residuals, slope, n_boot=800, seed=7)
        adj = wild_bootstrap(fitted, residuals, slope, n_boot=800, seed=7, leverage=hat)
        self.assertGreater(float(adj.ci_high) - float(adj.ci_low), float(raw.ci_high) - float(raw.ci_low))
        with self.assertRaisesRegex(ValueError, "hat diagonals"):
            wild_bootstrap(fitted, residuals, slope, n_boot=10, leverage=np.ones(n))
