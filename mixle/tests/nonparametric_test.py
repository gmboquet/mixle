"""Classical nonparametric (rank-based) hypothesis tests -- verified against scipy.stats."""

import unittest

import numpy as np
import scipy.stats as ss

from mixle.inference.nonparametric import (
    brunner_munzel,
    cliffs_delta,
    dunn_test,
    friedman_test,
    jonckheere_terpstra,
    kruskal_wallis,
    ks_1samp,
    ks_2samp,
    mann_whitney_u,
    mood_median_test,
    page_trend_test,
    runs_test,
    sign_test,
    wilcoxon_signed_rank,
)


class AgainstScipyTest(unittest.TestCase):
    """The asymptotic statistics/p-values must match scipy's asymptotic mode."""

    def setUp(self):
        rng = np.random.RandomState(7)
        self.x = rng.normal(0, 1, 40)
        self.y = rng.normal(0.5, 1.3, 45)
        self.xp = rng.normal(0, 1, 30)
        self.yp = self.xp + rng.normal(0.2, 1, 30)
        self.groups = [rng.normal(m, 1, n) for m, n in ((0, 20), (0.4, 25), (0.9, 22))]
        self.cols = [(rng.normal(0, 1, (18, 4)) + np.arange(4) * 0.3)[:, j] for j in range(4)]

    def test_mann_whitney_u(self):
        for alt in ("two-sided", "greater", "less"):
            r = mann_whitney_u(self.x, self.y, alternative=alt)
            s = ss.mannwhitneyu(self.x, self.y, alternative=alt, method="asymptotic")
            self.assertAlmostEqual(r.statistic, s.statistic, places=9)
            self.assertAlmostEqual(r.pvalue, s.pvalue, places=9)
        # rank-biserial is bounded and signed like the location shift
        self.assertTrue(-1.0 <= r.rank_biserial <= 1.0)

    def test_wilcoxon(self):
        r = wilcoxon_signed_rank(self.xp, self.yp)
        s = ss.wilcoxon(self.xp, self.yp, method="approx")
        self.assertAlmostEqual(r.statistic, s.statistic, places=9)
        self.assertAlmostEqual(r.pvalue, s.pvalue, places=9)

    def test_kruskal(self):
        r = kruskal_wallis(*self.groups)
        s = ss.kruskal(*self.groups)
        self.assertAlmostEqual(r.statistic, s.statistic, places=9)
        self.assertAlmostEqual(r.pvalue, s.pvalue, places=9)
        self.assertIn("epsilon_squared", r.extra)

    def test_friedman(self):
        r = friedman_test(*self.cols)
        s = ss.friedmanchisquare(*self.cols)
        self.assertAlmostEqual(r.statistic, s.statistic, places=9)
        self.assertAlmostEqual(r.pvalue, s.pvalue, places=9)
        self.assertTrue(0.0 <= r.extra["kendalls_w"] <= 1.0)

    def test_brunner_munzel(self):
        r = brunner_munzel(self.x, self.y)
        s = ss.brunnermunzel(self.x, self.y)
        self.assertAlmostEqual(r.statistic, s.statistic, places=6)
        self.assertAlmostEqual(r.pvalue, s.pvalue, places=6)

    def test_mood_median(self):
        r = mood_median_test(*self.groups)
        s = ss.median_test(*self.groups, correction=False)
        self.assertAlmostEqual(r.statistic, s[0], places=9)
        self.assertAlmostEqual(r.pvalue, s[1], places=9)

    def test_ks(self):
        r2 = ks_2samp(self.x, self.y)
        s2 = ss.ks_2samp(self.x, self.y, method="asymp")
        self.assertAlmostEqual(r2.statistic, s2.statistic, places=9)
        self.assertAlmostEqual(r2.pvalue, float(s2.pvalue), places=9)
        r1 = ks_1samp(self.x, ss.norm.cdf)
        s1 = ss.ks_1samp(self.x, ss.norm.cdf, method="asymp")
        self.assertAlmostEqual(r1.statistic, s1.statistic, places=9)
        self.assertAlmostEqual(r1.pvalue, float(s1.pvalue), places=9)

    def test_page(self):
        r = page_trend_test(*self.cols)
        s = ss.page_trend_test(np.column_stack(self.cols))
        self.assertAlmostEqual(r.statistic, s.statistic, places=9)
        self.assertAlmostEqual(r.pvalue, s.pvalue, places=4)

    def test_sign(self):
        d = self.xp - self.yp
        r = sign_test(self.xp, self.yp)
        s = ss.binomtest(int(np.sum(d > 0)), int(np.sum(d != 0)), 0.5)
        self.assertAlmostEqual(r.pvalue, s.pvalue, places=9)


class BehaviorTest(unittest.TestCase):
    """The tests without a direct scipy counterpart must behave correctly on designed inputs."""

    def test_jonckheere_detects_monotone_trend(self):
        rng = np.random.RandomState(1)
        ordered = [rng.normal(m, 1, 15) for m in (0, 1, 2, 3)]
        flat = [rng.normal(0, 1, 15) for _ in range(4)]
        self.assertLess(jonckheere_terpstra(*ordered, alternative="increasing").pvalue, 0.001)
        self.assertGreater(jonckheere_terpstra(*flat).pvalue, 0.05)

    def test_dunn_flags_the_extreme_pair(self):
        rng = np.random.RandomState(2)
        g = [rng.normal(m, 1, 25) for m in (0, 0.2, 3.0)]
        d = dunn_test(*g, p_adjust="holm")
        self.assertLess(d.pvalues[d.comparisons.index((0, 2))], 0.01)  # 0 vs 3 separated
        self.assertGreater(d.pvalues[d.comparisons.index((0, 1))], 0.05)  # 0 vs 0.2 not

    def test_runs_test(self):
        alternating = np.array([0, 1] * 25, dtype=float)
        self.assertLess(runs_test(alternating).pvalue, 0.001)  # far too many runs
        clustered = np.array([0] * 25 + [1] * 25, dtype=float)
        self.assertLess(runs_test(clustered).pvalue, 0.001)  # far too few runs

    def test_cliffs_delta(self):
        rng = np.random.RandomState(3)
        self.assertGreater(cliffs_delta(rng.normal(2, 1, 60), rng.normal(0, 1, 60)), 0.6)
        z = rng.normal(0, 1, 50)
        self.assertAlmostEqual(cliffs_delta(z, z), 0.0, places=6)  # identical -> 0 (ties)


class JonckheereNullVarianceTest(unittest.TestCase):
    """MXR-080-1599: the null variance must be the published Jonckheere-Terpstra one.

    The implementation used ``n(n-1)(2n+3) - sum n_i(n_i-1)(2n_i+3)``, a hybrid of the two equivalent
    published expressions (``n(n-1)(2n+5) - ...`` and ``n^2(2n+3) - ...``) that matches neither, and
    so did not reduce to the Mann-Whitney variance for two ordered groups. These tests pin the
    two-group case against an INDEPENDENT scipy reference rather than against a hand-copied constant.
    """

    def test_two_group_variance_equals_the_mann_whitney_variance(self):
        # audit repro: two groups of five with complete separation
        low = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        high = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        result = jonckheere_terpstra(low, high, alternative="increasing")

        # J is the full n1*n2 pairwise count under complete separation, so mu and z are pinned by the
        # required variance n1*n2*(n+1)/12 = 22.9166... alone
        self.assertAlmostEqual(result.statistic, 25.0)
        expected_var = 5 * 5 * (10 + 1) / 12.0
        expected_z = (25.0 - (10**2 - 5**2 - 5**2) / 4.0) / np.sqrt(expected_var)
        self.assertAlmostEqual(result.extra["zscore"], expected_z, places=10)
        # the pre-fix variance was 21.5278, giving z=2.69408 instead of 2.61116
        self.assertNotAlmostEqual(result.extra["zscore"], 2.69408, places=4)

    def test_two_untied_groups_match_the_rank_sum_normal_approximation(self):
        rng = np.random.RandomState(17)
        for _ in range(8):
            low = rng.normal(0.0, 1.0, 6)
            high = rng.normal(0.6, 1.0, 9)
            got = jonckheere_terpstra(low, high, alternative="increasing")
            expected = ss.ranksums(high, low, alternative="greater")
            self.assertAlmostEqual(got.extra["zscore"], float(expected.statistic), places=10)
            self.assertAlmostEqual(got.pvalue, float(expected.pvalue), places=12)

    def test_two_tied_groups_match_the_tie_corrected_mann_whitney_variance(self):
        """The tied branch shares the same base terms, so it inherited the same error. Checked against
        the standard tie-corrected Mann-Whitney normal approximation, computed independently here
        because scipy's own ``ranksums`` applies no tie correction at all."""
        rng = np.random.RandomState(23)
        for _ in range(8):
            low = rng.randint(0, 4, 7).astype(float)
            high = rng.randint(0, 4, 6).astype(float)
            pooled = np.concatenate([low, high])
            if np.unique(pooled).size == pooled.size:
                continue  # this case must actually contain ties to be testing the tied branch
            n1, n2 = high.size, low.size
            n = n1 + n2
            _, counts = np.unique(pooled, return_counts=True)
            tie_term = sum(c**3 - c for c in counts)
            var_u = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
            u = ss.mannwhitneyu(high, low, alternative="greater", use_continuity=False, method="asymptotic")
            expected_z = (float(u.statistic) - n1 * n2 / 2.0) / np.sqrt(var_u)

            got = jonckheere_terpstra(low, high, alternative="increasing")
            self.assertAlmostEqual(got.extra["zscore"], expected_z, places=10)

    def test_multi_group_trend_detection_is_unaffected(self):
        """Negative control: the corrected variance is slightly LARGER, so it can only make the test
        more conservative -- a genuine monotone trend must still be detected and flat groups must
        still not be."""
        rng = np.random.RandomState(1)
        ordered = [rng.normal(m, 1, 15) for m in (0, 1, 2, 3)]
        flat = [rng.normal(0, 1, 15) for _ in range(4)]
        self.assertLess(jonckheere_terpstra(*ordered, alternative="increasing").pvalue, 0.001)
        self.assertGreater(jonckheere_terpstra(*flat).pvalue, 0.05)


if __name__ == "__main__":
    unittest.main()
