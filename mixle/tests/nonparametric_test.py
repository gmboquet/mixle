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
        # method='auto' is the reference since the NP-3 fix: exact at small samples (as here),
        # asymptotic at large -- the same regime switch scipy itself makes
        s2 = ss.ks_2samp(self.x, self.y)
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


class OrderedAndSmallSampleHonestyTest(unittest.TestCase):
    """Audit NP-6/7/8/11: Page tie variance, BM separation bound, Mood exact route, JT cross-check."""

    def test_page_tie_variance_reduces_to_textbook_without_ties(self):
        rng = np.random.RandomState(0)
        cols = [rng.standard_normal(12) for _ in range(4)]
        res = page_trend_test(*cols)
        n, k = 12, 4
        mu = n * k * (k + 1) ** 2 / 4.0
        var_textbook = n * k**2 * (k + 1) * (k**2 - 1) / 144.0
        implied = ((res.statistic - mu) / res.extra["zscore"]) ** 2
        self.assertAlmostEqual(implied, var_textbook, places=6)

    def test_page_is_level_correct_under_heavy_ties(self):
        # The hardcoded no-tie variance overstated Var(L) for tied midranks (conservative, z
        # biased toward 0); the exact per-block permutation variance restores the level.
        rej, used = 0, 0
        for r in range(1500):
            rs = np.random.RandomState(50_000 + r)
            data = rs.randint(0, 3, size=(8, 4)).astype(float)
            try:
                pv = page_trend_test(*[data[:, j] for j in range(4)]).pvalue
            except ValueError:
                continue
            used += 1
            rej += pv < 0.05
        self.assertGreater(used, 1400)
        self.assertGreater(rej / used, 0.03)
        self.assertLess(rej / used, 0.075)

    def test_brunner_munzel_separation_reports_no_p_under_its_own_null(self):
        # STAT-RR19-04: the permutation tail 1/C(n, n1) is exact only under FULL exchangeability;
        # X = +/-1 equiprobable vs Y = 0 satisfies stochastic equality yet separates with
        # probability 2^(-n1), so the old report rejected 12.5% of that null at nominal 5%.
        # The p-value is now honestly NaN; the stronger-null bound ships under its true name.
        from math import comb

        r = brunner_munzel([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
        self.assertTrue(np.isinf(r.statistic))
        self.assertEqual(r.extra["method"], "separation-no-valid-p-under-stochastic-equality")
        self.assertTrue(np.isnan(r.pvalue))
        self.assertAlmostEqual(r.extra["p_exchangeability"], 2.0 / comb(10, 5), places=12)
        supported = brunner_munzel([1, 2, 3], [5, 6, 7], alternative="less")
        opposed = brunner_munzel([1, 2, 3], [5, 6, 7], alternative="greater")
        self.assertTrue(np.isnan(supported.pvalue) and np.isnan(opposed.pvalue))
        self.assertAlmostEqual(supported.extra["p_exchangeability"], 1.0 / comb(6, 3), places=12)
        self.assertEqual(opposed.extra["p_exchangeability"], 1.0)
        self.assertIn("exchangeability", r.extra["note"])

    def test_mood_routes_small_two_group_tables_to_fisher(self):
        m = mood_median_test([1.0, 2.0, 3.0, 10.0], [8.0, 9.0, 11.0, 12.0])
        self.assertEqual(m.extra["method"], "fisher-exact")
        self.assertLess(m.extra["min_expected_count"], 5.0)
        big = np.random.RandomState(1).standard_normal(60)
        m2 = mood_median_test(list(big), list(big + 0.1))
        self.assertEqual(m2.extra["method"], "chi-square")

    def test_jt_matches_mwu_exactly_with_continuity_off(self):
        jt = jonckheere_terpstra([1, 2, 3, 4, 5], [10, 11, 12, 13, 14], alternative="increasing")
        mw = mann_whitney_u([1, 2, 3, 4, 5], [10, 11, 12, 13, 14], alternative="less", use_continuity=False)
        self.assertAlmostEqual(abs(jt.extra["zscore"]), abs(mw.zscore), places=9)
        self.assertAlmostEqual(jt.pvalue, mw.pvalue, places=9)


if __name__ == "__main__":
    unittest.main()


class ExactSmallSampleNullTest(unittest.TestCase):
    """Audit NP-1/NP-3: exact small-sample nulls where SciPy/R use them."""

    def test_wilcoxon_exact_branch_restores_the_level(self):
        from itertools import product

        from mixle.inference.nonparametric import wilcoxon_signed_rank

        # the most extreme n=5 outcome: exact two-sided p is 2/32, and the old normal branch
        # reported 0.043 -- a guaranteed 6.25% type-I rate at nominal 5%
        r = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(r.pvalue, 0.0625, places=12)
        # over ALL 32 sign patterns at n=5, no attainable p may sit below the exact minimum,
        # so a 5% test can never reject under this null (the level violation is structural)
        magnitudes = [1.0, 2.0, 3.0, 4.0, 5.0]
        smallest = min(
            wilcoxon_signed_rank([m * s for m, s in zip(magnitudes, signs)]).pvalue
            for signs in product((1.0, -1.0), repeat=5)
        )
        self.assertAlmostEqual(smallest, 0.0625, places=12)
        # one-sided exact tails: all-positive differences give P(T+ >= 15) = 1/32
        one_sided = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0], alternative="greater")
        self.assertAlmostEqual(one_sided.pvalue, 1.0 / 32.0, places=12)

    def test_ks_2samp_small_sample_p_is_exact(self):
        import numpy as np
        from scipy import stats as scipy_stats

        from mixle.inference.nonparametric import ks_2samp

        # complete separation at n1=n2=3: the exact permutation p is 2/C(6,3) = 0.1; the old
        # one-sample-law substitution returned 0.0 -- certainty from the weakest possible evidence
        r = ks_2samp([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        self.assertAlmostEqual(r.pvalue, 0.1, places=9)
        # large samples agree with scipy's auto method to numerical precision
        x = np.random.RandomState(2).randn(80)
        y = np.random.RandomState(3).randn(90) + 0.5
        self.assertAlmostEqual(ks_2samp(list(x), list(y)).pvalue, scipy_stats.ks_2samp(x, y).pvalue, places=9)


class ExactCeilingTest(unittest.TestCase):
    """STAT-RR19-13/-14: the exact-null ceilings were historical, not computational."""

    def test_wilcoxon_n26_far_tail_is_exact(self):
        # All-positive untied n=26 got normal p = 8.3e-6 where the exact tail is 2.98e-8 -- a
        # 278x evidence overstatement. The subset-sum DP is O(n^3); the ceiling is now 300.
        import scipy.stats as ss

        d = np.arange(1.0, 27.0)
        r = wilcoxon_signed_rank(d)
        self.assertEqual(r.method, "exact")
        self.assertAlmostEqual(r.pvalue, ss.wilcoxon(d, method="exact").pvalue, places=15)
        rng = np.random.RandomState(0)
        d2 = rng.standard_normal(60) + 0.3
        self.assertEqual(wilcoxon_signed_rank(d2).method, "exact")
        self.assertAlmostEqual(wilcoxon_signed_rank(d2).pvalue, ss.wilcoxon(d2, method="exact").pvalue, places=12)

    def test_runs_test_stays_exact_past_sixty(self):
        # Exhaustive enumeration of every C(61, 3) sequence measured 9.83% size at nominal 5% on
        # the normal branch; the closed-form pmf is exact big-integer arithmetic at any n.
        seq = np.zeros(61)
        seq[[5, 30, 55]] = 1.0
        res = runs_test(seq, cutoff=0.5)
        self.assertEqual(res.extra["method"], "exact")
        big = np.tile([0.0, 1.0], 150)  # n=300: still exact
        self.assertEqual(runs_test(big, cutoff=0.5).extra["method"], "exact")
