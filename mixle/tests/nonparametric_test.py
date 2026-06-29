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


if __name__ == "__main__":
    unittest.main()
