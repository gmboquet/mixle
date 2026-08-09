"""Hierarchical within-subject event study / DiD — the statistics must be provably correct.

The load-bearing test is `concurrent_shock`: a difference-in-differences must CANCEL a shock common to
treated and control and recover only the treatment effect. If that fails the identification claim is void.
"""

import unittest

import numpy as np

from mixle.inference import (
    gaussian_effect,
    hierarchical_event_study,
    poisson_lograte_effect,
    poisson_lograte_effects,
    poisson_pooled_rate_ratio,
    tipping_drift,
)


def _study(n, true_effect, shock, seed, base_var=1.0, n_obs=40):
    """Simulate treated+control subjects: control drifts by `shock`, treated by `shock + true_effect`."""
    rng = np.random.RandomState(seed)
    te, tv, ce, cv = [], [], [], []
    for _ in range(n):
        base = rng.normal(0, 2)  # a time-INVARIANT subject trait (differenced out)
        pre_t = base + rng.normal(0, 1, n_obs)
        post_t = base + shock + true_effect + rng.normal(0, 1, n_obs)
        e, v = gaussian_effect(pre_t, post_t)
        te.append(e)
        tv.append(v)
        base_c = rng.normal(0, 2)
        pre_c = base_c + rng.normal(0, 1, n_obs)
        post_c = base_c + shock + rng.normal(0, 1, n_obs)
        e, v = gaussian_effect(pre_c, post_c)
        ce.append(e)
        cv.append(v)
    return np.array(te), np.array(tv), np.array(ce), np.array(cv)


class EventStudyTest(unittest.TestCase):
    def test_recovers_a_known_treatment_effect(self):
        te, tv, ce, cv = _study(400, true_effect=0.5, shock=0.0, seed=0)
        r = hierarchical_event_study(te, tv, ce, cv)
        self.assertAlmostEqual(r.effect, 0.5, delta=0.08)  # recovers the true ATT
        self.assertAlmostEqual(r.control_mean, 0.0, delta=0.08)  # control did not move
        self.assertLess(r.p_value, 1e-6)  # and it is significant
        self.assertTrue(r.ci[0] < 0.5 < r.ci[1])  # CI covers truth

    def test_difference_in_differences_cancels_a_concurrent_shock(self):
        # a shock of +3 hits EVERYONE at the event time; only the +0.5 treatment differential is real
        te, tv, ce, cv = _study(400, true_effect=0.5, shock=3.0, seed=1)
        r = hierarchical_event_study(te, tv, ce, cv)
        self.assertAlmostEqual(r.treated_mean, 3.5, delta=0.12)  # treated raw shift = shock + effect
        self.assertAlmostEqual(r.control_mean, 3.0, delta=0.12)  # control raw shift = shock alone
        self.assertAlmostEqual(r.effect, 0.5, delta=0.1)  # DiD cancels the shock, keeps the effect
        # WITHOUT the control, the naive within-subject estimate is catastrophically confounded (3.5)
        naive = hierarchical_event_study(te, tv)
        self.assertGreater(naive.effect, 3.0)

    def test_null_effect_is_not_significant(self):
        te, tv, ce, cv = _study(300, true_effect=0.0, shock=1.5, seed=2)
        r = hierarchical_event_study(te, tv, ce, cv)
        self.assertLess(abs(r.effect), 0.1)
        self.assertGreater(r.p_value, 0.05)  # correctly finds no influence

    def test_confidence_interval_is_calibrated(self):
        covered = 0
        trials = 120
        for s in range(trials):
            te, tv, ce, cv = _study(120, true_effect=0.4, shock=0.0, seed=100 + s, n_obs=25)
            r = hierarchical_event_study(te, tv, ce, cv)
            covered += r.ci[0] <= 0.4 <= r.ci[1]
        self.assertGreater(covered / trials, 0.88)  # ~95% nominal; allow MC slack

    def test_heterogeneity_detected_when_effects_vary(self):
        rng = np.random.RandomState(7)

        # homogeneous effects -> tau^2 ~ 0; heterogeneous -> tau^2 > 0
        def run(spread):
            te, tv = [], []
            for _ in range(300):
                eff = 0.5 + rng.normal(0, spread)
                pre = rng.normal(0, 1, 40)
                post = eff + rng.normal(0, 1, 40)
                e, v = gaussian_effect(pre, post)
                te.append(e)
                tv.append(v)
            return hierarchical_event_study(np.array(te), np.array(tv)).tau2_treated

        self.assertLess(run(0.0), 0.02)
        self.assertGreater(run(0.6), 0.15)

    def test_poisson_lograte_effect_recovers_a_rate_ratio(self):
        rng = np.random.RandomState(3)
        te, tv = [], []
        for _ in range(500):
            rate = rng.gamma(3, 1)  # subject baseline rate (invariant)
            k_pre = rng.poisson(rate * 50)
            k_post = rng.poisson(rate * 2.0 * 50)  # true rate DOUBLES -> log-effect = ln 2
            e, v = poisson_lograte_effect(k_pre, 50, k_post, 50)
            te.append(e)
            tv.append(v)
        r = hierarchical_event_study(np.array(te), np.array(tv))
        self.assertAlmostEqual(r.effect, np.log(2), delta=0.05)

    def test_rejects_nonpositive_variance_and_empty_input(self):
        # a zero or negative per-subject sampling variance is physically impossible -- before this
        # guard, it silently produced a fully-formed, confidently-wrong "significant" result instead
        # of surfacing the impossible input.
        with self.assertRaises(ValueError):
            hierarchical_event_study(np.array([0.5, 0.3]), np.array([1.0, 0.0]))  # zero variance
        with self.assertRaises(ValueError):
            hierarchical_event_study(np.array([0.5, 0.3]), np.array([1.0, -0.2]))  # negative variance
        with self.assertRaises(ValueError):
            hierarchical_event_study(np.array([]), np.array([]))  # n=0

    def test_gaussian_effect_and_tipping_drift(self):
        e, v = gaussian_effect([1.0, 2.0, 1.5, 1.2], [3.0, 3.5, 2.8, 3.1])
        self.assertAlmostEqual(e, np.mean([3.0, 3.5, 2.8, 3.1]) - np.mean([1.0, 2.0, 1.5, 1.2]))
        self.assertGreater(v, 0)
        te, tv, ce, cv = _study(300, true_effect=0.5, shock=0.0, seed=5)
        r = hierarchical_event_study(te, tv, ce, cv)
        sens = tipping_drift(r)
        self.assertAlmostEqual(sens["drift_to_nullify_point"], r.effect)  # drift = effect nullifies it
        self.assertTrue(abs(sens["drift_to_nullify_ci"]) < abs(r.effect))  # CI edge is a smaller drift


if __name__ == "__main__":
    unittest.main()


class SparsePoissonRegimeTest(unittest.TestCase):
    """STAT-RR17-09: the sparse regime refuses Gaussianization; the exact pooled route replaces it."""

    def test_zero_count_windows_are_refused(self):
        with self.assertRaisesRegex(ValueError, "zero-count window"):
            poisson_lograte_effect(0, 1.0, 2, 2.0)

    def test_sparse_batches_are_refused_below_the_measured_floor(self):
        rng = np.random.RandomState(0)
        with self.assertRaisesRegex(ValueError, "measured floor"):
            poisson_lograte_effects(rng.poisson(0.5, 200), 1.0, rng.poisson(1.0, 200), 2.0)

    def test_pooled_exact_route_holds_its_level_on_the_reviewers_null(self):
        # rate .5 in both windows, exposures 1 and 2: the Gaussianized pipeline reached
        # p = 1.37e-12 at n=1000 under this exact null; the conditional exact test holds level
        rejections = 0
        for i in range(400):
            rng = np.random.RandomState(20_000 + i)
            out = poisson_pooled_rate_ratio(rng.poisson(0.5, 200), 1.0, rng.poisson(1.0, 200), 2.0)
            rejections += out["p_value_ratio_equals_1"] <= 0.05
        self.assertLessEqual(rejections / 400.0, 0.075)

    def test_pooled_route_recovers_a_true_ratio(self):
        rng = np.random.RandomState(7)
        out = poisson_pooled_rate_ratio(rng.poisson(0.5, 1000), 1.0, rng.poisson(2.0, 1000), 2.0)
        self.assertLess(out["ci"][0], 2.0)
        self.assertGreater(out["ci"][1], 1.5)
        self.assertLess(out["p_value_ratio_equals_1"], 1e-10)
        self.assertIn("common", out["estimand"])

    def test_dense_regime_still_gaussianizes(self):
        rng = np.random.RandomState(3)
        effects, variances = poisson_lograte_effects(rng.poisson(8.0, 300), 1.0, rng.poisson(8.0, 300), 1.0)
        self.assertEqual(effects.shape, (300,))
        z = effects.mean() / np.sqrt(variances.mean() / 300.0)
        self.assertLess(abs(z), 3.0)

    def test_batch_variances_depend_only_on_subject_totals(self):
        # Pass-19 blocker: per-subject variances built from the realized counts let inverse-
        # variance pooling weight low-y subjects up (corr(1/v, y) = -0.72 measured), dragging a
        # TRUE-NULL weighted mean to -0.15 and z to -7.8 at n=1000. The conditional route's
        # variance is the exact Binomial(n_i, p_bar) logit variance -- a deterministic function
        # of the subject's TOTAL alone -- so equal totals get identical weights and the noise
        # channel is closed by construction.
        rng = np.random.RandomState(11)
        k_pre = rng.poisson(4.6, 800)
        k_post = rng.poisson(13.8, 800)
        effects, variances = poisson_lograte_effects(k_pre, 1.0, k_post, 3.0)
        totals = (k_pre + k_post)[(k_pre + k_post) > 0]
        self.assertEqual(effects.shape, totals.shape)
        for total in np.unique(totals):
            self.assertEqual(np.unique(variances[totals == total]).size, 1)

    def test_batch_null_is_level_correct_under_heterogeneous_baselines(self):
        # STAT-RR19-03: the arm-mean debias assumed one shared baseline rate; with half the
        # subjects at rate 0.1 and half at 9.1 (exposures 1:3) and every true ratio exactly 1 it
        # rejected 400/400 with mean z -20.75. Conditioning on each subject's total makes the
        # baseline rate cancel exactly; measured 0.055 at n=1000 (400 reps) after the fix.
        from mixle.inference.event_study import _random_effects

        rejections = 0
        reps = 80
        for rep in range(reps):
            rs = np.random.RandomState(70_000 + rep)
            lam = np.where(rs.rand(1000) < 0.5, 0.1, 9.1)
            y, v = poisson_lograte_effects(rs.poisson(lam), 1.0, rs.poisson(lam * 3.0), 3.0)
            mean, var, _, _ = _random_effects(y, v)
            rejections += abs(mean / np.sqrt(var)) > 1.959963984540054
        self.assertLess(rejections / reps, 0.15)

    def test_batch_null_is_level_correct_after_debiasing(self):
        # The Haldane offset is removed by exact pmf-summation debiasing; through the real
        # DL pool the unequal-exposure true null must reject at ~5%, not 49-100%.
        from mixle.inference.event_study import _random_effects

        rejections = 0
        reps = 120
        for rep in range(reps):
            rs = np.random.RandomState(60_000 + rep)
            k_pre = rs.poisson(4.6, 1000)
            k_post = rs.poisson(13.8, 1000)
            y, v = poisson_lograte_effects(k_pre, 1.0, k_post, 3.0)
            mean, var, _, _ = _random_effects(y, v)
            rejections += abs(mean / np.sqrt(var)) > 1.959963984540054
        self.assertLess(rejections / reps, 0.12)
        self.assertGreater(rejections / reps, 0.005)
