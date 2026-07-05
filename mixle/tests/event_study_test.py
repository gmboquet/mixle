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
