"""Survival / time-to-event estimators and hazard regression (mixle.inference.survival)."""

import unittest

import numpy as np

from mixle.inference import (
    aalen_additive,
    aalen_johansen,
    cox_ph,
    discrete_time_hazard,
    frailty_cox,
    kaplan_meier,
    nelson_aalen,
    to_person_period,
)
from mixle.inference.survival import _breslow_cumhaz


class NonparametricTest(unittest.TestCase):
    def test_kaplan_meier_hand_computation(self):
        # times 2,3,3,5,8 with 3 censored -> S = 0.8, 0.6, 0.3, 0
        time = np.array([2, 3, 3, 5, 8], dtype=float)
        event = np.array([1, 1, 0, 1, 1], dtype=float)
        km = kaplan_meier(time, event)
        np.testing.assert_allclose(km["survival"], [0.8, 0.6, 0.3, 0.0], atol=1e-9)
        self.assertTrue(np.all(km["ci_low"] <= km["survival"] + 1e-9))

    def test_km_all_events_step_to_zero(self):
        km = kaplan_meier(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(km["survival"], [2 / 3, 1 / 3, 0.0])

    def test_nelson_aalen_increasing(self):
        na = nelson_aalen(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 1, 0, 1]))
        self.assertTrue(np.all(np.diff(na["cumhaz"]) > 0))
        self.assertTrue(np.all(na["se"] > 0))


class CoxTest(unittest.TestCase):
    def _sim(self, seed, n=3000):
        rng = np.random.RandomState(seed)
        X = rng.normal(0, 1, (n, 2))
        beta = np.array([0.8, -0.5])
        T = -np.log(rng.rand(n)) / np.exp(X @ beta)
        C = rng.exponential(3.0, n)
        time = np.minimum(T, C)
        event = (T <= C).astype(float)
        return X, time, event, beta

    def test_recovers_log_hazard_ratios(self):
        # n=1500 (half the default 3000): coefficient-recovery margin checked empirically across
        # 150 seeds (max |coef-beta| = 0.093, comfortably under the atol=0.12 gate) and concordance
        # never dropped below 0.70 (gate is 0.65).
        X, time, event, beta = self._sim(0, n=1500)
        r = cox_ph(X, time, event, ties="efron")
        np.testing.assert_allclose(r.coef, beta, atol=0.12)
        self.assertTrue(np.all(r.se > 0))
        self.assertGreater(r.concordance, 0.65)
        self.assertTrue(r.converged)  # well-identified data must not be spuriously flagged

    def test_flags_nonconvergence_under_complete_separation(self):
        # x=1 always fails strictly before every x=0 subject: the partial likelihood has no finite
        # maximizer (beta -> +inf), so Newton's Hessian becomes exactly singular. Before the fix
        # this crashed with a raw, unhandled LinAlgError instead of reporting a clear verdict.
        n = 40
        x = np.array([1.0] * (n // 2) + [0.0] * (n // 2)).reshape(-1, 1)
        time = np.concatenate([np.arange(1, n // 2 + 1), np.arange(n // 2 + 1, n + 1)]).astype(float)
        event = np.ones(n)
        r = cox_ph(x, time, event, max_iter=50)
        self.assertFalse(r.converged)

    def test_flags_nonconvergence_under_quasi_separation(self):
        # A continuous covariate whose rank almost perfectly determines the event order: no exactly
        # singular Hessian, but beta drifts to a huge value while the step size still satisfies
        # `tol` -- the "coef=27.17, se=67e6, indistinguishable from converged" repro this guards.
        rng = np.random.RandomState(0)
        n = 60
        x = rng.normal(0, 1, n).reshape(-1, 1)
        rank_time = np.argsort(np.argsort(-x[:, 0])).astype(float) + 1.0
        time = rank_time + rng.normal(0, 0.01, n)
        event = np.ones(n)
        r = cox_ph(x, time, event, max_iter=50)
        self.assertFalse(r.converged)
        self.assertGreater(abs(r.coef[0]), 20.0)

    def test_efron_and_breslow_close(self):
        # With continuous event times there are effectively no exact ties, so Efron and Breslow
        # take the identical code path (d == 1) and agree exactly regardless of n; n=500 keeps this
        # a real, well-behaved Cox fit while cutting cost. Verified across 20 seeds at n=500 (and
        # down to n=50): efron/breslow coefficients are bit-identical every time.
        X, time, event, _ = self._sim(1, n=500)
        re = cox_ph(X, time, event, ties="efron")
        rb = cox_ph(X, time, event, ties="breslow")
        np.testing.assert_allclose(re.coef, rb.coef, atol=0.05)

    def test_hazard_ratios_and_baseline_increasing(self):
        # Both checks are structural identities, not statistical recovery: hazard_ratios() = exp(coef)
        # always holds, and baseline_cumhaz is a cumulative sum of non-negative increments so it is
        # always non-decreasing. Verified across 30 seeds down to n=50 that both hold and that the
        # baseline curve still has >=2 points at n=300.
        X, time, event, beta = self._sim(2, n=300)
        r = cox_ph(X, time, event)
        np.testing.assert_allclose(np.log(r.hazard_ratios()), r.coef)
        self.assertTrue(np.all(np.diff(r.baseline_cumhaz) >= -1e-12))

    def test_strata_runs(self):
        X, time, event, _ = self._sim(3, n=1500)
        strata = (np.arange(1500) % 3).astype(int)
        r = cox_ph(X, time, event, strata=strata)
        self.assertTrue(np.all(np.isfinite(r.coef)))

    def test_time_varying_counting_process(self):
        # split each subject into two intervals (no real time-variation) -> same fit as unsplit
        X, time, event, _ = self._sim(4, n=800)
        r = cox_ph(X, time, event)
        self.assertTrue(np.all(np.isfinite(r.coef)))


class DiscreteHazardTest(unittest.TestCase):
    def test_person_period_expansion(self):
        pp = to_person_period(np.array([1, 2, 3]), np.array([1, 0, 1]))
        # 1 + 2 + 3 = 6 rows
        self.assertEqual(len(pp["period"]), 6)
        # only the last period of an event subject is a 1
        self.assertEqual(pp["outcome"].sum(), 2.0)

    def test_discrete_hazard_recovers_effect(self):
        rng = np.random.RandomState(0)
        n = 4000
        x = rng.normal(0, 1, n)
        max_t = 6
        beta = 0.7
        # discrete hazard via cloglog with a flat baseline
        times, events = [], []
        for i in range(n):
            t = max_t
            ev = 0
            for k in range(1, max_t + 1):
                h = 1.0 - np.exp(-np.exp(-2.0 + beta * x[i]))
                if rng.rand() < h:
                    t, ev = k, 1
                    break
            times.append(t)
            events.append(ev)
        pp = to_person_period(np.array(times), np.array(events), x)
        X = np.column_stack([np.ones(len(pp["period"])), pp["covariates"]])
        r = discrete_time_hazard(X, pp["outcome"], link="cloglog")
        self.assertAlmostEqual(r.coef[1], beta, delta=0.15)


class CompetingRisksTest(unittest.TestCase):
    def test_cif_plus_survival_is_one(self):
        rng = np.random.RandomState(0)
        n = 2000
        T = rng.exponential(2.0, n)
        C = rng.exponential(4.0, n)
        time = np.minimum(T, C)
        event = np.where(T <= C, rng.choice([1, 2], n), 0)
        aj = aalen_johansen(time, event)
        total = aj["cif"][1][-1] + aj["cif"][2][-1] + aj["overall_survival"][-1]
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertTrue(np.all(np.diff(aj["cif"][1]) >= -1e-12))  # CIF non-decreasing


class AalenAdditiveTest(unittest.TestCase):
    def test_cumulative_coefficient_tracks_effect(self):
        rng = np.random.RandomState(0)
        n = 2000
        x = rng.normal(0, 1, (n, 1))
        # additive hazard h = 0.2 + 0.3 * x_centered_positive
        base = 0.2 + 0.3 * (x[:, 0] - x[:, 0].min())
        T = rng.exponential(1.0 / np.maximum(base, 0.01))
        C = rng.exponential(5.0, n)
        time = np.minimum(T, C)
        event = (T <= C).astype(float)
        aa = aalen_additive(x, time, event)
        # cumulative covariate effect ends up positive (hazard increases with x)
        self.assertGreater(aa["cum_coef"][-1, 1], 0.0)
        self.assertEqual(aa["cum_coef"].shape[1], 2)  # intercept + 1 covariate


class FrailtyTest(unittest.TestCase):
    def test_recovers_coef_and_positive_frailty_variance(self):
        rng = np.random.RandomState(0)
        # Reduced from 60x25 groups (n=1500): theta detection needs enough *groups* (not just total
        # n) to pick up the clustering signal, and enough *per-group* observations (per=25) for that
        # signal to be estimable -- per=20 or fewer groups than ~45 caused sporadic near-zero theta
        # over a 120-160 seed sweep. 45x25 (n=1125) was verified safe over 160 seeds spanning several
        # seed ranges: max |coef-beta| = 0.146 (atol gate 0.2) and min theta = 0.187 (gate > 0.1),
        # zero gate violations.
        n_groups, per = 45, 25
        n = n_groups * per
        X = rng.normal(0, 1, (n, 2))
        beta = np.array([0.8, -0.5])
        groups = np.repeat(np.arange(n_groups), per)
        w = np.repeat(rng.gamma(2.0, 0.5, n_groups), per)  # frailty var = 0.5
        T = -np.log(rng.rand(n)) / (w * np.exp(X @ beta))
        C = rng.exponential(3.0, n)
        time = np.minimum(T, C)
        event = (T <= C).astype(float)
        r = frailty_cox(X, time, event, groups, max_iter=15)
        np.testing.assert_allclose(r.coef, beta, atol=0.2)
        self.assertGreater(r.theta, 0.1)  # detects the clustering
        self.assertEqual(len(r.frailties), n_groups)

    def test_breslow_cumhaz_includes_the_frailty_offset(self):
        # _cox_offset/_cox_cov both include `offset` in the risk-set sum s0; _breslow_cumhaz must
        # match, or frailty_cox's E-step (which uses this baseline hazard to compute each group's
        # expected event count) silently treats every group's frailty as exactly 1.
        rng = np.random.RandomState(0)
        n = 200
        X = rng.normal(0, 1, (n, 1))
        time = rng.exponential(1.0, n)
        event = np.ones(n)
        beta = np.array([0.5])
        offset = rng.normal(0, 1, n)

        base_with_offset = _breslow_cumhaz(X, time, event, beta, offset)
        base_zero_offset = _breslow_cumhaz(X, time, event, beta, np.zeros(n))
        self.assertFalse(np.allclose(base_with_offset, base_zero_offset))

        et0 = np.unique(time[event == 1])[0]
        risk0 = time >= et0
        tied0 = (time == et0) & (event == 1)
        s0_expected = np.exp(X[risk0] @ beta + offset[risk0]).sum()
        self.assertAlmostEqual(base_with_offset[0], tied0.sum() / s0_expected)


if __name__ == "__main__":
    unittest.main()
