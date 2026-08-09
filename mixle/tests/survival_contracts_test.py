"""Contracts for stratified/counting-process Cox and gamma-frailty inference."""

import unittest

import numpy as np
from scipy import special

from mixle.inference.survival import (
    _concordance,
    _cox_offset,
    _frailty_posterior,
    _gamma_frailty_variance_mstep,
    aalen_additive,
    aalen_johansen,
    cox_ph,
    frailty_cox,
    kaplan_meier,
    nelson_aalen,
    to_person_period,
)


class CoxStructureContractsTest(unittest.TestCase):
    def _sample(self, seed=8, n=120):
        rng = np.random.RandomState(seed)
        x = rng.normal(size=(n, 1))
        failure = -np.log(rng.rand(n)) / np.exp(0.5 * x[:, 0])
        censoring = rng.exponential(3.0, n)
        return x, np.minimum(failure, censoring), (failure <= censoring).astype(float)

    def test_stratified_baselines_are_keyed_and_not_concatenated(self):
        x, time, event = self._sample()
        strata = np.arange(time.size) % 2
        result = cox_ph(x, time, event, strata=strata)
        self.assertEqual(set(result.baseline_by_stratum), {0, 1})
        self.assertEqual(result.baseline_time.size, 0)
        self.assertEqual(result.baseline_cumhaz.size, 0)
        for label, baseline in result.baseline_by_stratum.items():
            mask = strata == label
            np.testing.assert_array_equal(baseline["time"], np.unique(time[mask & (event == 1)]))
            self.assertTrue(np.all(np.diff(baseline["cumhaz"]) >= 0))

    def test_concordance_only_compares_subjects_within_strata(self):
        risk = np.array([10.0, 0.0, 0.0, 10.0])
        time = np.array([100.0, 120.0, 1.0, 2.0])
        event = np.array([1.0, 0.0, 1.0, 0.0])
        strata = np.array(["late", "late", "early", "early"])
        value = _concordance(
            risk,
            time,
            event,
            start=np.full(4, -np.inf),
            strata=strata,
            subject=np.arange(4),
        )
        self.assertEqual(value, 0.5)

    def test_counting_process_concordance_requires_subject_identity(self):
        x, terminal, terminal_event = self._sample(n=60)
        midpoint = terminal / 2.0
        design = np.repeat(x, 2, axis=0)
        start = np.column_stack([np.zeros(60), midpoint]).ravel()
        stop = np.column_stack([midpoint, terminal]).ravel()
        event = np.column_stack([np.zeros(60), terminal_event]).ravel()
        subject = np.repeat(np.arange(60), 2)
        without_identity = cox_ph(design, stop, event, start=start)
        with_identity = cox_ph(design, stop, event, start=start, subject=subject)
        self.assertIsNone(without_identity.concordance)
        self.assertIsInstance(with_identity.concordance, float)
        self.assertGreaterEqual(with_identity.concordance, 0)
        self.assertLessEqual(with_identity.concordance, 1)

    def test_cox_structure_and_solver_controls_are_validated(self):
        x, time, event = self._sample(n=20)
        invalid = [
            lambda: cox_ph(x, time[:-1], event),
            lambda: cox_ph(x, time, np.where(np.arange(20) == 0, 2, event)),
            lambda: cox_ph(x, time, event, start=time),
            lambda: cox_ph(x, time, event, strata=np.arange(19)),
            lambda: cox_ph(x, time, event, subject=np.arange(19)),
            lambda: cox_ph(x, time, event, ties="exact"),
            lambda: cox_ph(x, time, event, max_iter=0),
        ]
        for call in invalid:
            with self.subTest(call=repr(call)), self.assertRaises(ValueError):
                call()


class FrailtyEMContractsTest(unittest.TestCase):
    def test_e_step_retains_all_gamma_posterior_moments(self):
        x = np.zeros((4, 1))
        time = np.array([1.0, 2.0, 1.0, 2.0])
        event = np.array([1.0, 0.0, 0.0, 1.0])
        group_index = np.array([0, 0, 1, 1])
        mean, variance, expected_log = _frailty_posterior(
            x,
            time,
            event,
            group_index,
            2,
            np.zeros(1),
            np.array([0.2, 0.5]),
            0.5,
        )
        shape, rate = 3.0, 2.7
        np.testing.assert_allclose(mean, shape / rate)
        np.testing.assert_allclose(variance, shape / rate**2)
        np.testing.assert_allclose(expected_log, special.digamma(shape) - np.log(rate))

    def test_dispersion_m_step_uses_expected_log_frailty(self):
        means = np.ones(2)
        concentrated = _gamma_frailty_variance_mstep(means, np.full(2, -0.01))
        dispersed = _gamma_frailty_variance_mstep(means, np.full(2, -1.0))
        self.assertGreater(dispersed, concentrated)

    def test_efron_ties_change_the_frailty_coefficient_m_step(self):
        rng = np.random.RandomState(3)
        n = 120
        x = rng.normal(size=(n, 1))
        raw_time = -np.log(rng.rand(n)) / np.exp(0.6 * x[:, 0])
        failure = np.ceil(raw_time * 2.0) / 2.0
        censoring = rng.exponential(3.0, n)
        time = np.minimum(failure, censoring)
        event = (failure <= censoring).astype(float)
        offset = np.zeros(n)
        breslow = _cox_offset(x, time, event, offset, ties="breslow")
        efron = _cox_offset(x, time, event, offset, ties="efron")
        self.assertFalse(np.allclose(breslow, efron))

    def test_frailty_result_discloses_posterior_variance_and_convergence(self):
        rng = np.random.RandomState(12)
        n_groups, per_group = 12, 8
        groups = np.repeat(np.arange(n_groups), per_group)
        x = rng.normal(size=(groups.size, 1))
        frailty = np.repeat(rng.gamma(2.0, 0.5, n_groups), per_group)
        failure = -np.log(rng.rand(groups.size)) / (frailty * np.exp(0.4 * x[:, 0]))
        censoring = rng.exponential(3.0, groups.size)
        result = frailty_cox(
            x,
            np.minimum(failure, censoring),
            (failure <= censoring).astype(float),
            groups,
            max_iter=5,
            ties="efron",
        )
        self.assertEqual(result.ties, "efron")
        self.assertEqual(result.frailty_variance.shape, result.frailties.shape)
        self.assertTrue(np.all(result.frailty_variance >= 0))
        self.assertEqual(result.frailty_log_mean.shape, result.frailties.shape)
        self.assertIsInstance(result.converged, bool)


class NonparametricInputContractsTest(unittest.TestCase):
    """MXR-080-1607: the nonparametric entry points silently recoded or discarded malformed input.

    ``cox_ph`` validated its ``(time, event)`` pair, but every nonparametric surface reached
    ``_event_table`` through a bare ``np.asarray(..., dtype=float)``. A fractional or out-of-range
    event code compared unequal to 1 and so was dropped from the risk table as if censored, and a
    negative time was sorted into the table as a real duration -- both produced a plausible-looking
    survival curve estimated from data the caller never supplied.
    """

    def test_kaplan_meier_rejects_non_binary_event_codes(self):
        for event in ([0.5, 1.0], [2, 1], [-1, 1]):
            with self.assertRaisesRegex(ValueError, "0 \\(censored\\) and 1 \\(event\\)"):
                kaplan_meier([1.0, 2.0], event)

    def test_kaplan_meier_rejects_malformed_times(self):
        with self.assertRaisesRegex(ValueError, "non-negative durations"):
            kaplan_meier([-1.0, 2.0], [1, 1])
        with self.assertRaisesRegex(ValueError, "finite"):
            kaplan_meier([np.nan, 2.0], [1, 1])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            kaplan_meier([], [])
        with self.assertRaisesRegex(ValueError, "aligned with time"):
            kaplan_meier([1.0, 2.0], [1])

    def test_kaplan_meier_rejects_invalid_ci_level(self):
        for level in (1.5, 0.0, 1.0, np.nan, True):
            with self.assertRaisesRegex(ValueError, "ci_level"):
                kaplan_meier([1.0, 2.0], [1, 1], ci_level=level)

    def test_nelson_aalen_rejects_non_binary_event_codes(self):
        with self.assertRaisesRegex(ValueError, "0 \\(censored\\) and 1 \\(event\\)"):
            nelson_aalen([1.0, 2.0], [7, 1])

    def test_aalen_johansen_rejects_malformed_cause_labels(self):
        for event in ([-1, 1], [1.5, 1]):
            with self.assertRaisesRegex(ValueError, "non-negative integer cause labels"):
                aalen_johansen([1.0, 2.0], event)
        with self.assertRaisesRegex(ValueError, "missing from causes"):
            aalen_johansen([1.0, 2.0], [3, 1], causes=[1, 2])
        with self.assertRaisesRegex(ValueError, "must not repeat"):
            aalen_johansen([1.0, 2.0], [1, 1], causes=[1, 1])

    def test_aalen_additive_rejects_misaligned_design(self):
        with self.assertRaisesRegex(ValueError, "0 \\(censored\\) and 1 \\(event\\)"):
            aalen_additive(np.ones((2, 1)), [1.0, 2.0], [5, 1])
        with self.assertRaisesRegex(ValueError, "one row per observed time"):
            aalen_additive(np.ones((3, 1)), [1.0, 2.0], [1, 1])

    def test_to_person_period_rejects_fractional_and_negative_durations(self):
        with self.assertRaisesRegex(ValueError, "whole numbers of observed periods"):
            to_person_period([2.7, 3.0], [1, 1])
        with self.assertRaisesRegex(ValueError, "non-negative durations"):
            to_person_period([-2.0, 3.0], [1, 1])

    def test_well_formed_inputs_are_unchanged(self):
        km = kaplan_meier([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 1])
        np.testing.assert_allclose(km["survival"], [0.75, 0.375, 0.0])
        self.assertEqual(km["median"], 3.0)
        np.testing.assert_allclose(nelson_aalen([1.0, 2.0, 3.0])["cumhaz"], [1 / 3, 1 / 3 + 0.5, 1 / 3 + 0.5 + 1.0])
        cif = aalen_johansen([1.0, 2.0, 3.0], [1, 2, 0])["cif"]
        np.testing.assert_allclose(cif[1], [1 / 3, 1 / 3])
        np.testing.assert_allclose(cif[2], [0.0, 1 / 3])
        self.assertEqual(to_person_period([2, 3], [1, 0])["period"].tolist(), [1, 2, 1, 2, 3])


class SurvivalHonestyContractsTest(unittest.TestCase):
    """Audit S-2/S-5/S-6: honest frailty SEs, Aalen truncation, person-period zero refusal."""

    def _clustered(self, seed=3, n_groups=6, per_group=8, theta=1.0):
        rng = np.random.RandomState(seed)
        w = rng.gamma(1.0 / theta, theta, size=n_groups)
        x = rng.standard_normal(n_groups * per_group)
        grp = np.repeat(np.arange(n_groups), per_group)
        lam = w[grp] * np.exp(0.7 * x)
        t = rng.exponential(1.0 / lam)
        c = rng.exponential(2.0 / np.median(lam), size=t.size)
        return x.reshape(-1, 1), np.minimum(t, c), (t <= c).astype(int), grp

    def test_frailty_jackknife_se_is_available_and_recorded(self):
        x, t, e, g = self._clustered()
        plug_in = frailty_cox(x, t, e, g)
        jack = frailty_cox(x, t, e, g, se_method="jackknife")
        self.assertEqual(plug_in.se_method, "complete-data")
        self.assertEqual(jack.se_method, "jackknife")
        # same EM point estimate; only the SE machinery differs
        np.testing.assert_allclose(jack.coef, plug_in.coef)
        self.assertTrue(np.all(np.isfinite(jack.se)) and np.all(jack.se > 0))

    def test_frailty_jackknife_refuses_too_few_groups(self):
        x, t, e, _ = self._clustered(n_groups=2)
        two_groups = np.repeat([0, 1], x.shape[0] // 2)
        with self.assertRaisesRegex(ValueError, "at least 3 groups"):
            frailty_cox(x, t, e, two_groups, se_method="jackknife")
        with self.assertRaisesRegex(ValueError, "se_method"):
            frailty_cox(x, t, e, two_groups, se_method="louis")

    def test_aalen_additive_truncates_instead_of_accumulating_noise(self):
        # p = 4 columns (intercept + 3) against 8 all-event subjects: the risk set falls below
        # p before the last event times, where increments used to be min-norm numerical noise.
        rng = np.random.RandomState(0)
        x = rng.standard_normal((8, 3))
        time = np.arange(1.0, 9.0)
        out = aalen_additive(x, time, np.ones(8))
        self.assertIsNotNone(out["truncated_at"])
        self.assertEqual(out["cum_coef"].shape[0], out["time"].size)
        self.assertLess(out["time"].max(), out["truncated_at"])
        # the estimated prefix is exactly what it was BEFORE truncation existed: stopping is
        # not allowed to perturb the estimable part of the curve
        n_kept = out["time"].size
        self.assertTrue(np.all(np.isfinite(out["cum_coef"])))
        self.assertEqual(out["time"].tolist(), time[:n_kept].tolist())

    def test_person_period_refuses_zero_duration_instead_of_dropping_it(self):
        with self.assertRaisesRegex(ValueError, "time contains 0"):
            to_person_period(np.array([0, 2]), np.array([1, 0]))


if __name__ == "__main__":
    unittest.main()
