"""Contracts for stratified/counting-process Cox and gamma-frailty inference."""

import unittest

import numpy as np
from scipy import special

from mixle.inference.survival import (
    _concordance,
    _cox_offset,
    _frailty_posterior,
    _gamma_frailty_variance_mstep,
    cox_ph,
    frailty_cox,
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
            with self.subTest(call=call), self.assertRaises(ValueError):
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


if __name__ == "__main__":
    unittest.main()
