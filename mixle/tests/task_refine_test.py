"""Diagnosis-directed correction vs blind structure search on a planted-fault benchmark (REFINE-a).

Kill criterion (stated before the comparison, per the card): diagnosis-directed correction must reach
the held-out target in FEWER trials than blind search over the same edit space, or the negative result
is what these tests would need to record instead."""

import itertools
import unittest

import numpy as np

from mixle.inference.explain import diagnose
from mixle.task.refine import blind_structure_search, diagnosis_directed_correction, fit_independent_baseline

# 4 fields: 0 and 3 are independent noise; 1 (a) and 2 (b) carry a genuine, tight linear correlation
# (b = a + 0.05 + small noise) -- the planted fault is modeling 1 and 2 as independent.
_N = 300


def _generate(seed: int, n: int = _N) -> list[tuple]:
    rng = np.random.default_rng(seed)
    a = rng.uniform(-3.0, 3.0, size=n)
    b = a + 0.05 + rng.normal(0.0, 0.15, size=n)
    f0 = rng.normal(0.0, 1.0, size=n)
    f3 = rng.normal(0.0, 1.0, size=n)
    return list(zip(f0.tolist(), a.tolist(), b.tolist(), f3.tolist()))


def _split_cases(train_data: list[tuple]) -> tuple[list[tuple], list[tuple]]:
    background = [x for x in train_data if abs(x[1]) <= 1.5]
    failing = [x for x in train_data if abs(x[1]) > 2.2]
    return background, failing


_EDIT_SPACE = list(itertools.combinations(range(4), 2))  # canonical enumeration order, not cherry-picked


class PlantedFaultSetupTest(unittest.TestCase):
    def test_diagnose_names_the_planted_pair_on_the_fitted_baseline(self):
        train_data = _generate(seed=0)
        baseline = fit_independent_baseline(train_data)
        background, failing = _split_cases(train_data)

        report = diagnose(baseline, failing, background=background)
        self.assertEqual(report.suggested_fix, "add_edge")
        self.assertIn("field[1]", report.dominant)
        self.assertIn("field[2]", report.dominant)


class DiagnosisVsBlindSearchTest(unittest.TestCase):
    def setUp(self):
        self.train_data = _generate(seed=0)
        self.held_out = _generate(seed=1)
        self.baseline = fit_independent_baseline(self.train_data)
        self.background, self.failing = _split_cases(self.train_data)
        baseline_score = float(np.mean([self.baseline.log_density(x) for x in self.held_out]))
        self.target = baseline_score + 1.0  # only the true edge (~+2.4 nats) clears this bar

    def test_diagnosis_directed_correction_finds_the_true_edge_in_one_or_two_trials(self):
        outcome = diagnosis_directed_correction(
            self.baseline, self.train_data, self.failing, self.held_out, background=self.background, target=self.target
        )
        self.assertIsNotNone(outcome.found_edge)
        self.assertEqual(set(outcome.found_edge), {1, 2})
        self.assertLessEqual(outcome.trials, 2)
        self.assertTrue(outcome.history[-1].verified)

    def test_blind_search_needs_more_trials_over_the_same_edit_space(self):
        outcome = blind_structure_search(self.baseline, self.train_data, self.held_out, _EDIT_SPACE, target=self.target)
        self.assertIsNotNone(outcome.found_edge)
        self.assertEqual(set(outcome.found_edge), {1, 2})
        self.assertGreater(outcome.trials, 2)  # the canonical order tries several wrong edges first

    def test_diagnosis_beats_blind_search_on_trials_to_target(self):
        directed = diagnosis_directed_correction(
            self.baseline, self.train_data, self.failing, self.held_out, background=self.background, target=self.target
        )
        blind = blind_structure_search(self.baseline, self.train_data, self.held_out, _EDIT_SPACE, target=self.target)

        # the card's literal acceptance test: fewer trials than blind search, or record the negative result.
        self.assertLess(directed.trials, blind.trials)

    def test_wrong_edges_do_not_clear_the_target_only_the_true_pair_does(self):
        outcome = blind_structure_search(self.baseline, self.train_data, self.held_out, _EDIT_SPACE, target=self.target)
        for trial in outcome.history:
            if set(trial.edge) != {1, 2}:
                self.assertFalse(trial.verified)


class NoFaultFoundTest(unittest.TestCase):
    def test_a_well_specified_model_yields_no_diagnosis_and_zero_trials(self):
        # correct from the start: fields modeled with the true edge already in place
        train_data = _generate(seed=0)
        from mixle.inference.bayesian_network import (
            HeterogeneousBayesianNetwork,
            _LinearGaussianFactor,
            _MarginalFactor,
        )
        from mixle.stats import GaussianDistribution

        cols = [list(col) for col in zip(*train_data)]
        factors = [
            _MarginalFactor(0, GaussianDistribution(float(np.mean(cols[0])), float(np.var(cols[0])))),
            _MarginalFactor(1, GaussianDistribution(float(np.mean(cols[1])), float(np.var(cols[1])))),
            _LinearGaussianFactor.fit(2, [1], cols, discrete={}),
            _MarginalFactor(3, GaussianDistribution(float(np.mean(cols[3])), float(np.var(cols[3])))),
        ]
        well_specified = HeterogeneousBayesianNetwork(factors)
        background, failing = _split_cases(train_data)

        outcome = diagnosis_directed_correction(
            well_specified, train_data, failing, train_data, background=background, target=0.0
        )
        self.assertEqual(outcome.trials, 0)
        self.assertIsNone(outcome.found_edge)


if __name__ == "__main__":
    unittest.main()
