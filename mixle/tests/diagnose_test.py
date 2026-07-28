"""DIAGNOSE-a: diagnose() aggregates explain() ledgers over failing cases into a FaultReport --
planted-fault recovery for a missing dependency, and a well-specified model showing nothing dominant.

The merged H1-a `explain`/`explain_margin` do NOT take the (models, priors) dict-of-classifiers shape
the original card sketched before H1-a landed; the real API is `explain(model, x) -> Explanation`
(see mixle/inference/explain.py). `diagnose` composes with THAT real signature, calling `explain` once
per case and aggregating -- same spirit (H1's ledger feeds H5's critic), adapted to the shipped API.
"""

import unittest

import numpy as np

from mixle.inference import diagnose
from mixle.inference.bayesian_network import HeterogeneousBayesianNetwork, _LinearGaussianFactor, _MarginalFactor
from mixle.stats import GaussianDistribution


def _split_cases():
    """A grid of (a, b) pairs where b = a + 0.05 -- a genuine, tight correlation between the two
    fields. 'background' is the interior (typical) range; 'failing' is the tail, where a model that
    does not know about the correlation is most surprised."""
    a_grid = np.linspace(-3.0, 3.0, 41)
    pairs = [(float(a), float(a) + 0.05) for a in a_grid]
    background = [p for p in pairs if abs(p[0]) <= 1.5]
    failing = [p for p in pairs if abs(p[0]) > 2.2]
    return background, failing


def _buggy_net():
    """The planted fault: b is truly b = a + noise, but modeled as fully independent of a."""
    return HeterogeneousBayesianNetwork(
        [
            _MarginalFactor(0, GaussianDistribution(0.0, 1.0)),
            _MarginalFactor(1, GaussianDistribution(0.0, 1.0)),
        ]
    )


def _well_specified_net():
    """The SAME data, correctly modeled: b's factor is conditioned on a."""
    return HeterogeneousBayesianNetwork(
        [
            _MarginalFactor(0, GaussianDistribution(0.0, 1.0)),
            _LinearGaussianFactor(1, [0], {}, np.array([1.0, 0.0]), 0.1),
        ]
    )


class PlantedFaultRecoveryTest(unittest.TestCase):
    def test_missing_edge_is_named_dominant_with_add_edge(self):
        background, failing = _split_cases()
        report = diagnose(_buggy_net(), failing, background=background)

        self.assertEqual(report.suggested_fix, "add_edge")
        self.assertIn("field[0]", report.dominant)
        self.assertIn("field[1]", report.dominant)
        self.assertGreater(report.receipt["severity"], 0.9)  # both fields adverse on (almost) every case

    def test_well_specified_model_on_same_data_reports_nothing_dominant(self):
        background, failing = _split_cases()
        report = diagnose(_well_specified_net(), failing, background=background)

        self.assertEqual(report.dominant, "")
        self.assertEqual(report.suggested_fix, "")
        self.assertEqual(report.receipt["severity"], 0.0)

    def test_evidence_is_ranked_and_exact_field_names_present(self):
        background, failing = _split_cases()
        report = diagnose(_buggy_net(), failing, background=background)
        names = [n for n, _ in report.evidence]
        self.assertEqual(names, sorted(names, key=lambda n: -dict(report.evidence)[n]))
        self.assertEqual({n.split("|")[0] for n in names}, {"field[0]", "field[1]"})


class DeterminismTest(unittest.TestCase):
    def test_same_seeded_cases_give_bit_identical_reports(self):
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)

        def make_cases(rng):
            a = rng.uniform(-3.0, 3.0, size=30)
            return [(float(x), float(x) + 0.05) for x in a]

        cases1, cases2 = make_cases(rng1), make_cases(rng2)
        background, _ = _split_cases()

        r1 = diagnose(_buggy_net(), cases1, background=background)
        r2 = diagnose(_buggy_net(), cases2, background=background)

        self.assertEqual(r1.dominant, r2.dominant)
        self.assertEqual(r1.suggested_fix, r2.suggested_fix)
        self.assertEqual(r1.evidence, r2.evidence)
        self.assertEqual(r1.receipt, r2.receipt)


class EmptyInputTest(unittest.TestCase):
    def test_no_cases_returns_empty_report_not_a_crash(self):
        report = diagnose(_buggy_net(), [])
        self.assertEqual(report.dominant, "")
        self.assertEqual(report.evidence, [])


class DegenerateSampleSizeTest(unittest.TestCase):
    """A background too small to estimate a scale (MAD collapses to ~0, exploding z-scores) or a
    case count too small for co-occurrence to mean anything must not manufacture a confident,
    maximum-severity finding out of pure numerical noise."""

    def test_degenerate_background_reports_not_enough_data_not_a_confident_finding(self):
        report = diagnose(_buggy_net(), [(5.0, 5.0)] * 3, background=[(0.0, 0.0)])
        self.assertEqual(report.dominant, "")
        self.assertEqual(report.evidence, [])

    def test_single_case_does_not_manufacture_add_edge_from_zero_co_occurrence_evidence(self):
        background = [(0.1, 0.05), (-0.2, -0.15), (0.05, 0.1), (-0.1, -0.05)]
        report = diagnose(_buggy_net(), [(5.0, 5.0)], background=background)
        self.assertEqual(report.dominant, "")
        self.assertEqual(report.suggested_fix, "")
        self.assertEqual(report.receipt["severity"], 0.0)


def _single_field_anomaly_cases():
    """Cases adverse in field[0] only -- ``either`` is nonzero but ``both`` is always zero, so there is
    no co-anomalous evidence for a missing edge between the two fields."""
    background = [(0.1, 0.05), (-0.2, -0.15), (0.05, 0.1), (-0.1, -0.05), (0.0, 0.0), (0.2, -0.1)]
    failing = [(6.0, 0.02), (5.5, -0.03), (6.2, 0.01)]
    return background, failing


class ThresholdValidationTest(unittest.TestCase):
    """``min_z`` and ``co_occurrence_threshold`` are cutoffs on a clipped z-score and on a rate; values
    outside those ranges make both comparisons tautologies and prescribe a dependency that no case
    supports, so they are rejected at the boundary instead of quietly changing the verdict."""

    def test_negative_co_occurrence_threshold_is_rejected(self):
        background, failing = _single_field_anomaly_cases()
        with self.assertRaises(ValueError):
            diagnose(_buggy_net(), failing, background=background, co_occurrence_threshold=-1.0)

    def test_co_occurrence_threshold_above_one_is_rejected(self):
        background, failing = _single_field_anomaly_cases()
        with self.assertRaises(ValueError):
            diagnose(_buggy_net(), failing, background=background, co_occurrence_threshold=1.5)

    def test_negative_min_z_is_rejected(self):
        background, failing = _single_field_anomaly_cases()
        with self.assertRaises(ValueError):
            diagnose(_buggy_net(), failing, background=background, min_z=-1.0)

    def test_non_finite_cutoffs_are_rejected(self):
        background, failing = _single_field_anomaly_cases()
        with self.assertRaises(ValueError):
            diagnose(_buggy_net(), failing, background=background, min_z=float("nan"))
        with self.assertRaises(ValueError):
            diagnose(_buggy_net(), failing, background=background, co_occurrence_threshold=float("nan"))

    def test_in_range_cutoffs_are_still_accepted(self):
        background, failing = _split_cases()
        report = diagnose(_buggy_net(), failing, background=background, min_z=0.0, co_occurrence_threshold=1.0)
        self.assertEqual(report.suggested_fix, "add_edge")


class CoOccurrenceEvidenceTest(unittest.TestCase):
    def test_zero_threshold_still_requires_a_co_anomalous_case(self):
        """A rate cutoff of 0.0 means "any co-occurrence counts", not "no co-occurrence needed": with
        every case adverse in one field only, there is no pair to explain and no edge to add."""
        background, failing = _single_field_anomaly_cases()
        report = diagnose(_buggy_net(), failing, background=background, co_occurrence_threshold=0.0)
        self.assertEqual(report.dominant, "")
        self.assertEqual(report.suggested_fix, "")
        self.assertEqual(report.receipt["severity"], 0.0)


class ArrayInputTest(unittest.TestCase):
    def test_numpy_case_table_is_accepted_like_a_list_of_rows(self):
        background, failing = _split_cases()
        report = diagnose(_buggy_net(), np.asarray(failing), background=np.asarray(background))
        self.assertEqual(report.suggested_fix, "add_edge")

    def test_empty_numpy_case_table_reports_no_cases_instead_of_raising(self):
        report = diagnose(_buggy_net(), np.zeros((0, 2)), background=_split_cases()[0])
        self.assertEqual(report.dominant, "")
        self.assertEqual(report.evidence, [])
        self.assertEqual(report.receipt["n_cases"], 0)


if __name__ == "__main__":
    unittest.main()
