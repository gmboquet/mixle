"""Regression tests for the exact copula support, shape, parameter, and fit contracts."""

import unittest

import numpy as np

from mixle.stats.multivariate._copula_common import (
    BufferedUScoreAccumulator,
    UScoreEncoder,
    weighted_kendall_tau,
)
from mixle.stats.multivariate.clayton_copula import ClaytonCopulaDistribution
from mixle.stats.multivariate.frank_copula import FrankCopulaDistribution
from mixle.stats.multivariate.gaussian_copula import (
    GaussianCopulaDistribution,
    GaussianCopulaEstimator,
)
from mixle.stats.multivariate.gumbel_copula import GumbelCopulaDistribution
from mixle.stats.multivariate.rvine_copula import (
    RVineCopulaDistribution,
    _Edge,
    _max_spanning_tree,
)
from mixle.stats.multivariate.student_t_copula import (
    StudentTCopulaDistribution,
    StudentTCopulaEstimator,
)
from mixle.stats.multivariate.vine_copula import (
    ClaytonPairCopula,
    CVineCopulaDistribution,
    DVineCopulaDistribution,
    FrankPairCopula,
    GaussianPairCopula,
    GumbelPairCopula,
    IndependencePairCopula,
    StudentTPairCopula,
)


class CopulaShapeAndParameterContractTest(unittest.TestCase):
    def test_dimensioned_encoders_are_not_interchangeable(self):
        enc2 = UScoreEncoder(2)
        enc3 = UScoreEncoder(3)
        self.assertNotEqual(enc2, enc3)
        self.assertIn("dim=2", str(enc2))
        with self.assertRaises(ValueError):
            enc2.seq_encode([[0.2, 0.3, 0.4]])
        with self.assertRaises(ValueError):
            enc2.seq_encode([0.2, 0.3])
        with self.assertRaises(ValueError):
            enc2.seq_encode([[0.0, 0.3]])

    def test_archimedean_scorers_require_exact_width(self):
        for distribution in (
            ClaytonCopulaDistribution(3, 1.0),
            FrankCopulaDistribution(2, 1.0),
            GumbelCopulaDistribution(2, 2.0),
            CVineCopulaDistribution(2, {}),
            DVineCopulaDistribution(2, {}),
            RVineCopulaDistribution.independence(2),
        ):
            with self.subTest(distribution=type(distribution).__name__):
                wrong = np.full((1, distribution.dim + 1), 0.5)
                with self.assertRaises(ValueError):
                    distribution.seq_log_density(wrong)

    def test_dimensions_are_not_truncated(self):
        for constructor, theta in (
            (ClaytonCopulaDistribution, 1.0),
            (FrankCopulaDistribution, 1.0),
            (GumbelCopulaDistribution, 2.0),
        ):
            with self.subTest(constructor=constructor.__name__):
                with self.assertRaises(TypeError):
                    constructor(2.9, theta)

    def test_invalid_or_nonfinite_parameters_are_rejected(self):
        for call in (
            lambda: ClaytonCopulaDistribution(2, -1.0),
            lambda: ClaytonCopulaDistribution(2, np.nan),
            lambda: FrankCopulaDistribution(2, np.inf),
            lambda: GumbelCopulaDistribution(2, 0.5),
            lambda: GumbelCopulaDistribution(2, np.nan),
            lambda: StudentTCopulaDistribution(np.eye(2), np.inf),
        ):
            with self.subTest(call=call), self.assertRaises((TypeError, ValueError)):
                call()

    def test_extreme_finite_archimedean_scores_do_not_become_nan(self):
        point = np.array([[0.2, 0.8]])
        for distribution in (
            ClaytonCopulaDistribution(2, 1.0e-14),
            ClaytonCopulaDistribution(2, 1.0e3),
            FrankCopulaDistribution(2, 1.0e3),
            FrankCopulaDistribution(2, -1.0e3),
            GumbelCopulaDistribution(2, 1.0e3),
        ):
            with self.subTest(distribution=str(distribution)):
                self.assertFalse(np.isnan(distribution.seq_log_density(point)[0]))

    def test_correlation_inputs_and_properties_are_owned(self):
        original = np.array([[1.0, 0.6], [0.6, 1.0]])
        gaussian = GaussianCopulaDistribution(original)
        student = StudentTCopulaDistribution(original, 4.0)
        point = np.array([0.2, 0.7])
        expected_gaussian = gaussian.log_density(point)
        expected_student = student.log_density(point)
        original[0, 1] = original[1, 0] = -0.6
        gaussian.corr[0, 1] = -0.9
        student.corr[0, 1] = -0.9
        self.assertEqual(gaussian.log_density(point), expected_gaussian)
        self.assertEqual(student.log_density(point), expected_student)
        np.testing.assert_allclose(gaussian.corr, [[1.0, 0.6], [0.6, 1.0]])
        np.testing.assert_allclose(student.corr, [[1.0, 0.6], [0.6, 1.0]])

    def test_unsupported_pseudo_counts_fail_explicitly(self):
        for distribution in (
            GaussianCopulaDistribution(np.eye(2)),
            StudentTCopulaDistribution(np.eye(2), 4.0),
            ClaytonCopulaDistribution(2, 1.0),
            FrankCopulaDistribution(2, 1.0),
            GumbelCopulaDistribution(2, 2.0),
        ):
            with self.subTest(distribution=type(distribution).__name__):
                with self.assertRaises(ValueError):
                    distribution.estimator(pseudo_count=1.0)


class CopulaFitContractTest(unittest.TestCase):
    def test_buffer_validates_rows_weights_and_serialized_alignment(self):
        accumulator = BufferedUScoreAccumulator(2)
        for row, weight in (
            ([0.2], 1.0),
            ([0.2, np.nan], 1.0),
            ([0.2, 0.3], -1.0),
            ([0.2, 0.3], np.inf),
        ):
            with self.subTest(row=row, weight=weight), self.assertRaises(ValueError):
                accumulator.update(np.asarray(row), weight, None)
        with self.assertRaises(ValueError):
            accumulator.seq_update(np.array([[0.2, 0.3], [0.4, 0.5]]), np.ones(1), None)
        with self.assertRaises(ValueError):
            accumulator.from_value((np.array([[0.2, 0.3], [0.4, 0.5]]), np.ones(1)))

    def test_kendall_implementation_matches_tau_a_reference_with_ties(self):
        a = np.array([0.2, 0.2, 0.4, 0.8, 0.6])
        b = np.array([0.1, 0.7, 0.7, 0.3, 0.9])
        w = np.array([1.0, 0.5, 2.0, 1.5, 0.25])
        numerator = 0.0
        denominator = 0.0
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                numerator += w[i] * w[j] * np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
                denominator += w[i] * w[j]
        self.assertAlmostEqual(weighted_kendall_tau(a, b, w), numerator / denominator)

    def test_elliptical_controls_and_statistics_fail_closed(self):
        for constructor in (GaussianCopulaEstimator, StudentTCopulaEstimator):
            with self.subTest(constructor=constructor.__name__):
                for min_eig in (0.0, -1.0, np.inf, 1.0):
                    with self.assertRaises(ValueError):
                        constructor(2, min_eig=min_eig)
        gaussian = GaussianCopulaEstimator(2)
        invalid_gaussian_stats = (
            (np.zeros(1), np.eye(2), 1.0),
            (np.zeros(2), np.ones((2, 3)), 1.0),
            (np.zeros(2), np.eye(2), 0.0),
            (np.zeros(2), np.eye(2), np.inf),
            (np.zeros(2), np.array([[1.0, np.nan], [np.nan, 1.0]]), 1.0),
        )
        for statistic in invalid_gaussian_stats:
            with self.subTest(statistic=statistic), self.assertRaises(ValueError):
                gaussian.estimate(None, statistic)
        student = StudentTCopulaEstimator(2)
        with self.assertRaises(ValueError):
            student.estimate(None, (np.array([[0.2, 0.3]]), np.ones(1)))
        with self.assertRaises(ValueError):
            student.estimate(None, (np.array([[0.2, 0.3], [0.4, 0.5]]), np.zeros(2)))


class VineSelectionContractTest(unittest.TestCase):
    def test_exported_pair_copulas_enforce_open_unit_support(self):
        pairs = (
            IndependencePairCopula(),
            GaussianPairCopula(0.2),
            ClaytonPairCopula(1.0),
            FrankPairCopula(2.0),
            GumbelPairCopula(2.0),
            StudentTPairCopula(0.2, 4.0),
        )
        for pair in pairs:
            for method, first, second in (
                (pair.logpdf, -5.0e-10, 0.4),
                (pair.logpdf, 0.4, 1.0),
                (pair.h, np.nan, 0.4),
                (pair.h_inv, 0.4, 1.2),
            ):
                with self.subTest(family=pair.family, method=method.__name__):
                    with self.assertRaises(ValueError):
                        method(first, second)

    def test_pair_copula_parameters_fail_instead_of_clamping(self):
        invalid = (
            lambda: GaussianPairCopula(1.0),
            lambda: GaussianPairCopula(np.nan),
            lambda: ClaytonPairCopula(-1.0),
            lambda: FrankPairCopula(np.inf),
            lambda: GumbelPairCopula(0.5),
            lambda: StudentTPairCopula(-1.0, 4.0),
            lambda: StudentTPairCopula(0.2, np.nan),
        )
        for constructor in invalid:
            with self.subTest(constructor=constructor), self.assertRaises((TypeError, ValueError)):
                constructor()

    def test_candidate_sets_are_nonempty_known_and_unique(self):
        for candidates in ((), ("unknown",), ("gaussian", "gaussian")):
            for constructor in (
                lambda c=candidates: CVineCopulaDistribution(2, {}, candidates=c),
                lambda c=candidates: DVineCopulaDistribution(2, {}, candidates=c),
                lambda c=candidates: RVineCopulaDistribution.independence(2, candidates=c),
            ):
                with self.subTest(candidates=candidates, constructor=constructor):
                    with self.assertRaises(ValueError):
                        constructor()

    def test_pair_family_selection_exposes_aic_evidence(self):
        rng = np.random.RandomState(18)
        u = rng.uniform(0.05, 0.95, size=(200, 2))
        fitted = DVineCopulaDistribution(2, {}).estimator().estimate(None, (u, np.ones(len(u))))
        pair = fitted.pairs[(1, 1)]
        receipt = pair.selection_receipt
        self.assertEqual(receipt.schema_version, 1)
        self.assertEqual(receipt.criterion, "aic")
        self.assertEqual(receipt.selected_family, pair.family)
        self.assertEqual(receipt.edge_context, "D-vine tree 1 position 1")
        self.assertEqual({item.family for item in receipt.evidence}, set(fitted.candidates))
        selected = next(item for item in receipt.evidence if item.family == pair.family)
        self.assertIsNone(selected.fit_error)
        self.assertTrue(np.isfinite(selected.weighted_log_likelihood))
        self.assertTrue(np.isfinite(selected.aic))

    def test_disconnected_spanning_graph_is_rejected(self):
        with self.assertRaises(ValueError):
            _max_spanning_tree(3, {(0, 1): 1.0}, {(0, 1)})

    def test_rvine_requires_a_complete_proximal_tree_sequence(self):
        with self.assertRaises(ValueError):
            RVineCopulaDistribution(3, [])
        valid = RVineCopulaDistribution.independence(3)
        with self.assertRaises(ValueError):
            RVineCopulaDistribution(3, valid.trees[:1])
        malformed_second_tree = [
            valid.trees[0],
            [
                _Edge(
                    1,
                    2,
                    frozenset({0}),
                    IndependencePairCopula(),
                    {1: (1, 1), 2: (0, 2)},
                )
            ],
        ]
        with self.assertRaises(ValueError):
            RVineCopulaDistribution(3, malformed_second_tree)

    def test_explicit_independence_rvine_is_complete_and_samples_its_law(self):
        model = RVineCopulaDistribution.independence(4)
        self.assertEqual([len(tree) for tree in model.trees], [3, 2, 1])
        draws = model.sampler(91).sample(3000)
        self.assertEqual(draws.shape, (3000, 4))
        np.testing.assert_allclose(draws.mean(axis=0), 0.5, atol=0.025)
        np.testing.assert_allclose(np.corrcoef(draws.T), np.eye(4), atol=0.06)


if __name__ == "__main__":
    unittest.main()
