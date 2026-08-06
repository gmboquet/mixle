import unittest

import numpy as np
from numpy.random import RandomState

from mixle.stats.bayes.dirichlet_process_mixture import (
    DirichletProcessMixtureAccumulator,
    DirichletProcessMixtureDistribution,
    DirichletProcessMixtureEstimator,
    _prior_cross_entropy,
)
from mixle.stats.bayes.normal_gamma import NormalGammaDistribution
from mixle.stats.compute.mixture_evidence import InvalidMixtureEvidenceError
from mixle.stats.compute.posterior import ImpossiblePosteriorError
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.stats.univariate.continuous.gaussian import (
    GaussianAccumulator,
    GaussianDistribution,
    GaussianEstimator,
)
from mixle.stats.univariate.discrete.poisson import PoissonDistribution
from mixle.utils.special import betaln, digamma


class _FixedScoreGaussian(GaussianDistribution):
    def __init__(self, score):
        super().__init__(0.0, 1.0)
        self.score = float(score)

    def log_density(self, x):
        return self.score

    def expected_log_density(self, x):
        return self.score

    def seq_log_density(self, x):
        return np.full(len(x), self.score)

    def seq_expected_log_density(self, x):
        return np.full(len(x), self.score)


def _fixed_model(scores, *, prior=None):
    components = [_FixedScoreGaussian(score) for score in scores]
    count = len(components)
    return DirichletProcessMixtureDistribution(
        components,
        np.ones(count) / count,
        1.5,
        np.ones((count, 2)),
        [None] * count,
        prior=prior,
    )


class DirichletProcessStateContractTestCase(unittest.TestCase):
    def test_constructor_rejects_malformed_mixture_state(self):
        gaussian = GaussianDistribution(0.0, 1.0)
        cases = (
            ([], [], 1.0, np.empty((0, 2)), []),
            ([gaussian], [0.5, 0.5], 1.0, np.ones((1, 2)), [None]),
            ([gaussian], [np.nan], 1.0, np.ones((1, 2)), [None]),
            ([gaussian], [0.5], 1.0, np.ones((1, 2)), [None]),
            ([gaussian], [1.0], 0.0, np.ones((1, 2)), [None]),
            ([gaussian], [1.0], np.inf, np.ones((1, 2)), [None]),
            ([gaussian], [1.0], 1.0, np.ones((2, 2)), [None]),
            ([gaussian], [1.0], 1.0, np.asarray([[1.0, 0.0]]), [None]),
            ([gaussian], [1.0], 1.0, np.ones((1, 2)), []),
            (
                [gaussian],
                [1.0],
                1.0,
                np.ones((1, 2)),
                [NormalGammaDistribution(0.0, 1.0, 1.0, 1.0)],
            ),
        )
        for args in cases:
            with self.subTest(args=repr(args)), self.assertRaises((TypeError, ValueError)):
                DirichletProcessMixtureDistribution(*args)
        with self.assertRaises(ValueError):
            DirichletProcessMixtureDistribution(
                [GaussianDistribution(0.0, 1.0), PoissonDistribution(1.0)],
                [0.5, 0.5],
                1.0,
                np.ones((2, 2)),
                [None, None],
            )
        with self.assertRaises(TypeError):
            _fixed_model([0.0], prior=object())

    def test_parameter_setter_is_atomic_and_structure_bound(self):
        model = _fixed_model([0.0, -1.0])
        original_score = model.log_density(0.0)
        with self.assertRaises(ValueError):
            model.set_parameters((2.0, [0.25, 0.25], [_FixedScoreGaussian(2.0), _FixedScoreGaussian(3.0)]))
        self.assertEqual(model.a, 1.5)
        self.assertAlmostEqual(model.log_density(0.0), original_score)

        model.set_parameters((2.0, [0.25, 0.75], [_FixedScoreGaussian(2.0), _FixedScoreGaussian(3.0)]))
        self.assertEqual(model.a, 2.0)
        np.testing.assert_allclose(model.w, [0.25, 0.75])
        snapshot = model.get_parameters()
        snapshot[1][0] = 0.9
        snapshot[2][0].score = -99.0
        np.testing.assert_allclose(model.w, [0.25, 0.75])
        self.assertEqual(model.components[0].score, 2.0)

        model.components.append(_FixedScoreGaussian(0.0))
        with self.assertRaisesRegex(RuntimeError, "structure changed"):
            model.log_density(0.0)

    def test_string_round_trip_preserves_complete_state(self):
        model = DirichletProcessMixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 2.0)],
            [0.4, 0.6],
            1.3,
            [[2.0, 3.0], [1.0, 1.0]],
            [None, None],
            name="dpm",
            prior=None,
        )
        restored = eval(str(model))
        np.testing.assert_allclose(restored.w, model.w)
        np.testing.assert_allclose(restored.g, model.g)
        self.assertEqual(restored.a, model.a)
        self.assertEqual(restored.name, model.name)
        self.assertIsNone(restored.prior)

    def test_estimator_rejects_empty_or_unsupported_configuration(self):
        with self.assertRaises(ValueError):
            DirichletProcessMixtureEstimator([])
        with self.assertRaises(TypeError):
            DirichletProcessMixtureEstimator([GaussianEstimator()], prior=object())
        with self.assertRaises(ValueError):
            DirichletProcessMixtureEstimator([GaussianEstimator()], pseudo_count=1.0)

    def test_prior_tree_cross_entropy_requires_exact_structure(self):
        prior = NormalGammaDistribution(0.0, 1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            _prior_cross_entropy((prior, prior), (prior,))
        with self.assertRaises(TypeError):
            _prior_cross_entropy((prior,), prior)
        with self.assertRaises(ValueError):
            _prior_cross_entropy(prior, None)


class DirichletProcessEvidenceContractTestCase(unittest.TestCase):
    def test_impossible_evidence_is_consistent_across_public_paths(self):
        model = _fixed_model([-np.inf, -np.inf])
        encoded = model.dist_to_encoder().seq_encode([0.0, 1.0])
        self.assertTrue(np.isneginf(model.log_density(0.0)))
        self.assertTrue(np.isneginf(model.expected_log_density(0.0)))
        np.testing.assert_array_equal(model.seq_log_density(encoded), [-np.inf, -np.inf])
        np.testing.assert_array_equal(
            model.seq_expected_log_density(encoded),
            [-np.inf, -np.inf],
        )
        np.testing.assert_array_equal(model.seq_local_elbo(encoded), [-np.inf, -np.inf])
        np.testing.assert_array_equal(model.posterior(0.0), [0.0, 0.0])
        np.testing.assert_array_equal(model.seq_posterior(encoded), np.zeros((2, 2)))
        with self.assertRaises(ImpossiblePosteriorError):
            model.latent_posterior([0.0])

    def test_positive_infinity_has_one_winner_and_ambiguous_infinity_is_typed(self):
        unique = _fixed_model([np.inf, -np.inf])
        self.assertTrue(np.isposinf(unique.log_density(0.0)))
        np.testing.assert_array_equal(unique.posterior(0.0), [1.0, 0.0])
        encoded = unique.dist_to_encoder().seq_encode([0.0])
        np.testing.assert_array_equal(unique.seq_posterior(encoded), [[1.0, 0.0]])

        ambiguous = _fixed_model([np.inf, np.inf])
        with self.assertRaises(InvalidMixtureEvidenceError):
            ambiguous.posterior(0.0)
        with self.assertRaises(InvalidMixtureEvidenceError):
            ambiguous.seq_local_elbo(encoded)

    def test_zero_weight_component_cannot_contribute_positive_infinity(self):
        model = DirichletProcessMixtureDistribution(
            [_FixedScoreGaussian(np.inf), _FixedScoreGaussian(0.0)],
            [0.0, 1.0],
            1.0,
            np.ones((2, 2)),
            [None, None],
            prior=None,
        )
        self.assertEqual(model.log_density(0.0), 0.0)
        np.testing.assert_array_equal(model.posterior(0.0), [0.0, 1.0])

    def test_impossible_rows_do_not_create_accumulator_counts(self):
        model = _fixed_model([-np.inf, -np.inf])
        accumulator = DirichletProcessMixtureAccumulator([GaussianAccumulator(), GaussianAccumulator()])
        accumulator.update(0.0, 2.0, model)
        encoded = model.dist_to_encoder().seq_encode([0.0, 1.0])
        accumulator.seq_update(encoded, np.asarray([3.0, 4.0]), model)
        component_counts, beta_counts, *_ = accumulator.value()
        np.testing.assert_array_equal(component_counts, np.zeros(2))
        np.testing.assert_array_equal(beta_counts, np.zeros((2, 2)))


class DirichletProcessInitializationAndElboContractTestCase(unittest.TestCase):
    def test_zero_weight_initialization_creates_no_counts(self):
        accumulator = DirichletProcessMixtureAccumulator([GaussianAccumulator(), GaussianAccumulator()])
        encoded = GaussianDistribution(0.0, 1.0).dist_to_encoder().seq_encode([0.0])
        accumulator.seq_initialize(encoded, np.asarray([0.0]), RandomState(2))
        component_counts, beta_counts, *_ = accumulator.value()
        np.testing.assert_array_equal(component_counts, np.zeros(2))
        np.testing.assert_array_equal(beta_counts, np.zeros((2, 2)))

    def test_initialization_validates_weight_geometry(self):
        accumulator = DirichletProcessMixtureAccumulator([GaussianAccumulator(), GaussianAccumulator()])
        encoded = GaussianDistribution(0.0, 1.0).dist_to_encoder().seq_encode([0.0, 1.0])
        for weights in ([1.0], [[1.0], [1.0]], [1.0, -1.0], [1.0, np.nan]):
            with self.subTest(weights=repr(weights)), self.assertRaises(ValueError):
                accumulator.seq_initialize(encoded, weights, RandomState(2))
        np.testing.assert_array_equal(accumulator.comp_counts, np.zeros(2))

    def test_gamma_concentration_factor_is_present_in_global_elbo(self):
        q_alpha = GammaDistribution(3.0, 0.8)
        p_alpha = GammaDistribution(2.0, 1.5)
        gammas = np.asarray([[2.5, 3.5], [1.0, 1.0]])
        model = DirichletProcessMixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
            [0.6, 0.4],
            q_alpha.k * q_alpha.theta,
            gammas,
            [None, None],
            prior=q_alpha,
        )
        estimator = DirichletProcessMixtureEstimator(
            [GaussianEstimator(), GaussianEstimator()],
            prior=p_alpha,
        )

        gamma = gammas[0]
        gamma_sum = gamma.sum()
        expected_alpha = q_alpha.k * q_alpha.theta
        expected_log_alpha = digamma(q_alpha.k) + np.log(q_alpha.theta)
        expected_log_remaining = digamma(gamma[1]) - digamma(gamma_sum)
        stick_prior = expected_log_alpha + (expected_alpha - 1.0) * expected_log_remaining
        stick_entropy = (
            betaln(gamma[0], gamma[1])
            - (gamma[0] - 1.0) * digamma(gamma[0])
            - (gamma[1] - 1.0) * digamma(gamma[1])
            + (gamma_sum - 2.0) * digamma(gamma_sum)
        )
        concentration_term = -q_alpha.cross_entropy(p_alpha) + q_alpha.entropy()
        expected = stick_prior + stick_entropy + concentration_term
        self.assertAlmostEqual(estimator.model_log_density(model), expected, places=12)

    def test_serialized_count_geometry_is_rejected_atomically(self):
        accumulator = DirichletProcessMixtureAccumulator([GaussianAccumulator(), GaussianAccumulator()])
        before = accumulator.value()
        malformed = (
            (np.ones(3), np.ones((2, 2)), 1.0, 0.0, (None, None)),
            (np.asarray([1.0, -1.0]), np.zeros((2, 2)), 1.0, 0.0, (None, None)),
            (np.asarray([1.0, 0.0]), np.zeros((2, 2)), 1.0, 0.0, (None, None)),
            (
                np.asarray([0.5, 0.5]),
                np.asarray([[0.5, 0.0], [0.5, 0.0]]),
                1.0,
                0.0,
                (None, None),
            ),
        )
        for value in malformed:
            with self.subTest(value=repr(value)), self.assertRaises((TypeError, ValueError)):
                accumulator.from_value(value)
            after = accumulator.value()
            np.testing.assert_array_equal(after[0], before[0])
            np.testing.assert_array_equal(after[1], before[1])


class ConjugatePosteriorFamilyTest(unittest.TestCase):
    """A conjugate posterior may be a different CLASS from the prior it updates.

    A symmetric Dirichlet prior meeting asymmetric counts has a general Dirichlet posterior. The
    component-structure check compared exact classes, so the DP mixture rejected its own
    freshly-fitted output with "component prior structure differs at component 0" -- and
    `mixle.utils.automatic.get_dpm_mixture` produces exactly that for any categorical-bearing
    record, which is how two shipped notebooks hit it. The ELBO already handles the mixed pair;
    only the fingerprint objected.
    """

    def test_symmetric_and_general_dirichlet_share_a_structure_slot(self):
        from mixle.stats.bayes.dirichlet import DirichletDistribution
        from mixle.stats.bayes.dirichlet_process_mixture import _prior_structure
        from mixle.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution

        self.assertEqual(
            _prior_structure(SymmetricDirichletDistribution(1.0, 3)),
            _prior_structure(DirichletDistribution(np.array([2.0, 3.0, 4.0]))),
        )

    def test_an_unrelated_prior_family_is_still_rejected(self):
        from mixle.stats.bayes.dirichlet import DirichletDistribution
        from mixle.stats.bayes.dirichlet_process_mixture import _prior_structure

        self.assertNotEqual(
            _prior_structure(DirichletDistribution(np.array([1.0, 1.0]))),
            _prior_structure(GammaDistribution(1.0, 1.0)),
        )

    def test_the_automatic_dp_mixture_fits_sequence_records(self):
        import io

        from mixle.utils.automatic import get_dpm_mixture

        rng = RandomState(0)
        data = [[int(v) for v in rng.randint(0, 5, size=rng.randint(1, 6))] for _ in range(120)]
        model = get_dpm_mixture(data, rng=RandomState(1), max_components=4, max_its=5, out=io.StringIO())
        self.assertGreater(model.num_components, 0)


if __name__ == "__main__":
    unittest.main()
