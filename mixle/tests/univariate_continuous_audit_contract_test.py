"""Focused contracts for continuous-family prerelease audit repairs."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats import (
    BetaDistribution,
    BetaEstimator,
    GammaDistribution,
    GeneralizedExtremeValueDistribution,
    GeneralizedExtremeValueEstimator,
    GeneralizedGaussianDistribution,
    GeneralizedParetoDistribution,
    GumbelDistribution,
    LaplaceDistribution,
    LogisticDistribution,
    NakagamiDistribution,
    ParetoDistribution,
    RicianDistribution,
    SkewNormalDistribution,
    StudentTDistribution,
    StudentTEstimator,
    TweedieDistribution,
    TweedieSeriesResourceError,
)


class ContinuousParameterContractTest(unittest.TestCase):
    def test_location_family_constructors_reject_nonfinite_locations(self):
        constructors = (
            lambda value: GeneralizedGaussianDistribution(value, 1.0, 2.0),
            lambda value: GumbelDistribution(value, 1.0),
            lambda value: LaplaceDistribution(value, 1.0),
            lambda value: LogisticDistribution(value, 1.0),
            lambda value: StudentTDistribution(5.0, value, 1.0),
        )
        # subTest labels are serializable stand-ins, not the objects themselves: xdist ships every
        # subtest report through execnet, which cannot serialize a lambda or a distribution, so a
        # raw label crashed the report channel under the suite's own default -n invocation.
        for index, constructor in enumerate(constructors):
            for invalid in (np.nan, np.inf, -np.inf):
                with self.subTest(constructor=index, invalid=invalid), self.assertRaises(ValueError):
                    constructor(invalid)

    def test_student_moment_estimator_requires_existing_mean_and_variance(self):
        for df in (0.5, 1.0, 2.0, np.inf):
            with self.subTest(df=df), self.assertRaises(ValueError):
                StudentTEstimator(df=df)
        StudentTEstimator(df=2.0001)

    def test_zero_beta_pseudo_count_is_unsmoothed(self):
        estimator = BetaEstimator(pseudo_count=0.0)
        self.assertIsNone(estimator.pseudo_count)
        fitted = estimator.estimate(None, (0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual((fitted.a, fitted.b), (1.0, 1.0))

    def test_beta_estimator_rejects_invalid_controls_and_statistics(self):
        for pseudo_count in (-1.0, np.nan, np.inf, True):
            with self.subTest(pseudo_count=pseudo_count), self.assertRaises(ValueError):
                BetaEstimator(pseudo_count=pseudo_count)
        for prior in ((0.0,), (0.0, np.nan)):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                BetaEstimator(pseudo_count=1.0, suff_stat=prior)
        for statistics in (
            (-1.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, np.nan, 0.0, 0.0, 0.0),
        ):
            with self.subTest(statistics=statistics), self.assertRaises(ValueError):
                BetaEstimator().estimate(None, statistics)

    def test_gev_estimator_rejects_silently_ignored_or_incompatible_priors(self):
        for pseudo_count in (-1.0, np.nan, np.inf, True):
            with self.subTest(pseudo_count=pseudo_count), self.assertRaises(ValueError):
                GeneralizedExtremeValueEstimator(pseudo_count=pseudo_count)
        with self.assertRaises(ValueError):
            GeneralizedExtremeValueEstimator(pseudo_count=1.0)
        for prior in ((0.0, 1.0), (0.0, 1.0, np.inf)):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                GeneralizedExtremeValueEstimator(pseudo_count=1.0, suff_stat=prior)
        with self.assertRaises(ValueError):
            GeneralizedExtremeValueDistribution(0.0, 1.0, 1.0 / 3.0).estimator(pseudo_count=1.0)

        estimator = GeneralizedExtremeValueEstimator(pseudo_count=0.0)
        self.assertIsNone(estimator.pseudo_count)
        fitted = estimator.estimate(None, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual((fitted.loc, fitted.scale, fitted.shape), (0.0, 1.0, 0.0))


class ContinuousObservationContractTest(unittest.TestCase):
    def test_real_line_encoders_reject_nonfinite_observations(self):
        distributions = (
            GeneralizedExtremeValueDistribution(0.0, 1.0, 0.0),
            GeneralizedGaussianDistribution(0.0, 1.0, 2.0),
            SkewNormalDistribution(0.0, 1.0, 0.5),
            StudentTDistribution(5.0, 0.0, 1.0),
        )
        for distribution in distributions:
            encoder = distribution.dist_to_encoder()
            for invalid in ([0.0, np.nan], [np.inf], [-np.inf]):
                with self.subTest(distribution=repr(distribution), invalid=invalid), self.assertRaises(ValueError):
                    encoder.seq_encode(invalid)

    def test_nonnegative_amplitude_models_reject_negative_or_nonfinite_evidence(self):
        for distribution in (NakagamiDistribution(1.0, 2.0), RicianDistribution(1.0, 2.0)):
            encoder = distribution.dist_to_encoder()
            accumulator = distribution.estimator().accumulator_factory().make()
            for invalid in (-1.0, np.nan, np.inf):
                with self.subTest(distribution=repr(distribution), invalid=invalid), self.assertRaises(ValueError):
                    encoder.seq_encode([invalid])
                with self.subTest(distribution=repr(distribution), invalid=invalid), self.assertRaises(ValueError):
                    accumulator.update(invalid, 1.0, distribution)
                with self.subTest(distribution=repr(distribution), invalid=invalid), self.assertRaises(ValueError):
                    accumulator.seq_update(np.asarray([invalid]), np.ones(1), distribution)

    def test_generalized_pareto_fitting_and_encoding_respect_threshold_and_endpoint(self):
        distribution = GeneralizedParetoDistribution(2.0, -0.5, loc=1.0)
        encoder = distribution.dist_to_encoder()
        np.testing.assert_allclose(encoder.seq_encode([1.0, 3.0, 5.0]), [1.0, 3.0, 5.0])
        for invalid in ([0.999], [5.001], [np.nan], [np.inf]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                encoder.seq_encode(invalid)

        estimator = distribution.estimator()
        accumulator = estimator.accumulator_factory().make()
        with self.assertRaises(ValueError):
            accumulator.update(0.5, 1.0, None)
        with self.assertRaises(ValueError):
            accumulator.seq_update(np.asarray([0.5]), np.ones(1), None)
        with self.assertRaises(ValueError):
            accumulator.update(5.5, 1.0, distribution)

    def test_generalized_pareto_encoder_identity_includes_support(self):
        first = GeneralizedParetoDistribution(1.0, 0.0, loc=0.0).dist_to_encoder()
        shifted = GeneralizedParetoDistribution(1.0, 0.0, loc=2.0).dist_to_encoder()
        bounded = GeneralizedParetoDistribution(1.0, -0.5, loc=0.0).dist_to_encoder()
        self.assertNotEqual(first, shifted)
        self.assertNotEqual(first, bounded)


class ContinuousBoundaryDensityContractTest(unittest.TestCase):
    def _assert_scalar_numpy_backend_parity(self, distribution, values):
        encoded = distribution.dist_to_encoder().seq_encode(values)
        scalar = np.asarray([distribution.log_density(value) for value in values])
        batch = np.asarray(distribution.seq_log_density(encoded))
        backend = np.asarray(distribution.backend_seq_log_density(encoded, NUMPY_ENGINE))
        np.testing.assert_allclose(batch, scalar, rtol=1.0e-14, atol=1.0e-14)
        np.testing.assert_allclose(backend, scalar, rtol=1.0e-14, atol=1.0e-14)

    def test_beta_endpoint_limits(self):
        self.assertEqual(BetaDistribution(0.5, 2.0).log_density(0.0), np.inf)
        self.assertAlmostEqual(BetaDistribution(1.0, 2.0).density(0.0), 2.0)
        self.assertEqual(BetaDistribution(2.0, 2.0).log_density(0.0), -np.inf)
        self.assertEqual(BetaDistribution(2.0, 0.5).log_density(1.0), np.inf)
        self.assertAlmostEqual(BetaDistribution(2.0, 1.0).density(1.0), 2.0)
        self.assertEqual(BetaDistribution(2.0, 2.0).log_density(1.0), -np.inf)
        for distribution in (
            BetaDistribution(0.5, 2.0),
            BetaDistribution(1.0, 1.0),
            BetaDistribution(2.0, 0.5),
        ):
            self._assert_scalar_numpy_backend_parity(distribution, [0.0, 0.25, 1.0])

    def test_gamma_zero_limits(self):
        self.assertEqual(GammaDistribution(0.5, 2.0).log_density(0.0), np.inf)
        self.assertAlmostEqual(GammaDistribution(1.0, 2.0).density(0.0), 0.5)
        self.assertEqual(GammaDistribution(2.0, 2.0).log_density(0.0), -np.inf)
        for distribution in (
            GammaDistribution(0.5, 2.0),
            GammaDistribution(1.0, 2.0),
            GammaDistribution(2.0, 2.0),
        ):
            self._assert_scalar_numpy_backend_parity(distribution, [0.0, 0.25, 2.0])

    def test_nakagami_zero_limit(self):
        half = NakagamiDistribution(0.5, 2.0)
        self.assertTrue(np.isfinite(half.log_density(0.0)))
        self.assertAlmostEqual(half.log_density(0.0), half._log_const)
        self.assertEqual(NakagamiDistribution(1.0, 2.0).log_density(0.0), -np.inf)
        for distribution in (half, NakagamiDistribution(1.0, 2.0)):
            self._assert_scalar_numpy_backend_parity(distribution, [0.0, 0.25, 2.0])


class ContinuousResourceAndWeightContractTest(unittest.TestCase):
    def test_tweedie_series_fails_before_unbounded_allocation(self):
        distribution = TweedieDistribution(
            1.0,
            1.0e-12,
            1.5,
            max_series_terms=100,
            max_series_work=100,
        )
        with self.assertRaises(TweedieSeriesResourceError):
            distribution.log_density(1.0)
        with self.assertRaises(TweedieSeriesResourceError):
            distribution.backend_seq_log_density(np.asarray([1.0]), NUMPY_ENGINE)

    def test_tweedie_streaming_numpy_and_backend_scores_match(self):
        distribution = TweedieDistribution(2.5, 0.8, 1.4)
        values = np.asarray([0.0, 0.3, 1.0, 2.5, 7.0])
        encoded = distribution.dist_to_encoder().seq_encode(values)
        scalar = np.asarray([distribution.log_density(float(value)) for value in values])
        np.testing.assert_allclose(distribution.seq_log_density(encoded), scalar, atol=1.0e-10)
        np.testing.assert_allclose(
            np.asarray(distribution.backend_seq_log_density(encoded, NUMPY_ENGINE)),
            scalar,
            atol=1.0e-10,
        )

    def test_pareto_engine_and_numpy_apply_the_same_weight_policy(self):
        distribution = ParetoDistribution(1.0, 2.0)
        encoded = distribution.dist_to_encoder().seq_encode([1.5, 3.0])
        for method in ("seq_update", "seq_update_engine"):
            accumulator = distribution.estimator().accumulator_factory().make()
            with self.subTest(method=method), self.assertRaises(ValueError):
                if method == "seq_update":
                    accumulator.seq_update(encoded, np.asarray([1.0, -0.5]), distribution)
                else:
                    accumulator.seq_update_engine(
                        encoded,
                        np.asarray([1.0, -0.5]),
                        distribution,
                        NUMPY_ENGINE,
                    )

        numpy_accumulator = distribution.estimator().accumulator_factory().make()
        engine_accumulator = distribution.estimator().accumulator_factory().make()
        weights = np.asarray([1.0, 2.0])
        numpy_accumulator.seq_update(encoded, weights, distribution)
        engine_accumulator.seq_update_engine(encoded, weights, distribution, NUMPY_ENGINE)
        np.testing.assert_allclose(engine_accumulator.value(), numpy_accumulator.value())


if __name__ == "__main__":
    unittest.main()
