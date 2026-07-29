"""Regression contracts for Gaussian-family audit findings MXR-080-1154--1167."""

import numpy as np
import pytest

from mixle.stats.multivariate.diagonal_gaussian import (
    DiagonalGaussianAccumulator,
    DiagonalGaussianDataEncoder,
    DiagonalGaussianDistribution,
    DiagonalGaussianEstimator,
)
from mixle.stats.multivariate.multivariate_gaussian import (
    MultivariateGaussianAccumulator,
    MultivariateGaussianDataEncoder,
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
)
from mixle.stats.multivariate.multivariate_student_t import (
    MultivariateStudentTAccumulator,
    MultivariateStudentTDataEncoder,
    MultivariateStudentTDistribution,
    MultivariateStudentTEstimator,
)
from mixle.stats.univariate.continuous.gaussian import GaussianEstimator
from mixle.stats.univariate.continuous.log_gaussian import LogGaussianEstimator


@pytest.mark.parametrize(
    ("constructor", "args"),
    [
        (DiagonalGaussianDistribution, ([[0.0, 1.0]], [1.0, 1.0])),
        (DiagonalGaussianDistribution, ([0.0, 1.0], [1.0])),
        (DiagonalGaussianDistribution, ([0.0, np.nan], [1.0, 1.0])),
        (MultivariateGaussianDistribution, ([], np.empty((0, 0)))),
        (MultivariateGaussianDistribution, ([0.0, 1.0], np.eye(1))),
        (MultivariateGaussianDistribution, ([0.0, np.inf], np.eye(2))),
        (MultivariateStudentTDistribution, (5.0, [0.0, 1.0], np.eye(1))),
    ],
)
def test_vector_distribution_constructors_reject_incoherent_parameters(constructor, args):
    with pytest.raises((TypeError, ValueError)):
        constructor(*args)


def test_diagonal_gaussian_sequence_scalar_api_is_supported():
    dist = DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0])
    assert dist.log_density([0.0, 0.0]) == pytest.approx(dist.log_density(np.zeros(2)))
    accumulator = DiagonalGaussianAccumulator(dim=2)
    accumulator.update([1.0, 2.0], 0.5, dist)
    np.testing.assert_allclose(accumulator.value()[0], [0.5, 1.0])


@pytest.mark.parametrize(
    "encoder",
    [
        DiagonalGaussianDataEncoder(2),
        MultivariateGaussianDataEncoder(2),
        MultivariateStudentTDataEncoder(2),
    ],
)
def test_vector_encoders_never_reshape_event_structure(encoder):
    for malformed in ([1.0, 2.0], [[1.0, 2.0, 3.0, 4.0]], [[1.0, np.nan]]):
        with pytest.raises(ValueError):
            encoder.seq_encode(malformed)


@pytest.mark.parametrize(
    ("accumulator", "estimate"),
    [
        (DiagonalGaussianAccumulator(2), DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0])),
        (MultivariateGaussianAccumulator(2), MultivariateGaussianDistribution([0.0, 0.0], np.eye(2))),
        (MultivariateStudentTAccumulator(5.0, 2), MultivariateStudentTDistribution(5.0, [0.0, 0.0], np.eye(2))),
    ],
)
def test_vector_accumulators_reject_malformed_evidence(accumulator, estimate):
    with pytest.raises(ValueError):
        accumulator.update([1.0], 1.0, estimate)
    with pytest.raises(ValueError):
        accumulator.update([1.0, 2.0], -1.0, estimate)
    with pytest.raises(ValueError):
        accumulator.seq_update(np.ones((2, 2)), np.ones(1), estimate)


def test_student_t_accumulator_rejects_incompatible_estimates():
    accumulator = MultivariateStudentTAccumulator(5.0, 2)
    with pytest.raises(ValueError):
        accumulator.update(
            [0.0, 0.0],
            1.0,
            MultivariateStudentTDistribution(6.0, [0.0, 0.0], np.eye(2)),
        )


@pytest.mark.parametrize(
    "make_distribution",
    [
        lambda mu, scale: DiagonalGaussianDistribution(mu, np.diag(scale)),
        lambda mu, scale: MultivariateGaussianDistribution(mu, scale),
        lambda mu, scale: MultivariateStudentTDistribution(5.0, mu, scale),
    ],
)
def test_vector_distribution_parameters_are_owned_and_immutable(make_distribution):
    mu = np.array([0.0, 1.0])
    scale = np.array([[2.0, 0.2], [0.2, 1.5]])
    dist = make_distribution(mu, scale)
    before = dist.log_density([0.5, -0.25])
    mu[:] = 100.0
    scale[:] = 100.0
    assert dist.log_density([0.5, -0.25]) == pytest.approx(before)
    with pytest.raises(ValueError):
        dist.mu[0] = 3.0
    covariance = dist.covar if hasattr(dist, "covar") else dist.shape
    with pytest.raises(ValueError):
        covariance.flat[0] = 3.0


def test_public_multivariate_gaussian_rejects_singular_covariance():
    with pytest.raises(ValueError, match="positive definite"):
        MultivariateGaussianDistribution([0.0, 0.0], np.zeros((2, 2)))


@pytest.mark.parametrize(
    "dist",
    [
        DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0]),
        MultivariateGaussianDistribution([0.0, 0.0], np.eye(2)),
        MultivariateStudentTDistribution(5.0, [0.0, 0.0], np.eye(2)),
    ],
)
def test_vector_marginals_reject_duplicate_indices(dist):
    with pytest.raises(ValueError, match="unique"):
        dist.marginal([0, 0])


@pytest.mark.parametrize("estimator", [GaussianEstimator, LogGaussianEstimator])
def test_scalar_gaussian_pseudo_observations_pool_raw_moments(estimator):
    fitted = estimator(pseudo_count=1.0, suff_stat=(10.0, 1.0)).estimate(
        None,
        (0.0, 0.0, 1.0, 1.0),
    )
    assert fitted.mu == pytest.approx(5.0)
    assert fitted.sigma2 == pytest.approx(25.5)


def test_vector_gaussian_pseudo_observations_pool_raw_moments():
    diagonal = DiagonalGaussianEstimator(
        pseudo_count=1.0,
        suff_stat=(np.array([10.0]), np.array([1.0])),
        ridge=0.0,
    ).estimate(None, (np.array([0.0]), np.array([0.0]), 1.0))
    full = MultivariateGaussianEstimator(
        pseudo_count=1.0,
        suff_stat=(np.array([10.0]), np.array([[1.0]])),
        ridge=0.0,
    ).estimate(None, (np.array([0.0]), np.array([[0.0]]), 1.0))
    np.testing.assert_allclose(diagonal.mu, [5.0])
    np.testing.assert_allclose(diagonal.covar, [25.5])
    np.testing.assert_allclose(full.mu, [5.0])
    np.testing.assert_allclose(full.covar, [[25.5]], atol=2.0e-8)


def test_diagonal_estimator_infers_dimension_from_covariance_vector():
    estimator = DiagonalGaussianEstimator(suff_stat=(None, np.ones(4)))
    assert estimator.dim == 4


@pytest.mark.parametrize(
    "build",
    [
        lambda: GaussianEstimator(min_covar=np.nan),
        lambda: LogGaussianEstimator(pseudo_count=-1.0),
        lambda: DiagonalGaussianEstimator(dim=2, ridge=-1.0),
        lambda: MultivariateGaussianEstimator(dim=2, degenerate_ratio=2.0),
        lambda: MultivariateStudentTEstimator(dim=2, min_ridge=0.0),
    ],
)
def test_gaussian_estimator_controls_fail_closed(build):
    with pytest.raises((TypeError, ValueError)):
        build()


def test_gaussian_estimators_reject_invalid_reduced_statistics():
    with pytest.raises(ValueError):
        GaussianEstimator().estimate(None, (np.nan, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        DiagonalGaussianEstimator(dim=2).estimate(
            None,
            (np.ones(2), np.ones(2), -1.0),
        )
    with pytest.raises(ValueError):
        MultivariateGaussianEstimator(dim=2).estimate(
            None,
            (np.ones(2), np.full((2, 2), np.nan), 1.0),
        )


@pytest.mark.parametrize(
    "estimator",
    [
        DiagonalGaussianEstimator(dim=2, name="fit", keys="shared"),
        MultivariateGaussianEstimator(dim=2, name="fit", keys="shared"),
        MultivariateStudentTEstimator(dim=2, name="fit", keys="shared"),
    ],
)
def test_vector_fit_preserves_identity_metadata(estimator):
    accumulator = estimator.accumulator_factory().make()
    accumulator.seq_update(np.array([[0.0, 1.0], [1.0, 0.0]]), np.ones(2), None)
    fitted = estimator.estimate(None, accumulator.value())
    assert fitted.name == "fit"
    assert fitted.keys == "shared"


def test_student_t_restoration_copies_serialized_arrays():
    first = np.array([1.0, 2.0])
    second = np.eye(2)
    accumulator = MultivariateStudentTAccumulator(5.0, 2).from_value((1.0, 1.0, first, second))
    first[0] = 99.0
    second[0, 0] = 99.0
    assert accumulator.value()[2][0] == 1.0
    assert accumulator.value()[3][0, 0] == 1.0


def test_student_t_scalar_and_batch_shapes_are_explicit():
    dist = MultivariateStudentTDistribution(5.0, [0.0, 0.0], np.eye(2))
    with pytest.raises(ValueError):
        dist.log_density([[0.0, 0.0]])
    with pytest.raises(ValueError):
        dist.seq_log_density([0.0, 0.0])
    with pytest.raises(ValueError):
        dist.seq_log_density([[0.0, np.inf]])


def test_student_t_estimator_rejects_invalid_weighted_statistics():
    estimator = MultivariateStudentTEstimator(5.0, dim=2)
    with pytest.raises(ValueError):
        estimator.estimate(None, (1.0, -1.0, np.zeros(2), np.zeros((2, 2))))
    with pytest.raises(ValueError):
        estimator.estimate(None, (1.0, 1.0, np.ones(2), np.eye(1)))
