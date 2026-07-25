"""Adverse contract tests for Sigma-weighted structured projections."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from mixle.models.sigma_weighted_projection import (
    ButterflyProjection,
    CovarianceAdjustmentWarning,
    project,
    sigma_weighted_block_sparse,
    sigma_weighted_butterfly,
    sigma_weighted_error,
    sigma_weighted_low_rank,
    sigma_weighted_permutation,
)


@pytest.mark.parametrize(
    ("w", "sigma", "message"),
    [
        (np.ones(3), np.eye(3), "two-dimensional"),
        (np.empty((0, 3)), np.eye(3), "non-empty"),
        (np.asarray([[np.nan, 1.0]]), np.eye(2), "finite"),
        (np.ones((2, 2)), np.asarray([[1.0, 0.1], [0.0, 1.0]]), "symmetric"),
        (np.ones((2, 2)), np.diag([1.0, -0.1]), "positive semidefinite"),
        (np.ones((2, 2)), np.asarray([[1.0, 0.0], [0.0, np.inf]]), "finite"),
    ],
)
def test_all_public_objectives_validate_weight_covariance_domain(
    w: np.ndarray,
    sigma: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sigma_weighted_error(w, np.ones_like(w), sigma)


def test_roundoff_psd_projection_is_explicit_and_reported() -> None:
    w = np.eye(2)
    sigma = np.diag([1.0, -1.0e-13])
    with pytest.warns(CovarianceAdjustmentWarning, match="PSD cone"):
        _, report = project(w, sigma, structure="low_rank", rank=1)
    assert report.stats["covariance_psd_correction"] == pytest.approx(1.0e-13)


@pytest.mark.parametrize(
    "call",
    [
        lambda: sigma_weighted_low_rank(np.eye(2), np.eye(2), rank=-1),
        lambda: sigma_weighted_low_rank(np.eye(2), np.eye(2), rank=1.5),
        lambda: sigma_weighted_block_sparse(np.eye(4), np.eye(4), "2:4", max_iter=0),
        lambda: sigma_weighted_block_sparse(np.eye(4), np.eye(4), "2:4", tol=-1.0),
        lambda: sigma_weighted_butterfly(np.eye(4), np.eye(4), n_sweeps=0),
        lambda: sigma_weighted_butterfly(np.eye(4), np.eye(4), n_stages=3),
    ],
)
def test_solvers_reject_invalid_iteration_rank_and_tolerance_controls(call: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()


def test_permutation_is_an_exact_assignment_without_inert_sinkhorn_controls() -> None:
    w = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    profile = w[::-1]
    np.testing.assert_array_equal(sigma_weighted_permutation(w, np.eye(2), profile), w)

    with pytest.raises(TypeError, match="unexpected perm_profile options"):
        project(
            w,
            np.eye(2),
            structure="perm_profile",
            target_profile=profile,
            temperature=0.1,
        )


def test_butterfly_returns_compact_executable_pickle_stable_factors() -> None:
    rng = np.random.default_rng(42)
    w = rng.normal(size=(8, 8))
    sigma = np.eye(8)
    projection = sigma_weighted_butterfly(w, sigma, n_sweeps=2)

    assert isinstance(projection, ButterflyProjection)
    assert projection.parameter_count == 2 * projection.n * len(projection.strides)
    assert projection.parameter_nbytes == projection.parameter_count * np.dtype(np.float64).itemsize
    assert projection.parameter_count < np.prod(projection.shape)
    assert projection.serialized_nbytes == len(pickle.dumps(projection, protocol=5))

    x = rng.normal(size=(5, 8))
    dense = projection.to_dense()
    np.testing.assert_allclose(projection.apply(x), x @ dense.T)
    np.testing.assert_allclose(projection @ x.T, dense @ x.T)

    restored = pickle.loads(pickle.dumps(projection, protocol=5))
    np.testing.assert_allclose(restored.apply(x), projection.apply(x))


def test_butterfly_report_measures_physical_and_serialized_storage() -> None:
    projection, report = project(np.eye(8), np.eye(8), structure="butterfly", n_sweeps=1)
    assert report.stats["param_count"] == projection.parameter_count
    assert report.stats["parameter_nbytes"] == projection.parameter_nbytes
    assert report.stats["serialized_nbytes"] == projection.serialized_nbytes
    assert report.stats["dense_nbytes"] == np.asarray(projection).nbytes
