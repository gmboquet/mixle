"""Focused release-contract probes for optimal experimental design."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.doe.optimal import c_criterion, g_criterion, i_criterion, optimal_design

_POOL = np.array([[-1.0], [0.0], [1.0]])


@pytest.mark.parametrize(
    "name,value",
    [
        ("n", 2.9),
        ("n", True),
        ("n_candidates", 2.9),
        ("n_candidates", True),
        ("n_restarts", 0),
        ("n_restarts", 2.9),
        ("n_restarts", True),
        ("max_iter", 0),
        ("max_iter", 2.9),
        ("max_iter", True),
    ],
)
def test_optimal_design_counts_are_exact_positive_non_boolean(name, value):
    kwargs = {"n": 2, "candidates": _POOL}
    kwargs[name] = value
    with pytest.raises((TypeError, ValueError), match=name):
        optimal_design(None, **kwargs)


@pytest.mark.parametrize(
    "candidates",
    [
        np.array([0.0, 1.0]),
        np.array(1.0),
        np.empty((0, 1)),
        np.empty((2, 0)),
        np.array([[0.0], [np.nan]]),
    ],
)
def test_optimal_design_requires_a_finite_nonempty_candidate_matrix(candidates):
    with pytest.raises(ValueError, match="candidates"):
        optimal_design(None, 1, candidates=candidates)


@pytest.mark.parametrize(
    "model,match",
    [
        (lambda x: np.ones(x.shape[0]), "two-dimensional"),
        (lambda x: np.ones((x.shape[0] - 1, 1)), "one row per"),
        (lambda x: np.full((x.shape[0], 1), np.nan), "finite features"),
        (lambda x: np.empty((x.shape[0], 0)), "non-empty"),
    ],
)
def test_optimal_design_validates_the_model_matrix_sample_axis(model, match):
    with pytest.raises(ValueError, match=match):
        optimal_design(None, 2, candidates=_POOL, model=model)


@pytest.mark.parametrize("criterion", [i_criterion, g_criterion])
def test_prediction_criteria_reject_empty_reference_evidence(criterion):
    with pytest.raises(ValueError, match="non-empty"):
        criterion(np.eye(2), ref=np.empty((0, 2)))


@pytest.mark.parametrize("contrast", [np.array([]), np.array([np.nan]), np.array([np.inf])])
def test_c_criterion_rejects_empty_or_nonfinite_contrasts(contrast):
    with pytest.raises(ValueError, match="contrast"):
        c_criterion(contrast)


def test_optimal_design_rejects_empty_reference_model_matrix():
    with pytest.raises(ValueError, match="reference model matrix"):
        optimal_design(None, 2, candidates=_POOL, ref=np.empty((0, 2)))


@pytest.mark.parametrize("merit", [np.nan, np.inf, True, np.array([1.0])])
def test_optimizer_rejects_invalid_custom_criterion_merit(merit):
    def criterion(info, *, ref=None):
        return merit

    with pytest.raises(ValueError, match="merit"):
        optimal_design(None, 2, candidates=_POOL, criterion=criterion)
