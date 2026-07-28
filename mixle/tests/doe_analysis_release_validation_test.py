"""Focused release-contract probes for designed-experiment analysis."""

from __future__ import annotations

import numpy as np
import pytest

from mixle.doe.analysis import design_diagnostics, factorial_effects, response_surface


def test_opposite_factorial_aliases_publish_the_signed_estimable_contrast():
    design = np.array([[-1.0, 1.0], [1.0, -1.0]])
    response = 2.0 * design[:, 0] + 3.0 * design[:, 1]
    effects = factorial_effects(design, response, interactions=False, coded=True)
    assert set(effects.estimable_contrasts) == {"x0-x1"}
    assert effects.estimable_contrasts["x0-x1"][0] == pytest.approx(-2.0)


def test_underdetermined_quadratic_surface_is_not_classified():
    with pytest.raises(ValueError, match="not estimable.*rank is 1 of 3"):
        response_surface(np.array([[1.0]]), np.array([1.0]))


@pytest.mark.parametrize(
    "design,response,match",
    [
        (np.array([]), np.array([]), "two-dimensional"),
        (np.empty((0, 1)), np.array([]), "non-empty"),
        (np.array([[np.nan]]), np.array([1.0]), "finite design"),
        (np.array([[1.0], [2.0], [3.0]]), np.array([1.0, np.nan, 3.0]), "finite response"),
        (np.array([[1.0], [2.0], [3.0]]), np.array([[1.0], [2.0], [3.0]]), "one response"),
    ],
)
def test_response_surface_rejects_invalid_evidence_before_linear_algebra(design, response, match):
    with pytest.raises(ValueError, match=match):
        response_surface(design, response)


def test_identified_response_surface_reports_rank_and_degrees_of_freedom():
    x = np.linspace(-2.0, 2.0, 5)[:, None]
    surface = response_surface(x, 1.0 + 2.0 * x[:, 0] + 3.0 * x[:, 0] ** 2)
    assert surface.estimable is True
    assert surface.model_rank == surface.n_parameters == 3
    assert surface.degrees_of_freedom == 2


@pytest.mark.parametrize(
    "design,model,match",
    [
        (np.array([]), lambda x: x, "two-dimensional"),
        (np.empty((0, 1)), lambda x: x, "non-empty"),
        (np.array([[np.nan]]), lambda x: x, "finite"),
        (np.ones((2, 1)), lambda x: np.ones(2), "two-dimensional matrix"),
        (np.ones((2, 1)), lambda x: np.ones((1, 1)), "one row per design"),
        (np.ones((2, 1)), lambda x: np.full((2, 1), np.nan), "finite features"),
        (np.ones((2, 1)), lambda x: np.zeros((2, 1)), "rank zero"),
        (np.full((2, 1), 1e308), lambda x: x, "information matrix"),
    ],
)
def test_design_diagnostics_rejects_invalid_or_undefined_evidence(design, model, match):
    with pytest.raises(ValueError, match=match):
        design_diagnostics(design, model)


def test_rank_deficient_diagnostics_are_explicit_and_numerically_finite():
    design = np.column_stack((np.linspace(-1.0, 1.0, 8), np.ones(8)))

    def model(x):
        return np.column_stack((np.ones(x.shape[0]), x))

    diagnostics = design_diagnostics(design, model)
    assert diagnostics["full_rank"] is False
    assert diagnostics["rank"] == 2
    assert diagnostics["n_params"] == 3
    assert diagnostics["condition_number"] is None
    for key in ("d_efficiency", "a_efficiency", "g_efficiency", "effective_condition_number", "max_correlation"):
        assert np.isfinite(diagnostics[key])
