"""P13 (experimental) -- usable-information (V-information) receipts.

Receipts: on a linear-Gaussian task (true MI known), the linear family's V-information nearly
equals the available MI (small gap -- the grammar is sufficient); on a task whose generative law
sits just outside the linear grammar (quadratic), the linear family is blind (I_V ~ 0) while
adding the quadratic feature closes the gap. The ranking of the two candidate additions is stable
across seeds -- the card's kill criterion (a gap estimate too noisy to rank cannot steer anything).
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.experimental.v_information import (
    estimate_v_information,
    gaussian_mutual_information,
    usability_gap,
    v_information,
)


def _linear_task(rng, n, rho):
    x = rng.normal(0, 1, n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
    return x, y


def _quadratic_task(rng, n, noise=0.3):
    x = rng.normal(0, 1, n)
    y = x**2 + noise * rng.normal(0, 1, n)  # E[Y|X] = X^2 is uncorrelated with X (linear-blind)
    return x, y


def test_gaussian_mutual_information_closed_form() -> None:
    assert gaussian_mutual_information(0.0) == 0.0
    assert np.isclose(gaussian_mutual_information(0.7), -0.5 * np.log(1 - 0.49))
    assert gaussian_mutual_information(0.99) > gaussian_mutual_information(0.9)
    assert gaussian_mutual_information(1.0) == float("inf")
    assert gaussian_mutual_information(-1.0) == float("inf")


@pytest.mark.parametrize("rho", [2.0, -1.01, np.nan, np.inf, True])
def test_gaussian_mutual_information_rejects_invalid_correlations(rho) -> None:
    with pytest.raises(ValueError, match="correlation"):
        gaussian_mutual_information(rho)


def test_linear_family_captures_available_information() -> None:
    """I_V(linear) ~= I(X;Y) on a linear-Gaussian task: small usability gap."""
    rng = np.random.default_rng(0)
    x, y = _linear_task(rng, 4000, rho=0.7)
    true_mi = gaussian_mutual_information(0.7)
    iv = v_information(x, y, degree=1, seed=0)
    gap = usability_gap(true_mi, iv)
    assert abs(gap) < 0.03, f"linear family should capture the MI; gap was {gap:.4f}"


def test_gap_localizes_missing_capability_and_closes() -> None:
    """Generative law outside the linear grammar: linear is blind, quadratic closes the gap."""
    rng = np.random.default_rng(1)
    x, y = _quadratic_task(rng, 4000)
    iv_linear = v_information(x, y, degree=1, seed=0)
    iv_quad = v_information(x, y, degree=2, seed=0)
    assert abs(iv_linear) < 0.05, f"linear family should be ~blind on a quadratic law; got {iv_linear:.4f}"
    assert iv_quad > iv_linear + 0.5, f"adding the quadratic feature should close the gap; {iv_quad:.4f}"


def test_v_information_is_non_negative_when_family_captures() -> None:
    rng = np.random.default_rng(2)
    x, y = _linear_task(rng, 4000, rho=0.6)
    assert v_information(x, y, degree=1, seed=0) >= -0.02  # usable info non-negative (MC slack)


def test_ranking_is_consistent_across_seeds() -> None:
    """Kill criterion: the gap must rank the two candidate additions the same way every seed."""
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        x, y = _quadratic_task(rng, 3000)
        iv_linear = v_information(x, y, degree=1, seed=seed)
        iv_quad = v_information(x, y, degree=2, seed=seed)
        assert iv_quad > iv_linear, f"seed {seed}: ranking flipped ({iv_quad:.3f} !> {iv_linear:.3f})"


def test_determinism() -> None:
    rng = np.random.default_rng(3)
    x, y = _quadratic_task(rng, 2000)
    assert v_information(x, y, degree=2, seed=7) == v_information(x, y, degree=2, seed=7)


def test_repeated_split_estimate_reports_scoped_uncertainty() -> None:
    x, y = _linear_task(np.random.default_rng(4), 500, rho=0.5)
    receipt = estimate_v_information(x, y, degree=1, n_splits=5, confidence=0.9, seed=3)
    assert len(receipt.split_estimates) == 5
    assert receipt.standard_error >= 0.0
    assert receipt.interval[0] <= receipt.estimate <= receipt.interval[1]
    assert receipt.uncertainty_scope == "random_holdout_assignment_only"
    assert receipt.estimate == np.mean(receipt.split_estimates)


@pytest.mark.parametrize(
    ("x", "y", "kwargs", "message"),
    [
        (np.ones((3, 1)), np.ones(3), {}, "one-dimensional"),
        (np.ones(4), np.ones(3), {}, "same number"),
        (np.array([1.0, np.nan, 2.0]), np.ones(3), {}, "finite"),
        (np.ones(3), np.ones(3), {"holdout": 0.0}, "holdout"),
        (np.ones(3), np.ones(3), {"holdout": 1.0}, "holdout"),
        (np.ones(10), np.ones(10), {"degree": -1}, "degree"),
        (np.ones(10), np.ones(10), {"degree": 1.5}, "degree"),
        (np.ones(3), np.ones(3), {"degree": 1, "holdout": 0.3}, "training split"),
    ],
)
def test_v_information_rejects_invalid_or_unidentifiable_inputs(x, y, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        v_information(x, y, **kwargs)
