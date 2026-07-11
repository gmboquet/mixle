"""P8 (experimental) -- closed-loop equation discovery with an exact referee.

Receipts: the discovery loop recovers the governing operator's exact symbolic form and coefficients
across three worlds of increasing hardness (graded against ground truth); at a constrained
experiment budget, actively-chosen high-leverage probes beat random probing on discovery rate (the
card's "beat the random-experiment baseline on discovery-per-budget"); and the sparse fit does not
hallucinate spurious terms.
"""

from __future__ import annotations

import numpy as np

from mixle.experimental.equation_discovery import (
    active_experiments,
    discover,
    discovery_rate,
)

LINEAR = np.array([0.0, 1.5, 0.0, 0.0])  # dx/dt = 1.5 x
TERM_SELECT = np.array([0.0, 1.2, 0.0, 0.8])  # 1.2 x + 0.8 x^3
NONLINEAR = np.array([0.0, 2.0, 0.0, -1.0])  # 2 x - x^3


def test_recovers_operator_form_and_coefficients_exactly() -> None:
    for coef in (LINEAR, TERM_SELECT, NONLINEAR):
        r = discover(coef, active_experiments(16, 2.0), noise=0.15, threshold=0.3, seed=0)
        assert r.form_match, f"failed to recover the form of {coef}: got {sorted(r.recovered_terms)}"
        assert r.coef_error < 0.15, f"coefficient error {r.coef_error:.3f} too large for {coef}"


def test_exact_referee_grades_against_ground_truth() -> None:
    r = discover(NONLINEAR, active_experiments(16, 2.0), noise=0.1, threshold=0.3, seed=1)
    assert r.true_terms == {1, 3}  # x and x^3
    assert r.recovered_terms == r.true_terms
    assert isinstance(r.coef_error, float)


def test_active_beats_random_on_discovery_per_budget() -> None:
    seeds = range(60)
    active = discovery_rate(NONLINEAR, strategy="active", budget=5, radius=2.0, noise=0.4, threshold=0.4, seeds=seeds)
    random = discovery_rate(NONLINEAR, strategy="random", budget=5, radius=2.0, noise=0.4, threshold=0.4, seeds=seeds)
    assert active >= random + 0.25, f"active ({active:.2f}) did not clearly beat random ({random:.2f})"
    assert active >= 0.8, f"active discovery rate unexpectedly low: {active:.2f}"


def test_does_not_hallucinate_spurious_terms() -> None:
    """A purely linear world must recover exactly {x}, not a spurious cubic."""
    r = discover(LINEAR, active_experiments(16, 2.0), noise=0.1, threshold=0.3, seed=2)
    assert r.recovered_terms == {1}, f"hallucinated terms: {sorted(r.recovered_terms)}"


def test_harder_world_is_harder_at_low_budget() -> None:
    """The nonlinear world needs more/better probing than the linear one at a tiny budget."""
    seeds = range(60)
    lin = discovery_rate(LINEAR, strategy="random", budget=4, radius=2.0, noise=0.4, threshold=0.4, seeds=seeds)
    non = discovery_rate(NONLINEAR, strategy="random", budget=4, radius=2.0, noise=0.4, threshold=0.4, seeds=seeds)
    assert lin >= non, f"linear world should not be harder than nonlinear: lin={lin:.2f} non={non:.2f}"


def test_determinism() -> None:
    a = discover(NONLINEAR, active_experiments(12, 2.0), noise=0.2, threshold=0.3, seed=5)
    b = discover(NONLINEAR, active_experiments(12, 2.0), noise=0.2, threshold=0.3, seed=5)
    assert np.array_equal(a.recovered_coef, b.recovered_coef) and a.form_match == b.form_match
