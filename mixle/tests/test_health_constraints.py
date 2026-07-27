"""K6 DoD -- health/safety constraints + liabilities into J/H (notes/exec/workstream-K.md).

``exposure_constraints`` must mark a candidate operating option that breaches a named exposure limit
(here: an occupational silica-dust limit) ``feasible=False`` and name the breached limit; the H4
stochastic optimizer (``mixle.stochastic_opt.two_stage_stochastic_plan``) must then never see that
option at all -- a caller filters the feasible survivors down *before* building the blocks the
optimizer plans over, so the infeasible option is dropped from the plan by construction. This also
exercises ``health_liability`` (K6's other new symbol): pricing a K3 ``population_risk`` distribution
into a discounted dollar-liability distribution of the same shape.

Named with the ``test_*.py`` prefix (rather than this repo's own ``*_test.py`` `python_files`
convention -- see ``pyproject.toml``) because this exact path + node id is the frozen DoD command in
``notes/exec/workstream-K.md``; explicit pytest node ids are collected regardless of the
``python_files`` glob, so this does not conflict with the repo's discovery config.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.analysis.health_risk import DoseResponse, exposure_constraints, health_liability, population_risk
from mixle.reason.posterior_protocol import Posterior
from mixle.stochastic_opt import two_stage_stochastic_plan

SILICA_LIMIT = 0.05  # occupational 8-hour TWA, mg/m^3 (illustrative, not a regulatory citation)


class _PointGradePosterior:
    """A minimal IC-1 `Posterior`: a degenerate point mass at ``grades`` (deterministic H4 wiring)."""

    def __init__(self, grades: np.ndarray) -> None:
        self._grades = np.asarray(grades, dtype=float)

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.tile(self._grades, (n, 1))

    @property
    def mean(self) -> np.ndarray:
        return self._grades

    @property
    def cov(self) -> np.ndarray:
        return np.zeros((self._grades.size, self._grades.size))

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        return self._grades, self._grades

    def derived_quantity(self, fn, n, rng):
        draws = self.samples(n, rng)
        pushed = fn(draws)

        class _DQ:
            samples = pushed
            prior_dominated = False

            def credible_interval(self, level: float):
                a = (1.0 - level) / 2.0
                return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1.0 - a, axis=0)

        return _DQ()


def test_exposure_limit_removes_option():
    options = [
        {"name": "pit_a", "silica_pm4": 0.03, "block_cost": 10.0, "grade": 2.0},
        {"name": "pit_b", "silica_pm4": 0.12, "block_cost": 5.0, "grade": 3.0},  # breaches the limit
        {"name": "pit_c", "silica_pm4": 0.02, "block_cost": 8.0, "grade": 1.5},
    ]
    limits = {"silica_pm4": SILICA_LIMIT}

    annotated = exposure_constraints(options, limits)

    assert [o["feasible"] for o in annotated] == [True, False, True]
    assert annotated[1]["binding"] == ["silica_pm4"]
    assert annotated[0]["binding"] == []
    assert annotated[2]["binding"] == []
    # the original option dicts are untouched (a new list is returned)
    assert "feasible" not in options[0]

    # health_liability: a K3 case-count risk distribution -> a discounted dollar-liability distribution
    # of the same shape, the term J6's priced_liabilities/risk_adjusted_plan sums as "health_cost".
    dr = DoseResponse(model="loglinear", params={"beta": 0.01})
    risk = population_risk(np.array([o["silica_pm4"] for o in options]), dr, n=500, rng=np.random.default_rng(0))
    liability = health_liability(risk, cost_per_case=1_000_000.0, discount=0.0)
    assert liability.samples.shape == risk.samples.shape
    assert np.allclose(liability.samples, risk.samples * 1_000_000.0)
    assert liability.prior_dominated == risk.prior_dominated

    discounted = health_liability(risk, cost_per_case=1_000_000.0, discount=1.0)
    assert np.allclose(discounted.samples, liability.samples / 2.0)

    # Only the feasible survivors ever become candidate blocks for H4's stochastic optimizer -- the
    # infeasible option (pit_b) is dropped from the plan by never entering its inputs at all.
    feasible = [o for o in annotated if o["feasible"]]
    feasible_names = [o["name"] for o in feasible]
    assert "pit_b" not in feasible_names
    assert set(feasible_names) == {"pit_a", "pit_c"}

    block_cost = np.array([o["block_cost"] for o in feasible])
    grades = np.array([o["grade"] for o in feasible])
    posterior = _PointGradePosterior(grades)
    assert isinstance(posterior, Posterior)

    plan = two_stage_stochastic_plan(
        posterior, block_cost, price=100.0, k_scenarios=10, alpha=0.9, rng=np.random.default_rng(1)
    )

    # The plan is defined only over the feasible-option blocks -- pit_b's exposure-violating block cost
    # and grade were never assembled into `block_cost`/`posterior`, so it cannot appear in `extract`.
    assert plan.extract.shape == (len(feasible),)


def test_exposure_constraints_missing_and_nan_metrics_fail_closed():
    """MXR-080-0095: an option missing a regulated metric, or reporting it as NaN, is "unknown" and
    therefore infeasible by default -- neither is silently treated as "safe", since this is a hard
    exposure screen and NaN > limit is always False (the recurring NaN-comparison-is-False trap)."""
    limits = {"silica_pm4": SILICA_LIMIT}

    (missing,) = exposure_constraints([{"name": "no_data"}], limits)
    assert missing["status"] == "unknown"
    assert missing["feasible"] is False
    assert missing["binding"] == []
    assert missing["unmodeled"] == ["silica_pm4"]

    (nan_reading,) = exposure_constraints([{"name": "nan_reading", "silica_pm4": float("nan")}], limits)
    assert nan_reading["status"] == "unknown"
    assert nan_reading["feasible"] is False
    assert nan_reading["unmodeled"] == ["silica_pm4"]

    # a non-finite LIMIT (not just a non-finite measurement) is equally unevaluable
    (nan_limit,) = exposure_constraints([{"name": "ok", "silica_pm4": 0.02}], {"silica_pm4": float("nan")})
    assert nan_limit["status"] == "unknown"
    assert nan_limit["feasible"] is False


def test_exposure_constraints_violation_dominates_unknown():
    """A confirmed breach on one metric is never erased by a different metric's missing data."""
    limits = {"silica_pm4": SILICA_LIMIT, "noise_db": 85.0}
    # silica_pm4 breaches its limit; noise_db was never modeled for this option at all.
    (option,) = exposure_constraints([{"name": "loud_and_dusty", "silica_pm4": 0.20}], limits)
    assert option["status"] == "violating"
    assert option["feasible"] is False
    assert option["binding"] == ["silica_pm4"]
    assert option["unmodeled"] == ["noise_db"]


def test_exposure_constraints_policy_override_defaults_off():
    """The unmodeled-as-safe override is opt-in and off by default (MXR-080-0095)."""
    limits = {"silica_pm4": SILICA_LIMIT}
    options = [{"name": "no_data"}]

    (default_result,) = exposure_constraints(options, limits)
    assert default_result["feasible"] is False

    (overridden,) = exposure_constraints(options, limits, treat_unmodeled_as_safe=True)
    assert overridden["status"] == "unknown"  # still honestly reported as unmodeled...
    assert overridden["feasible"] is True  # ...but the caller explicitly chose to let it through

    # the override never rescues a genuine, confirmed violation
    (still_violating,) = exposure_constraints(
        [{"name": "over", "silica_pm4": 0.20}], limits, treat_unmodeled_as_safe=True
    )
    assert still_violating["status"] == "violating"
    assert still_violating["feasible"] is False


def test_health_liability_rejects_invalid_discount_and_cost():
    """MXR-080-0098: discount == -1 divided by zero (1/(1+discount)); anything below -1 silently
    flipped the liability's sign. cost_per_case must also be finite and non-negative."""
    risk = population_risk(
        np.array([0.02, 0.03]),
        DoseResponse(model="loglinear", params={"beta": 0.01}),
        n=10,
        rng=np.random.default_rng(0),
    )
    for bad_discount in (-1.0, -1.5, -2.0, float("nan"), float("-inf")):
        with pytest.raises(ValueError):
            health_liability(risk, cost_per_case=1_000_000.0, discount=bad_discount)
    for bad_cost in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            health_liability(risk, cost_per_case=bad_cost)


@pytest.mark.parametrize(
    "samples",
    [
        np.array([-1.0, 1.0]),
        np.array([]),
        np.array(np.nan),
        np.ones((2, 2, 2)),
    ],
)
def test_health_liability_rejects_invalid_risk_samples(samples):
    risk = type("Risk", (), {"samples": samples, "prior_dominated": False})()
    with pytest.raises(ValueError, match="risk.samples"):
        health_liability(risk, cost_per_case=10.0)


def test_health_liability_valid_discount_still_prices_correctly():
    """Negative control for MXR-080-0098: legitimate discounts (including a large but valid one) keep
    producing a correctly-scaled, non-negative liability."""
    risk = population_risk(
        np.array([0.02, 0.03]),
        DoseResponse(model="loglinear", params={"beta": 0.01}),
        n=10,
        rng=np.random.default_rng(0),
    )
    for discount in (0.0, 0.05, 1.0, 10.0):
        liability = health_liability(risk, cost_per_case=1_000_000.0, discount=discount)
        assert np.isfinite(liability.samples).all()
        assert (liability.samples >= 0.0).all()
        assert np.allclose(liability.samples, risk.samples * 1_000_000.0 / (1.0 + discount))


def test_exposure_constraints_genuine_safe_and_violating_unchanged():
    """Negative control for MXR-080-0095: fully-modeled, finite options behave exactly as before."""
    limits = {"silica_pm4": SILICA_LIMIT}
    safe, violating = exposure_constraints(
        [
            {"name": "ok", "silica_pm4": 0.02},
            {"name": "over", "silica_pm4": 0.12},
        ],
        limits,
    )
    assert safe["status"] == "safe"
    assert safe["feasible"] is True
    assert safe["unmodeled"] == []
    assert violating["status"] == "violating"
    assert violating["feasible"] is False
    assert violating["binding"] == ["silica_pm4"]
