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
