"""H4 DoD — stochastic / robust optimization under grade uncertainty (notes/exec/workstream-H.md).

Synthetic blocks with a *known* truth grade and a posterior stub whose ``.samples`` returns noisy
draws around it. Three groups: a few blocks that are always clearly profitable ("safe good"), a few
that are always clearly unprofitable ("safe bad"), and one "risky" block whose baseline grade is
mildly profitable but which carries a low-probability catastrophic grade shock (an ore/waste
misclassification tail). The risky block's *true* mean profit (computed from the exact mixture, no
sampling) is slightly negative — it is a true loser — but the specific fixed-seed 50-scenario draw
used for planning happens to under-realize the shock, so a plain point-estimate/expected-value-only
optimizer (the "deterministic-mean plan": threshold each block on the sample-mean profit of that same
draw, no risk term) is fooled into keeping it.

``two_stage_stochastic_plan``'s CVaR term looks at the *distribution* of the same 50 draws rather than
just their average, so it excludes the risky block: on a large held-out sample drawn from the true
generative process, this yields both a higher expected value (the risky block really was a net loser)
and a strictly better (lower) CVaR (its excluded tail risk).
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.reason.posterior_protocol import Posterior
from mixle.stochastic_opt import StochasticPlan, cvar_epigraph, two_stage_stochastic_plan

PRICE = 1.0
N_SAFE_GOOD = 3
N_SAFE_BAD = 3
N_BLOCKS = N_SAFE_GOOD + N_SAFE_BAD + 1  # + one risky block
SAFE_GOOD_GRADE = 3.0
SAFE_BAD_GRADE = 0.3
RISKY_MEAN = 1.06
RISKY_STD = 0.08
RISKY_SHOCK_PROB = 0.07
RISKY_SHOCK_GRADE = 0.1


class _BlockGradePosterior:
    """A minimal IC-1 `Posterior` over per-block ore grade: three known-truth groups (good/bad/risky)."""

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.zeros((n, N_BLOCKS))
        out[:, :N_SAFE_GOOD] = SAFE_GOOD_GRADE + rng.normal(0.0, 0.02, size=(n, N_SAFE_GOOD))
        out[:, N_SAFE_GOOD : N_SAFE_GOOD + N_SAFE_BAD] = SAFE_BAD_GRADE + rng.normal(0.0, 0.02, size=(n, N_SAFE_BAD))
        base = RISKY_MEAN + rng.normal(0.0, RISKY_STD, size=n)
        shocked = rng.random(n) < RISKY_SHOCK_PROB
        out[:, -1] = np.where(shocked, RISKY_SHOCK_GRADE, base)
        return np.clip(out, 0.0, None)

    @property
    def mean(self) -> np.ndarray:
        m = np.zeros(N_BLOCKS)
        m[:N_SAFE_GOOD] = SAFE_GOOD_GRADE
        m[N_SAFE_GOOD : N_SAFE_GOOD + N_SAFE_BAD] = SAFE_BAD_GRADE
        m[-1] = (1 - RISKY_SHOCK_PROB) * RISKY_MEAN + RISKY_SHOCK_PROB * RISKY_SHOCK_GRADE
        return m

    @property
    def cov(self) -> np.ndarray:
        return np.eye(N_BLOCKS)

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        return self.mean - 1.0, self.mean + 1.0

    def derived_quantity(self, fn, n, rng):
        s = fn(self.samples(n, rng))

        class _DQ:
            samples = s
            prior_dominated = False

            def credible_interval(self, level):
                a = (1.0 - level) / 2.0
                return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1 - a, axis=0)

        return _DQ()


def _held_out_cvar_and_ev(posterior, extract, cost, *, alpha=0.9, n_truth=200_000, seed=12345):
    """Empirical expected value + CVaR of a fixed extraction plan on a large held-out truth sample."""
    g_truth = posterior.samples(n_truth, np.random.default_rng(seed))
    profit = (PRICE * g_truth - cost[None, :]) @ extract.astype(np.float64)
    ev = float(profit.mean())
    loss = -profit
    n_tail = max(1, int(np.ceil((1.0 - alpha) * loss.size)))
    cvar = float(np.sort(loss)[::-1][:n_tail].mean())
    return ev, cvar


def test_posterior_stub_conforms_to_ic1():
    assert isinstance(_BlockGradePosterior(), Posterior)


def test_stochastic_plan_shape_and_types():
    posterior = _BlockGradePosterior()
    cost = np.ones(N_BLOCKS)
    plan = two_stage_stochastic_plan(posterior, cost, PRICE, k_scenarios=50, alpha=0.9, rng=np.random.default_rng(0))
    assert isinstance(plan, StochasticPlan)
    assert plan.extract.shape == (N_BLOCKS,)
    assert plan.extract.dtype == np.bool_
    assert plan.scenarios.shape == (50, N_BLOCKS)
    assert isinstance(plan.expected_value, float)
    assert isinstance(plan.cvar, float)


def test_stochastic_plan_beats_deterministic_mean_plan_on_held_out_truth():
    posterior = _BlockGradePosterior()
    cost = np.ones(N_BLOCKS)

    plan = two_stage_stochastic_plan(posterior, cost, PRICE, k_scenarios=50, alpha=0.9, rng=np.random.default_rng(0))

    # Safe blocks are always correctly classified: good ones extracted, bad ones not.
    assert bool(plan.extract[:N_SAFE_GOOD].all())
    assert not bool(plan.extract[N_SAFE_GOOD : N_SAFE_GOOD + N_SAFE_BAD].any())

    # The deterministic-mean plan: threshold each block on the sample-mean profit of the very same
    # 50-scenario draw the stochastic plan saw, ignoring the distribution around that mean entirely.
    sample_mean_profit = PRICE * plan.scenarios.mean(axis=0) - cost
    det_extract = sample_mean_profit > 0.0

    # The fixed-seed draw fools the naive mean-only plan into keeping the (truly losing) risky block,
    # while the CVaR-aware plan — looking at the tail, not just the average — excludes it.
    assert bool(det_extract[-1]) is True
    assert bool(plan.extract[-1]) is False

    ev_stochastic, cvar_stochastic = _held_out_cvar_and_ev(posterior, plan.extract, cost)
    ev_det, cvar_det = _held_out_cvar_and_ev(posterior, det_extract, cost)

    assert ev_stochastic >= ev_det - 1e-9
    assert cvar_stochastic < cvar_det


def test_cvar_epigraph_shapes_and_constraint():
    rng = np.random.default_rng(1)
    losses = rng.normal(size=(20, 4))
    alpha = 0.9
    c_add, a_ub_rows, b_ub, var_index = cvar_epigraph(losses, alpha)

    n, k = 4, 20
    assert var_index == n
    assert c_add.shape == (n + 1 + k,)
    assert a_ub_rows.shape == (k, n + 1 + k)
    assert b_ub.shape == (k,)
    assert c_add[var_index] == pytest.approx(1.0)
    assert np.allclose(c_add[var_index + 1 :], 1.0 / ((1.0 - alpha) * k))

    # For any x, the row for scenario m must encode losses[m] @ x - eta - u_m <= 0.
    x = rng.normal(size=n)
    eta = 0.3
    u = np.maximum(0.0, losses @ x - eta)
    full = np.concatenate([x, [eta], u])
    assert np.all(a_ub_rows @ full <= b_ub + 1e-9)


def test_cvar_epigraph_rejects_bad_alpha():
    with pytest.raises(ValueError):
        cvar_epigraph(np.zeros((3, 2)), 1.5)
    with pytest.raises(ValueError):
        cvar_epigraph(np.zeros((3, 2)), 0.0)


def test_cvar_epigraph_rejects_zero_scenarios():
    # coef = 1 / ((1 - alpha) * K) divides by K with no guard -- K=0 must raise, not ZeroDivisionError.
    with pytest.raises(ValueError):
        cvar_epigraph(np.zeros((0, 2)), 0.95)


class _ConstantPosterior:
    """A `Posterior` whose scenarios are a fixed per-item value repeated K times."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def samples(self, n: int, rng) -> np.ndarray:
        return np.tile(self._values, (n, 1))

    @property
    def mean(self) -> np.ndarray:
        return self._values

    @property
    def cov(self) -> np.ndarray:
        return np.eye(self._values.size)

    def credible_interval(self, level: float):
        return self._values, self._values

    def derived_quantity(self, fn, n, rng):
        raise NotImplementedError


def test_cvar_epigraph_rejects_non_finite_losses():
    # MXR-080-1721: only rank and alpha were checked, so [[NaN]] produced an ordinary-looking
    # constraint row containing NaN -- a row no solver can interpret and no caller stated.
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            cvar_epigraph(np.array([[bad]]), 0.9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cost": np.array([np.nan, 1.0])},
        {"cost": np.array([np.inf, 1.0])},
        {"cost": np.ones((2, 3))},  # a matrix must not be flattened into a 6-item vector
        {"cost": np.zeros(0)},
        {"price": np.nan},
        {"price": "1.0"},
        {"k_scenarios": 0},
        {"k_scenarios": True},
        {"k_scenarios": 4.5},
    ],
)
def test_two_stage_plan_rejects_undefined_scenario_and_economic_inputs(kwargs):
    # MXR-080-1721: the MILP was built from whatever came in -- non-finite prices/costs, a
    # `.size`-flattened cost matrix, and Boolean/fractional scenario counts all reached the solver.
    call = {"cost": np.ones(2), "price": 1.0, "k_scenarios": 3, **kwargs}
    with pytest.raises(ValueError):
        two_stage_stochastic_plan(
            _ConstantPosterior(np.full(np.asarray(call["cost"]).size or 1, 5.0)),
            call["cost"],
            call["price"],
            k_scenarios=call["k_scenarios"],
            alpha=0.5,
            rng=np.random.default_rng(0),
        )


def test_reported_risk_describes_the_returned_selection():
    # MXR-080-1723: expected_value was recomputed from the rounded mask but cvar was copied out of
    # the solver's eta/u block, so the two could describe different solutions. Both must now agree
    # with a direct empirical computation over the returned mask.
    posterior = _BlockGradePosterior()
    cost = np.ones(N_BLOCKS)
    alpha = 0.9
    plan = two_stage_stochastic_plan(posterior, cost, PRICE, k_scenarios=50, alpha=alpha, rng=np.random.default_rng(0))

    profit = (PRICE * plan.scenarios - cost[None, :]) @ plan.extract.astype(float)
    loss = -profit
    coef = 1.0 / ((1.0 - alpha) * loss.size)
    direct = min(float(eta + coef * np.maximum(loss - eta, 0.0).sum()) for eta in loss)
    assert plan.cvar == pytest.approx(direct, abs=1e-9)
    assert plan.expected_value == pytest.approx(float(profit.mean()), abs=1e-9)


def test_empty_selection_reports_the_risk_of_selecting_nothing():
    # Every item is a guaranteed loser, so the plan selects nothing: the loss of an empty selection
    # is identically zero and so is its CVaR.
    plan = two_stage_stochastic_plan(
        _ConstantPosterior([0.1, 0.1]),
        np.array([10.0, 10.0]),
        1.0,
        k_scenarios=5,
        alpha=0.5,
        rng=np.random.default_rng(0),
    )
    assert not plan.extract.any()
    assert plan.expected_value == pytest.approx(0.0, abs=1e-12)
    assert plan.cvar == pytest.approx(0.0, abs=1e-12)


def test_module_does_not_claim_recourse_decisions_it_does_not_model():
    # MXR-080-1720: there is one shared decision vector and no scenario-indexed second-stage
    # variables, recourse constraints or information stages anywhere in the program.
    import mixle.stochastic_opt as mod

    doc = " ".join((mod.__doc__ or "").split())
    assert "not a two-stage stochastic program with recourse" in doc
    assert "single-stage* sample-average risk-adjusted selection model" in doc
