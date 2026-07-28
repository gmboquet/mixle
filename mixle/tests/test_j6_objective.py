"""J6 DoD -- economic objective integration, the grand synthesis (notes/exec/workstream-J.md).

Six synthetic blocks, all with identical grade/cost/price so the *only* thing that can move the optimal
plan is a priced liability or a hard constraint -- isolating each effect cleanly:

  * Blocks 0-1 are "high-emission": raising ``carbon_price`` far enough must remove exactly them from
    the optimal plan (the other four stay, since their emissions -- and hence their carbon liability --
    are negligible by comparison).
  * A no-mine polygon (a G9-style hard constraint) enclosing blocks 2-3 must remove exactly those two
    blocks, regardless of the (zero, in this test) carbon price.

Both scenarios are asserted against the zero-liability/zero-constraint baseline plan, which -- with
every block identically profitable -- extracts all six: proving grade/cost/carbon/enviro terms all trade
against each other on the one `risk_adjusted_plan` objective, per the task's Algorithm and DoD text.
"""

from __future__ import annotations

import numpy as np
import pytest

from mixle.analysis.objective import hard_constraints, priced_liabilities
from mixle.reason.posterior_protocol import Posterior
from mixle.stochastic_opt import StochasticPlan, risk_adjusted_plan

N_BLOCKS = 6
PRICE = 10.0
GRADE = 1.0  # identical for every block
COST = 2.0  # identical for every block -> raw per-block profit = PRICE * GRADE - COST = 8.0 everywhere
RAW_PROFIT = PRICE * GRADE - COST

HIGH_EMISSION_IDX = np.array([0, 1])
LOW_EMISSION_IDX = np.array([2, 3, 4, 5])
EMISSIONS = np.array([5.0, 5.0, 0.2, 0.2, 0.2, 0.2])
EXPOSURE = np.zeros(N_BLOCKS)  # health_cost is a no-op in this test; isolates the carbon effect

NO_MINE_IDX = np.array([2, 3])


class _FlatGradePosterior:
    """A minimal IC-1 `Posterior`: every block's grade is ``GRADE`` plus tiny iid scenario noise.

    The noise is small enough that the CVaR term tracks the mean profit closely and does not itself
    drive any block's inclusion/exclusion -- this DoD isolates the liability/constraint wiring, not
    H4's own grade-uncertainty risk aversion (that is H4's DoD, not J6's).
    """

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.clip(GRADE + rng.normal(0.0, 0.01, size=(n, N_BLOCKS)), 0.0, None)

    @property
    def mean(self) -> np.ndarray:
        return np.full(N_BLOCKS, GRADE)

    @property
    def cov(self) -> np.ndarray:
        return np.eye(N_BLOCKS) * 0.01**2

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        return self.mean - 0.1, self.mean + 0.1

    def derived_quantity(self, fn, n, rng):
        s = fn(self.samples(n, rng))

        class _DQ:
            samples = s
            prior_dominated = False

            def credible_interval(self, level):
                a = (1.0 - level) / 2.0
                return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1 - a, axis=0)

        return _DQ()


def _plan(emissions: np.ndarray) -> dict:
    return {"grade": np.full(N_BLOCKS, GRADE), "exposure": EXPOSURE, "emissions": emissions}


def _no_cost(_arr: np.ndarray) -> np.ndarray:
    return np.zeros(N_BLOCKS)


def test_posterior_stub_conforms_to_ic1():
    assert isinstance(_FlatGradePosterior(), Posterior)


def test_priced_liabilities_shape_and_additivity():
    liabilities = priced_liabilities(
        _plan(EMISSIONS), carbon_price=3.0, health_cost=_no_cost, remediation_cost=_no_cost
    )
    assert liabilities["carbon"].shape == (N_BLOCKS,)
    assert np.allclose(liabilities["carbon"], 3.0 * EMISSIONS)
    assert np.allclose(liabilities["remediation"], 0.0)
    assert np.allclose(liabilities["health"], 0.0)
    assert np.allclose(liabilities["total"], liabilities["remediation"] + liabilities["health"] + liabilities["carbon"])
    assert liabilities["carbon_total"] == pytest.approx(float((3.0 * EMISSIONS).sum()))


def test_baseline_plan_extracts_every_block():
    posterior = _FlatGradePosterior()
    cost = np.full(N_BLOCKS, COST)
    baseline_liabilities = priced_liabilities(
        _plan(EMISSIONS), carbon_price=0.0, health_cost=_no_cost, remediation_cost=_no_cost
    )

    baseline_plan = risk_adjusted_plan(
        posterior,
        cost,
        PRICE,
        baseline_liabilities,
        {},
        k_scenarios=50,
        alpha=0.9,
        rng=np.random.default_rng(0),
    )
    assert isinstance(baseline_plan, StochasticPlan)
    assert baseline_plan.extract.shape == (N_BLOCKS,)
    assert baseline_plan.extract.dtype == np.bool_
    assert bool(baseline_plan.extract.all())


def test_raising_carbon_price_removes_exactly_the_high_emission_blocks():
    posterior = _FlatGradePosterior()
    cost = np.full(N_BLOCKS, COST)

    baseline_liabilities = priced_liabilities(
        _plan(EMISSIONS), carbon_price=0.0, health_cost=_no_cost, remediation_cost=_no_cost
    )
    baseline_plan = risk_adjusted_plan(
        posterior, cost, PRICE, baseline_liabilities, {}, k_scenarios=50, alpha=0.9, rng=np.random.default_rng(0)
    )
    assert bool(baseline_plan.extract[HIGH_EMISSION_IDX].all())

    # carbon_price = 3.0: high-emission blocks' liability (3.0 * 5.0 = 15.0) exceeds their raw profit
    # (8.0), while low-emission blocks' liability (3.0 * 0.2 = 0.6) does not -- so only the high-emission
    # blocks should flip from extracted to excluded.
    high_carbon_liabilities = priced_liabilities(
        _plan(EMISSIONS), carbon_price=3.0, health_cost=_no_cost, remediation_cost=_no_cost
    )
    assert RAW_PROFIT - float(high_carbon_liabilities["carbon"][0]) < 0.0
    assert RAW_PROFIT - float(high_carbon_liabilities["carbon"][2]) > 0.0

    high_carbon_plan = risk_adjusted_plan(
        posterior,
        cost,
        PRICE,
        high_carbon_liabilities,
        {},
        k_scenarios=50,
        alpha=0.9,
        rng=np.random.default_rng(0),
    )
    assert not bool(high_carbon_plan.extract[HIGH_EMISSION_IDX].any())
    assert bool(high_carbon_plan.extract[LOW_EMISSION_IDX].all())
    assert high_carbon_plan.expected_value < baseline_plan.expected_value


def test_no_mine_polygon_removes_exactly_its_enclosed_blocks():
    posterior = _FlatGradePosterior()
    cost = np.full(N_BLOCKS, COST)
    liabilities = priced_liabilities(
        _plan(EMISSIONS), carbon_price=0.0, health_cost=_no_cost, remediation_cost=_no_cost
    )

    baseline_plan = risk_adjusted_plan(
        posterior, cost, PRICE, liabilities, {}, k_scenarios=50, alpha=0.9, rng=np.random.default_rng(0)
    )
    assert bool(baseline_plan.extract[NO_MINE_IDX].all())

    no_mine_mask = np.zeros(N_BLOCKS, dtype=bool)
    no_mine_mask[NO_MINE_IDX] = True
    constraints = hard_constraints(no_mine_mask=no_mine_mask)

    no_mine_plan = risk_adjusted_plan(
        posterior, cost, PRICE, liabilities, constraints, k_scenarios=50, alpha=0.9, rng=np.random.default_rng(0)
    )
    assert not bool(no_mine_plan.extract[NO_MINE_IDX].any())
    other_idx = np.setdiff1d(np.arange(N_BLOCKS), NO_MINE_IDX)
    assert bool(no_mine_plan.extract[other_idx].all())


def test_exposure_cap_constrains_selection():
    """A K6-style exposure cap (via `hard_constraints(caps=...)`) forces fewer blocks than the
    unconstrained baseline when every block contributes equally to the capped resource."""
    posterior = _FlatGradePosterior()
    cost = np.full(N_BLOCKS, COST)
    liabilities = priced_liabilities(
        _plan(EMISSIONS), carbon_price=0.0, health_cost=_no_cost, remediation_cost=_no_cost
    )

    # sum(x) <= 3 (a per-block-equal "exposure unit" budget) with 6 identically profitable blocks --
    # exactly 3 must be selected instead of all 6.
    constraints = hard_constraints(caps=[{"coeffs": np.ones(N_BLOCKS), "bound": 3.0}])

    capped_plan = risk_adjusted_plan(
        posterior, cost, PRICE, liabilities, constraints, k_scenarios=50, alpha=0.9, rng=np.random.default_rng(0)
    )
    assert int(capped_plan.extract.sum()) == 3


def test_hard_constraints_rejects_bad_cap_sense():
    with pytest.raises(ValueError):
        hard_constraints(caps=[{"coeffs": np.ones(N_BLOCKS), "bound": 1.0, "sense": "=="}])


# ---------------------------------------------------------------------------
# MXR-080-0106 -- liability assembly must validate economic sign/finiteness and reject malformed
# (non-boolean, non-finite, non-1-D) mask/constraint data instead of silently converting and trusting it.
# ---------------------------------------------------------------------------


def test_priced_liabilities_rejects_negative_carbon_price():
    """A negative carbon "price" would net a negative liability -- subtracted from revenue, that reads
    as a profit *increase*, not a cost."""
    with pytest.raises(ValueError):
        priced_liabilities(_plan(EMISSIONS), carbon_price=-5.0, health_cost=_no_cost, remediation_cost=_no_cost)


def test_priced_liabilities_rejects_negative_remediation_cost_output():
    with pytest.raises(ValueError):
        priced_liabilities(
            _plan(EMISSIONS),
            carbon_price=0.0,
            health_cost=_no_cost,
            remediation_cost=lambda grade: -np.ones_like(grade),
        )


def test_priced_liabilities_rejects_negative_health_cost_output():
    with pytest.raises(ValueError):
        priced_liabilities(
            _plan(EMISSIONS),
            carbon_price=0.0,
            health_cost=lambda exposure: -np.ones_like(exposure),
            remediation_cost=_no_cost,
        )


def test_priced_liabilities_rejects_nan_emissions():
    """A NaN block emission must not be allowed to enter the objective unguarded."""
    bad_emissions = EMISSIONS.copy()
    bad_emissions[0] = np.nan
    with pytest.raises(ValueError):
        priced_liabilities(_plan(bad_emissions), carbon_price=1.0, health_cost=_no_cost, remediation_cost=_no_cost)


def test_priced_liabilities_rejects_nan_cost_model_output():
    """A NaN produced by a caller-supplied cost callable (not just a NaN input) must also be caught."""
    with pytest.raises(ValueError):
        priced_liabilities(
            _plan(EMISSIONS),
            carbon_price=0.0,
            health_cost=_no_cost,
            remediation_cost=lambda grade: np.full_like(grade, np.nan),
        )


def test_hard_constraints_rejects_nan_mask_entry():
    """`np.asarray(mask, dtype=bool)` would silently coerce a NaN entry to a hard ``True`` exclusion
    (``bool(float('nan'))`` is ``True``); malformed non-boolean mask data must be rejected instead."""
    mask = np.zeros(N_BLOCKS)
    mask[1] = np.nan
    with pytest.raises(ValueError):
        hard_constraints(no_mine_mask=mask)


def test_hard_constraints_rejects_fractional_mask():
    """A fractional mask (e.g. a confidence score) is not a hard exclusion and must not be silently
    coerced into one via nonzero-ness."""
    with pytest.raises(ValueError):
        hard_constraints(no_mine_mask=np.array([0.0, 0.5, 1.0]))


def test_hard_constraints_rejects_non_boolean_dtype_mask():
    """An integer 0/1 array is truthy-coercible to bool but is not the documented boolean contract."""
    with pytest.raises(ValueError):
        hard_constraints(no_mine_mask=np.array([0, 1, 0]))


def test_hard_constraints_rejects_non_finite_cap_coeffs():
    with pytest.raises(ValueError):
        hard_constraints(caps=[{"coeffs": np.array([1.0, np.nan]), "bound": 1.0}])


def test_hard_constraints_rejects_non_finite_cap_bound():
    with pytest.raises(ValueError):
        hard_constraints(caps=[{"coeffs": np.ones(N_BLOCKS), "bound": float("inf")}])


def test_hard_constraints_rejects_non_1d_cap_coeffs():
    with pytest.raises(ValueError):
        hard_constraints(caps=[{"coeffs": np.ones((N_BLOCKS, 2)), "bound": 1.0}])


def test_priced_liabilities_and_hard_constraints_accept_legitimate_inputs():
    """Negative control: legitimate nonnegative finite inputs with a real boolean mask must still
    assemble a sensible objective -- the new validation must not reject valid data."""
    liabilities = priced_liabilities(
        _plan(EMISSIONS),
        carbon_price=3.0,
        health_cost=lambda exposure: np.full_like(exposure, 0.5),
        remediation_cost=lambda grade: np.full_like(grade, 1.5),
    )
    expected_total = 1.5 + 0.5 + 3.0 * EMISSIONS
    assert np.allclose(liabilities["total"], expected_total)
    assert liabilities["grand_total"] == pytest.approx(float(expected_total.sum()))

    mask = np.zeros(N_BLOCKS, dtype=bool)
    mask[NO_MINE_IDX] = True
    constraints = hard_constraints(no_mine_mask=mask, caps=[{"coeffs": np.ones(N_BLOCKS), "bound": 3.0}])
    assert constraints["no_mine_mask"].dtype == np.bool_
    assert np.array_equal(constraints["no_mine_mask"], mask)
    assert np.allclose(constraints["caps"][0]["coeffs"], np.ones(N_BLOCKS))
    assert constraints["caps"][0]["bound"] == 3.0


# --------------------------------------------------------------------------------------------------
# MXR-080-1573: priced_liabilities validated emissions and the two callback OUTPUTS, but not the
# `grade`/`exposure` values it fed into those opaque callbacks, and never re-checked the assembled
# `total`/roll-ups. Invalid plan evidence could therefore vanish behind a cost model that ignores or
# saturates its argument, and individually finite costs could overflow into an infinite liability.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["grade", "exposure"])
def test_priced_liabilities_rejects_invalid_plan_evidence_before_calling_cost_models(field, bad_value):
    # audit repro: a plan with a NaN grade/exposure plus callbacks returning zeros used to produce a
    # completely finite, ordinary-looking liability record -- the invalid source evidence disappeared.
    plan = _plan(EMISSIONS)
    values = np.array(plan[field], dtype=float).copy()
    values[0] = bad_value
    plan[field] = values
    with pytest.raises(ValueError, match=field):
        priced_liabilities(plan, carbon_price=1.0, health_cost=_no_cost, remediation_cost=_no_cost)


def test_priced_liabilities_does_not_invoke_cost_models_on_invalid_evidence():
    """The callbacks are opaque user code: they must never see a value already known to be invalid."""
    seen: list[str] = []

    def _recording(name):
        def cost(arr: np.ndarray) -> np.ndarray:
            seen.append(name)
            return np.zeros_like(arr)

        return cost

    plan = _plan(EMISSIONS)
    plan["grade"] = np.full(N_BLOCKS, np.nan)
    with pytest.raises(ValueError):
        priced_liabilities(
            plan, carbon_price=1.0, health_cost=_recording("health"), remediation_cost=_recording("remediation")
        )
    assert seen == []


def test_priced_liabilities_rejects_overflow_in_the_netted_total():
    # three individually finite, individually valid per-block costs whose sum is not representable
    huge = 1e308
    plan = {"grade": np.array([1.0]), "exposure": np.array([1.0]), "emissions": np.array([huge])}
    with pytest.raises(ValueError):
        priced_liabilities(
            plan,
            carbon_price=1.0,
            health_cost=lambda exposure: np.full_like(exposure, huge),
            remediation_cost=lambda grade: np.full_like(grade, huge),
        )


def test_priced_liabilities_rejects_overflow_in_a_scalar_roll_up():
    # every per-block total is representable; only the reduction into a roll-up overflows
    huge = 1e308
    n = 4
    plan = {"grade": np.ones(n), "exposure": np.zeros(n), "emissions": np.zeros(n)}
    with pytest.raises(ValueError):
        priced_liabilities(
            plan,
            carbon_price=0.0,
            health_cost=lambda exposure: np.zeros_like(exposure),
            remediation_cost=lambda grade: np.full_like(grade, huge),
        )


def test_priced_liabilities_still_accepts_negative_grade_and_exposure_proxies():
    """Negative control: grade/exposure are declared arbitrary geology/exposure PROXIES, so a negative
    reading is legitimate data -- only the prices carry the non-negativity convention. The new
    validation must check finiteness only, and must not start rejecting valid negative proxies."""
    plan = {
        "grade": np.array([2.0, -1.0]),
        "exposure": np.array([3.0, -0.5]),
        "emissions": np.array([1.0, 2.0]),
    }
    out = priced_liabilities(
        plan,
        carbon_price=4.0,
        health_cost=lambda exposure: np.abs(exposure) * 2.0,
        remediation_cost=lambda grade: np.abs(grade) * 3.0,
    )
    assert np.allclose(out["remediation"], [6.0, 3.0])
    assert np.allclose(out["health"], [6.0, 1.0])
    assert np.allclose(out["carbon"], [4.0, 8.0])
    assert np.allclose(out["total"], [16.0, 12.0])
    assert out["grand_total"] == pytest.approx(28.0)


def _plain_plan_call(constraints):
    return risk_adjusted_plan(
        _FlatGradePosterior(),
        np.full(N_BLOCKS, COST),
        PRICE,
        {},
        constraints,
        k_scenarios=5,
        alpha=0.9,
        rng=np.random.default_rng(0),
    )


def test_hard_exclusions_require_a_genuine_boolean_mask():
    # MXR-080-1722: the mask was coerced with dtype=bool, which maps every nonzero object to True.
    # The string "False" -- and a NaN, which is truthy -- therefore became a permanent hard
    # exclusion of that item, the exact opposite of what the data says.
    for bad in (
        np.array(["False"] * N_BLOCKS, dtype=object),
        np.full(N_BLOCKS, np.nan),
        np.zeros(N_BLOCKS, dtype=float),
        np.zeros(N_BLOCKS, dtype=int),
    ):
        with pytest.raises(ValueError):
            _plain_plan_call({"no_mine_mask": bad})

    honest = np.zeros(N_BLOCKS, dtype=bool)
    assert bool(_plain_plan_call({"no_mine_mask": honest}).extract.all())


def test_unknown_constraint_and_cap_keys_are_refused_not_ignored():
    # MXR-080-1722: constraint/cap dicts had no closed schema, so a misspelled control was dropped
    # in silence and the caller believed a restriction was in force that never reached the solver.
    with pytest.raises(ValueError):
        _plain_plan_call({"no_min_mask": np.ones(N_BLOCKS, dtype=bool)})
    with pytest.raises(ValueError):
        _plain_plan_call({"caps": [{"coeffs": np.ones(N_BLOCKS), "bound": 3.0, "sence": "<="}]})


def test_cap_geometry_must_be_finite():
    for cap in (
        {"coeffs": np.full(N_BLOCKS, np.nan), "bound": 3.0},
        {"coeffs": np.ones(N_BLOCKS), "bound": np.inf},
        {"coeffs": np.ones((2, N_BLOCKS)), "bound": 3.0},
    ):
        with pytest.raises(ValueError):
            _plain_plan_call({"caps": [cap]})
