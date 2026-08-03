"""MXR-080-1900: the ``mixle.analysis`` scientific surfaces must not report answers they do not have.

Six independently reproduced defects, all silent -- every one of them produced a number that looked
exactly like a real result:

1. ``emissions.climate_terms`` reported a definite ``water_feasible=True`` for a water budget that
   carried no shortfall evidence at all (attribute absent, or ``None``), because
   ``getattr(water, "shortfall_m3", 0.0)`` defaulted to a numeric zero. The function's own docstring
   promises missing water evidence is always an explicit unknown.
2. ``emissions.TransitionRiskResult`` accepted a ``scenario_mean`` and a ``ranking`` that contradicted
   the ``samples`` they claim to summarize, so the point estimate and ``credible_interval()``
   described two different distributions.
3. Result records copied ``provenance`` only one level deep, leaving nested containers aliased to the
   caller's own objects -- a receipt that could be rewritten after it was reported.
4. A Boolean reached a numeric physical slot -- ``exposure=True`` priced as a 1 mg/kg-day dose,
   ``{"diesel_L": True}`` priced as one litre -- because ``float(True) == 1.0``.
5. ``health_risk.ExceedanceReport`` and ``epidemiology.CohortAttribution`` admitted impossible
   probabilities and shapes: a ``prob_exceed`` of ``5.0``, a ``false_alarm_target`` of ``7.5``, arrays
   of mismatched length, a negative hazard ratio, an attributable fraction above 1.
6. Bayesian risk and real-option paths accepted fewer posterior draws than the caller requested and
   reported the result as if it were the full-strength one.

Every test below names the behaviour it reproduces. Negative controls are included wherever the fix
adds a guard, because the dominant defect class in this package is a guard that refuses states the
library legitimately produces.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mixle.analysis.carcinogenic_risk import SlopeFactor, excess_lifetime_cancer_risk, radon_wlm_risk
from mixle.analysis.developmental_risk import BMDResult, rfd_exceedance
from mixle.analysis.emissions import (
    EmissionFactors,
    Footprint,
    TransitionRiskResult,
    climate_terms,
    emissions_footprint,
    transition_risk,
)
from mixle.analysis.epidemiology import CohortAttribution
from mixle.analysis.health_risk import ExceedanceReport
from mixle.analysis.max_stable import SmithMaxStable, SmithMaxStableFit
from mixle.analysis.real_options import voi_estimate
from mixle.analysis.sdm import HabitatModel

FOOTPRINT = Footprint(scope1=1.0, scope2=2.0, scope3=3.0, total=6.0)


class _ShortPosterior:
    """An IC-1 ``Posterior`` that thins its chain: asked for ``n`` draws, it delivers ``n // 4``.

    A perfectly ordinary implementation detail (a thinned MCMC chain, a rejection filter, a cached
    draw set) -- and the protocol's promise of exactly ``n`` draws was never enforced by any caller
    in ``mixle.analysis`` outside ``valuation``.
    """

    def __init__(self, d: int = 1, *, honour: int | None = None):
        self.d = d
        self.honour = honour  # a request for exactly this many draws is delivered in full

    def samples(self, n, rng):
        delivered = n if n == self.honour else max(n // 4, 1)
        return np.abs(rng.normal(size=(delivered, self.d)))

    @property
    def mean(self):
        return np.zeros(self.d)

    @property
    def cov(self):
        return np.eye(self.d)

    def credible_interval(self, level):
        return np.zeros(self.d), np.ones(self.d)

    def derived_quantity(self, fn, n, rng):
        pushed = np.asarray(fn(self.samples(n, rng)), dtype=float)
        return SimpleNamespace(
            samples=pushed,
            prior_dominated=False,
            credible_interval=lambda level: (
                np.quantile(pushed, (1.0 - level) / 2.0, axis=0),
                np.quantile(pushed, 1.0 - (1.0 - level) / 2.0, axis=0),
            ),
        )


class _ExactPosterior(_ShortPosterior):
    """Negative control: a conforming posterior that always delivers exactly what was requested."""

    def samples(self, n, rng):
        return np.abs(rng.normal(size=(n, self.d)))


# --- 1. missing water evidence must not read as feasible ------------------------------------------


class _NoShortfallAttribute:
    """A ``WaterBudget``-shaped summary object that carries no shortfall evidence at all."""


def test_absent_shortfall_attribute_is_unknown_not_feasible():
    # Reproduced: `climate_terms(fp, <object with no shortfall_m3>, carbon_price=1.0)` returned
    # {"water_feasible": True, "water_binding": False, "shortfall_time_fraction": 0.0} -- a definite
    # all-clear on a hard H4 constraint, from an object holding no water evidence whatsoever.
    terms = climate_terms(FOOTPRINT, _NoShortfallAttribute(), carbon_price=1.0)
    assert terms["water_feasible"] is None
    assert terms["water_binding"] is None
    assert terms["shortfall_time_fraction"] is None
    assert terms["carbon_cost"] == 6.0  # carbon never depended on water evidence


def test_none_shortfall_is_unknown_not_feasible():
    # Reproduced: an explicit `shortfall_m3=None` -- the canonical "not computed" marker -- was
    # coerced to 0.0 and reported as a confirmed non-binding constraint.
    terms = climate_terms(FOOTPRINT, SimpleNamespace(shortfall_m3=None), carbon_price=1.0)
    assert terms["water_feasible"] is None
    assert terms["water_binding"] is None
    assert terms["shortfall_time_fraction"] is None


@pytest.mark.parametrize("bad", [True, False, "12.5", object()])
def test_non_numeric_shortfall_is_unknown_not_a_volume(bad):
    # Reproduced for the Booleans: `float(True) == 1.0`, so a flag in the shortfall slot reported a
    # CONFIRMED one-cubic-metre shortfall (binding), and `False` a confirmed all-clear -- both off no
    # measurement. The string case is the same failure in the other direction: `float("12.5")`
    # succeeds and turns configuration text into a physical volume.
    terms = climate_terms(FOOTPRINT, SimpleNamespace(shortfall_m3=bad), carbon_price=1.0)
    assert terms["water_binding"] is None
    assert terms["water_feasible"] is None


def test_real_water_evidence_still_gives_definite_answers():
    # Negative control: the abstention must not swallow evidence that is actually there.
    clear = climate_terms(
        FOOTPRINT,
        SimpleNamespace(shortfall_m3=0.0, demand_m3=10.0, storage=np.full(4, 100.0)),
        carbon_price=1.0,
        water_limit_m3=1_000.0,
    )
    assert clear["water_feasible"] is True and clear["water_binding"] is False
    assert clear["shortfall_time_fraction"] == 0.0

    short = climate_terms(FOOTPRINT, SimpleNamespace(shortfall_m3=250.0), carbon_price=1.0)
    assert short["water_feasible"] is False and short["water_binding"] is True

    over_limit = climate_terms(
        FOOTPRINT, SimpleNamespace(shortfall_m3=0.0, demand_m3=5_000.0), carbon_price=1.0, water_limit_m3=100.0
    )
    assert over_limit["water_feasible"] is False and over_limit["water_binding"] is True


def test_storage_trajectory_is_still_used_when_the_shortfall_scalar_is_missing():
    # Negative control: a real per-step storage trajectory IS water evidence. Abstaining on the
    # missing scalar must not throw away the duration statistic the trajectory genuinely supports.
    terms = climate_terms(FOOTPRINT, SimpleNamespace(storage=np.array([10.0, 0.0, 0.0, 5.0])), carbon_price=1.0)
    assert terms["shortfall_time_fraction"] == pytest.approx(0.5)
    assert terms["water_binding"] is None  # the binding verdict still has no scalar to stand on


# --- 2. transition-risk summaries must agree with the samples they summarize ----------------------


def _consistent_result_kwargs(**overrides):
    samples = np.array([[10.0, 20.0], [30.0, 40.0]])
    base = dict(
        samples=samples,
        prior_dominated=False,
        scenario_mean=samples.mean(axis=0),
        ranking=(1, 0),  # scenario 1 has the higher mean, so it ranks first
        carbon_cost=np.array([0.5, 0.6]),
        provenance={},
    )
    base.update(overrides)
    return base


def test_scenario_mean_contradicting_samples_is_rejected():
    # Reproduced: TransitionRiskResult(samples=[[10,20],[30,40]], scenario_mean=[-999, 999], ...)
    # constructed cleanly. The true column means are [20, 30]; the record then reported a point
    # estimate from `scenario_mean` and an interval from `samples`, describing two distributions.
    with pytest.raises(ValueError, match="per-scenario mean of samples"):
        TransitionRiskResult(**_consistent_result_kwargs(scenario_mean=np.array([-999.0, 999.0])))


def test_ranking_contradicting_scenario_mean_is_rejected():
    # Reproduced: any permutation was accepted, so `ranking=(0, 1)` published scenario 0 as the best
    # option while its own scenario_mean said scenario 1 was.
    with pytest.raises(ValueError, match="best -> worst"):
        TransitionRiskResult(**_consistent_result_kwargs(ranking=(0, 1)))


def test_provenance_carbon_cost_contradicting_the_field_is_rejected():
    # Reproduced: `transition_risk` records the priced carbon cost twice -- as a field and in
    # provenance -- and nothing checked they agreed, so a consumer reading the receipt and one
    # reading the field could get different costs for the same result.
    with pytest.raises(ValueError, match=r"provenance\['carbon_cost'\]"):
        TransitionRiskResult(**_consistent_result_kwargs(provenance={"carbon_cost": [99.0, 99.0]}))


def test_consistent_transition_risk_records_still_construct():
    # Negative control, three ways: a hand-built consistent record, one with an agreeing provenance
    # receipt, and -- most important -- whatever `transition_risk` itself produces, which uses a
    # different (fsum) mean estimator than the check's fast path.
    plain = TransitionRiskResult(**_consistent_result_kwargs())
    assert plain.ranking == (1, 0)

    with_receipt = TransitionRiskResult(**_consistent_result_kwargs(provenance={"carbon_cost": [0.5, 0.6]}))
    assert with_receipt.provenance["carbon_cost"] == [0.5, 0.6]

    rng = np.random.default_rng(0)
    produced = transition_risk(
        Footprint(scope1=1_000.0, scope2=0.0, scope3=0.0, total=1_000.0, provenance={}),
        np.stack([np.full(10, 20.0), np.full(10, 120.0)]),
        npv_samples=rng.normal(loc=1_000_000.0, scale=50_000.0, size=5_000),
        discount=1.0 / 1.08 ** np.arange(10),
    )
    assert produced.scenario_mean[1] < produced.scenario_mean[0]
    assert produced.ranking == (0, 1)


def test_tied_scenario_means_accept_either_ranking_order():
    # Negative control against over-tightening: `np.argsort` gives no stable tie order, so demanding
    # one particular ordering of equal-mean scenarios would refuse rankings the producer emits.
    tied = np.array([[10.0, 10.0], [30.0, 30.0]])
    for ranking in ((0, 1), (1, 0)):
        record = TransitionRiskResult(
            samples=tied,
            prior_dominated=False,
            scenario_mean=np.array([20.0, 20.0]),
            ranking=ranking,
            carbon_cost=np.array([1.0, 1.0]),
            provenance={},
        )
        assert record.ranking == ranking


# --- 3. provenance must be a receipt, not a live view of the caller's dict ------------------------


def test_footprint_provenance_is_deeply_detached():
    # Reproduced: `dict(provenance)` copied only the top level, so mutating a nested list or dict
    # afterwards rewrote what an already-reported footprint claimed it was computed from.
    supplied = {"scopes": [1, 2, 3], "factor_model": {"source": "supplier_table"}}
    footprint = Footprint(1.0, 2.0, 3.0, 6.0, provenance=supplied)

    supplied["scopes"].append(99)
    supplied["factor_model"]["source"] = "tampered"

    assert footprint.provenance["scopes"] == [1, 2, 3]
    assert footprint.provenance["factor_model"]["source"] == "supplier_table"
    assert isinstance(footprint.provenance["scopes"], list)  # concrete types preserved


def test_cohort_attribution_provenance_is_deeply_detached():
    # Same defect on the attribution receipt, where `rng_state` exists precisely so a fit can be
    # replayed -- and could be edited out from under the record after the fact.
    supplied = {"coef": [0.1, 0.2], "rng_state": {"state": {"key": 1}}}
    result = CohortAttribution(
        hazard_ratio=2.0,
        hr_ci=(1.5, 2.7),
        attributable_fraction=0.5,
        af_ci=(0.3, 0.6),
        cif={},
        provenance=supplied,
    )

    supplied["coef"].append(99.0)
    supplied["rng_state"]["state"]["key"] = 999

    assert result.provenance["coef"] == [0.1, 0.2]
    assert result.provenance["rng_state"]["state"]["key"] == 1


def test_habitat_model_provenance_is_deeply_frozen():
    # Reproduced: `HabitatModel.provenance` is documented as an immutable receipt ("a receipt a caller
    # can mutate after the fact is not a receipt"), but `MappingProxyType(dict(...))` sealed only the
    # top level -- a nested dict stayed a live view of the caller's own object.
    supplied = {"nested": {"k": "orig"}, "digests": ["a", "b"]}
    model = HabitatModel(
        beta=np.zeros(2),
        beta_cov=np.eye(2),
        design=np.ones((3, 2)),
        cell_area=np.ones(3),
        species_id="lynx_rufus",
        provenance=supplied,
    )

    supplied["nested"]["k"] = "TAMPERED"
    supplied["digests"].append("c")

    assert model.provenance["nested"]["k"] == "orig"
    assert model.provenance["digests"] == ("a", "b")
    with pytest.raises(TypeError):  # the nested mapping is sealed too, not only the outer one
        model.provenance["nested"]["k"] = "TAMPERED"


def test_transition_risk_provenance_is_deeply_detached():
    supplied = {"carbon_cost": [0.5, 0.6], "notes": {"run": "a"}}
    result = TransitionRiskResult(**_consistent_result_kwargs(provenance=supplied))

    supplied["carbon_cost"][0] = -1.0
    supplied["notes"]["run"] = "b"

    assert result.provenance["carbon_cost"] == [0.5, 0.6]
    assert result.provenance["notes"]["run"] == "a"


# --- 4. a Boolean is not a physical quantity ------------------------------------------------------


def test_boolean_exposure_is_not_a_dose():
    # Reproduced: `excess_lifetime_cancer_risk(True, SlopeFactor(oral_csf=0.01))` returned
    # samples == [0.00995...], i.e. an exposure FLAG priced as a 1 mg/kg-day lifetime average dose,
    # indistinguishable in the result from a real measurement.
    sf = SlopeFactor(oral_csf=0.01)
    with pytest.raises(TypeError, match="Boolean"):
        excess_lifetime_cancer_risk(True, sf)
    with pytest.raises(TypeError, match="Boolean"):
        excess_lifetime_cancer_risk(np.array([True, False]), sf)
    with pytest.raises(TypeError, match="Boolean"):
        radon_wlm_risk(True)


def test_boolean_activity_and_factor_are_not_emission_quantities():
    # Reproduced: emissions_footprint({"diesel_L": True}, EmissionFactors(scope1={"diesel_L": True}))
    # returned total == 1.0 -- one litre of diesel at one unit CO2e per litre, off two flags.
    numeric_factors = EmissionFactors(scope1={"diesel_L": 2.68}, scope2={}, scope3={})
    with pytest.raises(TypeError, match="Boolean"):
        emissions_footprint({"diesel_L": True}, numeric_factors)
    with pytest.raises(TypeError, match="Boolean"):
        emissions_footprint({"diesel_L": 100.0}, EmissionFactors(scope1={"diesel_L": True}, scope2={}, scope3={}))
    with pytest.raises(TypeError, match="Boolean"):
        emissions_footprint(
            {"diesel_L": 100.0},
            EmissionFactors(scope1={"diesel_L": 2.68}, scope2={}, scope3={}, sigma={"diesel_L": True}),
        )


def test_numeric_zero_and_one_are_still_ordinary_quantities():
    # Negative control: only the Boolean TYPE is refused. An integer quantity of 1 -- and a genuine
    # zero -- are legitimate measurements and must keep working.
    sf = SlopeFactor(oral_csf=0.01)
    assert excess_lifetime_cancer_risk(1, sf).samples.size == 1
    assert excess_lifetime_cancer_risk(0.0, sf).samples[0] == 0.0
    assert (
        emissions_footprint({"diesel_L": 1}, EmissionFactors(scope1={"diesel_L": 1}, scope2={}, scope3={})).total == 1.0
    )


# --- 5. result records must not admit impossible probabilities or shapes --------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"prob_exceed": np.array([5.0, -3.0])}, id="outside_unit_interval"),
        pytest.param({"prob_exceed": np.array([0.5, np.nan])}, id="nan_probability"),
        pytest.param({"prob_exceed": np.array([0.5, 0.1, 0.2])}, id="probability_per_alert"),
        pytest.param({"warmed_up": np.array([True])}, id="warm_up_flag_per_alert"),
        pytest.param({"false_alarm_target": 7.5}, id="rate_above_one"),
        pytest.param({"false_alarm_target": 0.0}, id="rate_of_exactly_zero"),
    ],
)
def test_exceedance_report_rejects_impossible_probabilities_and_shapes(kwargs):
    # Reproduced: ExceedanceReport(alerts=[True, False], prob_exceed=[5.0, -3.0, nan],
    # false_alarm_target=7.5, warmed_up=[True]) constructed silently -- alerts whose "probability"
    # was not one, at a "target rate" that was not a rate, for timesteps that did not line up.
    base = dict(alerts=np.array([True, False]), prob_exceed=np.array([0.9, 0.1]), false_alarm_target=0.05)
    base.update(kwargs)
    with pytest.raises(ValueError):
        ExceedanceReport(**base)


def test_exceedance_report_still_accepts_an_unrecorded_warm_up():
    # Negative control: `warmed_up` defaults to an EMPTY array meaning "not recorded", and this
    # package's own tests construct reports that way. The shape check must not refuse them.
    report = ExceedanceReport(alerts=np.array([True]), prob_exceed=np.array([0.9]), false_alarm_target=0.05)
    assert report.warmed_up.size == 0
    assert report.prob_exceed.tolist() == [0.9]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hazard_ratio": -2.0},  # exp(coef) is never negative
        {"hazard_ratio": float("nan")},
        {"attributable_fraction": 4.2},  # (HR - 1) / HR approaches but never exceeds 1
        {"hr_ci": (9.0, 1.0)},  # inverted
        {"hr_ci": (-1.0, 2.0)},  # a hazard-ratio bound is never negative
        {"af_ci": (np.nan, 0.6)},  # half-NaN is not the documented no-evidence marker
        {"af_ci": (0.9, 0.1)},  # inverted
    ],
)
def test_cohort_attribution_rejects_impossible_effect_estimates(kwargs):
    # Reproduced: CohortAttribution(hazard_ratio=-2.0, hr_ci=(9.0, 1.0), attributable_fraction=4.2,
    # af_ci=(nan, nan), ...) constructed silently and was indistinguishable from a real Cox fit.
    base = dict(
        hazard_ratio=2.0,
        hr_ci=(1.5, 2.7),
        attributable_fraction=0.5,
        af_ci=(0.3, 0.6),
        cif={},
        provenance={},
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        CohortAttribution(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"model": None}, id="no_model"),
        pytest.param({"n_locations": -1, "n_pairs": -5}, id="negative_counts"),
        pytest.param({"n_replicates": 0}, id="no_replicates"),
        pytest.param({"n_pairs": 7}, id="pairs_not_n_choose_2"),
        pytest.param({"residual": float("nan")}, id="nan_residual"),
        pytest.param({"residual": -1.0}, id="negative_mean_squared_error"),
        pytest.param({"status": "converged"}, id="undefined_status"),
    ],
)
def test_smith_max_stable_fit_rejects_impossible_records(kwargs):
    # Reproduced: SmithMaxStableFit(model=None, n_locations=-1, n_replicates=0, n_pairs=-5,
    # residual=nan) constructed cleanly and reported `converged is True` -- a record claiming a
    # converged extremal-dependence fit with no model, no data, and no objective value.
    base = dict(
        model=SmithMaxStable(np.eye(2)),
        n_locations=3,
        n_replicates=20,
        n_pairs=3,
        residual=0.01,
    )
    base.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        SmithMaxStableFit(**base)


def test_smith_max_stable_fit_still_accepts_what_the_fitter_produces():
    # Negative control: the minimal legitimate fit (2 locations, 1 pair, an exactly-zero residual --
    # the documented one-pair/one-parameter case) and the "boundary" honesty status must both stand.
    minimal = SmithMaxStableFit(model=SmithMaxStable(np.eye(2)), n_locations=2, n_replicates=5, n_pairs=1, residual=0.0)
    assert minimal.converged is True
    boundary = SmithMaxStableFit(
        model=SmithMaxStable(np.eye(2)), n_locations=4, n_replicates=5, n_pairs=6, residual=0.5, status="boundary"
    )
    assert boundary.converged is False


def test_cohort_attribution_still_accepts_protective_exposures_and_no_bootstrap_evidence():
    # Negative control, two states the producer really emits: a PROTECTIVE exposure (HR < 1) has a
    # legitimately negative attributable fraction, and `(nan, nan)` is the documented marker for the
    # insufficient-bootstrap-evidence case. Neither may be refused.
    protective = CohortAttribution(
        hazard_ratio=0.5,
        hr_ci=(0.3, 0.8),
        attributable_fraction=-1.0,
        af_ci=(float("nan"), float("nan")),
        cif={},
        provenance={},
    )
    assert protective.attributable_fraction == -1.0
    assert np.isnan(protective.af_ci).all()


# --- 6. exact posterior-delivery receipts ---------------------------------------------------------


def test_cancer_risk_rejects_a_short_posterior_delivery():
    # Reproduced: with a posterior that thins 2000 requested draws to 500,
    # `excess_lifetime_cancer_risk(post, sf, n=2000)` returned samples of shape (500,) -- a risk
    # distribution built on a quarter of the requested evidence, with a mean and credible interval
    # that read exactly like the full-strength answer.
    with pytest.raises(ValueError, match="expected exactly n=2000"):
        excess_lifetime_cancer_risk(_ShortPosterior(), SlopeFactor(oral_csf=0.01), n=2000, rng=np.random.default_rng(0))


def test_radon_risk_rejects_a_short_posterior_delivery():
    # Reproduced identically for the radon path: requested n=2000, delivered (500,).
    with pytest.raises(ValueError, match="expected exactly n=2000"):
        radon_wlm_risk(_ShortPosterior(), n=2000, rng=np.random.default_rng(0))


def test_rfd_exceedance_rejects_a_short_derived_quantity_delivery():
    # `rfd_exceedance`'s pushforward already checked its INPUT draw count, so this exercises the
    # other half: a derived_quantity whose returned samples are shorter than the pushforward's own
    # output. `_SampleDerivedQuantity` validates non-empty/1-D/finite but never the count.
    class _ShortOutput(_ExactPosterior):
        def derived_quantity(self, fn, n, rng):
            pushed = np.asarray(fn(self.samples(n, rng)), dtype=float)
            return SimpleNamespace(samples=pushed[: max(n // 4, 1)], prior_dominated=False)

    bmd = BMDResult(bmd=1.0, bmdl=0.5, bmr=0.1, model="loglogistic", dof=3, bmd_se=0.1)
    with pytest.raises(ValueError, match="expected exactly n=100"):
        rfd_exceedance(_ShortOutput(), bmd, n=100, rng=np.random.default_rng(0))


def test_voi_estimate_rejects_a_short_inner_posterior_delivery():
    # Reproduced: the INNER draw was the silent one. `decision_fn(base)` reduces however many draws
    # it is handed to a single float and `centers[i] + inner_scale * (base - mean)` broadcasts at any
    # row count, so a posterior honouring the outer request but thinning the inner one produced a
    # dollar VOI computed on a quarter of the requested evidence -- and `VoiEstimate` carries no draw
    # count, so nothing downstream could detect it.
    with pytest.raises(ValueError, match="n_inner"):
        voi_estimate(
            _ShortPosterior(honour=8),
            lambda draws: float(np.mean(np.maximum(draws, 0.0))),
            {"method": "variance_rescaling_heuristic", "variance_reduction": 0.5},
            rng=np.random.default_rng(0),
            n_outer=8,
            n_inner=256,
        )


def test_voi_estimate_reports_the_outer_shortfall_by_name():
    # The outer shortfall previously surfaced only as an incidental `IndexError: index 2 is out of
    # bounds for axis 0 with size 2` from `centers[i]`, naming neither the posterior nor the count.
    with pytest.raises(ValueError, match="n_outer"):
        voi_estimate(
            _ShortPosterior(),
            lambda draws: float(np.mean(np.maximum(draws, 0.0))),
            {"method": "variance_rescaling_heuristic", "variance_reduction": 0.5},
            rng=np.random.default_rng(0),
            n_outer=8,
            n_inner=256,
        )


def test_gaussian_regime_screen_reports_a_short_probe_instead_of_silently_weakening():
    # The screen's skew/kurtosis thresholds are calibrated to its probe size, so a short probe made
    # it quietly weaker than documented. It warns (rather than raising) because it is a diagnostic:
    # the Gaussian-conjugate path never otherwise samples the posterior.
    with pytest.warns(UserWarning, match="probe draws"):
        with pytest.raises(ValueError):  # the estimator's own receipt still fires afterwards
            voi_estimate(
                _ShortPosterior(),
                lambda draws: float(np.mean(draws)),
                {"method": "variance_rescaling_heuristic", "variance_reduction": 0.5},
                rng=np.random.default_rng(0),
                n_outer=4,
                n_inner=8,
            )


def test_conforming_posteriors_are_untouched_by_the_delivery_receipts():
    # Negative control: a posterior that honours the protocol must be unaffected on every path the
    # receipts were added to.
    rng = np.random.default_rng(0)
    risk = excess_lifetime_cancer_risk(_ExactPosterior(), SlopeFactor(oral_csf=0.01), n=64, rng=rng)
    assert risk.samples.shape == (64,)

    radon = radon_wlm_risk(_ExactPosterior(), n=64, rng=rng)
    assert radon.samples.shape == (64,)

    bmd = BMDResult(bmd=1.0, bmdl=0.5, bmr=0.1, model="loglogistic", dof=3, bmd_se=0.1)
    assert rfd_exceedance(_ExactPosterior(), bmd, n=64, rng=rng).samples.shape == (64,)

    estimate = voi_estimate(
        _ExactPosterior(),
        lambda draws: float(np.mean(np.maximum(draws, 0.0))),
        {"method": "variance_rescaling_heuristic", "variance_reduction": 0.5},
        rng=np.random.default_rng(1),
        n_outer=8,
        n_inner=32,
    )
    assert np.isfinite(estimate.value)
