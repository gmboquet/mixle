"""MXR-080-1596: analysis result records must preserve evidence integrity.

Every public/returned record in ``mixle.analysis`` is a claim about a computation that already
happened -- a footprint against a factor table, a Cox fit against a cohort, a conformal-calibrated
alert series. While those records were plain mutable dataclasses holding writable arrays and shared
dicts, a caller could rewrite any part of one after construction: raise a total without touching its
provenance, flip a ``calibrated`` honesty flag, edit a ``cif`` curve through the array handle it was
built from. The summaries, status flags and provenance then described data the record no longer
held, and nothing anywhere reported a discrepancy.

``carcinogenic_risk.RiskQuantity`` and ``valuation.NPVDistribution`` were fixed first; this file is
the contract for the sibling result types in ``emissions``, ``epidemiology``, ``health_risk`` and
``developmental_risk``. Two properties per record: the fields cannot be rebound, and array fields
are owned read-only copies rather than aliases of whatever the caller passed in.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mixle.analysis.developmental_risk import BMDResult
from mixle.analysis.emissions import EmissionFactors, Footprint, TransitionRiskResult, emissions_footprint
from mixle.analysis.epidemiology import CohortAttribution
from mixle.analysis.health_risk import DoseResponse, ExceedanceReport

FACTORS = EmissionFactors(scope1={"diesel_L": 2.68}, scope2={"grid_kWh": 0.42}, scope3={})
ACTIVITY = {"diesel_L": 100.0, "grid_kWh": 500.0}

FROZEN_RECORDS = [Footprint, TransitionRiskResult, CohortAttribution, DoseResponse, ExceedanceReport, BMDResult]


@pytest.mark.parametrize("record", FROZEN_RECORDS, ids=lambda r: r.__name__)
def test_public_analysis_records_are_frozen(record):
    assert dataclasses.is_dataclass(record)
    assert record.__dataclass_params__.frozen, f"{record.__name__} must be a frozen dataclass"


def test_footprint_fields_cannot_be_rebound_after_construction():
    fp = emissions_footprint(ACTIVITY, FACTORS)
    original = fp.total
    with pytest.raises(dataclasses.FrozenInstanceError):
        fp.total = 999.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        fp.provenance = {}
    assert fp.total == original


def test_footprint_owns_its_provenance_and_interval():
    supplied = {"factor_source": "external_registry"}
    fp = Footprint(1.0, 2.0, 3.0, 6.0, ci=[5.0, 7.0], provenance=supplied)

    # The record took its own copy: editing the caller's dict afterwards cannot retroactively
    # rewrite what the footprint says it was computed from.
    supplied["factor_source"] = "tampered"
    assert fp.provenance["factor_source"] == "external_registry"

    # A list interval is normalized to an immutable (lo, hi) tuple.
    assert fp.ci == (5.0, 7.0)
    assert isinstance(fp.ci, tuple)


def test_transition_risk_result_owns_read_only_arrays():
    samples = np.array([[1.0, 2.0], [3.0, 4.0]])
    means = np.array([2.0, 3.0])
    costs = np.array([0.5, 0.25])
    result = TransitionRiskResult(
        samples=samples,
        prior_dominated=False,
        scenario_mean=means,
        ranking=(1, 0),
        carbon_cost=costs,
        provenance={},
    )

    # No aliasing: mutating the caller's arrays leaves the record's evidence untouched.
    samples[0, 0] = -1e9
    means[0] = -1e9
    costs[0] = -1e9
    assert result.samples[0, 0] == 1.0
    assert result.scenario_mean[0] == 2.0
    assert result.carbon_cost[0] == 0.5

    # And the record's own arrays are write-locked, so the ranking cannot fall out of agreement
    # with the scenario means it was derived from.
    for array in (result.samples, result.scenario_mean, result.carbon_cost):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.ranking = (0, 1)


def test_cohort_attribution_owns_its_cif_curves():
    curve = np.array([0.0, 0.1, 0.4])
    result = CohortAttribution(
        hazard_ratio=2.0,
        hr_ci=(1.5, 2.7),
        attributable_fraction=0.5,
        af_ci=(0.3, 0.6),
        cif={1: curve},
        provenance={"n": 300},
    )

    curve[2] = 99.0
    assert result.cif[1][2] == pytest.approx(0.4)
    assert not result.cif[1].flags.writeable
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.hazard_ratio = 10.0


def test_exceedance_report_honesty_flags_cannot_be_upgraded():
    # `calibrated=False` says the alerts carry no distribution-free false-alarm guarantee. Flipping
    # it to True (or rewriting `alerts`) used to turn a best-effort heuristic into an apparently
    # proven bound with nothing else about the report changing.
    alerts = np.array([True, False])
    report = ExceedanceReport(
        alerts=alerts,
        prob_exceed=np.array([0.9, 0.1]),
        false_alarm_target=0.05,
        calibrated=False,
        warmed_up=np.array([True, True]),
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        report.calibrated = True
    alerts[1] = True
    assert report.alerts.tolist() == [True, False]
    for array in (report.alerts, report.prob_exceed, report.warmed_up):
        assert not array.flags.writeable


def test_dose_response_params_cannot_drift_past_construction_validation():
    # Construction validates the coefficients eagerly; that gate is only meaningful if the validated
    # params are the ones every later response_fn() call rebuilds from.
    params = {"beta": 0.5}
    model = DoseResponse("loglinear", params)

    params["beta"] = -1.0  # would be rejected at construction
    assert model.params["beta"] == pytest.approx(0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.model = "hill"
    assert model.response_fn()(np.array([1.0]))[0] == pytest.approx(1.0 - np.exp(-0.5))
