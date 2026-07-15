from __future__ import annotations

import dataclasses
import json
import math

import pytest

from mixle.semantics import (
    CalibrationArtifact,
    CapabilityExtension,
    ConstraintSpec,
    DecisionArtifact,
    LikelihoodSpec,
    ObservationSpec,
    PosteriorArtifact,
    PredictiveArtifact,
    PriorSpec,
    TraceEvent,
    TraceSink,
    TransformKind,
    TransformSpec,
    UncertaintyComponent,
    UncertaintyKind,
    ValueRole,
    ValueSpec,
    canonical_json,
    load_reference_fixture,
    semantic_digest,
    to_record,
)


def _fixture_contracts():
    fixture = load_reference_fixture()
    value = ValueSpec.from_record(fixture["value"])
    likelihood = LikelihoodSpec.from_record(fixture["observation"]["likelihood"])
    observation = ObservationSpec.from_record(fixture["observation"])
    return fixture, value, likelihood, observation


def _posterior(**operational):
    fixture, value, likelihood, observation = _fixture_contracts()
    uncertainty = tuple(
        UncertaintyComponent(f"u-{kind}", UncertaintyKind(kind), "variance", value=index / 10, unit="(kg/s)^2")
        for index, kind in enumerate(fixture["inference"]["required_uncertainty_kinds"], start=1)
    )
    return PosteriorArtifact(
        id="source-rate-posterior",
        values=(value,),
        observations=(observation,),
        likelihood=likelihood,
        method=fixture["inference"]["method"],
        random_seed=fixture["inference"]["random_seed"],
        summary={"mean": 8.0, "std": 1.0},
        uncertainty=uncertainty,
        sample_digest="a" * 64,
        diagnostics={"r_hat": 1.0},
        **operational,
    )


def test_packaged_fixture_round_trips_value_semantics_without_loss():
    fixture, value, _, _ = _fixture_contracts()
    rebuilt = ValueSpec.from_record(json.loads(canonical_json(value, semantic=False)))
    assert rebuilt == value
    assert to_record(value) == fixture["value"]
    assert value.unit == value.prior.unit == "kg/s"


@pytest.mark.parametrize("role", [ValueRole.FREE, ValueRole.LATENT])
def test_unknown_roles_require_one_declared_prior(role):
    with pytest.raises(ValueError, match="require a prior"):
        ValueSpec("unknown", role, "1")


def test_fixed_controlled_derived_and_observed_states_cannot_smuggle_priors():
    prior = PriorSpec("p", "normal", {"mean": 0, "std": 1})
    with pytest.raises(ValueError, match="require a value"):
        ValueSpec("fixed", ValueRole.FIXED, "1")
    with pytest.raises(ValueError, match="cannot declare a prior"):
        ValueSpec("observed", ValueRole.OBSERVED, "1", prior=prior)
    with pytest.raises(ValueError, match="expression and dependencies"):
        ValueSpec("derived", ValueRole.DERIVED, "1")


def test_constraints_reject_invalid_bounds_and_values():
    with pytest.raises(ValueError, match="exceeds"):
        ConstraintSpec(lower=2, upper=1)
    constraint = ConstraintSpec(lower=0, upper=1, lower_inclusive=False)
    assert constraint.accepts(0.5)
    assert not constraint.accepts(0)
    with pytest.raises(ValueError, match="violates"):
        ValueSpec("fixed", ValueRole.FIXED, "1", constraint=constraint, value=2)


@pytest.mark.parametrize(
    ("transform", "natural"),
    [
        (TransformSpec(TransformKind.IDENTITY), 2.0),
        (TransformSpec(TransformKind.LOG), 2.0),
        (TransformSpec(TransformKind.LOGIT, lower=-1, upper=3), 1.0),
        (TransformSpec(TransformKind.AFFINE, scale=-2, offset=4), 1.5),
    ],
)
def test_transforms_round_trip_and_report_the_correct_jacobian(transform, natural):
    transformed = transform.forward(natural)
    assert transform.inverse(transformed) == pytest.approx(natural)
    epsilon = 1e-6
    numerical = abs((transform.forward(natural + epsilon) - transform.forward(natural - epsilon)) / (2 * epsilon))
    assert math.exp(transform.log_abs_det_jacobian(natural)) == pytest.approx(numerical, rel=1e-5)


def test_transform_domains_and_parameters_fail_loudly():
    with pytest.raises(ValueError, match="positive"):
        TransformSpec(TransformKind.LOG).forward(0)
    with pytest.raises(ValueError, match="open transform interval"):
        TransformSpec(TransformKind.LOGIT).forward(1)
    with pytest.raises(ValueError, match="cannot be zero"):
        TransformSpec(TransformKind.AFFINE, scale=0)


def test_posterior_semantic_identity_ignores_backend_job_and_location_only():
    local = _posterior(sample_ref="file:///tmp/samples", backend_id="numpy", job_id="local-1")
    remote = _posterior(sample_ref="s3://bucket/samples", backend_id="jax", job_id="cluster-9")
    assert local.identity == remote.identity
    assert to_record(local)["backend_id"] == "numpy"
    assert canonical_json(local, semantic=False) != canonical_json(remote, semantic=False)
    moved_observation = dataclasses.replace(local.observations[0], data_ref="s3://other/observation")
    moved = dataclasses.replace(local, observations=(moved_observation,))
    assert moved.identity == local.identity
    assert PosteriorArtifact.from_record(to_record(local)) == local


def test_posterior_requires_closed_value_observation_and_likelihood_references():
    fixture, value, likelihood, observation = _fixture_contracts()
    wrong = LikelihoodSpec("wrong", "normal", ("other",))
    with pytest.raises(ValueError, match="exactly its observations"):
        PosteriorArtifact(
            "bad",
            (value,),
            (observation,),
            wrong,
            fixture["inference"]["method"],
            17,
            {"mean": 1},
            (UncertaintyComponent("u", UncertaintyKind.EPISTEMIC, "variance", value=1),),
        )
    assert likelihood.observation_ids == (observation.id,)


def test_uncertainty_components_are_typed_and_have_exactly_one_payload():
    with pytest.raises(ValueError, match="exactly one"):
        UncertaintyComponent("u", UncertaintyKind.NUMERICAL, "error")
    with pytest.raises(ValueError, match="nonnegative"):
        UncertaintyComponent("u", UncertaintyKind.NUMERICAL, "error", value=-1)
    artifact = UncertaintyComponent("u", UncertaintyKind.NUMERICAL, "field", artifact_digest="f" * 64)
    assert artifact.kind is UncertaintyKind.NUMERICAL


def test_predictive_calibration_and_decision_artifacts_retain_posterior_closure():
    posterior = _posterior()
    predictive = PredictiveArtifact(
        "forecast",
        posterior.identity,
        ("source-rate",),
        "b" * 64,
        (UncertaintyComponent("u", UncertaintyKind.EPISTEMIC, "variance", value=1),),
        "posterior-predictive",
        backend_id="numpy",
    )
    calibration = CalibrationArtifact("cal", predictive.content_digest, "coverage", {"coverage_90": 0.91})
    decision = DecisionArtifact(
        "decision",
        ("monitor", "stop"),
        "monitor",
        {"monitor": 2.0, "stop": 1.0},
        posterior.identity,
        "expected-utility",
    )
    assert calibration.target_identity == predictive.content_digest
    assert decision.selected == max(decision.utility, key=decision.utility.get)


def test_extension_and_trace_contracts_are_structural_and_provider_neutral():
    extension = CapabilityExtension(
        "posterior-reader",
        "PRJ-INQUIRY",
        "mixle://schema/posterior-query/1",
        "mixle://schema/posterior-answer/1",
        "development",
    )

    class Sink:
        def emit(self, event):
            self.event = event

    sink = Sink()
    assert isinstance(sink, TraceSink)
    event = TraceEvent(
        "trace",
        0,
        "posterior_read",
        semantic_digest(extension),
        {"owner": extension.owner_project},
        "2026-07-15T00:00:00Z",
    )
    sink.emit(event)
    assert sink.event.semantic_identity == semantic_digest(extension)
