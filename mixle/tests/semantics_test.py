from __future__ import annotations

import dataclasses
import json
import math

import pytest

from mixle.semantics import (
    SEMANTICS_SCHEMA_VERSION,
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


def test_value_spec_direct_construction_rejects_unrecognized_role_strings():
    with pytest.raises(ValueError, match="value role"):
        ValueSpec("bogus", "bogus", "1")


def test_value_spec_direct_construction_enforces_role_rules_for_plain_string_roles():
    # Regression guard: before the fix, a plain string role skipped every role-specific check
    # below via identity comparisons that never matched (e.g. "derived" is not ValueRole.DERIVED),
    # and the FREE/FIXED cases crashed with an incidental AttributeError (`str` has no `.value`)
    # instead of raising the intended ValueError.
    with pytest.raises(ValueError, match="expression and dependencies"):
        ValueSpec("derived", "derived", "1")
    with pytest.raises(ValueError, match="require a prior"):
        ValueSpec("free", "free", "1")
    with pytest.raises(ValueError, match="require a value"):
        ValueSpec("fixed", "fixed", "1")
    derived = ValueSpec("derived", "derived", "1", expression="a + b", dependencies=("a", "b"))
    assert derived.role is ValueRole.DERIVED
    assert derived == ValueSpec.from_record(to_record(derived))
    with pytest.raises(ValueError, match="value role"):
        ValueSpec.from_record({"id": "bogus", "role": "bogus", "unit": "1"})


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


def test_transform_spec_direct_construction_matches_from_record_coercion():
    # Regression guard: before the fix, kind="log" never identity-matched TransformKind.LOG
    # (a plain string is never `is` an enum member), so forward() silently fell through to the
    # affine fallback (scale * value + offset) instead of dispatching to log, and a nonsense
    # string like "bogus" took the same silent fallback instead of being rejected.
    coerced = TransformSpec(kind="log", scale=2, offset=3)
    assert coerced.kind is TransformKind.LOG
    assert coerced.forward(2) == pytest.approx(math.log(2))
    assert coerced == TransformSpec.from_record({"kind": "log", "scale": 2, "offset": 3})
    with pytest.raises(ValueError, match="transform kind"):
        TransformSpec(kind="bogus")
    with pytest.raises(ValueError, match="transform kind"):
        TransformSpec.from_record({"kind": "bogus"})


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


def test_uncertainty_component_direct_construction_rejects_unrecognized_kind_strings():
    # UncertaintyComponent.__post_init__ never inspected `kind` at all, so a plain string sailed
    # through direct construction with no validation while from_record already rejected it.
    with pytest.raises(ValueError, match="uncertainty kind"):
        UncertaintyComponent("u", "bogus", "variance", value=1)
    coerced = UncertaintyComponent("u", "epistemic", "variance", value=1)
    assert coerced.kind is UncertaintyKind.EPISTEMIC


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


def test_quantitative_contracts_reject_nonfinite_values():
    _, _, _, observation = _fixture_contracts()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        dataclasses.replace(observation, measurement_uncertainty=float("nan"))
    posterior = _posterior()
    with pytest.raises(ValueError, match="utility values must be finite"):
        DecisionArtifact(
            "decision",
            ("monitor", "stop"),
            "monitor",
            {"monitor": float("nan"), "stop": 1.0},
            posterior.identity,
            "expected-utility",
        )


def test_extensions_and_trace_events_reject_invalid_identity_fields():
    with pytest.raises(ValueError, match="extension id"):
        CapabilityExtension("", "PRJ-INQUIRY", "input", "output", "development")
    with pytest.raises(ValueError, match="schema URIs"):
        CapabilityExtension("reader", "PRJ-INQUIRY", "", "output", "development")
    with pytest.raises(ValueError, match="sequence"):
        TraceEvent("trace", -1, "read", "a" * 64, {}, "2026-07-15T00:00:00Z")
    with pytest.raises(ValueError, match="RFC 3339"):
        TraceEvent("trace", 0, "read", "a" * 64, {}, "yesterday")


def test_logit_inverse_is_stable_for_extreme_finite_inputs():
    transform = TransformSpec(TransformKind.LOGIT)
    assert transform.inverse(-1000.0) == 0.0
    assert transform.inverse(1000.0) == 1.0


# MXR-080-1707: every record carries schema_version but no constructor checked it, so a record
# declaring an unsupported version was interpreted under the 1.0 rules and given a real identity.
@pytest.mark.parametrize(
    "build",
    [
        lambda v: ConstraintSpec(schema_version=v),
        lambda v: TransformSpec(schema_version=v),
        lambda v: PriorSpec("p", "normal", {"mu": 0}, schema_version=v),
        lambda v: ValueSpec("x", ValueRole.FIXED, "1", value=1, schema_version=v),
        lambda v: LikelihoodSpec("l", "normal", ("o",), schema_version=v),
        lambda v: UncertaintyComponent("u", UncertaintyKind.EPISTEMIC, "variance", value=1.0, schema_version=v),
        lambda v: CapabilityExtension("r", "PRJ", "in", "out", "development", schema_version=v),
        lambda v: TraceEvent("t", 0, "read", "a" * 64, {}, "2026-07-15T00:00:00Z", schema_version=v),
    ],
)
def test_unsupported_schema_versions_are_unreadable_not_current_contracts(build):
    with pytest.raises(ValueError, match="schema_version"):
        build("future/999")
    build(SEMANTICS_SCHEMA_VERSION)  # the supported version still constructs


def test_posterior_and_derived_artifacts_also_reject_unsupported_schema_versions():
    posterior = _posterior()
    with pytest.raises(ValueError, match="schema_version"):
        dataclasses.replace(posterior, schema_version="future/999")
    with pytest.raises(ValueError, match="schema_version"):
        CalibrationArtifact("cal", "a" * 64, "coverage", {"c": 1.0}, schema_version="future/999")
    with pytest.raises(ValueError, match="schema_version"):
        DecisionArtifact(
            "d", ("a", "b"), "a", {"a": 1.0, "b": 0.0}, posterior.identity, "eu", schema_version="future/999"
        )
    with pytest.raises(ValueError, match="schema_version"):
        PredictiveArtifact(
            "f",
            posterior.identity,
            ("x",),
            "b" * 64,
            (UncertaintyComponent("u", UncertaintyKind.EPISTEMIC, "variance", value=1.0),),
            "posterior-predictive",
            schema_version="future/999",
        )


# MXR-080-1708: frozen records retained caller-owned mappings BY REFERENCE, so mutating the original
# mapping afterwards changed the same object's content digest.
def test_a_durable_identity_does_not_change_when_the_caller_mutates_its_source_mapping():
    parameters = {"mu": 0}
    prior = PriorSpec("p", "normal", parameters)
    before = prior.identity
    assert before == "29c20eb51d65e725cf7c1a8fb85f5c375ec44e1c071979f344b6f1e622c0f133"

    parameters["mu"] = 9
    assert prior.identity == before
    assert prior.parameters == {"mu": 0}  # the record owns its own copy

    nested = {"chain": {"depth": 1}}
    posterior = _posterior()
    moved = dataclasses.replace(posterior, diagnostics=nested)
    digest = moved.identity
    nested["chain"]["depth"] = 99
    assert moved.identity == digest
    assert moved.diagnostics == {"chain": {"depth": 1}}


# MXR-080-1709: bounded constraints certified NaN (both ordered comparisons against it are false) and
# empty vectors (the validation loop ran zero times) as admissible.
def test_bounded_constraints_reject_nan_empty_and_non_numeric_values():
    constraint = ConstraintSpec(lower=0, upper=1)
    assert not constraint.accepts(float("nan"))
    assert not constraint.accepts(float("inf"))
    assert not constraint.accepts([])  # a vacuous container satisfies nothing
    assert not constraint.accepts(())
    assert not constraint.accepts(True)  # bool only compares because it subclasses int
    assert not constraint.accepts("0.5")
    assert not constraint.accepts([0.5, float("nan")])
    # negative control: real in-domain values still pass, scalars and vectors alike
    assert constraint.accepts(0.5)
    assert constraint.accepts(0)
    assert constraint.accepts([0.0, 0.5, 1.0])


def test_an_unbounded_allowed_values_constraint_still_admits_its_own_vocabulary():
    constraint = ConstraintSpec(allowed_values=("red", "green"))
    assert constraint.accepts("red")
    assert constraint.accepts(["red", "green"])
    assert not constraint.accepts("blue")
    assert not constraint.accepts([])


def test_a_nan_value_cannot_pass_its_declared_constraint_into_a_value_spec():
    with pytest.raises(ValueError, match="violates"):
        ValueSpec("x", ValueRole.FIXED, "1", constraint=ConstraintSpec(lower=0, upper=1), value=float("nan"))


def test_a_genuinely_different_record_still_gets_a_different_identity():
    # negative control: freezing the digest must not make every record look identical
    a = PriorSpec("p", "normal", {"mu": 0})
    b = PriorSpec("p", "normal", {"mu": 9})
    assert a.identity != b.identity
    assert dataclasses.replace(a, family="lognormal").identity != a.identity
