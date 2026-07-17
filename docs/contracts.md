# Mixle Core contracts

Document ID: CORE-DOC-CONTRACTS-001
Version scope: 0.8.x development
Owner: PRJ-CORE

## Public Python surface

The supported import map is described by docs/package-map.rst and the generated
reference under docs/api/. The root package lazily exposes major namespaces and
must preserve the underlying exception when an existing module has a missing
dependency.

## Probability and inference

Distribution, estimator, sampler, encoder, posterior, engine, and capability
interfaces are defined in mixle.contracts, mixle.stats.compute, and their
owning modules. Capability inspection is preferred over assuming that every
model supports every operation or backend.

## Quantitative semantics

mixle.semantics version 1.0.0 defines portable value roles, constraints,
transforms, priors, observations, uncertainty components, posterior,
predictive, calibration, decision, extension, and trace records. Semantic
identity excludes explicitly operational fields while record serialization
retains them for audit.

Invalid identifiers, non-finite canonical quantities, broken references, and
malformed covariance or calibration inputs are contract violations, not values
to repair silently.

## Artifacts and receipts

Artifact and receipt surfaces live in mixle.task.artifact,
mixle.inference.receipt, mixle.reason.receipt, telemetry, and numerical error
receipts. Callers must record the exact artifact type and schema they exchange;
the word artifact alone does not imply one universal serialization format.

## Compatibility

Core follows semantic versioning. Pre-1.0 minor releases may evolve quickly,
but public changes still require a changelog entry, compatibility analysis,
and migration when callers must change. Private names beginning with an
underscore are implementation details unless a narrower contract explicitly
says otherwise.

Serialized objects, schemas, providers, CLIs, and cross-project fixtures each
need their own compatibility evidence. Successful import is not serialization
or behavioral compatibility.
