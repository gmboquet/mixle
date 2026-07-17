# Mixle Core operations

Document ID: CORE-DOC-OPERATIONS-001
Version scope: 0.8.x development
Owner: PRJ-CORE

Mixle Core is a library, not a hosted service. Applications embedding it own
their deployment, service-level objectives, access control, and incident
response. Core provides primitives that can carry execution, artifact, and
decision evidence.

## Deployment

Build and install an immutable candidate from a recorded source revision.
Declare Python, dependency, optional backend, hardware, precision, and
artifact compatibility. Run a clean-environment import and a focused smoke
test before promotion.

## Observability

Capture fit and inference status, termination reason, warnings, resource use,
backend and precision, seeds, artifact digests, calibration or validity
metrics, and downstream decision outcomes where permitted. Do not emit secrets
or private payloads.

## Budgets

Bound memory, compute, simulator calls, external requests, retries, and wall
time. A budget is part of the scientific and operational configuration.
Exhaustion returns a structured terminal state; it must not be confused with
convergence or success.

## Failure recovery

Preserve the last valid model or checkpoint where the algorithm supports it.
Record partial distributed state, make retries idempotent, and verify artifact
identity before resume. Roll back by restoring the prior package and compatible
artifact/schema set; do not rewrite shared history.

## Support

Supported versions and reporting channels are in ../SECURITY.md, ../README.md,
and the release notes. Optional backend support is bounded by the declared
matrix and evidence, not by the presence of an import adapter.
