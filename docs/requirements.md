# Mixle Core requirements

Document ID: CORE-DOC-REQUIREMENTS-001
Version scope: 0.8.x development
Owner: PRJ-CORE

The durable program requirements live in the Mixle status repository. This
file maps the requirements that directly constrain Core implementation.

| Requirement | Priority | Core acceptance |
| --- | --- | --- |
| REQ-TRACEABILITY | Required | Material behavior links intent, work, change, tests, revision, and remaining gaps. |
| REQ-REPRODUCIBILITY | Required | Randomness, environment, inputs, configuration, and portable hashes are controlled or recorded. |
| REQ-RELEASE-EVIDENCE | Required | No release claim precedes immutable source identity and accepted gate evidence. |
| REQ-DOCUMENTATION-ACCURACY | Required | Documentation separates released, observed, experimental, planned, and unknown behavior. |
| CORE-REQ-FAIL-CLOSED-001 | Required | Invalid shapes, non-finite quantities, malformed identities, and unsupported operations fail before producing a successful scientific verdict. |
| CORE-REQ-COMPATIBILITY-001 | Required | Public API, schema, artifact, and serialization changes preserve supported consumers or include an explicit migration. |
| CORE-REQ-COMPOSITION-001 | Required | Shared primitives remain domain-neutral and compose through declared protocols instead of hidden backend assumptions. |
| CORE-REQ-TEST-DISCOVERY-001 | Required | Both supported Python test filename conventions are collected by ordinary test commands. |

## Verification

Acceptance is claim-specific. Focused positive, negative, boundary, and
compatibility tests provide implementation evidence; hosted release checks
provide the broader supported-environment matrix. A generic passing suite does
not establish a scientific claim unless the evidence maps to its stated
invariant and tolerance.

## Traceability

Pull requests targeting 0.8.0 identify the active release, work ID, change ID,
requirements, focused validation, and limitations. The canonical requirement,
work, change, evidence, capability, and release records are maintained in the
Mixle status repository.
