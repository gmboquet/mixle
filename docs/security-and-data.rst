Security and Data
=================

Core Mixle should stay a modeling library. It may fit, score, sample, serialize,
and audit models, but it should not hide private data handling, credential
storage, deployment authority, or high-stakes decision policy inside numerical
helpers.

The core package can support evidence-rich workflows without becoming the
system of record for secrets, raw production data, or approval decisions. Public
documentation should keep those boundaries visible so users know where Mixle
ends and their application governance begins.

Data Boundary
-------------

Examples, tests, notebooks, and docs should use synthetic data, public data with
clear source information, or explicitly cleared derived data. Do not commit
private customer data, unreviewed field observations, credential-bearing logs,
or local absolute paths.

Missing data must remain semantically visible. ``NaN`` inputs represent missing
or undefined observations in the statistical surface; stability fixes should not
silently coerce them into ordinary values unless a documented model explicitly
defines that transformation.

When a guide shows a realistic workflow, prefer small synthetic records that
preserve the relevant structure without copying sensitive values. If an example
needs domain-like names, use synthetic identifiers that cannot be mistaken for
private records. If a model artifact is generated from real data outside the
repository, document the classification and validation status rather than
checking in the raw source data.

Data lineage should be clear enough for review. A fitted artifact or report
should identify whether it came from synthetic data, public data, internal
cleared data, or an external service call. That metadata does not by itself make
the artifact safe, but it prevents accidental promotion of an unknown training
source.

Secrets and Credentials
-----------------------

Core package code must not require API keys, bearer tokens, passwords, private
service URLs, or cloud credentials for base import, tests, or documentation
builds. Optional integrations should accept credentials through reviewed
configuration boundaries and keep base-package behavior usable without them.

Optional integrations should fail closed and explain what is missing. Importing
``mixle`` or building the Sphinx docs should not attempt network access, read a
developer's local credential store, or infer credentials from environment
variables. Examples may show where a caller passes credentials, but the example
should not include real tokens, account identifiers, private endpoints, or
paths into a developer machine.

Documentation examples should prefer explicit non-sensitive values such as
``example-token`` or ``https://example.invalid`` when a credential or endpoint
shape is needed. The surrounding text should make clear that those values are
not operational secrets.

Network and Service Calls
-------------------------

Base documentation builds and examples should be offline by default. A page
that demonstrates a hosted teacher, cloud backend, dataset download, or remote
artifact store should say which call is external, which credential or
configuration object is expected, and how the example behaves when the service
is unavailable.

Do not describe a workflow as "model-backed", "teacher-backed", or "served"
unless the documented path identifies the actual process or endpoint being
called. If the page uses a local stand-in, synthetic teacher, cached payload, or
mock transport, label it explicitly.

Model Artifacts
---------------

Serialized models and registry entries should carry enough provenance to make
them reviewable: package version, source commit when available, training or fit
command, data classification, parameter fingerprint, and validation status.
Artifact helpers should not imply that storing a fitted object is the same as
approving it for production use.

Before publishing artifacts, inspect both metadata and payloads. Provenance can
itself reveal sensitive paths, user names, host names, source-table names, or
service identifiers even when the model parameters are safe to share.

Artifacts should be treated as records, not authority. A registry entry can say
what was fit, when, from which source class, and with which validation result.
The consuming application still decides whether the artifact may serve traffic,
answer locally, escalate, or require human review. Documentation should avoid
phrases that collapse those steps into a single "approved" or "safe" claim.

Decision Boundaries
-------------------

Mixle supports uncertainty, abstention, routing, and decision-support workflows.
Those surfaces must not be documented as autonomous authority for high-stakes
decisions without an application-specific safety case. Keep limitations,
calibration status, escalation policy, and human review requirements visible in
the docs and release evidence.

For task-serving and LLM workflows, distinguish three questions:

* whether a model can produce an answer;
* whether the answer is calibrated, grounded, or covered by a verifier; and
* whether the application is allowed to act on that answer.

Mixle can help with the first two questions. The third belongs to the
application owner and should be documented as such.

For public examples, prefer abstention, review, or escalation language over
automatic-action language unless the action policy is itself the subject of a
reviewed application safety case.

Review Checklist
----------------

Before public release, inspect:

* examples and fixtures for private data or local paths;
* generated reports for credentials, account identifiers, and private URLs;
* optional integrations for guarded imports and explicit configuration;
* artifact metadata for provenance and validation status;
* serialized payloads for embedded source snippets or private metadata;
* missing-data behavior for ``NaN`` and impossible observations; and
* docs pages for claims that overstate maturity, deployment status, or
  decision authority.

If any of these checks finds a problem, fix the source or record the finding in
release evidence. Do not rely on the absence of committed private data as proof
that generated artifacts, screenshots, notebooks, or logs are safe; inspect the
actual files that will ship.

Documentation Claims
--------------------

Security-sensitive documentation should prefer precise operational claims:

* "records provenance" rather than "approves";
* "escalates when uncertain" rather than "guarantees safety";
* "requires a reviewed credential boundary" rather than "connects securely";
* "validated on this environment" rather than "supported everywhere".

These distinctions keep the core package's claims precise. Mixle can provide
evidence and guardrails, but the application that handles real data owns
authorization, retention, access control, and final action policy.
