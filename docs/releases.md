# Mixle Core releases

Document ID: CORE-DOC-RELEASES-001
Version scope: 0.8.x development
Owner: PRJ-CORE

Core uses semantic versioning. Version 0.8.0 is the current development target
and remains unreleased until all applicable gates have accepted evidence.

## Pull requests

Every release-bound pull request targets release/0.8.0, carries milestone
0.8.0, and declares:

~~~text
Target-Release: REL-PRJ-CORE-0.8.0
Release-Milestone: 0.8.0
Release-Scope: included
Work-Id: WORK-YYYYMMDD-NNNN
Change-Id: CHG-YYYYMMDD-NNNN
~~~

Use neutral branch and change names. Commits contain only owned work, required
traceability trailers, and ordinary maintainer identity metadata.

## Candidate checklist

1. Freeze scope, owners, exclusions, target environments, and compatibility.
2. Reconcile work, requirements, changes, gaps, and pull requests to the
   active release.
3. Confirm one version across package metadata, documentation, changelog, and
   candidate artifacts.
4. Pass applicable focused, negative, integration, scientific, lint, type,
   packaging, security, dependency, and supported-environment checks.
5. Review public APIs, schemas, serialization, providers, migrations, and
   rollback.
6. Build an immutable candidate with digest, provenance, dependency inventory,
   and clean-environment install evidence.
7. Obtain accountable approval, tag and publish, then independently verify the
   published artifact and downstream smoke tests.
8. Record monitoring, rollback readiness, lessons learned, and resulting work.

## User-facing records

../CHANGELOG.md records notable changes. docs/migrations/ records required caller
actions and deprecations. Release notes identify supported versions, known
limitations, scientific validity bounds, security notes, and artifact
destinations.

## Rollback

Restore the last supported package and its compatible schemas and artifacts.
If data or artifact migration occurred, follow the recorded reverse order or
stop when rollback is not safe. Preserve the failed candidate and evidence for
diagnosis.
