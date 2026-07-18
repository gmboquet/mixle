# Mixle Core security

Document ID: CORE-DOC-SECURITY-001
Version scope: 0.8.x development
Owner: PRJ-CORE

The public vulnerability-reporting policy is in ../SECURITY.md. Do not report
a vulnerability through a public issue.

## Threat model

Core processes caller data, serialized model artifacts, optional backend
objects, external data-source configuration, generated expressions, and
distributed execution metadata. Relevant threats include unsafe
deserialization, path traversal, secret leakage, dependency compromise,
resource exhaustion, untrusted code execution, and corrupted or substituted
artifacts.

## Data handling

- Treat caller data and artifact metadata as untrusted at boundaries.
- Do not log secrets, raw credentials, private datasets, or unrestricted model
  payloads.
- Validate paths, sizes, shapes, dtypes, schemas, and digests before use.
- Prefer safe structured formats. Never load an untrusted pickle or executable
  artifact merely because it has a Mixle filename.
- Record provenance and applicable license or privacy constraints when data or
  models leave the caller process.

## Dependencies and optional backends

Base dependencies are declared in pyproject.toml; optional backends expand the
attack and supply-chain surface. Pin release environments, review dependency
changes, generate applicable inventories, and test the installed combination.
A missing optional dependency must fail at its boundary without masking a
different nested import failure.

## Disclosure and response

Use the private reporting channel in ../SECURITY.md. Preserve reproduction
details and affected versions, limit disclosure, create a scoped fix and
regression test, assess downstream consumers, and publish an advisory and
patched release when required.
