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
- Code-execution deserialization is opt-in, per call, as of 0.8.0. Every loader
  whose payload can execute code on load requires the caller to pass the literal
  `True` for its trust flag -- `load_encoded(path, encoder=..., trusted=True)`
  for encoded-data payloads, and the corresponding explicit flags on embedder
  and artifact loads. Truthy stand-ins (`1`, `"true"`, `numpy.True_`) are
  rejected so a config value or environment string cannot silently become
  consent, and the flag is not sticky: nothing a library call does can turn it
  on for a later call. Pass it only for a path you control; the stored integrity
  digest detects truncation and tampering, not a file substituted wholesale.
- Record provenance and applicable license or privacy constraints when data or
  models leave the caller process.

## Dependencies and optional backends

Base dependencies are declared in pyproject.toml; optional backends expand the
attack and supply-chain surface. Pin release environments, review dependency
changes, generate applicable inventories, and test the installed combination.
A missing optional dependency must fail at its boundary without masking a
different nested import failure.

Release security automation audits both the isolated base-wheel environment
and the machine-checked ``all`` runtime-feature union, retaining separate
candidate/wheel-bound CycloneDX inventories. A scoped Bandit gate reviews
Mixle and release-script source for medium-or-higher confidence/severity
findings. Dependency advisories and source analysis are distinct gates.

The secret gate runs Gitleaks against the complete Git history from a
full-depth checkout. The fast source-tree gate additionally recognizes common
cloud, registry, payment, collaboration, JWT, and private-key credential
formats in the current tracked tree. Synthetic redaction fixtures are
allowlisted only by exact value.

Every third-party action in a release workflow is pinned to a full commit
identifier. Its adjacent version comment is review context, not executable
identity.

## Disclosure and response

Use the private reporting channel in ../SECURITY.md. Preserve reproduction
details and affected versions, limit disclosure, create a scoped fix and
regression test, assess downstream consumers, and publish an advisory and
patched release when required.
