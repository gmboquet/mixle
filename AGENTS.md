---
id: CORE-AGENTS-001
schema_version: 1.0.0
document_version: 1.0.0
status: active
owner_project: PRJ-CORE
effective_at: 2026-07-17T00:00:00Z
reviewed_at: 2026-07-17T00:00:00Z
review_due: 2026-08-17T00:00:00Z
---

# Mixle Core working rules

Mixle Core provides composable probability, inference, execution, evaluation,
and evidence primitives shared across the Mixle portfolio. Domain physics,
data acquisition, deployment orchestration, and project-specific policy belong
in their owning projects.

Before governed work, read and follow
/Users/grantboquet/mixle/status/AGENTS.md. Resolve the active release and task
profile from status instead of inferring them from a branch name. As of this
document version, development targets release 0.8.0 on release/0.8.0.

## Local boundaries

- Preserve unrelated dirty work and use a clean worktree when ownership is
  unclear.
- Keep public behavior composable, typed, provider-neutral, and fail-closed at
  invalid scientific boundaries.
- Do not treat experimental modules, plans, filenames, or generated API pages
  as proof of maturity.
- Keep compatibility, uncertainty, unsupported cases, and remaining gaps
  explicit.
- Local test commands must be narrowly selected and terminated at 30 seconds.
  Broader matrices belong in hosted checks.
- Target the active release branch, include the required release/work/change
  metadata in every pull request, and use neutral branch and change names.
- Commits and identity metadata describe the software change only and contain
  no tool-generation or automated authorship attribution.

## Documentation map

- Project purpose and boundaries: docs/charter.md
- Architecture and failure modes: docs/architecture.md
- Public contracts and compatibility: docs/contracts.md
- Development and testing: docs/development.md and docs/testing.md
- Security and scientific validity: docs/security.md and
  docs/scientific-validity.md
- Operations, releases, and migrations: docs/operations.md,
  docs/releases.md, and docs/migrations/
- User guides and generated API reference: docs/index.rst and docs/api/

Source, tests, immutable revisions, and accepted evidence establish current
capability. Status connects those facts to requirements and release scope.
