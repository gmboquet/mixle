# Contributing to mixle

## Development setup

```sh
git clone https://github.com/gmboquet/mixle.git
cd mixle
python -m venv .venv
source .venv/bin/activate           # or .venv\Scripts\activate on Windows
pip install -e ".[test,lint,docs]"  # add other extras (torch, numba, spark, ...) as needed
git config core.hooksPath .githooks # auto-format + lint-fix staged files before every commit
```

Python 3.11 or 3.12 is required; those are the complete 0.8.0 release matrix.

<!-- BEGIN GENERATED DEVELOPMENT POLICY -->
## Authoritative development policy

This summary is generated from `manifests/development_policy.json`; edit the manifest and rerun
`python scripts/render_contributing_policy.py` rather than changing this block by hand.

Current work targets `release/0.8.0` and milestone `0.8.0`. Automated
dependency updates target the same branch. Retarget both the manifest and Dependabot deliberately
when the release line changes.

Local diagnostics must select the affected node and finish within
30 seconds. Hosted validation owns broader execution.
The execution tiers are `smoke`, `core`, `full`, `optional`, `numerical`, `benchmark`, `hardware`.

Blocking validation:

- ruff format
- ruff check
- import-linter architecture contracts
- typed-core mypy
- purpose-named hosted test tiers

Advisory validation:

- whole-tree mypy

Public API maturity has three levels:

- **stable** — Deprecate before removal and preserve the old behavior for at least two minor releases.
- **provisional** — Usable and tested; signature or defaults may change within a minor release with a changelog entry.
- **experimental** — No compatibility guarantee; stable modules must not depend on it.

Stable deprecations remain functional for at least
2 minor releases after announcement, emit
`DeprecationWarning`, name their replacement/removal release, and ship migration guidance. Genuine
security or data-corruption repairs may fail closed immediately, but must be documented explicitly.
<!-- END GENERATED DEVELOPMENT POLICY -->

## Pull request conventions

- Keep a PR small enough that its purpose is obvious from the file list — one logical change per PR.
- Write the commit/PR title and body around *why*, not just *what*; the diff already shows what
  changed.
- Include a test plan: what you ran, what passed, what you narrowed the test selection to and why.
- Resolve the active target from the authoritative policy manifest; do not infer it from the default
  branch.
- Update `CHANGELOG.md`'s `[0.8.0] — Unreleased` section for any user-visible change (new public API, fixed
  bug, behavior change). Purely internal refactors with no visible effect don't need an entry.
- Required hosted checks must be green before merge. Optional-backend and security evidence is required
  when the changed surface or release gate makes it applicable.

## Reporting bugs / requesting features

Open a GitHub issue. For a bug report, include a minimal reproduction, the mixle version
(`python -c "import mixle; print(mixle.__version__)"`), the Python version, and the full traceback.
For a security vulnerability, see [SECURITY.md](SECURITY.md) instead — do not open a public issue.
