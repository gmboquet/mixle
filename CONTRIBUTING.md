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

Python 3.10+ is required; CI tests against 3.10, 3.11, and 3.12.

## Running tests

```sh
pytest                                        # the fast gate (default addopts): < 30s
pytest -m "not optional and not benchmark"    # the full non-optional suite (what CI's `full` job runs)
pytest -m "optional or torch or numba or jax" # optional-extras tests (needs the matching extras installed)
pytest path/to/some_test.py -n0 -m ""         # a single file, serial, ignoring the fast-gate filter
pytest --cov --cov-report=term-missing        # with coverage
```

New tests default to the `fast` gate automatically (see `mixle/tests/conftest.py`) unless they need a
heavier tag (`slow`, `optional`, `torch`, `numba`, `jax`, `benchmark`, ...) — tag anything that takes
more than a couple seconds or needs an optional dependency, so the default `pytest` invocation stays
fast for everyone.

## Linting and formatting

```sh
ruff format mixle    # auto-format
ruff check mixle      # lint (enforced in CI)
mypy                  # type check (advisory in CI, not yet enforced)
```

The pre-commit hook (once enabled via `git config core.hooksPath .githooks`) runs `ruff format` +
`ruff check --fix` on staged Python files automatically, so most formatting/lint issues never reach a
commit.

## Pull request conventions

- Keep a PR small enough that its purpose is obvious from the file list — one logical change per PR.
- Write the commit/PR title and body around *why*, not just *what*; the diff already shows what
  changed.
- Include a test plan: what you ran, what passed, what you narrowed the test selection to and why.
- Target `main` unless you're told otherwise; release branches (`release/X.Y.Z-*`) are used for
  large in-flight bodies of work and merged back per the checklist in
  [`release-checklists/`](release-checklists/README.md).
- Update `CHANGELOG.md`'s `[Unreleased]` section for any user-visible change (new public API, fixed
  bug, behavior change). Purely internal refactors with no visible effect don't need an entry.
- CI must be green (lint, fast, full) before merge; the `optional` and `security` jobs are informative
  but not currently required.

## Deprecation policy

mixle is pre-1.0 and iterates quickly, but a deprecation should still give users a chance to react
before it becomes a hard break, not just an entry in the changelog they might not read in time:

1. **Deprecate first.** Mark the old API with a `DeprecationWarning` (via `warnings.warn(...,
   DeprecationWarning, stacklevel=2)`) that names the replacement. Document it in the docstring and in
   `CHANGELOG.md` under `Changed` (or a `Deprecated` heading if there are several in one release).
2. **Keep it working for one minor release.** The deprecated path should still function (not just
   warn) for at least one full minor version after the warning is introduced, so a user pinned to a
   slightly-behind version isn't broken by upgrading within the same minor line.
3. **Remove in a later minor release**, with a `CHANGELOG.md` entry under `Removed` naming what was
   removed and what replaced it, and (if the removal is significant) a one-line migration note.
4. **Exception:** a genuine security fix or a bug fix where the "working" behavior was itself unsafe
   (e.g. the `Boolean.coerce` / `conform_record` silent-corruption fixes in 0.7.0) does not have to
   follow this cycle — correctness and safety win over compatibility, but the change must still be
   called out clearly in `CHANGELOG.md` under `Fixed` so nobody is silently surprised.

This project does not yet publish a formal API-stability tier (e.g. "stable" vs. "experimental"
namespaces); until it does, treat everything under `mixle.*` as covered by this policy except names
prefixed with a single underscore, which are implementation details and may change without notice.

## Reporting bugs / requesting features

Open a GitHub issue. For a bug report, include a minimal reproduction, the mixle version
(`python -c "import mixle; print(mixle.__version__)"`), the Python version, and the full traceback.
For a security vulnerability, see [SECURITY.md](SECURITY.md) instead — do not open a public issue.
