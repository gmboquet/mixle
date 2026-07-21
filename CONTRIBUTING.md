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

Python 3.11+ is required; the release matrix tests Python 3.11 and 3.12.

## Running tests

```sh
# Local work: select the affected node and enforce a 30-second deadline.
perl -e 'alarm shift; exec @ARGV' 30 python -m pytest -q -n0 -m "" path/to/some_test.py::test_name

# Hosted checks run the broader fast, full, packaging, and environment matrix.
python -m pytest -m fast -n auto
python -m pytest -m "not optional and not benchmark" -n auto
```

New tests default to the `fast` gate automatically (see `mixle/tests/conftest.py`) unless they need a
heavier tag (`slow`, `optional`, `torch`, `numba`, `jax`, `benchmark`, ...). Local
diagnostics must remain narrowly selected and terminate at 30 seconds; broader suites belong in hosted
checks.

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
- Resolve the active target from the Mixle status repository. Current 0.8.0 work targets
  `release/0.8.0` and milestone `0.8.0`.
- Update `CHANGELOG.md`'s `[0.8.0] — Unreleased` section for any user-visible change (new public API, fixed
  bug, behavior change). Purely internal refactors with no visible effect don't need an entry.
- Required hosted checks must be green before merge. Optional-backend and security evidence is required
  when the changed surface or release gate makes it applicable.

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
