# Mixle Core development

Document ID: CORE-DOC-DEVELOPMENT-001
Version scope: 0.8.x development
Owner: PRJ-CORE

## Environment

From the repository root:

~~~sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,lint,docs]"
git config core.hooksPath .githooks
~~~

Python 3.11 or newer is required. The release matrix currently exercises
Python 3.11 and 3.12. Install only the optional extras needed by the changed
surface.

## Focused validation

Local test commands must target the affected behavior and have a hard
30-second process deadline. On macOS or Linux, one portable invocation is:

~~~sh
perl -e 'alarm shift; exec @ARGV' 30 python -m pytest -q -n0 -m '' path/to/file_test.py::test_name
~~~

If a useful test cannot finish within that deadline, stop it, record the
limitation, and rely on the appropriate hosted job. Do not run the full suite
locally during ordinary focused work.

Run style checks only on owned files:

~~~sh
python -m ruff check path/to/changed.py
python -m ruff format --check path/to/changed.py
~~~

## Debugging

Start with the smallest failing node, invalid input, or import boundary. Record
the exact command, Python version, revision, and result. Preserve the original
exception and data; do not broaden exception handling merely to make a test
green.

## Documentation

Update public behavior, contracts, limitations, changelog, and migration notes
in the same change. Sphinx sources live under docs/, while generated API pages
are not the only explanation of supported behavior.

## Change control

Resolve the active release and governed work/change records before editing.
Use an isolated worktree if the primary checkout is dirty. Stage explicit
owned paths, review the staged diff and identity, commit one coherent change,
and open a pull request against the active release branch with the required
metadata.
