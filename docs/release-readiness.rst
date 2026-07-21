Release Readiness
=================

This page is the public-release checklist for the core ``mixle`` package. It
does not replace tests or the release manifest; it tells reviewers what
evidence must exist before the branch can be treated as a releasable artifact.

Supported Environment
---------------------

The package metadata declares Python 3.11 and newer. The effective release
matrix is the intersection of that declaration and the optional dependencies
used by a given surface:

* base probability, inference, and data-structure paths should install without
  Torch, JAX, Spark, Dask, MPI, pandas, Arrow, or symbolic packages;
* optional extras should be validated only when their dependencies are present;
* heavy backends must remain guarded so ``import mixle`` works from the base
  install; and
* release notes should record the Python and operating-system combinations
  actually tested.

At minimum, record one base environment and any optional environment that backs
public release claims. For example, a release note that highlights Torch-backed
neural leaves should include Torch import and smoke evidence; a release note
that highlights Spark or Dask should include the relevant backend smoke
evidence or mark that gate as skipped.

Package Artifact Gates
----------------------

Before publishing a core release, record:

* the final version in ``pyproject.toml`` and any exposed ``__version__``;
* ``python -m build`` and ``twine check dist/*``;
* installation from the built wheel in a fresh virtual environment;
* an import sweep over public modules with optional-dependency guards;
* package-data inspection for schemas, templates, or non-Python assets; and
* a resolver check for the base package and documented extras.

Artifact checks should run outside the repository checkout. A source-tree import
can hide missing package data or undeclared dependencies. Keep the wheel path,
virtual-environment path, Python version, and command output in the release
evidence.

Behavior Gates
--------------

The final release candidate should run more than the fast local subset:

* the full test suite against the clean wheel where practical;
* focused numerical tests for any changed model, estimator, latent model, PPL
  route, DOE selector, or capability contract;
* NaN and missing-data stress checks for touched statistical paths;
* examples and notebooks that are linked from public docs; and
* deterministic checks for stochastic tests that gate release confidence.

Numerical gates should include negative evidence as well as success paths:
impossible observations should stay impossible, missing values should preserve
the documented missing-data contract, and optional backends should either match
the NumPy path within tolerance or decline clearly.

Documentation Gates
-------------------

Documentation release evidence should include:

* Sphinx build with warnings treated as errors and no undocumented warning
  suppressions;
* clean-archive Sphinx build from tracked files only;
* regenerated and committed API source pages when modules are added, removed,
  or renamed;
* example execution status recorded through :doc:`example-execution-manifest`;
* current changelog and release notes; and
* explicit migration notes for any removed or renamed public surface.

Review the built navigation as well as the source files. Top-level menu labels
should be stable and user-facing; release-specific information belongs in
release notes, changelog entries, validation records, or previous-release pages.
Generated API pages should support narrative docs, not replace them.

The API reference is generated from many public and implementation modules.
Doctest is not a release gate unless examples have been curated into explicit
doctest blocks; otherwise upstream NumPy/SciPy docstring examples can fail on
version-specific scalar representations without indicating a Mixle behavior
regression.

If ``conf.py`` suppresses a warning class to tolerate inherited or upstream
docstring formatting, release evidence must name that suppression and explain why
it does not hide author-maintained page defects. A fully clean warning run should
remain the target.

Docstrings exposed through autodoc should be reviewed for the same professional
standard as hand-written pages: no local notes, no private paths, no roadmap
labels, no notes-to-self, and no claims that overstate maturity or deployment
authority.

For a documentation update PR, record the exact source pages changed, whether
generated API pages were regenerated or only carried forward, and the command
used for the strict build. If a page describes new behavior without a matching
runtime change in the PR, cite the existing implementation or test that already
supports the claim.

Family Release Gate
-------------------

The core package is the dependency of several sister packages. A public family
release is not ready until the coordinated manifest records the exact commit,
version, resolver result, and cross-package smoke evidence for every package in
the release set.

See :doc:`family-release` for the public-facing family release process:
package roles, version policy, support matrix, co-install gate, publication
order, rollback expectations, and documentation responsibilities.
See :doc:`support-policy` for supported-runtime, compatibility, dependency
bound, and reproducibility expectations.

Readiness States
----------------

Use explicit readiness labels in release evidence:

``ready``
    The gate passed against the final release candidate artifact or checkout.

``needs rerun``
    The gate passed earlier but not against the final candidate.

``skipped``
    The gate was intentionally omitted with a reason and owner.

``blocked``
    The gate could not run because of unavailable hardware, private data,
    external credentials, or another condition outside the source tree.

``not applicable``
    The gate does not apply to the changed surface.

Avoid ambiguous status like "looks good" or "probably fine"; it is not useful
when a release has to be audited later.
