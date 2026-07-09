Development
===========

This page is for contributors working in the repository. It covers environment
setup, tests, documentation, optional dependencies, and the expectations for
changes to public modeling behavior.

Environment
-----------

From a clone:

.. code-block:: sh

   python -m venv .venv
   . .venv/bin/activate
   pip install -e ".[test,lint]"

Use all optional dependencies only when you need to exercise integrations:

.. code-block:: sh

   pip install -e ".[all,test,lint]"

Keep optional imports lazy. The base install should remain usable without
Torch, Spark, Dask, MPI, pandas, Arrow, JAX, NetworkX, or symbolic packages.

When debugging optional integrations, record the extra or dependency set used.
A failure under ``.[all]`` is not always a base-package failure, and a passing
base install does not validate Torch, Spark, MPI, JAX, or data-source behavior.

Test Gates
----------

Run the default gate:

.. code-block:: sh

   python -m pytest

Run the broader non-optional suite:

.. code-block:: sh

   python -m pytest -m "not optional and not benchmark"

When touching a distribution family, include focused tests for:

* sampler reproducibility;
* scalar versus vectorized density parity;
* estimator recovery on generated data;
* convergence through ``optimize``;
* capability behavior such as enumeration, posterior queries, or conjugacy.

When touching workflow layers, test the contract that users call:

* PPL changes should cover lowering and the selected fitter route.
* DOE changes should cover selector output, budget accounting, and oracle
  validation.
* Task changes should cover local answer, escalation, calibration, and saved
  artifact behavior.
* Production changes should cover provenance, registry, serving, and drift
  report objects.

Linting and Typing
------------------

.. code-block:: sh

   ruff format --check .
   ruff check .
   mypy mixle

The project allows incremental typing in places. Treat existing configuration
in ``pyproject.toml`` as canonical rather than broadening ignores locally.

Formatting-only changes should be kept separate from behavioral or
documentation changes when possible. This keeps review focused and helps
release notes name what actually changed.

Documentation Build
-------------------

Update curated guide pages whenever behavior or public workflow changes. Do
not rely on generated API reference pages to explain a feature.

Regenerate API pages after adding, removing, or renaming modules:

.. code-block:: sh

   make -C docs apidoc

The generated ``docs/api/*.rst`` files are documentation sources. Include them
in the PR when they change; only ``docs/_build`` is local output.

Build docs with warnings treated as errors:

.. code-block:: sh

   make -C docs html SPHINXOPTS="-W --keep-going"

Generated HTML lands in ``docs/_build/html``. A release-ready build should not
depend on untracked source pages, local ``PYTHONPATH`` state that differs from
the installed package, or undocumented warning suppressions.

Docstrings are part of the public documentation when autodoc imports a module.
Avoid notes-to-self, local file paths, roadmap labels, and private design-note
references in docstrings. If a behavior is experimental, say what is supported
and what raises clearly; do not describe it as an internal work item.

Repository Conventions
----------------------

* Keep distribution families near related support groups under ``mixle.stats``.
* Add abstractions only when several families or workflows need the same
  behavior.
* Prefer capability checks over concrete class checks.
* Keep sample shapes explicit in examples and tests.
* Avoid importing heavy optional packages at module import time.
* Preserve existing serialization and provenance behavior when changing fitted
  model state.
* Preserve missing-data semantics. A numerical-stability fix must not silently
  turn ``NaN`` into an ordinary value unless the model documents that policy.
* Keep generated files predictable. API ``.rst`` files are source; HTML build
  output is not.

Adding Public APIs
------------------

When adding a public symbol:

1. export it from the relevant package ``__init__.py`` when it is meant to be
   user-facing;
2. add it to ``__all__``;
3. add a guide-page mention if it changes a workflow;
4. update generated API reference pages when module coverage changes;
5. include tests showing expected behavior.

When changing an existing public symbol, also check whether serialized artifacts,
registry entries, examples, and tutorials mention its old name or call shape.
Compatibility shims are preferred when users may have stored fully-qualified
class names in artifacts.

Release Checklist
-----------------

Before cutting or preparing a release-like change:

* run the relevant tests;
* build docs with ``-W``;
* regenerate API reference pages when module coverage changes;
* check optional dependency behavior for touched integrations;
* verify examples that advertise the changed workflow;
* update version or packaging metadata only as part of an intentional release.

Do not claim a release is complete from a source-tree-only check. A releasable
change needs artifact evidence: built wheel, fresh install, import sweep, tests,
docs, examples, and any family-package coordination that applies.
