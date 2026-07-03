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

Linting and Typing
------------------

.. code-block:: sh

   ruff format --check .
   ruff check .
   mypy mixle

The project allows incremental typing in places. Treat existing configuration
in ``pyproject.toml`` as canonical rather than broadening ignores locally.

Documentation Build
-------------------

Update curated guide pages whenever behavior or public workflow changes. Do
not rely on generated API reference pages to explain a feature.

Regenerate API pages after adding, removing, or renaming modules:

.. code-block:: sh

   make -C docs apidoc

Build docs with warnings treated as errors:

.. code-block:: sh

   .venv/bin/sphinx-build -W -b html docs docs/_build/html

Generated HTML lands in ``docs/_build/html``.

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

Adding Public APIs
------------------

When adding a public symbol:

1. export it from the relevant package ``__init__.py`` when it is meant to be
   user-facing;
2. add it to ``__all__``;
3. add a guide-page mention if it changes a workflow;
4. update generated API reference pages when module coverage changes;
5. include tests showing expected behavior.

Release Checklist
-----------------

Before cutting or preparing a release-like change:

* run the relevant tests;
* build docs with ``-W``;
* regenerate API reference pages when module coverage changes;
* check optional dependency behavior for touched integrations;
* verify examples that advertise the changed workflow;
* update version or packaging metadata only as part of an intentional release.
