Test Tiers
==========

The test suite is organized into tiers by *purpose* and *time budget*, so a
contributor can run the right subset for the moment and CI can gate each tier at
a known cost. The named tiers are derived centrally during collection from the
reviewed domain-marker registry. Files do not self-assign execution tiers. The
legacy ``fast`` alias remains available for local compatibility, while hosted
gates use the purpose-named tiers below.

Each tier is a pytest marker declared in ``pyproject.toml``. The latency-bounded
local smoke command is ``python scripts/run_smoke.py``. Do not use a broad
``pytest -m smoke`` invocation as latency evidence: pytest imports every
discovered test module before marker deselection.

.. list-table::
   :header-rows: 1
   :widths: 12 36 22 30

   * - Tier
     - Purpose
     - Budget
     - Status
   * - ``smoke``
     - Import, public-API, and critical-fit-path checks -- "is the package
       fundamentally working?"
     - <= 30 s local, <= 60 s in CI
     - Enforced now -- 4 tests in an explicit one-file manifest.
   * - ``core``
     - Stable base-install correctness (no optional extras).
     - <= 12 min per Python CI job
     - Populated centrally: quick, non-experimental, non-stochastic tests with
       no optional-backend marker.
   * - ``full``
     - All non-optional correctness, including the ``slow`` stochastic,
       integration, and exhaustive tests.
     - <= 90 min
     - Populated centrally: every non-optional, non-benchmark correctness test.
   * - ``optional``
     - Tests that require optional extras or external executables; one job per
       installed backend group.
     - per backend group
     - Usable now. Unlike the other tiers, the collected count depends on which
       extras are installed, so no single number describes it: this tree
       collects roughly 1100 with the ``torch`` extra present and roughly 340
       without it.
   * - ``numerical``
     - Repeated-seed and numerical-stress tests.
     - <= 30 min; scheduled/manual, not a per-commit gate
     - Populated centrally from the ``stochastic`` purpose marker.
   * - ``benchmark``
     - Timing-oriented performance tests. Never mixed with correctness
       assertions except explicit parity gates.
     - performance only
     - Usable now -- 7 tests collected.
   * - ``hardware``
     - Real MPI / multi-process receipts.
     - <= 20 min; scheduled or manually gated
     - Populated centrally from MPI and torchrun tests. No GPU result is
       inferred from this CPU-hosted lane.

Current Status
---------------

All named tiers are populated. ``smoke`` remains an explicit four-test,
collection-light manifest. ``core`` and ``full`` are derived from the central
triage registry, ``optional`` includes every optional-backend test, ``numerical``
includes stochastic stress tests, and ``hardware`` contains the MPI/torchrun
multi-process cases. ``benchmark`` remains timing-only. The smoke surface is
``mixle/tests/smoke-manifest.txt`` and is run without broad collection by
``scripts/run_smoke.py``.

``scripts/run_test_tier.py`` owns hosted marker selection, enforces the documented
hard subprocess deadline, binds the command and duration to the full candidate
commit, and emits a retained JSON receipt. Job-level deadlines also bound setup
or teardown outside pytest. A tier that times out fails; its budget cannot
silently become a target.

Guidance
--------

* **smoke** must stay genuinely fast and dependency-light: it runs on a base
  install with no optional extras, and its job is to fail loudly and quickly when
  something is fundamentally broken (an import cycle, a broken public entry
  point, a critical fit path that no longer converges). Its runner rejects
  directories, globs, repository escapes, duplicate entries, and budgets above
  30 seconds.
* **numerical** and **benchmark** tests are deliberately kept out of the
  per-commit gates. A stochastic assertion that fails one run in fifty does not
  belong in the signal a contributor reads on every push; it belongs in a
  nightly tier where a failure is investigated deliberately.
* **hardware** tests produce real MPI and multi-process receipts. GPU validation
  requires a separately identified GPU runner and is not inferred from this
  CPU-hosted tier.
* A test may carry more than one marker (for example ``smoke`` and
  ``serialization``). The tier answers *when it runs*; the domain markers answer
  *what it covers*.

Every executable tier has both a hard subprocess budget and an enclosing hosted
job deadline. Receipts are retained as workflow artifacts.
