Test Tiers
==========

The test suite is organized into tiers by *purpose* and *time budget*, so a
contributor can run the right subset for the moment and CI can gate each tier at
a known cost. The single broad ``fast`` marker is being replaced by these named
tiers; ``fast`` remains as the current per-commit default while the migration
proceeds.

Each tier is a pytest marker declared in ``pyproject.toml``. Select a tier on the
command line with ``-m``, for example ``pytest -m smoke``.

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
     - Usable now -- 4 tests collected.
   * - ``core``
     - Stable base-install correctness (no optional extras).
     - <= 4 min per Python CI job
     - Declared, not yet populated -- 0 tests collected.
   * - ``full``
     - All non-optional correctness, including the ``slow`` stochastic,
       integration, and exhaustive tests.
     - <= 20 min
     - Declared, not yet populated -- 0 tests collected.
   * - ``optional``
     - Tests that require optional extras or external executables; one job per
       installed backend group.
     - per backend group
     - Usable now -- 277 tests collected.
   * - ``numerical``
     - Repeated-seed and numerical-stress tests.
     - nightly (not a per-commit gate)
     - Declared, not yet populated -- 0 tests collected.
   * - ``benchmark``
     - Timing-oriented performance tests. Never mixed with correctness
       assertions except explicit parity gates.
     - performance only
     - Usable now -- 7 tests collected.
   * - ``hardware``
     - Real MPI / GPU / multi-process receipts.
     - scheduled or manually gated
     - Declared, not yet populated -- 0 tests collected.

Current Status
---------------

Three tiers are usable today and select real tests: ``smoke`` (4 tests, all in
``mixle/tests/smoke_test.py``), plus the pre-existing ``optional`` (277 tests) and
``benchmark`` (7 tests) tiers. Verified directly with ``pytest -m <tier>
--collect-only``.

The four tiers T3.1 introduced alongside ``smoke`` -- ``core``, ``full``,
``numerical``, and ``hardware`` -- are declared in ``pyproject.toml`` (so
``--strict-markers`` makes them usable and typo-proof) and documented above with
their intended purpose and budget, but **no test in the suite carries any of
these four markers yet**. Running ``pytest -m core``, ``pytest -m full``,
``pytest -m numerical``, or ``pytest -m hardware`` each currently reports "no
tests collected". This is not an abandoned feature: the module docstring of
``mixle/tests/test_tiers_test.py`` states plainly that T3.1 "does not re-mark the
whole suite (that migration is incremental); it guarantees the vocabulary and the
smoke tier are real and stay in sync." Re-marking the ~7,200 existing tests into
``core``/``full``/``numerical``/``hardware`` is tracked as follow-up work, not a
quick pass -- treat these four names as reserved and typo-proofed, not as usable
subsets, until this note is updated.

Guidance
--------

* **smoke** must stay genuinely fast and dependency-light: it runs on a base
  install with no optional extras, and its job is to fail loudly and quickly when
  something is fundamentally broken (an import cycle, a broken public entry
  point, a critical fit path that no longer converges).
* **numerical** and **benchmark** tests are deliberately kept out of the
  per-commit gates. A stochastic assertion that fails one run in fifty does not
  belong in the signal a contributor reads on every push; it belongs in a
  nightly tier where a failure is investigated deliberately.
* **hardware** tests produce real receipts (MPI reduction equivalence, GPU
  parity, multi-process invariance). They are gated on the hardware being present
  and are scheduled or triggered manually, never assumed in the base gate.
* A test may carry more than one marker (for example ``smoke`` and
  ``serialization``). The tier answers *when it runs*; the domain markers answer
  *what it covers*.

Budgets are targets, not hard limits enforced per-test; they exist so a tier that
grows past its budget is noticed and split rather than silently becoming the new
slow path.
