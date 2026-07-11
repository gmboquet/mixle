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
   :widths: 14 46 40

   * - Tier
     - Purpose
     - Budget
   * - ``smoke``
     - Import, public-API, and critical-fit-path checks -- "is the package
       fundamentally working?"
     - <= 30 s local, <= 60 s in CI
   * - ``core``
     - Stable base-install correctness (no optional extras).
     - <= 4 min per Python CI job
   * - ``full``
     - All non-optional correctness, including the ``slow`` stochastic,
       integration, and exhaustive tests.
     - <= 20 min
   * - ``optional``
     - Tests that require optional extras or external executables; one job per
       installed backend group.
     - per backend group
   * - ``numerical``
     - Repeated-seed and numerical-stress tests.
     - nightly (not a per-commit gate)
   * - ``benchmark``
     - Timing-oriented performance tests. Never mixed with correctness
       assertions except explicit parity gates.
     - performance only
   * - ``hardware``
     - Real MPI / GPU / multi-process receipts.
     - scheduled or manually gated

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
