The Stable Surface
==================

This is the reviewed, deliberately short list of what 0.8.0 supports as **stable** (worklist A1.3): the
surfaces covered by the compatibility policy in :doc:`support-policy`, whose behavior is pinned by tests and
whose changes follow the deprecation lifecycle. It is short on purpose -- short enough to test exhaustively
and read on one page. Everything not on this list is ``provisional`` or ``experimental`` per the machine
registry in :mod:`mixle.maturity`; a whole namespace is **never** declared stable when only a subset is
mature.

The rule: a surface is stable only with **artifact-level evidence** -- a test or gate that fails if the
behavior regresses. Each row below names that evidence.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Stable surface
     - Evidence (fails on regression)
   * - Core distribution contracts (density, sampler, estimator, encoder; scalar/vectorized agreement)
     - the invariant catalog (``invariant_catalog_test``), the distribution interface suites, and
       ``scipy_golden_test`` for densities against SciPy
   * - Common scalar and multivariate distributions
     - the invariant catalog + per-family suites; parameter-recovery and weighted-estimation contracts
   * - Composite / record / optional / sequence combinators
     - the combinator suites; heterogeneous-record automatic modeling round-trips
   * - Mixtures and the principal HMM path
     - mixture stability + fused-EM suites; the HMM numerical stress panel (long-sequence underflow,
       impossible-observation ``-inf``); parity against ``hmmlearn`` (sunspots flagship) and against
       scikit-learn's ``GaussianMixture`` (fitted-parameter parity)
   * - Direct MLE / EM / conjugate fitting through ``optimize``
     - the weighted-estimation contract (weights == replicated sufficient statistics), the fit-seed
       determinism suite, and EM monotonicity/quiet-by-default behavior
   * - Base NumPy execution (no optional backend installed)
     - the blocking clean-wheel install + import-sweep job and the base-install optional-import guard
   * - Serialization paths explicitly covered by compatibility tests
     - the cross-version load fixtures (0.7.0 artifacts), the serialization schema manifest + drift gate,
       and atomic-write / safe-JSON deployment tests

Not stable
----------

Everything else is explicitly **not** covered by the stable compatibility promise, even where it is useful
and tested:

* **provisional** (usable, may change within a minor release) -- ``mixle.ppl``, ``mixle.process``,
  ``mixle.models`` (neural leaves, GPs, grammars, ...), ``mixle.task`` / ``mixle.reason``,
  ``mixle.enumeration`` / ``mixle.ops`` beyond the capability-gated core, ``mixle.doe`` / ``mixle.evolve``,
  the runtime layers (``mixle.substrate`` / ``mixle.pool`` / ``mixle.telemetry`` / ``mixle.scientist``), and
  ``mixle.inference.production`` (see :doc:`maturity`);
* **experimental** (no compatibility guarantee) -- everything under ``mixle.experimental`` and the
  standalone frontier-training mechanism prototypes (muP, 2:4 sparsity, scaling laws, simulated TP/PP/CP,
  and fault injection). The executable distributed backend and complete-state checkpoint APIs are
  provisional ``mixle.utils.parallel`` surfaces, not stable compatibility promises.

To check a surface's tier programmatically::

   from mixle.maturity import maturity_of
   maturity_of("mixle.stats.latent.hidden_markov")   # -> Maturity.STABLE
   maturity_of("mixle.ppl")                           # -> Maturity.PROVISIONAL

The stable list and the machine registry are kept consistent by a test, so this page cannot silently claim
more than the registry backs.
