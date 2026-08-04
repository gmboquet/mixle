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
   * - Gaussian, Gamma, Exponential, LogGaussian, Poisson, Geometric, and Categorical distribution
       modules
     - the complete invariant catalog for this allowlist, their per-family interface and recovery
       suites, and ``scipy_golden_test`` density parity
   * - ``mixle.stats.parameter_packing``
     - exact pack/unpack shape, ordering, and round-trip contracts
   * - ``mixle.semantics`` (see :doc:`quantitative-semantics`)
     - the packaged ``fixtures/quantitative-semantics-v1.json`` cross-project contract round-tripping
       without loss, the semantic-identity boundary suite (operational fields -- sample location,
       backend, job -- excluded from identity while priors, units, transforms, constraints and
       observations are not), and the refusal of any ``schema_version`` other than the supported one
   * - Direct MLE / EM / conjugate fitting through ``optimize``
     - the weighted-estimation contract (weights == replicated sufficient statistics), the fit-seed
       determinism suite, and EM monotonicity/quiet-by-default behavior
   * - Base NumPy execution (no optional backend installed)
     - the blocking clean-wheel install + import-sweep job and the base-install optional-import guard
   * - Serialization paths explicitly covered by compatibility tests
     - the five explicitly stable persistence schemas (Gaussian, Poisson, Exponential, Categorical,
       and nested Mixture), their digest-bound 0.7.0 load fixtures and ``0->1`` migrations, the
       dependency-profile schema drift gate, and atomic-write / safe-JSON deployment tests. Other
       registered types are versioned but provisionally persistent.

Not stable
----------

Everything else is explicitly **not** covered by the stable compatibility promise, even where it is useful
and tested:

* **provisional** (usable, may change within a minor release) -- all other ``mixle.stats`` families
  (including combinators, mixtures, HMMs, graphs, processes, and Bayesian catalogs), ``mixle.ppl``, ``mixle.process``,
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
   maturity_of("mixle.stats.univariate.continuous.gaussian")  # -> Maturity.STABLE
   maturity_of("mixle.stats.latent.hidden_markov")            # -> Maturity.PROVISIONAL
   maturity_of("mixle.ppl")                                   # -> Maturity.PROVISIONAL

The stable list and the machine registry are kept consistent by a test, so this page cannot silently claim
more than the registry backs.
