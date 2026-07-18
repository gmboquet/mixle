Project Maturity
================

``mixle`` is not one uniformly mature surface. The core probability library is
the reliable center of the project, while several applied, neural, task,
reasoning, design, and production-oriented namespaces are still moving quickly.
That does not make those surfaces peripheral. It means they need clearer
expectations, examples, and validation habits.

Use this page to decide where to start and how much verification to require
before depending on a feature. Use :doc:`index` and :doc:`package-map` for the
full scope of what Mixle can do.

Maturity Map
------------

This map has a machine-readable mirror in :mod:`mixle.maturity`: ``maturity_of("mixle.stats.latent")``
returns the tier (``stable`` / ``provisional`` / ``experimental``, matching the deprecation policy in
:doc:`support-policy`) for any dotted module name, resolving by longest prefix. A test keeps the two in
sync, so this table and the registry cannot drift apart.

.. list-table::
   :header-rows: 1

   * - Surface
     - Current status
     - Best use
   * - ``mixle.stats``
     - Stable core
     - Distribution families, estimators, samplers, encoders, combinators,
       mixtures, HMMs, and common Bayesian families.
   * - ``mixle.inference.optimize`` and direct estimation helpers
     - Stable core
     - MLE/EM/conjugate fitting for ordinary distribution and latent-model
       workflows.
   * - ``mixle.semantics``
     - Stable core shared contract
     - Domain-neutral value roles, units, transforms, priors, observations,
       posterior/predictive identity, uncertainty, calibration, and decisions.
   * - ``mixle.enumeration`` and core ``mixle.ops``
     - Usable, evolving
     - Ranking, top-k, support traversal, quantization, conditioning,
       projection, and capability-gated distribution transformations.
   * - ``mixle.ppl``
     - Active development
     - Compact symbolic model expressions that lower to the stats/inference
       layer. Good for experiments, but check the generated model and route.
   * - ``mixle.process``
     - Active development
     - Stochastic-process families, temporal/event models, and CTMCs.
   * - ``mixle.models``
     - Incubating applied helpers
     - Neural leaves, language-model helpers, Gaussian processes, random
       forests, graph models, induced grammars, POMDPs, and truncated DPM
       helpers. These objects do not all share the maturity or exact contract
       coverage of ``mixle.stats``.
   * - ``mixle.task`` and ``mixle.reason``
     - Active application/research workflows
     - Task distillation, LLM uncertainty, semantic entropy, cascades,
       extraction, graph-producing LLMs, typed ontologies, evidence fusion,
       and reasoning workflows.
   * - ``mixle.substrate``, ``mixle.pool``, and ``mixle.telemetry``
     - New local application runtime
     - Provenanced local knowledge stores, action-based reasoners, reusable
       skills, local-or-pool job boundaries, and decision telemetry. Validate
       retrieval, routing, scope, and governance behavior in the target
       application.
   * - ``mixle.scientist``
     - Optional assembled workflow
     - Local scientific reasoning with cached encoders, certified heads, and
       substrate-backed answering. Requires optional heavy dependencies and
       local model weights.
   * - ``mixle.doe`` and ``mixle.evolve``
     - Active application/research workflows
     - Scientific design, Bayesian optimization, model-improvement loops, and
       anti-regression experiments.
   * - ``mixle.blending``, ``mixle.mine_planning``, and ``mixle.pipeline_twin``
     - Active mine-planning workflows (worklist H2, H3, H8)
     - Blend-to-spec LP/MILP with IIS feasibility diagnostics; ultimate-pit
       and time-phased extraction scheduling; and digital-twin, period-stepped
       re-solve simulation of the mine-to-plant-to-customer pipeline.
   * - ``mixle.inference.production``
     - Practical helpers, not a platform
     - Provenance headers, filesystem registries, scoring wrappers, activity
       logs, and drift reports. Treat these as building blocks around a fitted
       model, not as a full deployment system.

What Is Safe to Build on First
------------------------------

For ordinary work, start with:

* ``mixle.stats`` for model families and estimators;
* ``mixle.inference.optimize`` for fitting;
* ``mixle.semantics`` for cross-package values, priors, observations, and results;
* ``mixle.describe`` to inspect what the fitted object supports;
* ``mixle.enumeration`` and ``mixle.ops`` only after checking capabilities.

That path exercises the oldest and most coherent part of the codebase.

Evidence Ladder
---------------

Choose validation depth from the maturity of the surface and the consequence
of the decision:

``exploration``
    A notebook or local experiment can rely on a smoke fit, shape inspection,
    and a small diagnostic plot, provided the result is not described as
    production evidence.

``candidate``
    A model or workflow being compared for real use needs held-out scoring,
    capability checks, deterministic seeds where relevant, and recorded
    assumptions about missing data, optional dependencies, and artifacts.

``promotion``
    A promoted model needs provenance, restart or stability evidence for latent
    models, calibration or drift evidence when probabilities drive decisions,
    and a clear record of rejected alternatives.

``release``
    A public release needs strict Sphinx docs, wheel build and clean install
    evidence, import sweeps, tests/examples/notebooks that match the claimed
    surfaces, and the coordinated release manifest described in
    :doc:`release-readiness` and :doc:`family-release`.

How to Treat ``mixle.models``
-----------------------------

``mixle.models`` is useful, but it is not the conceptual center of the package.
It is a collection of applied helpers that connect specialized model families
to the rest of Mixle when that is practical.

The namespace currently mixes several levels of maturity:

* tested utilities such as Gaussian-process helpers, random graph models, and
  random-forest conditionals;
* neural leaves and compact language-model helpers that are useful for
  experiments but have more moving parts and optional dependencies;
* research-oriented helpers for dependence discovery, induced grammars,
  knowledge graphs, POMDPs, DPMs, training search, and continual learning.

Use ``mixle.models`` when the specialized family is genuinely the right tool.
For ordinary distribution modeling, start with ``mixle.stats`` and
``mixle.inference``. Reach for ``mixle.models`` when the problem specifically
needs a neural leaf, Gaussian process, graph model, grammar, random forest,
DPM, POMDP, or other applied helper.

How Documentation Should Signal Maturity
----------------------------------------

Documentation should make maturity visible without turning every page into a
warning label. Use stable, factual language:

* stable core pages should document contracts, supported shapes, and expected
  failure behavior;
* active-development pages should include route explanations, validation
  requirements, and unsupported combinations;
* incubating pages should name optional dependencies, artifact checks, and
  held-out evidence before deployment claims;
* release notes should distinguish shipped behavior from work still under
  review.

Avoid implying that a surface is production-ready because it appears in the API
reference. Generated reference coverage proves import visibility, not maturity,
calibration, or deployment readiness.

How to Treat Runtime Surfaces
-----------------------------

The substrate, reasoner, pool, telemetry, and scientist layers are application
surfaces. They are valuable because they connect fitted models to knowledge,
skills, evidence, and deployment decisions, but they need application-level
validation:

* check retrieval quality and abstention thresholds with representative
  questions;
* audit scope and sharing behavior before storing sensitive data;
* treat pool placement as a priced decision and keep explicit confirmation for
  billable backends;
* inspect telemetry logs before training learned routing or placement policy;
* reload and re-score neural artifacts after serialization;
* keep local model weights and optional dependencies pinned in deployment
  environments.
