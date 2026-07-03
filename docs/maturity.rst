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
     - Stochastic-process families and temporal/event models.
   * - ``mixle.models``
     - Incubating applied helpers
     - Neural leaves, language-model helpers, Gaussian processes, random
       forests, graph models, induced grammars, POMDPs, and truncated DPM
       helpers. These objects do not all share the maturity or exact contract
       coverage of ``mixle.stats``.
   * - ``mixle.task`` and ``mixle.reason``
     - Active application/research workflows
     - Task distillation, LLM uncertainty, semantic entropy, cascades,
       extraction, graph-producing LLMs, evidence fusion, and reasoning
       workflows.
   * - ``mixle.doe`` and ``mixle.evolve``
     - Active application/research workflows
     - Scientific design, Bayesian optimization, model-improvement loops, and
       anti-regression experiments.
   * - ``mixle.inference.production``
     - Practical helpers, not a platform
     - Provenance headers, filesystem registries, scoring wrappers, activity
       logs, and drift reports. Treat these as building blocks around a fitted
       model, not as a full deployment system.

What Is Safe To Build On First
------------------------------

For ordinary work, start with:

* ``mixle.stats`` for model families and estimators;
* ``mixle.inference.optimize`` for fitting;
* ``mixle.describe`` to inspect what the fitted object supports;
* ``mixle.enumeration`` and ``mixle.ops`` only after checking capabilities.

That path exercises the oldest and most coherent part of the codebase.

How To Treat ``mixle.models``
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
