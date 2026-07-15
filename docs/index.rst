mixle
=====

.. raw:: html

   <section class="mixle-hero">
     <p class="mixle-eyebrow">Automatic inference for composable models of heterogeneous data</p>
     <p class="mixle-hero-copy">
       Mixle is a Python framework for building probabilistic systems over data
       that is mixed, structured, temporal, neural, or produced by language
       models. Its center is a composable distribution and estimator contract:
       describe the model shape, fit it through a common inference surface, and
       carry the fitted object into scoring, sampling, calibration, audit, and
       deployment workflows.
     </p>
     <p class="mixle-hero-copy">
       The long-term direction is broader than a distribution catalog. Mixle is
       being developed as a unified modeling layer for heterogeneous records,
       mixtures, HMMs, Transformer leaves, task distillation, uncertainty-aware
       LLM systems, local reasoning workflows, design of experiments,
       structured decisions, and production evidence.
     </p>
     <p class="mixle-actions">
       <a class="mixle-button mixle-button-primary" href="quickstart.html">Start with the quickstart</a>
       <a class="mixle-button" href="maturity.html">Read the maturity guide</a>
       <a class="mixle-button" href="api-overview.html">Find an API</a>
     </p>
   </section>

What Mixle Provides
-------------------

Mixle is designed for applications where a single estimator class is not enough.
One workflow may include a structured probability model, a latent state model,
a calibrated decision rule, and a neural or LLM component. Mixle keeps those
pieces connected through explicit model structure and capability-checked
operations.

At the stable center are distribution families, estimators, samplers, encoders,
combinators, mixtures, HMMs, and the ``optimize`` inference entry point. Around
that center are newer workflow layers for automatic model recommendation,
probabilistic-programming expressions, neural leaves, task replacement, LLM
uncertainty, design loops, model evolution, and production metadata. Maturity
is called out where it matters so users can choose the right validation
standard.

Core Principles
---------------

Composition over special cases
    Build larger models from distributional components instead of writing a new
    training loop for every data shape.

Inference from structure
    The estimator or prototype defines the route: direct estimation, EM,
    conjugate updates, gradient fitting, Bayesian objectives, PPL lowering, or
    calibrated task workflows.

Operational uncertainty
    Posterior queries, conformal prediction, semantic entropy, density gates,
    abstention policies, and escalation rules are modeled as part of system
    behavior rather than post-processing.

Inspectable automation
    Automatic estimator selection, model recommendation, and LLM-designed
    specifications expose assumptions, validation checks, confidence gaps, and
    fallback behavior.

Common Workflows
----------------

Fit a heterogeneous probability model
    Start with :doc:`quickstart`, then read :doc:`concepts`,
    :doc:`distributions`, :doc:`stats-structured`, and :doc:`hmms-latent`.

Infer a first model from raw data
    Read :doc:`automatic-inference` and
    :doc:`automatic-modeling-internals`. These pages cover
    ``get_estimator(data)``, prototype-driven fitting, model recommendation,
    validation, and fallback behavior.

Work with neural or language-model components
    Read :doc:`neural-llm`, :doc:`torch-modules`, :doc:`representation`,
    :doc:`task-distillation`, :doc:`task-serving`, and
    :doc:`agentic-task-distillation`.

Add uncertainty to LLM or reasoning systems
    Read :doc:`uncertainty` and :doc:`reasoning-systems` for semantic entropy,
    claim reliability, calibrated abstention, graph-producing LLMs, and
    cross-modal evidence.

Scale, serve, or audit fitted models
    Read :doc:`engines`, :doc:`compute-layer`,
    :doc:`utilities-and-parallelism`, :doc:`data`, :doc:`production`,
    :doc:`lifecycle`, and :doc:`training-at-scale` for MFU/anomaly receipts
    and the pilot-ladder GO/NO-GO staging.

Build a local reasoning workflow
    Read :doc:`reasoning-ecosystem` for substrate storage, skills, reasoner
    actions, pool jobs, telemetry, and the optional ``Scientist`` workflow.

Explore scientific design and structured decisions
    Read :doc:`doe`, :doc:`analysis`, :doc:`evolution`,
    :doc:`relations`, :doc:`operations`, and :doc:`enumeration`.

Minimal Example
---------------

The public fitting surface accepts an estimator, a prototype distribution, or
an estimator inferred from data:

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats import GaussianDistribution, MixtureDistribution
   from mixle.utils.automatic import get_estimator

   reals = [-1.2, -0.9, -1.1, 0.8, 1.2, 1.1]
   rows = [("free", 4), ("paid", 19), ("free", 5), ("paid", 23)]

   proto = MixtureDistribution(
       [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
       [0.5, 0.5],
   )

   fitted_from_shape = optimize(reals, proto, prev_estimate=proto, out=None)

   inferred_estimator = get_estimator(rows)
   fitted_from_data = optimize(rows, inferred_estimator, out=None)

Passing ``proto`` as the model argument gives ``optimize`` the family shape.
Passing the same object as ``prev_estimate`` also uses its parameter values as
the starting point, which is usually what you want for a mixture example. See
:doc:`automatic-inference` for the full route.

Project Direction
-----------------

Mixle is moving toward a single modeling interface for hybrid probabilistic,
neural, task, and reasoning systems. The stable distribution library remains
the foundation. The forward-looking work is the connective tissue: automatic
model design, uncertainty-aware LLM behavior, neural leaves, representation
learning, active design, self-improvement loops, structured decisions, and
production evidence carried through the same inspectable model lifecycle.

The standard is practical: if a system can be described as a composition of
evidence, latent structure, learned components, and calibrated decisions, Mixle
should make it possible to fit, inspect, compare, and deploy that system without
breaking the abstraction apart.

Manual Map
----------

The foundational pages are :doc:`installation`, :doc:`quickstart`,
:doc:`concepts`, :doc:`maturity`, and :doc:`package-map`. The tutorial index in
:doc:`tutorials/index` provides task-sized walkthroughs. The generated
reference under :doc:`api/modules` covers the broad public module surface;
:doc:`api-overview` is the human map for finding the right import.

.. toctree::
   :caption: Start Here
   :hidden:
   :maxdepth: 2

   installation
   maturity
   stable-surface
   what-mixle-is-not
   whats-new-0-6-2
   quickstart
   concepts
   quantitative-semantics
   module-ownership
   package-map
   lifecycle
   capability-lifecycle
   tutorials/index

.. toctree::
   :caption: Core Workflows
   :hidden:
   :maxdepth: 2

   neural-boundary
   neural-llm
   torch-modules
   automatic-inference
   models
   representation
   task-distillation
   task-serving
   bring_your_own_model
   training-at-scale
   agentic-task-distillation
   uncertainty
   reasoning-systems
   reasoning-ecosystem
   hmms-latent
   processes
   automatic-modeling-contract
   automatic-modeling-internals
   cookbook

.. toctree::
   :caption: Release And Validation
   :hidden:
   :maxdepth: 2

   release-readiness
   claim-evidence-ledger
   validation
   scale-out-economics
   test-tiers
   performance-crossover
   benchmark-methodology
   reproduction
   support-policy
   backend-support
   security-and-data
   stability-and-missing-data
   family-release
   release-notes
   example-execution-manifest
   changelog

.. toctree::
   :caption: Reference Guides
   :hidden:
   :maxdepth: 2

   api-overview
   capabilities-contracts
   compute-layer
   distributions
   stats-univariate
   stats-structured
   stats-latent-bayes
   inference
   inference-toolkit
   ppl
   operations
   relations
   engines
   enumeration
   data
   doe
   analysis
   evolution
   production
   utilities-and-parallelism
   experimental-program
   examples
   examples_gallery
   troubleshooting
   glossary
   extending
   development

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 2

   api/modules

.. toctree::
   :caption: Architecture Notes
   :hidden:
   :maxdepth: 2

   design-notes
   frontier-integration-note
   large-module-audit
