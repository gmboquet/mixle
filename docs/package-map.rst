Package Map
===========

``mixle`` is broad because it is meant to keep heterogeneous modeling work in a
single composable environment. The package is easier to navigate if you think of
it as nine layers:

1. the user-facing lifecycle and recommendation surface;
2. cross-project contracts for capability, semantics, and causal claims;
3. composable probability distributions and estimators;
4. inference, enumeration, uncertainty, and operations over those models;
5. applied neural, LLM, task, reasoning, design, and evolution workflows built
   on the same contracts where practical;
6. operations-research building blocks -- blending, precedence scheduling,
   stochastic optimization, network simulation, and forecast-to-dispatch
   routing -- reducing to a shared handful of combinatorial solvers;
7. substrate, skill, pool, telemetry, and system-facade surfaces around local
   applications;
8. compute, serialization, data, and parallel runtime support;
9. optional assembled applications such as ``mixle.scientist``.

The generated API reference is available at :doc:`api/modules`. This page is a
human map for choosing where to start.

The maturity level is not uniform across this map. Core distribution and
inference contracts should be treated as stable unless marked otherwise.
Applied neural, task, reasoning, DOE, evolution, and assembled application
surfaces need explicit validation evidence before they carry production claims.

Top-Level Facade
----------------

Use the top-level package when you want discoverability:

.. code-block:: python

   import mixle

   model = mixle.Model().fit(rows)
   print(model.evaluate(holdout))
   print(mixle.describe(model.fitted))

Important top-level exports include:

``mixle.Model``
    Lifecycle wrapper for fit, evaluation, posterior queries, distillation,
    artifact handling, and explanation. See :doc:`lifecycle`.

``mixle.propose``
    Builds a verified candidate frontier from automatic recommendation, an
    independence baseline, and optionally an LLM-designed model. Treat it as an
    exploratory helper and validate the result on held-out data.

``mixle.describe`` / ``mixle.capabilities`` / ``mixle.supports``
    Capability inspection. Use these before assuming that a model can enumerate,
    condition, marginalize, produce exact densities, expose latent posteriors,
    or run on a backend. See :doc:`capabilities-contracts`.

``mixle.stats``
    High-level distribution and estimator namespace. This is the first import
    location for most probabilistic modeling code.

Cross-Project Contracts and Governance
--------------------------------------

A few modules exist to keep claims -- about identification, capability, or
meaning -- honest across project boundaries, rather than to compute anything
themselves:

.. list-table::
   :header-rows: 1

   * - Namespace
     - Use it for
   * - ``mixle.contracts``
     - Every ABC/Protocol Mixle defines, gathered into one import:
       ``Distribution``, ``Enumerable``, ``Conditionable``, ``Relation``,
       ``ComputeEngine``, ``Surrogate``, ``EMStrategy``, and related roles.
   * - ``mixle.semantics``
     - Domain-neutral contracts for what an uncertain value or inference
       result *means*, independent of solver, backend, or storage location.
   * - ``mixle.causal``
     - Dependency-free schema for causal claims -- estimands, assumptions,
       interventions, counterfactual queries, and identification status --
       so a result cannot silently claim more than it identified.
   * - ``mixle.capability_lifecycle``
     - Governed lifecycle records (evaluation, maturity, availability,
       authorization, corroboration) for a capability shared across
       projects, kept as independent facts rather than one collapsed
       status. See :doc:`capability-lifecycle`.

Maturity is tracked the same way: ``mixle.maturity`` is the machine-readable
registry behind :doc:`maturity` and the generated API manifest, resolving a
dotted module name to a stability tier by longest matching prefix.

Use :doc:`capabilities-contracts` for the capability-protocol narrative guide.

Distribution and Model Objects
------------------------------

``mixle.stats`` contains the composable probability model substrate:

.. list-table::
   :header-rows: 1

   * - Namespace
     - Use it for
   * - ``mixle.stats``
     - Common distribution, estimator, sampler, encoder, and combinator imports.
   * - ``mixle.stats.univariate``
     - Continuous and discrete scalar families.
   * - ``mixle.stats.bayes``
     - Conjugate priors and Bayesian estimators.
   * - ``mixle.stats.compute``
     - The protocol and vectorized compute layer beneath the public families.
   * - ``mixle.models``
     - Incubating applied helpers: neural leaves, Transformer helpers, random
       forests, GPs, graphs, grammars, DPMs, POMDPs, and related utilities.

Use :doc:`distributions`, :doc:`stats-univariate`, :doc:`stats-structured`, and
:doc:`stats-latent-bayes` for the stable family catalog. Use :doc:`models`
when you specifically need an applied helper that is not an ordinary
distribution family.

``mixle.dist`` is a bare re-export of ``mixle.stats`` (``from mixle.stats
import *``), kept as a friendly object-namespace alias from the
concern-oriented reorg. Use whichever reads better at the call site; both
names stay in sync automatically since one is a re-export of the other.

Inference and Probability Operations
------------------------------------

Inference is split by kind of question rather than by data type:

.. list-table::
   :header-rows: 1

   * - Namespace
     - Use it for
   * - ``mixle.inference``
     - ``optimize``, EM, streaming estimation, priors, calibration,
       conformal prediction, MCMC, model comparison, forecasting,
       diagnostics, certified model creation, simulation, verified synthesis,
       reproducibility receipts, skills, placement plans, and UQ dispatch.
   * - ``mixle.enumeration``
     - Top-k, rank, seek, nucleus traversal, HMM paths, assignments, graph
       structures, and quantized support traversal.
   * - ``mixle.ops``
     - Conditioning, marginalization, projections, transforms, mixtures,
       quantization, products of experts, and distribution algebra.
   * - ``mixle.ppl``
     - Symbolic random-variable expressions that lower to Mixle estimators,
       distributions, and inference targets.

The main guides are :doc:`automatic-inference`, :doc:`inference`,
:doc:`inference-toolkit`, :doc:`enumeration`, :doc:`operations`,
:doc:`reasoning-ecosystem`, and :doc:`ppl`.

Applied Neural, LLM, and Task Workflows
---------------------------------------

The neural and LLM-facing layers reuse the same model vocabulary where
practical. They are newer and more uneven than the core distribution layer, but
they are part of the main Mixle story rather than miscellaneous extras:

.. list-table::
   :header-rows: 1

   * - Namespace
     - Use it for
   * - ``mixle.models``
     - Incubating Transformer leaves, direct language models, DPO leaves,
       neural leaves, learned embeddings, and applied model helpers.
   * - ``mixle.task``
     - LLM labelers, task distillation, one-call replacement for label,
       numeric, multi-label, and structured-output functions, active learning,
       local students, cascades, routers, edge search, quantization, tool
       calling, planning, artifacts, and scorecards.
   * - ``mixle.reason``
     - LLM uncertainty, semantic entropy, claim reliability, finite-hypothesis
       reasoning, graph-producing LLMs, typed ontologies, and cross-modal
       latent evidence.
   * - ``mixle.represent``
     - Segmenters, embeddings, heterogeneous encoders, vector quantizers, and
       cross-modal representations, including posterior-affinity retrieval.

Use :doc:`neural-llm`, :doc:`task-distillation`, :doc:`task-serving`,
:doc:`uncertainty`, :doc:`reasoning-systems`, and :doc:`representation` for the
narrative guides. Use :doc:`maturity` to decide how much validation each
surface needs.

For these applied namespaces, a useful artifact record includes the teacher or
training source, calibration split, optional dependency versions, escalation
policy, and reload check. The package boundary should not hide those facts.

Local Application Runtime
-------------------------

Local application surfaces connect fitted models to operational workflows:

.. list-table::
   :header-rows: 1

   * - Namespace
     - Use it for
   * - ``mixle.system``
     - The answer/ingest/improve facade for a local application: a cost
       ledger (``Spend``), named degradation modes instead of silent
       fallbacks (``with_fallback``, ``abstain_on_timeout``, ``route_past``),
       a held-out scorecard with regression detection, a meta-improvement
       budget allocator, and the model registry fitted task models are read
       from and written into. Formerly six cross-importing flat modules;
       ``mixle.spend``, ``mixle.fault``, ``mixle.scorecard``, ``mixle.meta``,
       and ``mixle.registry`` still work as deprecated aliases that forward
       here with a warning.
   * - ``mixle.substrate``
     - Typed, scoped, provenanced storage for documents, records, artifacts,
       traces, context packets, graph facts, retrieval, multi-hop evidence,
       factuality checks, lineage audits, secret scans, sharing, and
       governance.
   * - ``mixle.inference.skill``
     - Packaging a fitted model, created artifact, or callable as a named
       reusable capability with certificate/provenance metadata.
   * - ``mixle.pool``
     - Local-or-pool job submission, budget checks, explicit confirmation for
       billable backends, and artifact return.
   * - ``mixle.telemetry``
     - Local JSONL decision events for fits, placement, routing, context,
       reasoning, escalation, pool jobs, and drift.
   * - ``mixle.scientist``
     - Optional assembled local scientific workflow using cached open-weight
       encoders, certified heads, substrate-backed reasoning, and edge
       distillation receipts.

Use :doc:`reasoning-ecosystem` for the narrative guide and
:doc:`api/mixle.system`, :doc:`api/mixle.substrate`, :doc:`api/mixle.pool`,
:doc:`api/mixle.telemetry`, and :doc:`api/mixle.scientist` for API reference.

Scientific Design and Analysis
------------------------------

Mixle also includes scientific modeling support:

.. list-table::
   :header-rows: 1

   * - Namespace
     - Use it for
   * - ``mixle.process``
     - Temporal point processes, renewal processes, Hawkes processes,
       birth-death processes, continuous-time Markov chains, and random
       partitions.
   * - ``mixle.relations``
     - Assignment, shortest path, edit distance, Viterbi path, spanning tree,
       subset, and ranking relation solvers.
   * - ``mixle.analysis``
     - Coverage, extremes, KDE, kriging, spatial mixtures, rank aggregation,
       covariance shrinkage, and related diagnostics.
   * - ``mixle.doe``
     - Design of experiments, Bayesian optimization, active design,
       multi-fidelity design, sensitivity, propagation, and constrained
       optimization.
   * - ``mixle.evolve``
     - Objective-led model improvement, structure search, verification, and
       anti-regression ledgers.
   * - ``mixle.epistemic``
     - Hypothesis portfolios with open-world mass, distribution discrepancy
       (KL/JS/Wasserstein/MMD), EIG-driven action selection, and a
       replayable decision journal -- the epistemic-loop control flow, built
       on ``mixle.inference``, ``mixle.doe``, and ``mixle.evolve``.

Use :doc:`processes`, :doc:`relations`, :doc:`analysis`, :doc:`doe`, and
:doc:`evolution` when the modeling task is closer to scientific design,
structured optimization, or repeated model improvement.

Scientific workflows should record the design objective, constraints, random
seed, and validation metric. A selector, optimizer, or evolutionary search
result is only as trustworthy as the evidence attached to the chosen candidate.

Operations-Research Building Blocks
-----------------------------------

A family of applied modules reduce to the same handful of combinatorial
solvers in ``mixle.relations`` (max-flow/min-cut, min-cost flow, branch-and-
bound MILP). Each names one recurring OR problem shape; the worked example in
each module (ore blending, mine scheduling, ...) is a domain instantiation of
a construction built entirely from caller-supplied sources, capacities, and
costs, not something specific to that domain:

.. list-table::
   :header-rows: 1

   * - Namespace
     - Use it for
   * - ``mixle.blending``
     - Minimum-cost mixture of several sources to hit a target attribute
       window -- fuel, feed, alloy, and ore blending all reduce to the same
       LP/MILP.
   * - ``mixle.precedence_scheduling``
     - Maximum-weight closure (Picard's min-cut construction) and
       time-phased, capacity-constrained scheduling under a precedence
       relation between items.
   * - ``mixle.stochastic_opt``
     - Two-stage stochastic programming with CVaR risk (Rockafellar-Uryasev)
       over a calibrated posterior of uncertain per-item value -- the same
       construction as portfolio or project selection under uncertain
       returns.
   * - ``mixle.pipeline_twin``
     - Digital-twin, period-stepped re-simulation of a min-cost-flow network
       under stochastic arrivals, with named scenario interventions,
       wrapped as a ``mixle.inference.simulate.Simulator``.
   * - ``mixle.fulfillment``
     - Demand forecasting (a regime-switching HMM recalibrated with split
       conformal prediction) chained into minimum-cost supply routing
       against that forecast.

See ``examples/precedence_scheduling_example.py`` for a worked walkthrough.

Data, Engines, and Runtime
--------------------------

The lower layers keep model code independent from storage location, array
library, and deployment format:

.. list-table::
   :header-rows: 1

   * - Namespace
     - Use it for
   * - ``mixle.data``
     - Schemas, validation, exchangeability checks, hashes, encoded IO,
       structured data sources, and stream token sources.
   * - ``mixle.engines``
     - NumPy, Torch, JAX, symbolic, bit-packed, high-precision, LNS, and
       precision-planning engines.
   * - ``mixle.stats.compute``
     - Protocol classes, encoded data, generated kernels, backend scoring,
       sufficient statistics, and decomposition metadata.
   * - ``mixle.utils``
     - Automatic model typing, serialization, optional dependency gates,
       metrics, vector utilities, HVIS embeddings, and runtime helpers.
   * - ``mixle.utils.parallel``
     - Encoded-data backends, resource planning, model sharding, distributed
       folding, and model-parallel estimators.

Use :doc:`data`, :doc:`engines`, :doc:`compute-layer`, and
:doc:`utilities-and-parallelism` when performance, persistence, or deployment
boundaries matter.

Experimental Surface
--------------------

``mixle.experimental`` contains exploratory APIs that are useful for research
but are not the stable entry point for new production code. The current program
surface is documented in :doc:`experimental-program`. Prefer stable surfaces in
``mixle.stats`` and ``mixle.inference`` when possible. Use ``mixle.ppl``,
``mixle.models``, and ``mixle.task`` deliberately when their higher-level or
applied workflow is the reason for the code.

Choosing an Entry Point
-----------------------

.. list-table::
   :header-rows: 1

   * - Situation
     - Start with
   * - You know the estimator tree
     - ``mixle.inference.optimize`` and :doc:`concepts`.
   * - You have heterogeneous raw rows and want a first model
     - ``mixle.propose`` or ``mixle.task.recommend_model``.
   * - You need a lifecycle object for a demo or application boundary
     - ``mixle.Model`` and :doc:`lifecycle`.
   * - You need to replace an LLM or rule with a local model
     - ``mixle.task.solve`` and :doc:`task-serving`.
   * - You need uncertainty-aware LLM behavior
     - ``mixle.reason`` and :doc:`uncertainty`.
   * - You need a local knowledge/reasoning application shell
     - ``mixle.substrate``, ``mixle.inference.skill``, ``mixle.pool``,
       ``mixle.telemetry``, and :doc:`reasoning-ecosystem`.
   * - You need speed or scale
     - :doc:`compute-layer`, :doc:`engines`, and
       :doc:`utilities-and-parallelism`.

If you are unsure where a feature belongs, start from the most stable layer
that can express the problem. Move upward to applied helpers only when their
additional assumptions are part of the task.
