Tutorials
=========

These tutorials are task-oriented walkthroughs. They are meant to show how the
pieces fit together, not to replace the reference pages. Each tutorial names the
model shape, the inference route, and the point where you should inspect or
validate the result.

Recommended First Route
-----------------------

If you are new to Mixle, read these in order:

1. :doc:`heterogeneous-records`
2. :doc:`ppl-mixture`
3. :doc:`enumeration-ranking`
4. :doc:`production-artifacts`

That path starts with the stable ``mixle.stats`` and ``mixle.inference`` core,
then moves through symbolic model expressions, support traversal, and artifact
handling.

Choose By Problem
-----------------

.. list-table::
   :header-rows: 1

   * - If you need to
     - Read
     - Surface
   * - model records with mixed field types
     - :doc:`heterogeneous-records`
     - Stable core
   * - express a model with the PPL layer
     - :doc:`ppl-mixture`
     - Active development
   * - enumerate top-k support values
     - :doc:`enumeration-ranking`
     - Stable/evolving core
   * - save, serve, and monitor model artifacts
     - :doc:`production-artifacts`
     - Practical production helpers
   * - build a hybrid neural/probabilistic event model
     - :doc:`hybrid-llm-events`
     - Incubating neural leaf
   * - replace repeated LLM calls with a calibrated local model
     - :doc:`llm-distillation-cascade`
     - Active task workflow
   * - decide when an LLM should abstain
     - :doc:`llm-uncertainty`
     - Active reasoning workflow
   * - build shared representations for multiple modalities
     - :doc:`representation-and-models`
     - Active representation workflow
   * - combine distribution operations with structured decisions
     - :doc:`relations-and-operations`
     - Stable/evolving core
   * - run an auditable model improvement loop
     - :doc:`evolution-and-analysis`
     - Active design/evolution workflow

Learning Tracks
---------------

Core probabilistic modeling
    Start with :doc:`heterogeneous-records`, then read
    :doc:`ppl-mixture`, :doc:`enumeration-ranking`, and
    :doc:`relations-and-operations`.

LLM and task replacement
    Start with :doc:`llm-distillation-cascade`, then read
    :doc:`llm-uncertainty`. For numeric replacement, multi-label tagging,
    structured outputs, tool calls, and planning, continue with
    :doc:`/task-serving` and
    :doc:`/agentic-task-distillation`.

Neural and representation workflows
    Start with :doc:`hybrid-llm-events`, then read
    :doc:`representation-and-models`, :doc:`/neural-llm`, and
    :doc:`/representation`.

Production and improvement
    Start with :doc:`production-artifacts`, then read
    :doc:`evolution-and-analysis`, :doc:`/production`, and :doc:`/lifecycle`.

What Each Tutorial Demonstrates
-------------------------------

.. list-table::
   :header-rows: 1

   * - Tutorial
     - Main idea
     - Read next
   * - :doc:`heterogeneous-records`
     - A tuple-shaped row becomes a composite estimator, and a mixture adds a
       latent cluster over the whole record.
     - :doc:`/concepts`, :doc:`/stats-structured`
   * - :doc:`ppl-mixture`
     - ``free`` parameters and ``Mix`` lower to the same estimator/distribution
       contract as the core API.
     - :doc:`/ppl`, :doc:`/automatic-inference`
   * - :doc:`enumeration-ranking`
     - A fitted model can expose ranked support traversal when the capability
       is available.
     - :doc:`/enumeration`, :doc:`/capabilities-contracts`
   * - :doc:`production-artifacts`
     - Fitted models need provenance, registry metadata, serving wrappers, and
       drift checks.
     - :doc:`/production`, :doc:`/lifecycle`
   * - :doc:`hybrid-llm-events`
     - A Transformer event leaf and a Gamma timing model can form one joint
       score for event streams.
     - :doc:`/neural-llm`, :doc:`/models`
   * - :doc:`llm-distillation-cascade`
     - A teacher labels examples, a local model learns the task, and calibrated
       confidence decides whether to answer or escalate.
     - :doc:`/task-distillation`, :doc:`/task-serving`
   * - :doc:`llm-uncertainty`
     - Repeated LLM samples become semantic entropy, answer confidence, and
       abstention decisions.
     - :doc:`/uncertainty`, :doc:`/reasoning-systems`
   * - :doc:`representation-and-models`
     - Segmenters, embeddings, and vector quantizers turn heterogeneous
       modalities into a shared modeling stream.
     - :doc:`/representation`, :doc:`/models`
   * - :doc:`relations-and-operations`
     - Distribution operations and structured feasible-set solvers solve
       different parts of a decision workflow.
     - :doc:`/operations`, :doc:`/relations`
   * - :doc:`evolution-and-analysis`
     - Diagnostics and objective-led search promote challengers only when they
       pass a verification gate.
     - :doc:`/analysis`, :doc:`/evolution`

Reference Bridges
-----------------

For broader reference material related to these tutorials, use:

* :doc:`/quickstart` and :doc:`/concepts` for the core data-shape and
  estimator-shape rule.
* :doc:`/maturity` for the boundary between stable, active, and incubating
  surfaces.
* :doc:`/stats-univariate`, :doc:`/stats-structured`, and
  :doc:`/stats-latent-bayes` for the distribution catalog.
* :doc:`/inference-toolkit` for scoring, calibration, conformal prediction,
  comparison, resampling, survival, and robust inference.
* :doc:`/task-serving` for one-call task replacement, routers, scorecards, edge
  deployment, and quantized local students.
* :doc:`/reasoning-systems` for finite-hypothesis reasoning, graph-producing
  LLMs, cross-modal retrieval, and learned shared latents.
* :doc:`/api-overview` when you know what you want to import but not which
  namespace owns it.

.. toctree::
   :maxdepth: 1

   heterogeneous-records
   ppl-mixture
   enumeration-ranking
   production-artifacts
   hybrid-llm-events
   llm-distillation-cascade
   llm-uncertainty
   representation-and-models
   relations-and-operations
   evolution-and-analysis
