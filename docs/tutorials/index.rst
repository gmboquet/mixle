Tutorials
=========

These tutorials are task-oriented walkthroughs. Each one names the model shape,
the inference route, and the point where the result should be inspected or
validated.

How to Use These Tutorials
--------------------------

Each tutorial is written as a workflow, not as a benchmark claim. The code
blocks show the smallest useful path through the API; the surrounding text
names the checks that make the result credible in a real project.

For exploratory work, start with the model shape and verify that fitting,
scoring, and sampling run without warnings. For production-facing work, keep a
separate validation split, record the estimator or program configuration, and
save the diagnostic output that would explain a later promotion decision.

The tutorials intentionally point back to the reference guides. Use the
walkthroughs to assemble a workflow, then use the guides to check edge cases,
capability contracts, optional dependencies, and release-readiness expectations.

Choose by Problem
-----------------

.. list-table::
   :header-rows: 1

   * - If you need to
     - Walkthrough
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

What Each Tutorial Demonstrates
-------------------------------

.. list-table::
   :header-rows: 1

   * - Tutorial
     - Main idea
     - Related guides
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

.. toctree::
   :maxdepth: 1

   heterogeneous-records
   ppl-mixture
   enumeration-ranking
   production-artifacts
   llm-distillation-cascade
   llm-uncertainty
   representation-and-models
   relations-and-operations
   evolution-and-analysis
