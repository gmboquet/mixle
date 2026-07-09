What Is New In 0.6.2
====================

Version 0.6.2 broadens Mixle from a composable statistical modeling library
into a more complete runtime for heterogeneous modeling, local reasoning, and
auditable deployment. The stable center is still the distribution/estimator
contract and ``optimize``. The new work adds higher-level creation,
reproducibility, placement, reasoning, telemetry, neural, and task-distillation
surfaces around that contract.

Use this page as a map of the release. The topic guides linked from each
section explain the workflow details.

Local Reasoning Runtime
-----------------------

0.6.2 adds a local application layer for building retrieval, reasoning, and
decision workflows around fitted models:

* ``mixle.substrate`` stores text, records, vectors, graph facts, mounted
  tools, and provenance-bearing evidence.
* ``mixle.substrate.Reasoner`` and ``investigate`` plan over retrieve, compute,
  simulate, and delegate actions while preserving abstention and citations.
* ``mixle.inference.skill`` registers reusable typed capabilities that a
  reasoner or application can discover.
* ``mixle.pool`` provides a small pool/job abstraction for local or remote
  execution decisions.
* ``mixle.telemetry`` records decisions, cost, latency, and outcomes; the
  dashboard helpers summarize those receipts.
* ``mixle.scientist`` assembles the local pieces into an optional, offline
  scientific-assistant workflow when the ``scientist`` extra and local model
  weights are available.

Read :doc:`reasoning-ecosystem`, :doc:`reasoning-systems`, and
:doc:`production`.

Certified Creation, Simulation, and Reproducibility
---------------------------------------------------

The inference layer now includes higher-level artifact workflows in addition to
ordinary fitting:

* ``create(data, ...)`` fits a model, records provenance, optional calibration,
  uncertainty, exchangeability diagnostics, and a certificate.
* ``simulate(model, ...)`` draws from models that expose compatible sampler
  behavior.
* ``synthesize(source, ...)`` creates checked synthetic artifacts from a model,
  dataset, callable, or task surface.
* ``record_fit`` and ``verify_reproducible`` store and check reproducibility
  receipts.
* ``certify`` and ``plan_placement`` expose fit guarantees and local/pool
  placement decisions.
* ``uq`` dispatches uncertainty queries across fitted models, point predictors,
  ensembles, and LLM-style callables.
* ``hierarchical_event_study`` estimates confirmed-exposure influence with
  within-subject shifts, random-effects pooling, difference-in-differences
  contrast, and sensitivity bounds.

Read :doc:`inference`, :doc:`quickstart`, :doc:`uncertainty`, and
:doc:`lifecycle`.

Data, Structure, and Process Families
-------------------------------------

0.6.2 adds several modeling and diagnostics surfaces:

* ``exchangeability_check`` reports whether rows look exchangeable, shifted, or
  trended before a workflow assumes that "more rows like these" is a valid
  sampling story.
* ``mixle.represent.modality`` supplies deterministic image and signal feature
  helpers for cross-modal examples and tests.
* ``Ontology`` and ``OntologyConstrainedKG`` add typed graph constraints for
  knowledge-graph and reasoning workflows.
* ``ContinuousTimeMarkovChainDistribution`` and its estimator model fully
  observed CTMC trajectories with a closed-form generator MLE that certifies as
  ``GLOBAL_UNIQUE``.
* Multivariate Gaussian fitting uses a BLAS-backed covariance accumulation
  path and a robust Cholesky fallback so float32/GPU EM fits can recover from
  small numerical indefiniteness instead of crashing.
* Hidden Markov distributions default to the Numba encoder when Numba is
  installed, matching the estimator default while still respecting
  ``use_numba=False``.

Read :doc:`data`, :doc:`representation`, :doc:`reasoning-systems`,
:doc:`processes`, :doc:`stats-structured`, and :doc:`hmms-latent`.

Neural and Task Surfaces
------------------------

The neural and task layers gained more durable, explicit behavior:

* ``Flow``, ``MAF``, ``VAE``, and ``DiscreteAR`` are constructible neural
  density families that can appear directly in model trees.
* Neural leaves, direct language models, streaming Transformer leaves, DPO
  models, energy models, and density leaves now have broader serialization
  support.
* Streaming Transformer and DPO accumulators preserve sample weights and EM
  responsibilities during gradient updates.
* ``mixle.task.distill_methods`` adds Torch-to-Torch response, multi-teacher,
  hint, attention-transfer, relational, and sequence-level distillation
  helpers.
* Task artifact durability improved for JSON-safe ``qhat=inf`` reloads, empty
  inputs, and int4 packed quantized weights.

Read :doc:`neural-llm`, :doc:`task-distillation`, and :doc:`task-serving`.

Engines, Placement, and Operational Hardening
---------------------------------------------

0.6.2 also tightens lower-level execution and production behavior:

* Torch DTensor component sharding is gated to Torch 2.5 or newer, where the
  needed DTensor strategies exist. Older Torch versions get a clear error that
  points to ``backend="model_parallel"``.
* DTensor import handling recognizes both the public and older private Torch
  module locations.
* The production registry constrains names, versions, and aliases to safe path
  components and raises clearer errors for unknown model names or versions.
* Benchmark and distributed-stress harnesses that are useful for maintainers
  moved out of tracked ``examples/`` paths so the examples page focuses on
  runnable documentation.

Read :doc:`engines`, :doc:`utilities-and-parallelism`, :doc:`examples`, and
:doc:`production`.

API Reference Coverage
----------------------

The generated API reference now includes the new top-level packages and modules
added in this release, including ``mixle.substrate``, ``mixle.pool``,
``mixle.telemetry``, ``mixle.scientist``, ``mixle.inference.create``,
``mixle.inference.simulate``, ``mixle.inference.synthesize``,
``mixle.inference.reproduce``, ``mixle.inference.skill``,
``mixle.inference.placement``, ``mixle.inference.planning``,
``mixle.inference.orchestration``, ``mixle.inference.event_study``,
``mixle.inference.uq``, ``mixle.data.exchangeability``,
``mixle.represent.modality``, ``mixle.reason.ontology``,
``mixle.stats.processes.ctmc``, ``mixle.models.neural_families``, and
``mixle.task.distill_methods``.

