Production Workflows
====================

Production support lives primarily in ``mixle.inference.production`` and
``mixle.task``. The goal is not to replace deployment infrastructure. The goal
is to make fitted probabilistic models self-describing, versionable,
monitorable, and callable behind a service or cascade.

This page describes practical helpers, not a full production platform. Validate
serialization, scoring latency, drift thresholds, rollback behavior, and
runtime dependencies in the environment where the model will actually run.

Production concerns are grouped into five areas:

* provenance headers;
* verifiable training lineage;
* registry and alias promotion;
* scoring services with activity logs;
* drift detection and retrain/swap monitoring.

Fit with Provenance
-------------------

``fit_with_provenance`` fits a model and attaches a structured ``Header``.

.. code-block:: python

   from mixle.inference.production import fit_with_provenance, verify_lineage
   from mixle.stats import GaussianEstimator

   model, header = fit_with_provenance(
       data,
       GaussianEstimator(),
       max_its=30,
       seed=1,
   )

   print(header.dataset_hash)
   print(header.model_hash)
   print(header.final_loglik)
   print(verify_lineage(header))

The header records:

* model type and summary;
* inferred schema when available;
* record count and dataset hash;
* final log likelihood;
* model hash;
* training settings and convergence trace;
* timing, resource usage, environment, and git commit when available.

Build Headers Explicitly
------------------------

Use ``build_header`` when fitting happened elsewhere but you still want a
provenance record.

.. code-block:: python

   from mixle.inference.production import build_header

   header = build_header(
       model,
       data,
       training={"source": "external-fit"},
       final_loglik="auto",
   )

Headers are plain serializable records through ``Header.to_dict()``.

Registry
--------

``Registry`` stores models by name and version. It also supports aliases such
as ``production`` so serving code can load the currently promoted version.

.. code-block:: python

   from mixle.inference.production import Registry

   registry = Registry("/tmp/mixle-registry")
   version = registry.register(model, "events", header=header)

   print(registry.names())
   print(registry.versions("events"))

   registry.promote("events", version, alias="production")
   prod_model, prod_header = registry.current("events", alias="production")

The registry is filesystem-backed and serializes models through mixle's
serialization registry.

Checkpointing Long Fits
-----------------------

Registry checkpointers can snapshot a model during optimization:

.. code-block:: python

   from mixle.inference import optimize
   from mixle.inference.production import Registry

   registry = Registry("/tmp/mixle-checkpoints")
   model = optimize(
       data,
       estimator,
       max_its=100,
       on_step=registry.checkpointer("run-2026-07-01", every=5),
       out=None,
   )

   print(registry.verify_chain("run-2026-07-01"))

Each checkpoint can carry lineage metadata, making interruption and audit
workflows explicit.

Service
-------

``Service`` wraps a fitted model and scores batches while recording activity.

.. code-block:: python

   from mixle.inference.production import Service

   service = Service(model, name="events", reference=reference_data)
   log_probs = service.score(current_batch)
   print(service.health())

Activity logs include record count, wall time, mean log likelihood, and the
number of unscorable records. A JSONL log path can be supplied for persistent
activity records.

Load from a Registry Alias
--------------------------

.. code-block:: python

   service = Service.from_registry(registry, "events", alias="production")
   scores = service.score(records)

This is the handoff point for deployment systems: promote a version in the
registry, and serving code reads the alias.

Drift Detection
---------------

``detect_drift`` compares reference data to current data using both model-native
score drift and per-feature shift.

.. code-block:: python

   from mixle.inference.production import detect_drift

   report = detect_drift(
       model,
       reference_data,
       current_data,
       psi_threshold=0.25,
       ks_threshold=0.2,
       loglik_shift_threshold=-0.5,
   )

   print(report.drift)
   print(report.score)
   print(report.per_feature)

Score drift looks at the model's log-density distribution. Feature drift uses
population stability index, Kolmogorov-Smirnov, and related summary statistics
where applicable.

Task Artifacts and Cascades
---------------------------

``mixle.task`` models are production-oriented artifacts for local task serving.
They can be saved, loaded, calibrated, and placed behind a ``Cascade`` that
escalates uncertain inputs to a teacher.

Use :doc:`task-distillation` when the production object is a classifier,
extractor, or local LLM-distilled model rather than a density model.

Practical Deployment Shape
--------------------------

1. Fit with provenance.
2. Register the model.
3. Promote a version to an alias.
4. Serve through ``Service`` or a task ``Cascade``.
5. Log scoring activity.
6. Run drift checks against a reference sample.
7. Retrain and promote a new version when drift or quality thresholds fail.

API Map
-------

.. list-table::
   :header-rows: 1

   * - Import
     - Purpose
   * - ``fit_with_provenance``
     - fit and produce a provenance header
   * - ``Header``, ``build_header``, ``environment_info``
     - provenance records and environment capture
   * - ``verify_lineage``
     - verify convergence/model-hash lineage
   * - ``Registry``
     - versioned model store and alias promotion
   * - ``Service``
     - batch scoring with activity logging and health summaries
   * - ``detect_drift``, ``score_drift``, ``DriftReport``
     - drift detection from model scores and feature shifts
   * - ``Monitor``
     - drift-triggered retrain/swap loop
