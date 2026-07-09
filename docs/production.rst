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
* drift detection and retrain/swap monitoring;
* reproducibility receipts and local telemetry where the broader runtime layer
  is used.

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

Treat missing header fields as explicit limitations. A header without a data
hash, environment record, or source commit can still be useful locally, but it
should not be described as complete release evidence.

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

When building a header around an externally fitted model, record the external
training command or system as precisely as possible. Otherwise the header only
documents storage, not lineage.

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
serialization registry. Model names, versions, and aliases are constrained to
single path components. An unsafe name such as ``"../model"`` raises
``ValueError``, and unknown names or versions raise clear ``KeyError``
messages rather than leaking store paths.

Registration and promotion are separate actions. Registering a model records a
version; promoting an alias should happen only after the validation and review
gate has passed.

Reproducibility Receipts
------------------------

Use ``record_fit`` when the fit itself must be replayable, not merely stored.

.. code-block:: python

   from mixle.inference import record_fit, verify_reproducible

   receipt = record_fit(model, data, seed=1, estimator=estimator)
   check = verify_reproducible(estimator, data, receipt, seed=1)

   print(receipt.as_dict())
   print(check["reproducible"])

The receipt records a data fingerprint, seed, estimator type, and parameter
fingerprint. It complements provenance headers: the header describes the
training run, while the reproducibility receipt checks whether the same fit can
be recovered.

Use receipts for deterministic or near-deterministic recovery checks, not as a
substitute for quality validation. A reproducible low-quality model is still a
low-quality model.

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

Checkpoint chains are development or long-run evidence. Promote only a final
validated model, not an intermediate checkpoint, unless the application has an
explicit rollback or resume policy for that checkpoint.

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

Unscorable records should remain visible. ``NaN``, ``-inf``, missing fields,
and impossible observations are operational signals, not values to coerce into
ordinary low scores.

For application-level route, placement, context, pool, and reasoning events,
use :mod:`mixle.telemetry` and the workflow in :doc:`reasoning-ecosystem`.

Load from a Registry Alias
--------------------------

.. code-block:: python

   service = Service.from_registry(registry, "events", alias="production")
   scores = service.score(records)

This is the handoff point for deployment systems: promote a version in the
registry, and serving code reads the alias.

The deployment system should record which alias it loaded and when. Serving
from an alias without recording the resolved version makes rollback and
incident review much harder.

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

Drift thresholds should be tied to an action policy: alert, collect labels,
shadow a challenger, retrain, rollback, or escalate for review. A drift report
without an action policy is a diagnostic, not an operational gate.

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

Release Evidence
----------------

For production-facing documentation or artifacts, preserve:

* fitted model and provenance header;
* registry name, version, alias, and promotion decision;
* clean-wheel load and scoring smoke result;
* unscorable-record behavior;
* activity-log and drift-threshold configuration;
* rollback or alias-resolution policy; and
* receipts or lineage checks when reproducibility is claimed.

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
     - versioned model store and alias promotion with safe path components
   * - ``Service``
     - batch scoring with activity logging and health summaries
   * - ``detect_drift``, ``score_drift``, ``DriftReport``
     - drift detection from model scores and feature shifts
   * - ``Monitor``
     - drift-triggered retrain/swap loop
   * - ``record_fit``, ``verify_reproducible``, ``ReproReceipt``
     - replay and verify fitted parameter recovery
   * - ``Telemetry``, ``record``
     - local decision events for reasoning, routing, placement, and pool jobs
