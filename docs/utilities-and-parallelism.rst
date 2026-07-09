Utilities and Parallelism
=========================

``mixle.utils`` contains support code that is important to real applications:
automatic estimator construction, serialization, optional dependency gates,
metrics, numerical helpers, heterogeneous visualization utilities, checkpoint
helpers, and parallel runtime planning.

This page covers the parts that are most likely to matter when moving from an
experiment to a durable workflow.

Serialization
-------------

``mixle.utils.serialization`` provides safe JSON-compatible serialization for
Mixle objects and selected callables:

``register_serializable_class(cls, type_id=None)``
    Register a class with a stable type identifier.

``register_serializable_callable(fn, callable_id=None)``
    Register a callable that may appear in serialized payloads.

``serializable_class_ids()``
    Inspect registered class identifiers.

``ensure_pysp_serialization_registry()``
    Load the built-in distribution registry.

``to_serializable(value)`` / ``from_serializable(payload)``
    Convert between Python objects and JSON-compatible payloads.

``to_json(value, **kwargs)`` / ``from_json(text)``
    Strict JSON round trip.

The probability-distribution base class delegates ``to_dict``, ``from_dict``,
``to_json``, and ``from_json`` to this module. Use it for artifacts and model
metadata that should survive process boundaries. Use pickle only when you need
full Python object fidelity and trust the environment.

Serialized payloads are part of the public contract once they are written to
disk. Keep stable type identifiers, avoid local callables in release artifacts,
and test a load round trip from a clean process before claiming artifact
durability.

Optional Dependencies
---------------------

``mixle.utils.optional_deps.require`` centralizes optional dependency errors.
Backends such as Spark, Dask, MPI, Ray, Torch, JAX, and database connectors
should call through optional dependency helpers so users get actionable errors
instead of import failures from deep inside a stack.

Optional dependency failures should name the extra or package the user needs.
They should not fire during base import, Sphinx builds, or workflows that do
not request the optional surface.

Evaluation and Metrics
----------------------

``mixle.utils.evaluation`` and ``mixle.utils.metrics`` collect lightweight
evaluation helpers used by model recommendation, task replacement, and tests.
Keep task-specific evaluation in the task layer, but use these utilities for
shared scoring, comparisons, and small metric calculations.

Numerical Helpers
-----------------

``mixle.utils.special`` and ``mixle.utils.vector`` contain special functions
and vector utilities that support distributions, inference, and detectors.
Prefer these shared helpers over copying numerical snippets into individual
families, especially when stability or broadcasting behavior matters.

HVIS Utilities
--------------

``mixle.utils.hvis`` supports heterogeneous visual inspection and embedding
workflows. Important surfaces include:

``htsne`` / ``humap`` / ``dpmsne``
    Embedding helpers for heterogeneous or model-derived affinities.

``model_log_affinity`` and ``get_pmat``
    Build model-based affinities or probability matrices.

``model_knn``
    Compute nearest neighbors under model-derived affinities.

Balanced, local, and Fisher factors
    Helpers for constructing useful affinity geometry from model behavior.

Use these tools for inspection and exploratory analysis. For deployment
decisions, validate with held-out likelihood, task metrics, calibration, and
monitoring rather than relying on a visualization alone.

Visual inspection artifacts should record the model, affinity, embedding
settings, and sampled rows used to produce them. A plot without that context is
not reproducible evidence.

Encoded-Data Parallelism
------------------------

``mixle.utils.parallel`` exposes the public parallel runtime helpers:

``encoded_data(data, estimator=..., model=..., backend=...)``
    Encode data into a backend handle.

``is_encoded_data_handle(obj)``
    Check whether an object is a parallel encoded-data handle.

``Resources``
    Describe CPU, memory, GPU, worker, and device resources.

``plan(data, estimator=..., resources=...)``
    Build a placement and chunking plan.

``model_sharding_plan(model, resources=...)``
    Decide how model work should be split across devices or workers.

Backend handles preserve the same high-level sequence-driver contract: they
support operations such as log-density sums and sufficient-statistic folding
without changing model code.

Run a local encoded-data parity check before trusting a new backend handle.
The same data and estimator should produce matching counts, log-density sums,
and sufficient-statistic updates within the expected numeric tolerance.

Resource and Calibration Catalogs
---------------------------------

The planner module includes calibration records and catalogs for keeping
runtime estimates explicit:

``DeviceSpec``
    Describes a local CPU, GPU, worker, or accelerator target.

``CalibrationRecord``
    Stores measured runtime and memory behavior for a model/data/backend shape.

``CalibrationCatalog``
    Reuses those measurements when planning future runs.

Planning should be treated as an estimate until measured on the target system.
Calibration records are how Mixle turns "this should fit" into "this shape has
fit before under these resource constraints."

Calibration records should expire or be revalidated when hardware, dependency
versions, model shape, or encoded payload size changes materially.

Model Parallelism
-----------------

``mixle.utils.parallel.model_parallel`` supports splitting model work rather
than only splitting data:

``ModelParallelEstimator``
    Wraps an estimator so folds can be computed over model shards.

``ModelParallelEncodedData``
    Encoded data handle aware of model-parallel folding.

``model_parallel_fold``
    Execute a fold across model shards.

``auto_parallel_estimator``
    Choose a model-parallel wrapper when the estimated model and data footprint
    call for it.

Use model parallelism when the model has large independent or nearly
independent component work, such as mixture components, ensembles, or structured
children that can be reduced safely.

The reduction must be associative and semantically equivalent to the local
estimator path. If a model shard changes the meaning of sufficient statistics,
the model should not use this route.

This is also the recommended fallback when Torch DTensor component sharding is
not available. Torch versions before 2.5 expose incomplete DTensor strategies
for the mixture operations Mixle needs, so ``TorchEngine`` rejects that path
with guidance instead of letting a low-level distributed tensor error surface.

Decomposition and Backend Modules
---------------------------------

Parallel support is split into focused modules:

.. list-table::
   :header-rows: 1

   * - Module
     - Purpose
   * - ``planner``
     - Resource planning, encoded-data registry, local/Spark/Dask/Ray style
       handles, and placement logic.
   * - ``multiprocessing``
     - Local process workers for encoded data.
   * - ``mpi`` and ``torchrun``
     - Distributed process coordination.
   * - ``lightning_data`` and ``ray_data``
     - Optional runtime integrations.
   * - ``model_decomposition``
     - Helpers for splitting model structures.
   * - ``balance``
     - Work-balancing plans and auto-balanced estimators.
   * - ``dcp_checkpoint``
     - Distributed checkpoint helpers.
   * - ``torch_neural``
     - Encoded-data support for streaming token and neural workloads.

Application code should normally use the top-level ``mixle.utils.parallel``
helpers rather than importing backend modules directly.

Operational Guidance
--------------------

For durable workflows:

* serialize model metadata with stable JSON payloads;
* persist the estimator or model specification used for fitting;
* record optional dependencies and backend choices;
* keep calibration records for large jobs;
* run scalar/vectorized parity checks before trusting a new backend;
* use scorecards, drift monitors, and provenance records for deployed task
  models.

Release Evidence
----------------

For utility and parallelism work, preserve:

* serialization round-trip evidence from a clean process;
* optional dependency guard behavior for base imports and missing extras;
* backend parity checks for encoded data and accumulators;
* resource calibration records with hardware and dependency versions;
* explicit fallback behavior for unavailable distributed runtimes; and
* artifact metadata for visualizations or diagnostic outputs.

API Reference
-------------

* :doc:`api/mixle.utils`
* :doc:`api/mixle.utils.serialization`
* :doc:`api/mixle.utils.optional_deps`
* :doc:`api/mixle.utils.evaluation`
* :doc:`api/mixle.utils.metrics`
* :doc:`api/mixle.utils.special`
* :doc:`api/mixle.utils.vector`
* :doc:`api/mixle.utils.hvis`
* :doc:`api/mixle.utils.parallel`
* :doc:`api/mixle.utils.parallel.planner`
* :doc:`api/mixle.utils.parallel.model_parallel`
