Data Layer
==========

``mixle.data`` is the bridge between external data and the encoder contract
used by distributions. It is optional: plain Python lists still work. Use the
data layer when you need typed schemas, lazy sources, reproducible hashes,
structure-aware partitioning, or reusable encoded payloads.

Core Objects
------------

``DataSource``
    A lazy, typed, structured source of records. A source exposes ``records()``
    and ``encode(...)`` and carries optional schema and sample-structure
    metadata.

``Schema``
    An ordered set of named fields with logical types such as ``Real``,
    ``Count``, ``Categorical``, ``Text``, ``Vector``, ``Optional``, and
    ``Nested``.

``SampleStructure``
    The exchangeability assumption for a dataset: ``IID``, ``EXCHANGEABLE``,
    ``SEQUENTIAL``, or a partially exchangeable grouping.

``exchangeability_check``
    A permutation-based diagnostic for whether numeric row order carries trend
    or regime-shift information.

``dataset_hash`` and ``model_hash``
    Stable identifiers used by provenance, registries, drift checks, and
    reproducibility workflows.

Schema Validation
-----------------

Schemas coerce records into the Python values expected by encoders and report
problems early.

.. code-block:: python

   from mixle.data import Field, Real, Schema, Text, check_dataset

   schema = Schema(
       (
           Field("country", Text()),
           Field("age", Real()),
           Field("spend", Real()),
       )
   )

   rows = [
       {"country": "US", "age": "41", "spend": 12.5},
       {"country": "CA", "age": 39, "spend": 8.0},
   ]

   conformed = schema.conform(rows)
   report = check_dataset(rows, schema=schema)

Use schema validation at the boundary of a pipeline. The fitted model can still
receive ordinary Python records after they are conformed.

Sources
-------

``as_source`` wraps in-memory data. ``open_source`` constructs a lazy source for
supported external formats when the relevant optional extra is installed.

.. code-block:: python

   from mixle.data import EXCHANGEABLE, as_source

   source = as_source(rows, structure=EXCHANGEABLE, schema=schema)
   for record in source.records():
       print(record)

The public source system includes adapters for pandas, Arrow, SQL, Mongo,
Hadoop, Spark, text, and graph data. Heavy integrations are imported lazily so
the base install stays small.

Sample Structure
----------------

Sample structure tells mixle how records may be partitioned or interpreted.

.. list-table::
   :header-rows: 1

   * - Structure
     - Use when
   * - ``IID``
     - independent and identically distributed samples
   * - ``EXCHANGEABLE``
     - order does not matter, but exact iid assumptions are not asserted
   * - ``SEQUENTIAL``
     - record order is meaningful, as in time series or event streams
   * - ``partially_exchangeable(by=...)``
     - samples are exchangeable within groups, such as users or sessions

Structure-aware partitioning matters for streaming, distributed fitting, and
validation. It prevents sequence or group boundaries from being broken
accidentally.

Exchangeability Diagnostics
---------------------------

Many high-level verbs assume rows can be pooled into one model or sampled as
"more rows like these." ``exchangeability_check`` tests that assumption for
numeric scalar or tuple/list fields by looking for order trends and first-half
versus second-half shifts.

.. code-block:: python

   from mixle.data import exchangeability_check

   report = exchangeability_check(values, alpha=0.01, seed=0)
   print(report.label)
   print(report.as_dict())

The report label is one of:

``exchangeable``
    No order signal was found at the tested level.

``trend``
    Values co-move with row position; fit a temporal or sequential model
    instead of pooling silently.

``shift``
    The early and late halves differ in location; treat the rows as a regime
    change unless the split is intended.

``mixle.inference.create`` and ``mixle.inference.synthesize`` run this check
when applicable and store the verdict in provenance. It is a warning signal,
not an automatic refusal, because some applications intentionally pool after
domain review.

Encoded Data
------------

Most users do not need to call encoders directly. When repeated fits should
reuse the same preprocessing boundary, save encoded data:

.. code-block:: python

   from mixle.data import load_encoded, save_encoded

   encoder = model.dist_to_encoder()
   encoded = encoder.seq_encode(rows)
   save_encoded("encoded.mixle", encoded)
   encoded_again = load_encoded("encoded.mixle")

The encoded payload is the same kind of data consumed by ``optimize`` internally.

Hashes and Provenance
---------------------

``dataset_hash`` and ``model_hash`` provide durable identifiers:

.. code-block:: python

   from mixle.data import dataset_hash, model_hash

   data_id = dataset_hash(rows)
   model_id = model_hash(model)

Production helpers use these values in model headers, registries, drift
reports, and lineage checks.

DataFrame and Spark Helpers
---------------------------

Optional adapters keep tabular and distributed data close to the same record
shape used by ordinary lists.

.. code-block:: python

   from mixle.data import dataframe_records

   rows = list(dataframe_records(df, fields=["country", "age", "spend"]))

Spark helpers include sampling functions for RDD-backed workflows. Use them
after the local model shape works on an in-memory sample.

Graph Data
----------

Graph adapters expose graph observations to graph distributions without making
graphs part of the scalar record path. ``GraphDataEncoder`` and
``GraphObservation`` are loaded lazily to avoid import cycles with graph
families.

Practical Workflow
------------------

1. Start with a plain list of representative records.
2. Fit the model locally and confirm the estimator shape.
3. Add a ``Schema`` to make field coercion explicit.
4. Wrap the data in a ``DataSource`` when you need lazy loading, external data,
   sample structure, or partitioning.
5. Add hashes and encoded-data persistence once the workflow becomes
   repeatable or production-facing.

API Map
-------

.. list-table::
   :header-rows: 1

   * - Import
     - Purpose
   * - ``DataSource``, ``MaterializedSource``, ``LazySource``, ``as_source``
     - source abstraction and in-memory/lazy wrappers
   * - ``Schema``, ``Field``, ``Real``, ``Count``, ``Categorical``, ``Text``
     - typed field schemas
   * - ``FieldType``, ``Boolean``, ``Timestamp``
     - additional schema types and the base field-type protocol
   * - ``IID``, ``EXCHANGEABLE``, ``SEQUENTIAL``, ``partially_exchangeable``
     - sample-structure declarations
   * - ``exchangeability_check``, ``ExchangeabilityReport``
     - row-order diagnostics used by creation and synthesis provenance
   * - ``check_dataset``, ``DataReport``
     - validation and diagnostics
   * - ``dataset_hash``, ``model_hash``
     - reproducibility identifiers
   * - ``save_encoded``, ``load_encoded``
     - persist encoded payloads
   * - ``open_source``, ``source_kinds``
     - external data source discovery and construction
   * - ``seq_encode_dataframe``, ``sample_rdd``, ``sample_seq_as_rdd``,
       ``take_sample``
     - tabular and Spark/RDD sampling or encoding helpers
