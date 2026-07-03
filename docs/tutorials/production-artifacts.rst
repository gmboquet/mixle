Production Artifacts
====================

This tutorial shows the minimum production loop around a fitted Mixle model:
record provenance, register a version, serve scores, and run a drift check.
It is intentionally modest. Mixle does not replace your deployment platform; it
provides model metadata and verification objects that are explicit enough to
put behind one.

1. Fit With Provenance
----------------------

``fit_with_provenance`` returns a fitted model plus a header containing hashes,
training settings, timing, environment information, and lineage metadata.

.. code-block:: python

   import numpy as np
   from mixle.inference.production import fit_with_provenance, verify_lineage
   from mixle.stats import GaussianEstimator

   rng = np.random.RandomState(0)
   data = rng.normal(3.0, 2.0, 4000).tolist()

   model, header = fit_with_provenance(
       data,
       GaussianEstimator(),
       max_its=30,
       seed=1,
       out=None,
   )

   print(header.dataset_hash)
   print(header.model_hash)
   print(verify_lineage(header))

Keep the header next to the model artifact. It is the record that answers
"what data and settings produced this object?"

2. Register A Version
---------------------

The registry is a lightweight local artifact registry. It is suitable for
tests, demos, and simple deployments; larger systems can store the same model
and header in their own registry.

.. code-block:: python

   from mixle.inference.production import Registry

   registry = Registry("/tmp/mixle-demo-registry")
   version = registry.register(
       model,
       "demo-gaussian",
       header=header,
       metadata={"owner": "docs", "purpose": "tutorial"},
   )
   registry.promote("demo-gaussian", version, alias="production")

Promotion should happen only after held-out scoring and any domain-specific
review gates have passed.

3. Serve Scores
---------------

``Service`` is a small scoring wrapper. It records recent calls, exposes a
health summary, and keeps scoring behavior close to the fitted distribution.

.. code-block:: python

   from mixle.inference.production import Service

   service = Service(model, name="demo-gaussian", reference=data)
   current = rng.normal(3.0, 2.0, 100).tolist()
   log_probs = service.score(current)

   print(log_probs[:5])
   print(service.health())

Use the service boundary to standardize logging and monitoring. Do not hide
model exceptions; failures are part of the operational signal.

4. Detect Drift
---------------

Drift checks compare reference data with current data under both feature-level
and model-score diagnostics where available.

.. code-block:: python

   from mixle.inference.production import detect_drift

   shifted = rng.normal(9.0, 2.0, 500).tolist()
   report = detect_drift(model, data, shifted)

   print(report.drift)
   print(report.score)
   print(report.per_feature)
   print(report.thresholds)

A drift report is not an automatic rollback command. Treat it as evidence for
a policy: alert, shadow a challenger, collect labels, retrain, or escalate to
manual review.

Operational Checklist
---------------------

For a production artifact, keep:

* the fitted model;
* the provenance header;
* the training and validation data hashes;
* the package version and optional dependency set;
* the promotion decision and reviewer;
* recent score distributions and drift reports.

Read :doc:`/production` for the full production API and :doc:`/lifecycle` for
the higher-level ``mixle.Model`` wrapper.
