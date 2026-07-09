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

The header is evidence, not decoration. If the model is later copied into a
registry, service, notebook, or report, copy the header or a stable reference to
it as well.

Treat missing provenance as a release blocker for promoted artifacts. A model
can still be useful for exploration without a complete header, but it should
not be described as reproducible, audited, or production-ready until the data
hashes, fitting settings, environment, and lineage are present.

2. Register a Version
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
       metadata={"owner": "ml-platform", "purpose": "density-monitoring"},
   )
   registry.promote("demo-gaussian", version, alias="production")

Promotion should happen only after held-out scoring and any domain-specific
review gates have passed.

The registry call records a version; it does not certify the version by itself.
Keep promotion criteria in the surrounding application or release evidence so a
future reviewer can distinguish "stored" from "approved for use."

Use aliases such as ``production`` only for explicit promotion decisions. A
registry entry can hold failed challengers, shadow models, and rollback
candidates; the alias should identify the artifact that cleared the gate, not
the artifact that happened to be registered last.

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

For probabilistic models, decide what the service should do with ``-inf`` or
``NaN`` scores before exposing it. Impossible observations, missing fields, and
invalid inputs should be visible in logs and metrics rather than collapsed into
ordinary low-confidence predictions.

Serving checks should include malformed records and impossible observations.
Those cases are part of the model contract: they show whether the service
preserves the fitted distribution's semantics or accidentally rewrites failures
into ordinary scores.

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

Use the same reference window when comparing drift reports over time. Changing
the reference data, thresholds, or score transformation without recording that
change makes drift trends difficult to interpret.

Drift evidence should identify which action it can trigger. Some reports are
alert-only diagnostics; others can start label collection, shadow evaluation,
or retraining. Record the action policy near the thresholds so operators do not
infer more authority from the report than it was designed to carry.

Operational Checklist
---------------------

For a production artifact, keep:

* the fitted model;
* the provenance header;
* the training and validation data hashes;
* the package version and optional dependency set;
* the promotion decision and reviewer;
* recent score distributions and drift reports.

Release Review Questions
------------------------

Before documenting an artifact as production-ready, answer:

* Was the artifact built from the package version that will be released?
* Was it loaded from the built wheel, not only from the source checkout?
* Are optional dependencies and hardware assumptions recorded?
* Does the serving wrapper preserve missing-data and impossible-observation
  behavior?
* Is promotion separate from registration?
* Can a reviewer reproduce the model hash, dataset hash, and validation result?
* Is every alias, rollback target, and drift threshold tied to an explicit
  decision record?

Read :doc:`/production` for the full production API and :doc:`/lifecycle` for
the higher-level ``mixle.Model`` wrapper.
