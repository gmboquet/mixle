PPL Mixture Workflow
====================

The PPL layer is useful when the statistical formula is clearer than an
estimator tree. This tutorial fits a two-component Gaussian mixture with
``free`` parameter slots and then inspects the concrete distribution that the
expression lowered into.

The important idea is that PPL expressions are not a separate modeling world.
They lower back to Mixle distributions, estimators, targets, and inference
routes.

1. Build Synthetic Data
-----------------------

.. code-block:: python

   import numpy as np

   rng = np.random.RandomState(1)
   data = list(
       np.concatenate(
           [
               rng.normal(-5.0, 1.0, 8000),
               rng.normal(5.0, 1.0, 8000),
           ]
       )
   )

The data has two clear modes. A one-Gaussian model would blur the structure;
a mixture can represent the latent group.

For an applied workflow, keep a held-out slice before fitting. PPL syntax makes the
model declaration compact, but it does not remove the need to compare against a
simpler baseline or repeat a mixture fit with more than one seed.

2. Declare the Model
--------------------

.. code-block:: python

   from mixle.ppl import Mix, Normal, free

   expr = Mix([Normal(free, free), Normal(free, free)])

``Normal(free, free)`` means both the mean and scale are estimated. ``Mix``
adds a latent component assignment, so ``fit`` routes to the same EM machinery
used by :class:`mixle.stats.MixtureEstimator`.

The ``free`` markers are part of the public model contract. If a parameter is
fixed, document the fixed value and why it is not estimated. If every parameter
is free, record the initialization and fitting route in provenance so the same
expression can be explained later.

3. Fit the Expression
---------------------

.. code-block:: python

   model = expr.fit(
       data,
       max_its=80,
       rng=np.random.RandomState(7),
   )

The returned object keeps the PPL-facing wrapper, while ``model.dist`` is the
underlying fitted distribution.

If fitting fails, inspect the selected route before changing the model formula.
A symbolic expression can lower to different concrete machinery depending on
free parameters, observed fields, custom potentials, and missing-data settings.

Use route inspection as a debugging tool, not as a release guarantee by itself.
The lowered route still needs the same numerical checks as the estimator API:
finite objective values, stable component responsibilities, and predictable
handling of impossible or missing observations.

For release-facing notebooks, save the route explanation next to the fitted
object. It is the compact record of whether the example produced an EM point
estimate, a MAP result, or posterior draws.

4. Inspect the Bound Distribution
---------------------------------

.. code-block:: python

   means = sorted(component.mu for component in model.dist.components)
   weights = model.dist.w

   print(means)
   print(weights)
   print(model.dist.log_density(0.0))

After fitting, use ordinary distribution methods for scoring, sampling, and
capability inspection.

Keep ``model.dist`` visible in notebooks and reports. It is the object that
answers distribution-level questions such as scoring, sampling, support, and
posterior responsibilities, and it is the object most reviewers will need to
compare with an estimator-built model.

When the PPL wrapper is serialized or passed between packages, keep a small
score check against ``model.dist`` in the receipt. That makes the symbolic
expression and the concrete fitted object auditable as one artifact.

5. Ask Posterior Questions
--------------------------

.. code-block:: python

   responsibilities = model.posterior([-5.0, 0.0, 5.0])
   print(responsibilities)

For a mixture, the posterior is responsibility mass over components. For an
HMM, the same idea becomes state posterior mass through time.

Responsibilities are not labels. Treat them as evidence from the fitted model
and check whether ambiguous points, such as values near zero in this example,
behave consistently across seeds and validation splits.

For label-sensitive downstream work, record how component ordering was chosen.
Sorting components by mean is reasonable for this simple one-dimensional
example; richer mixtures need a domain-specific alignment rule or a
label-invariant evaluation metric.

6. Add Observed Covariates
--------------------------

PPL expressions can also reference observed fields. The expression below is a
Poisson regression-style model whose rate depends on an observed feature.

.. code-block:: python

   from mixle.ppl import Field, Poisson

   regression = Poisson(free * Field("x") + free)
   fitted = regression.fit(counts, given={"x": x})

Use the PPL layer when it clarifies the model declaration. Use the estimator
API directly when you need exact control over every child estimator,
initialization, or backend detail.

For missing data, make the policy explicit. A PPL route may reject missing
values, marginalize them, or require an inference method that can represent
latent missing fields. Do not rely on ``NaN`` cleanup outside the model unless
that transformation is part of the documented data pipeline.

Validation Checklist
--------------------

For mixture-style PPL models:

* run more than one random seed or use a restart strategy for difficult data;
* inspect the lowered distribution, not only the wrapper;
* record the selected fitting route and unsupported route combinations;
* compare against a simpler baseline on held-out log score;
* check whether component labels are identifiable enough for the target
  interpretation;
* keep route-specific diagnostics with the artifact, such as objective traces
  or posterior convergence checks;
* record the missing-data policy and any unsupported route/feature combination;
* record the lowered model structure in provenance; and
* preserve non-finite score behavior in diagnostics instead of rewriting it
  during data preparation.

Read :doc:`/ppl` for the full expression language and :doc:`/inference` for
the lower-level fitting controls.
