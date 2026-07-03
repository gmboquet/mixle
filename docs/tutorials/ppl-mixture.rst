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

2. Declare The Model
--------------------

.. code-block:: python

   from mixle.ppl import Mix, Normal, free

   expr = Mix([Normal(free, free), Normal(free, free)])

``Normal(free, free)`` means both the mean and scale are estimated. ``Mix``
adds a latent component assignment, so ``fit`` routes to the same EM machinery
used by :class:`mixle.stats.MixtureEstimator`.

3. Fit The Expression
---------------------

.. code-block:: python

   model = expr.fit(
       data,
       max_its=80,
       rng=np.random.RandomState(7),
   )

The returned object keeps the PPL-facing wrapper, while ``model.dist`` is the
underlying fitted distribution.

4. Inspect The Bound Distribution
---------------------------------

.. code-block:: python

   means = sorted(component.mu for component in model.dist.components)
   weights = model.dist.w

   print(means)
   print(weights)
   print(model.dist.log_density(0.0))

After fitting, use ordinary distribution methods for scoring, sampling, and
capability inspection.

5. Ask Posterior Questions
--------------------------

.. code-block:: python

   responsibilities = model.posterior([-5.0, 0.0, 5.0])
   print(responsibilities)

For a mixture, the posterior is responsibility mass over components. For an
HMM, the same idea becomes state posterior mass through time.

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

Validation Checklist
--------------------

For mixture-style PPL models:

* run more than one random seed or use a restart strategy for difficult data;
* inspect the lowered distribution, not only the wrapper;
* compare against a simpler baseline on held-out log score;
* check whether component labels are identifiable enough for the intended
  interpretation;
* record the lowered model structure in provenance.

Read :doc:`/ppl` for the full expression language and :doc:`/inference` for
the lower-level fitting controls.
