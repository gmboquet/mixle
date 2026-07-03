Probabilistic Programming
=========================

``mixle.ppl`` is an expression layer over the distribution and inference
contracts. It lets model code read like the statistical model while lowering
back to ordinary ``mixle.stats`` distributions, estimators, or inference
targets.

The PPL is intentionally thin. The distribution families underneath are the
same families used by the explicit estimator API.

Parameter Slots
---------------

A parameter slot can hold:

* a fixed value;
* ``free``, meaning estimate this parameter;
* another distribution, meaning a prior or latent variable;
* an expression over fields, groups, named variables, or transforms.

.. code-block:: python

   from mixle.ppl import Normal, free

   fixed = Normal(0.0, 1.0)
   estimated = Normal(free, free)
   hierarchical = Normal(Normal(0.0, 10.0), 1.0)

Fitting
-------

.. code-block:: python

   from mixle.ppl import Field, Markov, Mix, Normal, Poisson, free

   clusters = Mix([Normal(free, free), Normal(free, free)]).fit(data)
   states = Markov(Normal(free, free), states=2).fit(sequences)
   counts = Poisson(free * Field("x") + free).fit(y, given={"x": x})

``Mix`` adds latent component assignments. ``Markov`` adds latent state through
time. ``Field`` reads covariates supplied through ``given=``.

Route Selection
---------------

The ``how=`` argument selects the inference route. ``how="auto"`` inspects the
lowered model and chooses a route family such as conjugate, EM, MAP, Laplace,
VI, VMP, MCMC, HMC, NUTS, ensemble, or hierarchical inference.

.. code-block:: python

   model = Mix([Normal(free, free), Normal(free, free)]).fit(data, how="auto")

Use explicit ``how=`` values when comparing routes or when an automatic choice
is not the one you want.

Named Variables and Constraints
-------------------------------

Named variables can be shared across a model and constrained by comparisons.

.. code-block:: python

   from mixle.ppl import Mix, Normal, constrain

   a = Normal(0.0, 10.0, name="a")
   b = Normal(0.0, 10.0, name="b")

   ordered = constrain(a < b, Mix([Normal(a, 1.0), Normal(b, 1.0)]))
   model = ordered.fit(data)

Use this for parameter tying, label-switching constraints, monotonicity, and
other structural assumptions.

Regression and Group Effects
----------------------------

Fields and groups make GLM-like expressions concise.

.. code-block:: python

   from mixle.ppl import Field, Group, Normal, Poisson, free

   y_model = Normal(free * Field("x") + free * Field("z") + free, free)
   fitted = y_model.fit(y, given={"x": x, "z": z})

   counts = Poisson(free * Field("x") + Group("site")).fit(
       count_y,
       given={"x": x, "site": site_ids},
   )

Random effects can also be expressed with ``.each()`` on priors when the data
are grouped.

Neural Predictors
-----------------

The PPL can use neural modules as predictors in distribution parameters.

.. code-block:: python

   from mixle.ppl import Categorical, Net, Transformer

   classifier = Categorical(logits=Net(hidden=[64], out=3)).fit(
       labels,
       given={"x": features},
       epochs=100,
   )

   next_token = Categorical(
       logits=Transformer(out=vocab, d_model=64, n_layer=2, n_head=4)
   ).fit(next_ids, given={"x": contexts}, epochs=40)

Use :doc:`neural-llm` for the estimator-level neural leaf workflow.

Posterior and Predictive Checks
-------------------------------

Depending on the route, fitted PPL objects can expose posterior summaries,
posterior predictive checks, prior predictive checks, WAIC/LOO-style
diagnostics, and comparison helpers.

.. code-block:: python

   from mixle.ppl import compare, posterior_predictive_check, posterior_summary

   summary = posterior_summary(model)
   check = posterior_predictive_check(model, data)
   ranking = compare([model_a, model_b], data)

Lowering
--------

``lower`` converts a PPL expression to a concrete target.

.. code-block:: python

   from mixle.ppl import lower

   dist = lower(rv, target="dist")
   estimator = lower(rv, target="estimator")

Extension work should usually add a lowering rule rather than branch inside a
fit loop.

Specialized PPL Surfaces
------------------------

The namespace also includes:

* field and spatial models such as ``GaussianField``, ``GP``, ``RBF``, and
  related kernels;
* conformal wrappers such as ``ConformalRegressor`` and
  ``ConformalClassifier``;
* survival helpers such as ``fit_censored`` and ``kaplan_meier``;
* posterior summaries such as ``hdi`` and ``posterior_summary``;
* guide-based routes such as ``structured_vi`` and ``admixture``.

When to Use PPL
---------------

Use the PPL when:

* the model is clearer as an equation than as an estimator tree;
* you need shared or constrained variables;
* you want priors directly in parameter slots;
* you want a concise latent model such as ``Mix`` or ``Markov``;
* you want to lower back to the same ``mixle.stats`` machinery.

Use explicit estimators when:

* you are writing production library code;
* you need direct control over encoders or estimators;
* you are extending a distribution family;
* you want the model tree to be explicit in ordinary Python objects.

API Map
-------

.. list-table::
   :header-rows: 1

   * - Area
     - Imports
   * - scalar families
     - ``Normal``, ``Poisson``, ``Gamma``, ``Categorical``, ``StudentT``, ...
   * - latent structure
     - ``Mix``, ``SemiMix``, ``Seq``, ``Markov``, ``LDA``
   * - parameters and covariates
     - ``free``, ``Field``, ``Group``, ``Embedding``
   * - constraints
     - ``constrain``, ``ordered``, ``increasing``, ``monotone``, ``potential``
   * - neural predictors
     - ``Net``, ``Conv``, ``Transformer``
   * - diagnostics
     - ``compare``, ``posterior_predictive_check``, ``posterior_summary``
   * - lowering
     - ``lower``
