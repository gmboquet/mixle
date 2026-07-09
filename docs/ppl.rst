Probabilistic Programming
=========================

``mixle.ppl`` is an expression layer over the distribution and inference
contracts. It lets model code read like the statistical model while lowering
back to ordinary ``mixle.stats`` distributions, estimators, or inference
targets.

The PPL is intentionally thin. The distribution families underneath are the
same families used by the explicit estimator API.

The public contract is therefore: declare with PPL when it makes the model
clearer, inspect the selected route, and validate the lowered object exactly
as you would validate an estimator tree written by hand.

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

Lowering Boundary
-----------------

The PPL layer does not define a second statistical runtime. It lowers a compact
expression into one of the same concrete surfaces used elsewhere in Mixle:

``target="dist"``
    A fitted or fixed distribution object that can score, sample, or expose
    capabilities according to the underlying family.

``target="estimator"``
    An estimator tree suitable for ``mixle.inference.optimize`` and the normal
    encoder/accumulator loop.

``target="objective"``
    A numerical target for MAP, variational, sampler, or custom-potential
    routes.

When documenting or saving a PPL model, record the lowered target class. A
small expression can lower to a closed-form estimator, a mixture EM route, a
state-space model, or a numerical posterior objective; those results have
different validation requirements.

Route Selection
---------------

The ``how=`` argument selects the inference route. ``how="auto"`` inspects the
lowered model and chooses a route family such as conjugate, EM, MAP, Laplace,
VI, VMP, MCMC, HMC, NUTS, ensemble, or hierarchical inference.

.. code-block:: python

   model = Mix([Normal(free, free), Normal(free, free)]).fit(data, how="auto")

Use explicit ``how=`` values when comparing routes or when an automatic choice
is not the one you want.

``explain_fit`` reports the route without fitting:

.. code-block:: python

   from mixle.ppl import Normal, free

   route = Normal(Normal(0.0, 10.0), free).explain_fit()
   print(route["route"], route["reason"])

Use this before relying on ``how="auto"`` in examples, notebooks, or production
artifacts. A prior does not always imply that ``auto`` returns a full posterior;
some models resolve to MAP and warn that the result is a point estimate.

For durable artifacts, store the route explanation with the model. The route is
part of the statistical claim: conjugate and closed-form routes, EM routes,
point-estimate MAP routes, and posterior sampling or variational routes support
different diagnostics and uncertainty summaries.

Route Guarantees
----------------

The selected route determines what downstream code may safely claim.

.. list-table::
   :header-rows: 1

   * - Route family
     - Primary result
     - Promotion evidence
   * - Conjugate / closed form
     - Analytic posterior or analytic fitted state.
     - Prior/likelihood pairing, posterior moments, and a reload check.
   * - EM / MLE
     - Maximum-likelihood point estimate.
     - Monotone objective trace, finite parameters, and held-out log score.
   * - MAP / Laplace
     - Point estimate, optionally with a local Gaussian approximation.
     - Optimization status, curvature diagnostics, and sensitivity to
       initialization.
   * - MCMC / HMC / NUTS / ensemble
     - Posterior draws.
     - Chain diagnostics, effective sample size, divergences or rejection
       rates, and posterior predictive checks.
   * - VI / VMP
     - Variational posterior approximation.
     - ELBO trace, approximation limits, and posterior predictive checks.

Do not present a point-estimate route as posterior uncertainty. If a model
needs posterior claims, use ``how="posterior"`` or an explicit posterior route
and keep the diagnostics with the artifact.

Parameterization Contract
-------------------------

PPL constructors use user-facing parameter names. A fitted result should report
parameters in the same conceptual parameterization the expression used, even
when an internal optimizer works on transformed coordinates. For example, a
positive scale may be optimized through an unconstrained transform, but the
artifact should still make the scale parameter visible as a scale rather than
as an implementation-only tensor slot.

This is not just formatting. Parameterization affects priors, constraints,
saved artifacts, and user interpretation. When adding a PPL distribution,
document:

* which arguments are fixed values versus estimable slots;
* which arguments are constrained, such as positive scales or simplex weights;
* what route ``how="auto"`` selects for fixed, free, and prior-bearing forms;
* how fitted parameters appear after save/load; and
* whether scalar and vectorized scoring have parity tests.

Missing Data and Non-Finite Inputs
----------------------------------

The default PPL behavior is strict: non-finite observed data is rejected. Pass
``missing="marginalize"`` only when ``NaN`` means "integrate this observation
out of the likelihood".

.. code-block:: python

   observed = [1.0, float("nan"), 2.0, 3.0]
   fitted = Normal(free, free).fit(observed, missing="marginalize")

This route fits from present observations without imputing replacement values.
On EM/MLE paths it wraps estimator leaves with the marginalizing missing-data
contract. On numerical posterior paths such as MAP, VI, MCMC, HMC, and NUTS,
support depends on the flat autograd target and available optional backend.
Unsupported closed-form routes raise rather than silently changing the model.
When the observed ``NaN`` is actual input that should remain observable to a
downstream model, do not enable marginalization. The PPL will not reinterpret
non-finite values as ordinary numbers or silently fill them in.

For the broader policy, including mixture responsibilities and DOE selectors,
see :doc:`stability-and-missing-data`.

Validation Standard
-------------------

Before promoting a PPL model, verify the lowered object and the fitted object:

* ``explain_fit`` or an equivalent route record names the selected inference
  family;
* ``lower(..., target="estimator")`` or the fitted object matches the intended
  estimator tree;
* unsupported priors, potentials, constraints, or missing-data modes fail
  loudly rather than falling through to a weaker route;
* scalar scoring and any vectorized or encoded scoring path agree on a small
  fixture;
* a saved artifact can be reloaded and rescored on a representative example.

This keeps PPL expressions convenient without making the symbolic surface a
place where fitting assumptions disappear.

Artifact Checklist
------------------

For release-facing PPL artifacts, preserve:

* the expression or a serialized representation of the lowered model;
* the selected ``how=`` route and ``explain_fit`` output;
* any ``given=`` fields, grouping keys, constraints, potentials, and missing
  data policy;
* random seeds, optimizer or sampler settings, and optional dependency
  requirements such as Torch;
* route-specific diagnostics, such as EM traces, optimization status,
  convergence diagnostics, or posterior predictive checks;
* a representative reload-and-rescore check.

This evidence is especially important when a compact PPL expression lowers to
a mixture, state-space model, custom potential, or neural predictor.

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

Lowering is also the boundary to inspect when debugging. If a PPL expression
fits unexpectedly, first check whether the lowered target is an estimator, a
distribution, a numerical objective, or a specialized family with its own
registered fitter.

Specialized PPL Surfaces
------------------------

The namespace also includes:

* field and spatial models such as ``GaussianField``, ``GP``, ``RBF``, and
  related kernels;
* conformal wrappers such as ``ConformalRegressor`` and
  ``ConformalClassifier``;
* survival helpers such as ``fit_censored`` and ``kaplan_meier``;
* posterior summaries such as ``hdi`` and ``posterior_summary``;
* guide-based routes such as ``structured_vi`` and ``admixture``;
* state-space families such as ``LocalLevel`` and ``AR1`` that expose a fitted
  distribution after ``.fit()`` for log-probability and simulation;
* indexed latent vectors such as ``theta[Field("g")]`` with MAP, MCMC, HMC, and
  NUTS routes for grouped observations;
* custom ``potential`` terms on flat and composite models when the route is
  numerical enough to include the extra log-joint term.

Custom Potentials
-----------------

Use ``potential`` for an extra differentiable or numeric log term that is not
expressible as an ordinary distribution slot.

.. code-block:: python

   from mixle.ppl import Mix, Normal, free, potential

   mu0 = Normal(0.0, 10.0, name="mu0")
   anchor = Normal(3.0, 0.05, name="anchor")
   model = Mix([Normal(mu0, free), Normal(free, free)])

   coupled = model.fit(
       data,
       how="map",
       potentials=potential(
           lambda mu, a: -20.0 * (mu - a) ** 2,
           mu0,
           anchor,
       ),
   )

Potentials require routes that score the numerical joint objective: ``map``,
``mcmc``, ``hmc``, ``nuts``, ``ensemble``, or ``auto`` when it resolves to one
of those. EM, conjugate, VMP, and closed-form variational routes reject
potentials because they would otherwise ignore the extra term.

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
