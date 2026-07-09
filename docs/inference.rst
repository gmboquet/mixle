Inference
=========

``mixle.inference`` is the concern for turning data and a model specification
into a fitted or posterior-bearing model. The core entry points share one
encoder/estimator loop and differ mainly in how much control you want over
initialization, objectives, streaming, restarts, and diagnostics.

Entry Points
------------

.. list-table::
   :header-rows: 1

   * - Function
     - Use when
   * - ``optimize(data, estimator)``
     - you want the standard fit route for an estimator, prototype, or inferred model
   * - ``fit(data, estimator)``
     - you want the posterior-oriented wrapper with Bayesian defaults
   * - ``initialize(data, estimator)``
     - you want the initial model before iteration
   * - ``estimate(data, estimator, prev_estimate=model)``
     - you want one explicit estimate pass
   * - ``best_of(...)``
     - you want repeated random starts for a latent model
   * - ``StreamingEstimator`` / ``BayesianStreamingEstimator`` / ``IncrementalEstimator``
     - data arrive in batches or posterior state should carry forward
   * - ``create(data, ...)``
     - you want a certified fitted artifact with optional calibration and UQ
       post-conditions
   * - ``simulate(model)``
     - you want to turn a fitted generative model into a reusable simulator
   * - ``synthesize(source, label=..., verify=...)``
     - you want a verified synthetic or teacher-labeled dataset
   * - ``certify(model)`` / ``plan_placement(certificate)``
     - you want an auditable estimate of how each block was solved and where
       it should run
   * - ``record_fit`` / ``verify_reproducible``
     - you want to replay a fit and check that the same parameters are
       recoverable
   * - ``uq(thing, data)``
     - you want method-selected uncertainty over a fitted model, point
       predictor, ensemble, or LLM-style callable

Estimator, Prototype, or Inferred Model
---------------------------------------

``optimize`` accepts three model specifications:

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureDistribution

   # explicit estimator
   m1 = optimize(data, GaussianEstimator(), out=None)

   # prototype distribution: derive the estimator shape from the model object
   proto = MixtureDistribution(
       [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
       [0.5, 0.5],
   )
   m2 = optimize(values, proto, prev_estimate=proto, out=None)

   # inferred estimator from raw data
   m3 = optimize(data, out=None)

Use explicit estimators for production or publication-quality workflows. Use
prototype or inferred estimators for exploration. Passing a prototype as the
second argument tells ``optimize`` which estimator tree to build. Passing the
same object as ``prev_estimate`` also uses its parameter values as the starting
estimate, which is important for latent models such as mixtures and HMMs.

What ``optimize`` Does
----------------------

At a high level, ``optimize``:

1. resolves the estimator;
2. chooses or reuses an encoder;
3. encodes raw records into chunks;
4. initializes a model;
5. repeats an E/M-style update loop;
6. scores convergence on the selected objective;
7. returns the best model under the training or validation objective.

The same outer loop supports closed-form leaves, mixtures, HMMs, variational
families, MAP objectives, neural leaves, and distributed encoded data.

Inspect the fitted route when the model will be reused. A useful run record
contains the estimator shape, objective, convergence trace or final score,
validation score when supplied, random seed, backend/engine, and any fallback
or warning emitted by automatic inference.

Fitted-Model Evidence
---------------------

Treat a fitted object as a statistical claim plus evidence. The minimum
evidence depends on the route:

.. list-table::
   :header-rows: 1

   * - Route
     - Evidence to keep with the model
   * - Closed-form estimator
     - Estimator class, sufficient-statistic totals, fitted parameters, and a
       small score fixture.
   * - EM or latent-variable estimator
     - Initialization, objective trace, component/state diagnostics, finite
       responsibilities, and validation score when available.
   * - MAP or gradient MLE
     - Objective, optimizer status, parameter constraints, seed, and gradient
       or curvature diagnostics when available.
   * - Bayesian posterior route
     - Prior specification, posterior summary or draws, convergence
       diagnostics, and posterior-predictive checks.
   * - Neural or external engine route
     - Backend, device, precision, training settings, seed, and reload/rescore
       evidence.

Generated API pages can name the functions, but they cannot prove that a fit
is trustworthy. Keep the route evidence with notebooks, examples, release
artifacts, and production handoffs.

Certified Creation
------------------

``create`` is the higher-level creation verb. It infers and fits a model,
attaches an estimation certificate, and can reserve held-out data for a
calibration report or attach a UQ handle.

.. code-block:: python

   from mixle.inference import create

   artifact = create(rows, calibrate=0.2, quantify_uq=True, seed=0)

   print(artifact.guarantee)
   print(artifact.why())
   print(artifact.is_calibrated())

The returned ``CreatedModel`` is deliberately not just the fitted distribution.
It carries:

* ``model``: the fitted model;
* ``certificate``: how each estimation block was solved;
* ``calibration``: optional held-out PIT/log-density report;
* ``uq``: optional uncertainty object;
* ``provenance``: record counts, seed, budget/device constraints, and the
  exchangeability check when applicable.

Use ``optimize`` when you need direct control over the estimator route. Use
``create`` when the artifact boundary and post-conditions matter.

Numerical Safety
----------------

Inference routes should fail loudly when the requested model cannot score the
observations. Non-finite observations are not repaired by the fit loop. Use an
explicit missing-data wrapper, PPL ``missing="marginalize"``, or an upstream
field transformation only when that is the intended statistical contract.

For latent models, monitor responsibilities, component weights, and validation
score across iterations. If a component collapses or a score becomes
non-finite, prefer a different initialization, stronger prior, simpler family,
or an explicit missing-data policy over accepting the fit.

Input Ownership and Missingness
-------------------------------

The inference loop may build encoded chunks, backend arrays, or standardized
working buffers. Those buffers are allowed to adapt data for computation. The
input object supplied by the caller is not the place where missingness is
repaired.

Use one of these explicit contracts:

* strict routes reject non-finite observations before fitting;
* missing-data wrappers marginalize or model absence while leaving caller data
  unchanged;
* PPL routes use ``missing="marginalize"`` only when ``NaN`` means "integrate
  this value out";
* preprocessing pipelines that impute or transform data return a new dataset
  and record the transformation.

This is especially important for NumPy arrays passed to repeated fits. A route
that silently fills ``NaN`` in place can make a later model appear stable while
changing the experiment.

Common Fit Knobs
----------------

``max_its``
    Maximum number of iterations.

``delta``
    Convergence tolerance. Use ``None`` when you want exactly ``max_its``
    iterations.

``rng``
    Random state for initialization and stochastic routes.

``out``
    Progress output stream. Pass ``out=None`` for quiet code.

``vdata``
    Validation data for selecting the best model.

``prev_estimate``
    Resume or continue from an existing fitted model. When fitting from a
    prototype distribution, pass ``prev_estimate=proto`` if the prototype's
    parameter values should seed the fit.

``backend``
    Encoded-data backend, such as ``local``, ``mp``, ``spark``, ``dask``, or
    ``mpi``.

``engine``
    Compute engine, such as ``TorchEngine(device="cuda")``.

``precision``
    Explicit precision, ``"auto"``, or ``"minimal"``.

``strategy``
    EM strategy object or callable for specialized update loops.

``on_step``
    Callback receiving per-iteration ``EMStep`` records, useful for
    checkpointing.

Objectives
----------

``objective=`` controls the convergence and selection objective.

.. list-table::
   :header-rows: 1

   * - Objective
     - Meaning
   * - ``"auto"``
     - choose MLE, MAP, or variational objective from model capabilities and priors
   * - ``"mle"``
     - observed-data likelihood
   * - ``"map"``
     - penalized likelihood with parameter priors
   * - ``"vb"``
     - variational evidence lower bound

The default ``"auto"`` is usually correct. Force an objective when comparing
routes or debugging prior behavior.

Latent Models and Restarts
--------------------------

Mixtures and HMMs can have local optima. Use ``best_of`` with a validation set
when random initialization matters.

.. code-block:: python

   import numpy as np
   from mixle.inference import best_of
   from mixle.stats import GaussianEstimator, MixtureEstimator

   est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])

   score, model = best_of(
       train,
       valid,
       est,
       trials=8,
       max_its=100,
       init_p=0.1,
       delta=1e-8,
       rng=np.random.RandomState(0),
       out=None,
   )

EM Strategies
-------------

``mixle.inference.em`` contains strategy objects for variants of the EM loop:
hard EM, annealed EM, generalized EM, monotonic EM, ECM, Monte-Carlo EM,
variational EM, online EM, accelerated EM, and restart EM.

Use strategies when the default exact E/M update is not the right numerical or
statistical route for the model.

Bayesian and Gradient Routes
----------------------------

Conjugate families use closed-form posterior updates through priors and
``conjugate_posterior`` helpers. Non-conjugate differentiable targets can use
MAP, Laplace, variational, HMC, or NUTS-oriented routes.
``is_conjugate_family`` is the guard used by the higher-level inference path to
decide whether a fitted estimator/prior pair can take the analytic posterior
route or should fall back to a numerical approximation.

.. code-block:: python

   from mixle.inference.priors import NormalGammaPrior
   from mixle.stats import GaussianEstimator

   est = GaussianEstimator(prior=NormalGammaPrior())
   posterior_model = optimize(data, est, objective="auto", out=None)

For gradient objectives, see ``mixle.inference.gradient_fit`` and
``mixle.inference.target``.

Certificates and Placement
--------------------------

``certify`` classifies the fitted model's estimation blocks along an ordered
guarantee ladder:

``HEURISTIC``
    Gradient descent or another heuristic local route.

``STATIONARY``
    EM or coordinate ascent fixed point.

``STATIONARY_ESCAPE_TESTED``
    EM with explicit restart or saddle-escape testing.

``GLOBAL``
    Convex objective such as least squares or IRLS.

``GLOBAL_UNIQUE``
    Closed-form unique optimum, such as many exponential-family and
    count-rate MLEs.

.. code-block:: python

   from mixle.inference import PoolSpec, certify, plan_placement

   certificate = certify(model)
   placement = plan_placement(certificate, PoolSpec(available=False))

   print(certificate.table())
   print(placement.report())

Placement is advisory: closed-form, convex, and EM blocks stay local; gradient
blocks can become pool-eligible when a pool is configured and the estimated
work clears the threshold.

Streaming
---------

Streaming estimators update across batches. Bayesian streaming can carry a
posterior forward as the next batch's prior.

.. code-block:: python

   from mixle.inference import BayesianStreamingEstimator

   stream = BayesianStreamingEstimator(estimator)
   for batch in batches:
       model = stream.update(batch)

Use streaming when the dataset is naturally batched, too large for one pass, or
needs recursive updating.

Simulation and Verified Synthesis
---------------------------------

``simulate`` packages a fitted generative model as a data generator. For
Bayesian-network-like models, named scenarios can apply interventions.

.. code-block:: python

   from mixle.inference import simulate

   sim = simulate(model)
   baseline_rows = sim.run(100, seed=0)

For learned Bayesian networks, ``sim.scenario(name, interventions)`` registers
a named ``do``-operator scenario and ``compare`` estimates scenario effects.

``synthesize`` builds a dataset by drawing inputs, optionally labeling them
with a teacher, and keeping only rows that pass a verifier:

.. code-block:: python

   from mixle.inference import synthesize

   def draw(rng):
       return float(rng.normal())

   dataset = synthesize(
       draw,
       label=lambda x: "positive" if x > 0 else "negative",
       verify=lambda x, y: y in {"positive", "negative"},
       n=50,
       seed=0,
   )

   print(dataset.acceptance_rate)
   print(dataset.recheck())

When the source is a list of real rows, ``synthesize`` records an
exchangeability check in the dataset provenance because sampling "more rows
like these" assumes the source rows can be pooled.

Reproducibility Receipts
------------------------

``record_fit`` captures the data fingerprint, seed, estimator type, and fitted
parameter fingerprint. ``verify_reproducible`` refits and checks whether the
same parameters are recovered.

.. code-block:: python

   from mixle.inference import record_fit, verify_reproducible

   receipt = record_fit(model, rows, seed=0, estimator=estimator)
   check = verify_reproducible(estimator, rows, receipt, seed=0)

   print(check["reproducible"])

Fingerprints round floating-point values to a fixed precision before hashing,
so last-bit platform noise does not invalidate an otherwise equivalent fit.

Receipts are strongest when they are paired with a small prediction or
log-density fixture. The fingerprint says the parameters round-trip; the
fixture says the fitted object still behaves the same after reload.

Uncertainty Dispatch
--------------------

``uq`` chooses an uncertainty route from the object it receives:

* fitted Mixle model plus fitting data: Laplace parameter posterior;
* point predictor or Torch module plus ``(X_cal, y_cal)``: split-conformal
  prediction intervals;
* list of predictors: ensemble disagreement plus conformal intervals;
* LLM-like callable: semantic entropy over sampled generations.

.. code-block:: python

   from mixle.inference import uq

   uncertainty = uq(model, rows)
   lo, hi = uncertainty.credible_interval(lambda m: float(m.log_density(rows[0])))

Use specialized UQ functions when you already know the route. Use ``uq`` when
the caller owns a heterogeneous object and wants a single front door.

Backends and Engines
--------------------

The same model can be fitted locally, on a device engine, or on a distributed
encoded-data backend.

.. code-block:: python

   from mixle.engines import TorchEngine
   from mixle.inference import optimize

   local = optimize(data, est, out=None)
   gpu = optimize(data, est, engine=TorchEngine(device="cuda"), out=None)
   mp = optimize(data, est, backend="mp", num_workers=4, out=None)

Use :doc:`engines` for engine details and :doc:`data` for sources and encoded
payloads.

Learned Orchestration
---------------------

``mixle.inference.orchestration`` learns from telemetry rows produced by the
runtime layer. The initial policies defer to static rules when the feature
region is thin, and use historical outcomes only where nearby examples support
the learned decision.

.. code-block:: python

   from mixle.inference import learn_action_policy
   from mixle.telemetry import Telemetry

   telemetry = Telemetry()
   rows = telemetry.training_rows("route")
   # policy = learn_action_policy(rows)

Use these helpers with :doc:`reasoning-ecosystem`. They are for application
routing, placement, and scheduling decisions, not for replacing the statistical
fit route itself.

Event Studies
-------------

``hierarchical_event_study`` estimates confirmed-exposure influence from
per-subject pre/post effects and optional exposed-non-actor controls. Helpers
compute Gaussian mean shifts and Poisson log-rate shifts, then pool them with a
random-effects meta-analysis and report a difference-in-differences contrast
when controls are present.

Use this for timestamped interventions with a defensible exposure time. The
result includes a sensitivity bound via ``tipping_drift``; it does not remove
the need for study-design assumptions.

Diagnostics and Comparison
--------------------------

The inference namespace also includes:

* calibration diagnostics and conformal prediction;
* cross-validation splitters;
* model comparison tests and ELPD comparison;
* MCMC diagnostics such as R-hat, ESS, Geweke, and MCSE;
* bootstrap and permutation inference;
* robust and sandwich covariance estimators;
* proper scoring rules.

See :doc:`inference-toolkit` for the detailed map of scoring rules,
calibration, conformal prediction, cross-validation, model comparison,
multiple testing, regression, nonparametric tests, survival models, posterior
helpers, MCMC diagnostics, resampling, and decision utilities.

Production
----------

``mixle.inference.production`` adds provenance headers, registries, scoring
services, drift detection, monitors, and checkpointing. See :doc:`production`.

API Map
-------

.. list-table::
   :header-rows: 1

   * - Area
     - Imports
   * - fitting
     - ``optimize``, ``fit``, ``initialize``, ``estimate``, ``best_of``
   * - EM
     - ``EMStrategy``, ``run_em`` and strategy classes in ``mixle.inference.em``
   * - streaming
     - ``StreamingEstimator``, ``BayesianStreamingEstimator``
   * - priors and Bayes
     - ``mixle.inference.priors``, ``conjugate_posterior``
   * - diagnostics
     - :doc:`inference-toolkit`, ``mixle.inference.diagnostics``, scoring, calibration, model comparison
   * - production
     - ``mixle.inference.production``
   * - creation and certificates
     - ``create``, ``CreatedModel``, ``certify``, ``plan_estimation``,
       ``schedule``, ``plan_placement``
   * - simulation and synthesis
     - ``simulate``, ``Simulator``, ``synthesize``, ``Dataset``
   * - reproducibility
     - ``record_fit``, ``verify_reproducible``, ``ReproReceipt``
   * - UQ dispatch
     - ``uq``, ``UQResult``
   * - learned orchestration
     - ``learn_action_policy``, ``learn_placement_policy``,
       ``learn_schedule_policy``, ``meta_improve``
   * - reusable capabilities
     - ``skill``, ``Skill``, ``SkillRegistry``
   * - event studies
     - ``gaussian_effect``, ``poisson_lograte_effect``,
       ``hierarchical_event_study``, ``tipping_drift``
