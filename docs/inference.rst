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
