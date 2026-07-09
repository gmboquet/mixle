Automatic Inference
===================

Automatic inference in ``mixle`` means the fitting route follows from explicit
model structure. It does not mean every decision is hidden. The library tries
to expose the chosen shape, confidence gaps, fallback path, and model
capabilities so you can decide whether to trust the result or collect better
data.

There are five common entry points:

.. list-table::
   :header-rows: 1

   * - You provide
     - Call
     - Result
   * - an estimator
     - ``optimize(data, estimator)``
     - fit exactly the structure you requested
   * - a prototype distribution
     - ``optimize(data, prototype)``
     - derive the matching estimator from the prototype shape
   * - raw data only
     - ``optimize(data)`` or ``get_estimator(data)``
     - infer a first estimator from observed types and profiles
   * - raw data plus an audit boundary
     - ``create(data, ...)``
     - fit a model and return an artifact with provenance, certificate,
       optional calibration, uncertainty, and exchangeability diagnostics
   * - raw data plus an LLM designer
     - ``design_model(data, llm)``
     - ask for an allowlisted spec, build it, fit-validate it, and fall back on failure

Explicit Estimator
------------------

Use this when you know the model you want.

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats import CompositeEstimator, GammaEstimator, PoissonEstimator

   est = CompositeEstimator((PoissonEstimator(), GammaEstimator()))
   model = optimize(rows, est, max_its=50, out=None)

The estimator chooses the model family. The structure chooses the inference
route: no latent wrapper means ordinary estimation; a mixture or HMM adds EM; a
neural child adds gradient training inside its M-step.

Prototype Distribution
----------------------

Use a prototype when the shape is clearer as a model than as an estimator.
``optimize`` coerces the prototype to its matching estimator. If the prototype
parameters should also be the initialization, pass the same object as
``prev_estimate``.

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats import GaussianDistribution, MixtureDistribution

   proto = MixtureDistribution(
       [GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0)],
       [0.5, 0.5],
   )
   model = optimize(reals, proto, prev_estimate=proto, max_its=100, out=None)

This is the pattern to use for latent models where initial component locations
matter. Passing only ``proto`` still runs, but it uses the prototype for shape
rather than guaranteeing those parameter values as the starting point.

Infer an Estimator from Data
----------------------------

For a first pass over heterogeneous Python data:

.. code-block:: python

   from mixle.inference import optimize
   from mixle.utils.automatic import get_estimator

   est = get_estimator(rows, pseudo_count=1.0e-4)
   model = optimize(rows, est, out=None)

Or use the shorthand:

.. code-block:: python

   model = optimize(rows, out=None)

Automatic typing is useful for exploration and baselines. For production, keep
the returned estimator or use ``recommend_model`` so the decision is visible.
Do not treat automatic typing as data cleaning. Missing-value handling, field
dropping, and dependency choices should be visible in the returned estimator,
profile, recommendation report, or artifact provenance.

Certified Artifact Creation
---------------------------

``create`` is the higher-level route when the fit itself should become an
auditable artifact. It still uses the same estimator and inference machinery,
but it packages the fitted model with provenance and optional validation
receipts.

.. code-block:: python

   from mixle.inference import create

   artifact = create(rows, calibrate=0.2, quantify_uq=True, seed=0)
   model = artifact.model

   print(artifact.certificate.level)
   print(artifact.provenance.get("exchangeability"))

Use ``optimize`` when you want the fitted model directly. Use ``create`` when a
workflow needs to retain how the model was built, what checks were run, and
which guarantees were available at creation time.

Model Recommendation
--------------------

``recommend_model`` wraps structural profiling in a report a program can act
on.

.. code-block:: python

   from mixle.task import recommend_model

   rec = recommend_model(rows)
   print(rec.estimator)

   for field in rec.fields:
       print(field.path, field.family, field.runner_up, field.gap_bits, field.confident)

   for line in rec.explain():
       print(line)

   model = rec.fit(rows, max_its=30, out=None)

The recommendation includes:

* a ready estimator;
* per-field family choices;
* a runner-up family when there is a real alternative;
* a bit-gap showing how clear the choice was;
* low-confidence fields where more data would sharpen the model;
* pairwise dependency hints that argue for joint modeling.

Use ``rec.low_confidence_fields()`` to find the columns where the model choice
is still fragile.

Promotion Evidence
------------------

An automatically proposed model is ready to promote only after it has evidence
outside the fitting pass. For a production-facing workflow, keep:

* the profile or recommendation report that selected the estimator;
* field-level runner-up families and score gaps;
* any warnings about ignored fields, missingness, or dependency hints;
* a held-out likelihood, calibration, or task metric against a simpler
  baseline;
* the exact estimator or model specification used for the promoted run;
* the random seed, split definition, and optional dependency versions needed to
  repeat the fit.

Small score gaps, dense dependency hints, and fields with high missingness are
not failures by themselves. They are review items. A mature workflow either
uses domain knowledge to make those choices explicit or collects more data
before allowing the automatic recommendation to become the deployed model.

LLM-Designed Models
-------------------

``design_model`` lets an LLM propose a model shape from a compact data profile.
The LLM does not get to execute code. It emits JSON from an allowlist, mixle
builds the estimator, and then mixle fits a sample before accepting it.

.. code-block:: python

   from mixle.task import design_model

   designed = design_model(rows, llm)
   print(designed.source)  # "llm" or "fallback"
   print(designed.spec)

   model = designed.fit(rows, max_its=30, out=None)

Allowed specs include scalar families, composites, and mixtures:

.. code-block:: python

   {"family": "gamma"}
   {"type": "composite", "fields": [{"family": "categorical"}, {"family": "gamma"}]}
   {"type": "mixture", "k": 3, "component": {"family": "student_t"}}

If parsing, building, or fit-validation fails, the result falls back to
``recommend_model`` when ``fallback=True``.

The returned ``source`` and ``note`` fields are part of the contract. A model
whose source is ``"fallback"`` should be documented as an automatic-profile
model, not as an LLM-designed model. If the LLM proposal is accepted, keep the
allowlisted JSON spec and the fit-validation result with the artifact so the
design can be reproduced without replaying an LLM conversation.

PPL Route Selection
-------------------

The PPL lowers formulas to the same estimator/target machinery. ``how="auto"``
chooses a route from the lowered model:

.. code-block:: python

   from mixle.ppl import Markov, Normal, free

   hmm = Markov(Normal(free, free), states=3).fit(sequences, how="auto")

Common route families include conjugate updates, EM, MAP, Laplace, variational
inference, MCMC, HMC, NUTS, ensembles, and hierarchical routes. Use
``explain_fit`` when available, or ``mixle.describe`` on the lowered/fitted
object, to inspect what the automatic route selected.

Record the route explanation when a PPL fit becomes an artifact. The important
distinction is whether ``auto`` produced an analytic update, an EM fixed point,
a point-estimate MAP route, or a posterior-bearing route. Those outcomes have
different uncertainty and reproducibility implications even when they share the
same model expression.

Objectives
----------

``optimize`` and ``fit`` accept ``objective=``:

.. list-table::
   :header-rows: 1

   * - Objective
     - Meaning
   * - ``"auto"``
     - prior/ELBO-aware default; choose MLE, MAP, or VB as appropriate
   * - ``"mle"``
     - maximize observed-data likelihood
   * - ``"map"``
     - maximize penalized likelihood when the estimator carries priors
   * - ``"vb"``
     - use variational evidence lower bound when the model exposes one

The default is usually the right choice. Force an objective when you are testing
or comparing routes.

Backends and Engines
--------------------

Automatic inference does not force a local CPU path. The model stays the same:

.. code-block:: python

   from mixle.engines import TorchEngine
   from mixle.inference import optimize

   gpu = optimize(data, est, engine=TorchEngine(device="cuda"), out=None)
   mp = optimize(data, est, backend="mp", num_workers=4, out=None)
   spark = optimize(data, est, backend="spark", out=None)

``engine=`` controls array/device math. ``backend=`` controls where encoded
data are folded.

Failure Modes to Watch
----------------------

.. list-table::
   :header-rows: 1

   * - Symptom
     - What to do
   * - low recommendation gap
     - collect more data or choose the family explicitly
   * - mixture fit changes across runs
     - use ``best_of`` with validation data
   * - LLM-designed spec falls back
     - inspect ``designed.note`` and the allowed-family list
   * - neural fit is slow
     - reduce model size first, then move to Torch/GPU or streaming
   * - automatic route lacks a capability
     - call ``mixle.describe(model)`` and choose a model that supports it

A good workflow is exploratory but auditable: let mixle propose a shape, look
at the confidence and capabilities, then make the important choices explicit
once the model matters.
