Model Lifecycle
===============

``mixle.Model`` is a convenience facade over the library's main lifecycle:
propose a model, fit it, evaluate it, inspect it, query it, distill it, and
deploy it. It does not introduce a separate inference engine. It gives users one
place to stand while delegating to the same distribution, inference,
capability, task, and artifact systems used elsewhere.

Use it when a workflow needs a durable object with consistent verbs. Use the
lower-level APIs directly when you are developing a new estimator, benchmarking
an inference route, or controlling each E-step and M-step explicitly.

Basic Flow
----------

.. code-block:: python

   import mixle

   model = mixle.Model().fit(rows)
   quality = model.evaluate(holdout)
   samples = model.sample(5, seed=0)
   explanation = model.explain()

   print(quality["mean_log_density"])
   print(explanation)

``Model(spec=None)`` means "infer the estimator from data at fit time." The
``spec`` may also be an estimator or a prototype distribution:

.. code-block:: python

   from mixle.stats import GaussianEstimator, GaussianDistribution

   from_estimator = mixle.Model(GaussianEstimator()).fit(values)
   from_prototype = mixle.Model(GaussianDistribution(0.0, 1.0)).fit(values)

Fitting delegates to ``mixle.inference.optimize``. Scoring, sampling,
enumeration, and posterior calls delegate to the fitted distribution.

The facade should not hide the fitted object. When a workflow depends on a
specific capability, inspect ``model.fitted`` with ``mixle.describe`` and store
the fitted distribution type in the artifact notes.

Certified Creation Alternative
------------------------------

Use ``mixle.inference.create`` when you want a fitted artifact plus explicit
post-conditions rather than a convenience lifecycle facade.

.. code-block:: python

   from mixle.inference import create

   artifact = create(rows, calibrate=0.2, quantify_uq=True, seed=0)
   print(artifact.certificate.table())

``CreatedModel`` carries the fitted model, an estimation certificate, optional
calibration and UQ objects, and provenance. ``Model`` is still the shorter
interactive facade; ``create`` is the stronger artifact boundary.

Use ``create`` for release-like evidence when the certificate, calibration, or
provenance will be reviewed independently from the notebook that produced the
fit.

Proposal Frontier
-----------------

``mixle.propose`` builds a candidate frontier and returns a ``Model`` whose
``spec`` is the best candidate:

.. code-block:: python

   proposed = mixle.propose(rows, fit=False)

   for candidate in proposed.frontier:
       print(candidate)

   proposed.fit(rows)

The frontier can include:

* ``recommend_model`` from ``mixle.task`` for dependency-aware structural
  recommendation;
* the plain automatic estimator from ``mixle.utils.automatic.get_estimator`` as
  an independence baseline;
* an LLM-designed model from ``mixle.task.design_model`` when an LLM handle is
  provided.

Each candidate is fit on a train split and scored on held-out data. Candidate
failures are reported in the frontier instead of being silently ignored. The
winning estimator becomes the model spec, while field-level recommendation
notes and dependency hints are stored in ``Model.notes``.

Keep the full frontier when model selection matters. The rejected candidates
and their failure notes explain why the chosen spec was accepted and prevent
the proposal step from becoming an opaque recommendation.

Automatic Restart Guard
-----------------------

Latent-variable models can land on symmetric saddle points, especially mixtures
whose components start identical. ``Model.fit`` uses ``restarts="auto"`` by
default:

.. code-block:: python

   model = mixle.Model(mixture_estimator).fit(rows)

The fit first runs the ordinary route. If the fitted object exposes a posterior
and components, Mixle checks whether every sampled observation has an almost
uniform component posterior. When that pattern suggests a symmetric saddle, the
model is refit with symmetry-breaking restarts and the better likelihood is
kept. Notes record whether the automatic restart changed the result.

Pass an integer to force a number of restarts, or ``restarts=None`` to keep the
raw single fit.

For latent models, record whether the restart guard changed the result. A
model that needed symmetry-breaking restarts should carry that fact into
provenance.

Query Verbs
-----------

Once fitted, the lifecycle object exposes common distribution queries:

.. code-block:: python

   logp = model(x)
   top = model.enumerate().top_k(10)
   z = model.posterior(x)
   forecast = model.forecast(history, horizon=7)
   effect = model.do({"treatment": "on"})
   parts = model.explain_prediction(x)

The available queries depend on the fitted model's capabilities. A continuous
Gaussian will not enumerate. A mixture can expose posterior responsibilities. A
Bayesian network can answer interventions when the necessary graph structure is
available. Use ``mixle.describe(model.fitted)`` when you need to know what the
object supports.

Query verbs should preserve capability failures. If a fitted object cannot
enumerate, condition, forecast, or answer an intervention, the lifecycle layer
should report that limitation rather than inventing an unsupported answer.

Distillation
------------

``Model.distill`` routes into ``mixle.task.solve``:

.. code-block:: python

   solution = model.distill(teacher, examples, seed=0)
   answer = solution(new_input)

If ``teacher`` is omitted, the fitted model teaches from its latent posterior:
inputs are labeled by the most probable latent component. This turns a fitted
mixture into a calibrated local classifier of its own clusters.

The returned object is a task ``Solution``. It answers locally when calibrated
and escalates to the teacher when it should not guess. See
:doc:`task-distillation` and :doc:`task-serving`.

When latent posteriors become task labels, document that the labels are
model-derived. They are useful for distillation, but they are not external
ground truth unless a separate review validates them.

Deployment
----------

``Model.deploy`` writes a durable artifact directory:

.. code-block:: python

   path = model.deploy("artifacts/customer-regime-model")
   restored = mixle.Model.load(path)

The artifact contains the fitted model and a manifest with family, creation
time, fit metadata, notes, and artifact schema name. This is a lightweight
lifecycle artifact, not a full model registry. For production registry,
provenance, drift, and serving concepts, see :doc:`production`.

Load the artifact in a fresh process before relying on it. A deploy directory
that only works from the source checkout is not durable.

Reusable Skills
---------------

``mixle.inference.skill`` wraps a fitted model, ``CreatedModel``, or callable
as a named capability. A skill can be registered, searched by query, indexed
into a substrate, and used as a compute action by a reasoner.

.. code-block:: python

   from mixle.inference import SkillRegistry, skill

   registry = SkillRegistry()
   sk = skill(
       "sample-customers",
       model.fitted,
       description="sample synthetic customer rows",
       registry=registry,
   )

   print(registry.find("customer sample"))

Use skills when the fitted artifact becomes an application verb. Use
``Model.deploy`` or ``Registry`` when the artifact is primarily a model
version.

When to Use Lower-Level APIs
----------------------------

Use the facade when:

* you want a short path from raw data to a fitted, inspectable object;
* you are building examples, notebooks, or application-level integrations;
* you want proposal notes and held-out frontier scoring in one place;
* the model lifecycle matters more than the exact optimizer steps.

Use lower-level APIs when:

* you need a fixed estimator tree with no recommendation step;
* you are controlling initialization, streaming updates, or restarts manually;
* you are testing a new distribution family, capability, or backend;
* you need explicit access to encoded data, accumulators, kernels, or engines.

Release Evidence
----------------

For lifecycle workflows, preserve:

* fitted distribution type and capability report;
* proposal frontier and rejected candidates when automatic proposal is used;
* restart policy and whether it changed a latent fit;
* held-out evaluation output;
* artifact load smoke test from a fresh process; and
* escalation or calibration evidence for distilled task solutions.

API Reference
-------------

* :doc:`api/mixle.lifecycle`
* :doc:`api/mixle.inference`
* :doc:`api/mixle.task.recommend`
* :doc:`api/mixle.task.design`
* :doc:`api/mixle.task.solve`
* :doc:`api/mixle.inference.create`
* :doc:`api/mixle.inference.skill`
