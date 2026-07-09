Cookbook
========

These recipes answer the questions that come up when you are writing code
against ``mixle`` rather than reading about it.

Treat the recipes as starting points. Before moving one into production, add a
held-out score, route explanation, missing-data policy, and artifact reload
check that match the risk of the workflow.

Fit a scalar family
-------------------

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats import GaussianEstimator

   model = optimize([1.0, 1.2, 0.9, 1.4], GaussianEstimator(), out=None)
   print(model.mu, model.sigma2)

Fit a heterogeneous row
-----------------------

Use ``CompositeEstimator`` for positional records and ``RecordEstimator`` for
named records. The estimator children must match the observation shape.

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats import CategoricalEstimator, CompositeEstimator, GaussianEstimator

   rows = [("us", 42.0), ("ca", 39.0), ("us", 44.0)]
   est = CompositeEstimator((CategoricalEstimator(), GaussianEstimator()))
   model = optimize(rows, est, out=None)
   print(model.log_density(("us", 41.0)))

Fit a variable-length sequence
------------------------------

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats import CategoricalEstimator, PoissonEstimator, SequenceEstimator

   sequences = [[3, 4, 5], [2, 3], [5, 4, 4, 6]]
   est = SequenceEstimator(PoissonEstimator(), len_estimator=CategoricalEstimator())
   model = optimize(sequences, est, out=None)

Use a prototype distribution
----------------------------

When you know the model shape, pass a prototype instead of an estimator. The
matching estimator is derived from that shape. For latent models, pass the
prototype as ``prev_estimate`` as well when its parameter values should seed the
fit.

.. code-block:: python

   from mixle.inference import optimize
   from mixle.stats import GaussianDistribution, MixtureDistribution

   proto = MixtureDistribution(
       [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
       [0.5, 0.5],
   )
   model = optimize(data, proto, prev_estimate=proto, out=None)

Let mixle infer an estimator
----------------------------

.. code-block:: python

   from mixle.inference import estimate, initialize
   from mixle.utils.automatic import get_estimator

   est = get_estimator(data, pseudo_count=1.0e-4)
   init = initialize(data, est, rng=1)
   model = estimate(data, est, prev_estimate=init)

Print or persist the inferred estimator before relying on it. For a fuller
field-by-field explanation, use ``recommend_model`` from the next recipe and
store low-confidence fields with the artifact.

Use multiple starts for a mixture
---------------------------------

Latent models can settle into local optima. Prefer ``best_of`` when the model is
small enough for multiple restarts.

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

Reduce a Gaussian mixture without sampling
------------------------------------------

Use ``reduce_mixture`` when a fitted Gaussian mixture has more components than
you want to serve or compare. The reduction is closed form for Gaussian
mixtures, preserves the overall first two moments, and avoids the sample/refit
loop used by ``mixle.ops.project``.

.. code-block:: python

   from mixle.inference import reduce_mixture

   compact = reduce_mixture(fitted_mixture, n_components=4)
   print(len(compact.w))

Share embeddings across language-model experts
-----------------------------------------------

This recipe uses ``mixle.models``, which is an incubating applied-helper
namespace. Use it when several language-model experts should share token
semantics instead of learning duplicate embedding tables.

.. code-block:: python

   from mixle.models import CategoricalEmbedding, TransformerLMEstimator
   from mixle.stats import MixtureEstimator

   embedding = CategoricalEmbedding(num_categories=8000, dim=256, name="word")
   experts = [
       TransformerLMEstimator(8000, d_model=256, n_layer=4, block=64, embedding=embedding)
       for _ in range(3)
   ]
   est = MixtureEstimator(experts)

Recommend a model from data
---------------------------

.. code-block:: python

   from mixle.task import recommend_model

   rec = recommend_model(rows)
   for field in rec.fields:
       print(field.path, field.family, field.runner_up, field.gap_bits)
   model = rec.fit(rows, max_its=30, out=None)

Keep the recommendation report with the model. A low ``gap_bits`` value means
the family choice is fragile and should be reviewed, not hidden inside the fit.

Distill an LLM labeler
----------------------

.. code-block:: python

   from mixle.task import CallableLLM, active_distill, llm_labeler

   teacher = llm_labeler(CallableLLM(generate), ["spam", "ham"])
   active = active_distill(teacher, unlabeled_texts, budget=60, seed_size=20, rounds=4)
   local_model = active.model

Serve a calibrated cascade
--------------------------

.. code-block:: python

   from mixle.task import CalibratedTaskModel, Cascade, CostModel

   calibrated = CalibratedTaskModel(local_model, alpha=0.1).calibrate(cal_x, teacher(cal_x))
   cascade = Cascade(calibrated, teacher, cost=CostModel(c_frontier=0.01, c_local=0.00001))
   predictions = cascade.serve(requests)
   print(cascade.report())

Before replacing a teacher, run a scorecard on held-out or shadow traffic and
check rare labels separately.

Replace an existing task function
---------------------------------

Use ``solve`` when you have code that already performs a task and want a
calibrated local model in front of it.

.. code-block:: python

   from mixle.task import solve

   def route(ticket):
       return "review" if ticket["amount"] > 5000 else "approve"

   solution = solve(route, historical_tickets, seed=0)
   print(solution(new_ticket))
   print(solution.report())

Replace a numeric task function
-------------------------------

Use ``solve_regression`` when the routine returns a scalar and local answers
are acceptable only within an application tolerance.

.. code-block:: python

   from mixle.task import solve_regression

   def price(item):
       base = {"basic": 20.0, "pro": 80.0, "max": 150.0}[item["tier"]]
       return base + 0.5 * item["size"]

   solution = solve_regression(price, historical_items, tol=5.0, alpha=0.1)
   value = solution(new_item)
   yhat, lo, hi = solution.interval(new_item)

Replace a multi-label tagger
----------------------------

Use ``solve_multilabel`` when the teacher returns zero or more tags. The local
student answers only when every label is decided as present or absent.

.. code-block:: python

   from mixle.task import solve_multilabel

   def flags(transaction):
       tags = []
       if transaction["amount"] > 400:
           tags.append("high-value")
       if transaction["region"] == "eu":
           tags.append("eu-rules")
       return tags

   solution = solve_multilabel(flags, historical_transactions, alpha=0.1)
   tags = solution(new_transaction)

Replace a structured output function
------------------------------------

Use ``solve_structured`` when the teacher returns a dictionary with a stable
schema. Numeric output fields need explicit tolerances.

.. code-block:: python

   from mixle.task import solve_structured

   def enrich(ticket):
       return {
           "route": "finance" if ticket["amount"] > 10_000 else "ops",
           "priority": "high" if ticket["age_hours"] > 24 else "normal",
           "reserve": ticket["amount"] * 0.15,
       }

   solution = solve_structured(enrich, historical_tickets, tol={"reserve": 25.0})
   output = solution(new_ticket)

Route across several task tiers
-------------------------------

.. code-block:: python

   from mixle.task import Router

   router = Router.from_solutions(
       [low_cost_solution, compact_solution],
       teacher=frontier_teacher,
       costs=[0.0001, 0.001, 0.03],
   )

   answer = router(request)
   print(router.report())

Quantize a distilled student
----------------------------

.. code-block:: python

   from mixle.task import quantize_mlp

   q_student = quantize_mlp(student, bits=8)
   q_student.save("student-int8")

Quantify LLM uncertainty
------------------------

.. code-block:: python

   from mixle.reason import LLMUncertainty

   uq = LLMUncertainty(generate, n=20)
   assessment = uq.assess(prompt)
   print(assessment.answer, assessment.confidence, assessment.semantic_entropy)

   uq.calibrate(calibration_examples, alpha=0.1)
   maybe_answer = uq.answer(prompt)

Fuse finite-hypothesis evidence
-------------------------------

.. code-block:: python

   import numpy as np
   from mixle.reason import reason_discrete

   answer = reason_discrete(
       ["ok", "fault"],
       [
           ("sensor", np.array([-0.1, -2.0])),
           ("operator_note", np.array([-1.5, -0.2])),
       ],
   )

   print(answer.top())

Use a graph-producing LLM
-------------------------

.. code-block:: python

   from mixle.reason import GraphLLM

   graph_llm = GraphLLM(generate, parse_triples, n=20)
   graph_dist = graph_llm.distribution(prompt)
   print(graph_dist.edge_marginals())

Sample from a fitted model
--------------------------

.. code-block:: python

   sampler = model.sampler(seed=0)
   simulated = sampler.sample(10)

Inspect why an operation is unavailable
---------------------------------------

.. code-block:: python

   import mixle

   print(mixle.describe(model))
   print(mixle.capabilities(model))

Run the same fit on a backend
-----------------------------

.. code-block:: python

   from mixle.inference import optimize

   local = optimize(data, estimator, backend="local", out=None)
   mp = optimize(data, estimator, backend="mp", num_workers=4, out=None)

Move array math to Torch
------------------------

.. code-block:: python

   from mixle.engines import TorchEngine
   from mixle.inference import optimize

   model = optimize(data, estimator, engine=TorchEngine(device="cuda"), out=None)

Get top-k support values
------------------------

.. code-block:: python

   from mixle.enumeration import top_k

   for value, log_p in top_k(distribution, 5):
       print(value, log_p)

Create a durable production fit
-------------------------------

.. code-block:: python

   from mixle.inference.production import fit_with_provenance

   model, header = fit_with_provenance(data, estimator, seed=1)
   print(header.dataset_hash, header.model_hash)

Verify a saved task artifact
----------------------------

.. code-block:: python

   from mixle.task import TaskModel

   local_model.save("student-artifact")
   restored = TaskModel.load("student-artifact")
   assert restored(example) == local_model(example)

Use the documented ``save``/``load`` helpers for the artifact family you are
serving. The point of the recipe is the check: a model is not ready for service
until a fresh object can reproduce representative scoring or prediction
behavior.
