Relations And Operations
========================

This tutorial shows how Mixle separates two ideas that often get tangled:
probability transformations and structured decision constraints.

* Use ``mixle.ops`` when you transform a distribution.
* Use ``mixle.relations`` when you enumerate structured feasible outputs.

Quantize A Continuous Model
---------------------------

Start with a continuous distribution and convert it into a finite support when
enumeration is required.

.. code-block:: python

   from mixle.ops import quantize
   from mixle.stats import GaussianDistribution

   response_time = GaussianDistribution(120.0, 25.0)
   finite_response_time = quantize(response_time, bits=6)

   likely_bins = finite_response_time.enumerator().top(5)

This is useful when a downstream decision rule expects ranked finite outcomes.
The original model remains continuous; the quantized version is an operational
artifact.

Pool Two Experts
----------------

For compatible tractable families, a product of experts combines evidence by
adding log densities.

.. code-block:: python

   from mixle.ops import product_of_experts
   from mixle.stats import CategoricalDistribution

   prior = CategoricalDistribution({"approve": 0.6, "review": 0.3, "deny": 0.1})
   policy = CategoricalDistribution({"approve": 0.4, "review": 0.5, "deny": 0.1})

   pooled = product_of_experts([prior, policy], weights=[1.0, 0.7])

The result is another categorical distribution. It can be scored, enumerated,
serialized, or used as a component in a larger model.

Rank A Structured Assignment
----------------------------

Now suppose a model produced a cost matrix for assigning alerts to analysts.
The decision must be one-to-one, so this is a relation.

.. code-block:: python

   from mixle.relations import Assignment

   costs = [
       [0.2, 1.4, 0.8],
       [1.1, 0.3, 0.7],
       [0.9, 0.5, 0.4],
   ]

   assignment = Assignment(costs)
   best = assignment.solve()
   alternatives = assignment.top(5)

The alternatives are ranked feasible assignments. They are not random samples.

Connect Model Scores To Relations
---------------------------------

The common pattern is:

1. Fit or call a model to produce local scores.
2. Convert scores into relation costs.
3. Enumerate the best globally feasible solutions.
4. Optionally score or calibrate the chosen solution downstream.

For entity matching, a record model might estimate ``p(match | pair)`` for each
candidate pair. The assignment relation then enforces that each entity is used
at most once.

Keep Provenance Clear
---------------------

When this workflow enters production, store both sides:

* the model artifact that produced scores;
* the operation artifacts, such as quantization or expert pooling;
* the relation type and objective used to choose the structured output;
* the top alternatives when ambiguity matters.

This makes later debugging far easier than storing only the final decision.

