Relations and Operations
========================

This tutorial shows how Mixle separates two ideas that often get tangled:
probability transformations and structured decision constraints.

* Use ``mixle.ops`` when you transform a distribution.
* Use ``mixle.relations`` when you enumerate structured feasible outputs.

Quantize a Continuous Model
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

Record both objects when reproducibility matters. The continuous model explains
the statistical assumption, while the quantized artifact explains the finite
decision surface that was actually enumerated.

Quantization settings are part of the decision artifact. If the number of bits,
bucket boundaries, or support window changes, the ranked outcomes can change
even when the source distribution is identical.

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

Pooling is not a vote unless the inputs were calibrated to mean the same thing.
Before using pooled scores in a decision rule, inspect whether each expert's
support, units, and confidence scale are compatible with the chosen weights.

When expert weights are tuned, keep the tuning data separate from the review
data. Otherwise the pooled distribution can appear better calibrated than it
will be on new cases.

Rank a Structured Assignment
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
Use them for review, fallback, or sensitivity analysis: if the best and second
best assignments are nearly tied, the decision may need calibration or human
inspection even when the solver returns a valid optimum.

Connect Model Scores to Relations
---------------------------------

The common pattern is:

1. Fit or call a model to produce local scores.
2. Convert scores into relation costs.
3. Enumerate the best globally feasible solutions.
4. Optionally score or calibrate the chosen solution downstream.

For entity matching, a record model might estimate ``p(match | pair)`` for each
candidate pair. The assignment relation then enforces that each entity is used
at most once.

The relation should not be asked to repair invalid local scores. If the pair model
is poorly calibrated or missing important candidates, the global optimum will
still be the best solution under a flawed objective. Validate the scoring model
and the feasible-set construction separately.

Also validate the cost direction. Some relation APIs minimize costs while
probabilistic models naturally produce log scores or probabilities. The
conversion should be explicit enough that a reviewer can tell whether larger
model confidence became lower cost, higher reward, or a filtered candidate.

Keep Provenance Clear
---------------------

When this workflow enters production, store both sides:

* the model artifact that produced scores;
* the operation artifacts, such as quantization or expert pooling;
* the relation type and objective used to choose the structured output;
* the top alternatives when ambiguity matters.

This makes later debugging far easier than storing only the final decision.

Validation Pattern
------------------

For a release or production review, keep four checks separate:

* distribution operations preserve documented semantics, such as normalized
  probabilities after pooling or finite support after quantization;
* relation solvers return feasible outputs and expose alternatives when the
  API promises ranking;
* score-to-cost conversion is documented and tested on boundary cases; and
* downstream calibration or review policy sees the ambiguity information it
  needs, not just the single winning assignment.

For release evidence, include at least one boundary case where the best and
second-best feasible outputs are close. Near ties are where the distinction
between model score, operation artifact, and relation constraint matters most.

This separation makes failures easier to assign. An incorrect final decision might
come from a model score, an operation, a relation constraint, or a deployment
policy. The artifacts above should make that distinction visible.
