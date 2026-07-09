Enumeration and Ranking
=======================

Many Mixle distributions can traverse their support in probability order. This
is useful for exact top-k decoding, support inspection, probability mass
summaries, and search procedures where the best few structured values matter
more than random samples.

Enumeration is capability-driven. Always ask whether a fitted object supports
the operation before writing code that depends on it.

The goal is not only to get a ranked list. The goal is to know what guarantee
the ranked list carries: exact traversal, finite support, approximate rank,
bounded tail mass, or relation-specific feasibility.

1. Start With a Categorical Distribution
----------------------------------------

.. code-block:: python

   import numpy as np
   from mixle.enumeration import supports_enumeration, top_k
   from mixle.stats import CategoricalDistribution

   dist = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})

   print(supports_enumeration(dist))
   for value, log_p in top_k(dist, 3):
       print(value, np.exp(log_p))

``top_k`` returns values and log probabilities. Keeping log probabilities
avoids underflow when structured values combine many factors.

Do not convert to ordinary probabilities until presentation time. For long
structured values, probability-space multiplication can underflow even when the
ranking itself is well defined in log space.

When the result becomes an artifact, store the log scores and the model version
that produced them. A ranked list without its scoring context is difficult to
audit or reproduce.

2. Compose Enumerable Children
------------------------------

If children are enumerable, structured records can be enumerable too.

.. code-block:: python

   from mixle.stats import CompositeDistribution, IntegerCategoricalDistribution

   record_dist = CompositeDistribution(
       [
           IntegerCategoricalDistribution(0, [0.6, 0.4]),
           CategoricalDistribution({"x": 0.7, "y": 0.3}),
       ]
   )

   for value, log_p in top_k(record_dist, 3):
       print(value, np.exp(log_p))

The output values are whole records, not independent per-field answers. The
first result is the most likely joint assignment.

This distinction matters when fields are later constrained together. If a
caller needs the best feasible assignment under a global constraint, use a
relation or constrained search surface rather than independently taking each
field's local argmax.

Add a boundary example when testing structured enumeration. A record whose
fields are individually likely but jointly less likely is a good way to verify
that the workflow is ranking whole records rather than field-wise summaries.

3. Inspect the Guarantee
------------------------

Decomposable distributions can often answer rank and seek queries exactly.
Latent marginal models, such as mixtures and HMMs, may return bounds or
certified estimates because exact marginal ranking can be much harder.

Use the capability and result objects rather than assuming every distribution
has the same guarantee:

.. code-block:: python

   import mixle

   print(mixle.describe(record_dist))
   print(mixle.capabilities(record_dist))

If a required guarantee is missing, either change the model family or introduce
an approximation explicitly in the workflow.

Approximate ranking should be documented as approximate in the artifact or
report that consumes it. Include the method, window size, bound, or sampler
settings that make the approximation reproducible.

If the guarantee is not strong enough for the decision, change the model family
or the decision workflow. Do not silently promote an approximate ranking into an
exact release claim.

4. Summarize Mass
-----------------

Top-k traversal is also a diagnostic: it tells you whether probability mass is
concentrated in a few outcomes or spread across a long tail.

.. code-block:: python

   top = top_k(record_dist, 4)
   mass = sum(np.exp(log_p) for _, log_p in top)
   print(mass)

For finite supports, top-k mass can be used to decide whether exact
enumeration is enough for a report or whether sampling/quantization is needed.

For infinite or very large supports, top-k mass is only a partial diagnostic.
Record whether the omitted tail is bounded, estimated, or simply unknown.

Score gaps matter as much as order. If the first few values are nearly tied,
the downstream policy should see that ambiguity instead of receiving only the
top value.

5. Use Relations for Feasible Sets
----------------------------------

``mixle.relations`` exposes the same best-first idea for feasible sets such as
paths, assignments, edit neighborhoods, spanning trees, and subset regression.
The relation is the specification; the enumerator yields ranked solutions.

Use distribution enumeration when the object is probabilistic support. Use
relations when the object is a structured feasible set or optimization problem.

Read :doc:`/enumeration` for traversal algorithms and :doc:`/relations` for
optimization-shaped ranking.

Validation Notes
----------------

Before relying on an enumerator in a public workflow:

* confirm the object advertises the enumeration or rank/seek capability you
  need;
* verify that returned values are feasible under the model or relation;
* compare top-k mass or score gaps against the decision threshold that will use
  the result;
* test near ties, impossible values, and constrained structured records;
* preserve log scores in artifacts when exact reproducibility matters; and
* name any approximation explicitly in reports and release evidence.
