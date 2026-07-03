Enumeration and Ranking
=======================

Many Mixle distributions can traverse their support in probability order. This
is useful for exact top-k decoding, support inspection, probability mass
summaries, and search procedures where the best few structured values matter
more than random samples.

Enumeration is capability-driven. Always ask whether a fitted object supports
the operation before writing code that depends on it.

1. Start With A Categorical Distribution
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

3. Inspect The Guarantee
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

5. Use Relations For Feasible Sets
----------------------------------

``mixle.relations`` exposes the same best-first idea for feasible sets such as
paths, assignments, edit neighborhoods, spanning trees, and subset regression.
The relation is the specification; the enumerator yields ranked solutions.

Use distribution enumeration when the object is probabilistic support. Use
relations when the object is a structured feasible set or optimization problem.

Read :doc:`/enumeration` for traversal algorithms and :doc:`/relations` for
optimization-shaped ranking.
