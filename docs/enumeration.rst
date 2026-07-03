Enumeration and Ranking
=======================

Enumeration is the concern for models that can traverse their support in
descending probability order. It is used for exact top-k answers, support
inspection, rank queries, nucleus sets, structured decoding, and combinatorial
optimization over probabilistic models.

Enumeration is capability-driven. A distribution, relation, or quantized model
participates when it implements an enumerator and advertises the appropriate
capability.

First Calls
-----------

.. code-block:: python

   import numpy as np
   from mixle.enumeration import supports_enumeration, top_k
   from mixle.stats import CategoricalDistribution

   dist = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})

   print(supports_enumeration(dist))
   for value, log_p in top_k(dist, 3):
       print(value, np.exp(log_p))

The values returned by enumeration are model values. For a composite
distribution, the value is a whole tuple or record. For a Markov model, it may
be a sequence or path.

Enumerator Surface
------------------

Many enumerable objects expose ``enumerator()``:

.. code-block:: python

   enum = dist.enumerator()
   top = enum.top_k(10)
   nucleus = enum.top_p(0.95)
   rank = enum.rank(value)

Common operations:

``top_k(k)``
    Most probable values with log probabilities.

``top_p(p)``
    Smallest high-probability set covering probability mass ``p``.

``rank(value)``
    Number of values with strictly higher probability, when supported.

``seek(i)``
    Value near a probability rank or structural index, when supported.

Not every enumerable model supports every operation. Use ``mixle.describe`` to
inspect exact capabilities.

Capabilities
------------

.. list-table::
   :header-rows: 1

   * - Capability
     - Meaning
   * - ``Enumerable``
     - can iterate support in descending-probability order
   * - ``FiniteSupport``
     - support size is finite and can be queried
   * - ``RankableByIndex``
     - supports rank/seek or count-budget unranking

Check support explicitly:

.. code-block:: python

   import mixle
   from mixle.enumeration import Enumerable, RankableByIndex

   print(mixle.supports(dist, Enumerable))
   print(mixle.supports(dist, RankableByIndex))

Composed Supports
-----------------

Combinators preserve enumeration when their children support it.

.. code-block:: python

   from mixle.enumeration import top_k
   from mixle.stats import CompositeDistribution, IntegerCategoricalDistribution

   record = CompositeDistribution(
       [
           IntegerCategoricalDistribution(0, [0.6, 0.4]),
           CategoricalDistribution({"x": 0.7, "y": 0.3}),
       ]
   )

   for value, log_p in top_k(record, 5):
       print(value, log_p)

The enumeration is over whole records, not independent per-field lists.

Quantized and Count-Budget Indexes
----------------------------------

Large or infinite supports often need an index. ``quantized_index`` and
``count_budget_index`` build seek/unrank structures that trade memory and
accuracy for access to high-probability regions.

.. code-block:: python

   from mixle.enumeration import count_budget_index, quantized_index

   q_index = quantized_index(dist, budget=4096)
   c_index = count_budget_index(dist, budget=4096)

Use these when top-k traversal is too slow but you still need structured access
to likely support values.

Latent Models and HMM Paths
---------------------------

Exact marginal ranking for mixtures and HMMs can be hard. Mixle provides
specialized algorithms and reports the guarantee rather than silently treating
an approximation as exact.

.. code-block:: python

   from mixle.enumeration import density_rank, hmm_best_paths

   rank_report = density_rank(model, value, n_samples=10000)
   paths = hmm_best_paths(hmm, observations, k=10)

Use :doc:`hmms-latent` for HMM modeling details.

Autoregressive Enumeration
--------------------------

``AutoregressiveEnumerable`` supports models that expose next-step log
probabilities rather than a closed finite support. The count index can then
perform thresholding and unranking over the generated tree.

.. code-block:: python

   from mixle.enumeration import AutoregressiveEnumerable, autoregressive_count_index

   enumerable = AutoregressiveEnumerable(next_logprobs, start_state)
   index = autoregressive_count_index(enumerable, budget=10000)

This is the bridge between token-like next-step models and structured
probability ranking.

Relations
---------

Enumeration also applies to feasible-set relations: assignments, paths,
spanning trees, edit neighborhoods, subset regression, and related
combinatorial objects. A relation defines the feasible structure; enumeration
produces ranked feasible values.

Practical Guidance
------------------

* Use ``top_k`` for small or clearly finite supports.
* Use ``mixle.describe`` before relying on rank, seek, or exact enumeration.
* Use quantized/count-budget indexes for large decomposable supports.
* Treat latent marginal ranking as a different problem from path enumeration.
* Prefer exact guarantees where available; inspect result objects when a route
  is approximate or bounded.

API Map
-------

.. list-table::
   :header-rows: 1

   * - Import
     - Purpose
   * - ``top_k``, ``supports_enumeration``
     - first calls for enumerable objects
   * - ``DistributionEnumerator``, ``EnumerationError``
     - enumeration contract and errors
   * - ``Enumerable``, ``FiniteSupport``, ``RankableByIndex``
     - capability markers
   * - ``density_rank``, ``DensityRankResult``
     - rank/cumulative queries for density models
   * - ``quantized_index``, ``count_budget_index``
     - high-probability seek/unrank indexes
   * - ``best_first_union``, ``merge_enumerators``, ``ProductEnumerator``
     - generic best-first enumeration utilities
   * - ``hmm_best_paths``
     - k-best HMM state paths
   * - ``AutoregressiveEnumerable``, ``autoregressive_count_index``
     - next-logprob enumeration
