Quickstart
==========

This quickstart uses the stable center of Mixle: ``mixle.stats`` distributions
and ``mixle.inference.optimize``. It fits a heterogeneous record model, scores
new rows, samples from the fitted distribution, and inspects the fitted
object's capabilities.

The example deliberately avoids ``mixle.models`` at first. Neural leaves and
other applied helpers are useful, but they are an incubating surface. Learn the
distribution/estimator shape rule before reaching for them.

Install
-------

.. code-block:: sh

   pip install mixle

From a repository checkout:

.. code-block:: sh

   pip install -e .

Data Shape
----------

One observation is a Python tuple:

.. code-block:: text

   (segment, value, count_sequence)

Example:

.. code-block:: python

   ("steady", 11.8, [3, 4, 2])

The fields are intentionally mixed:

* ``segment`` is categorical;
* ``value`` is real-valued;
* ``count_sequence`` is a variable-length sequence of counts.

Make Synthetic Rows
-------------------

.. code-block:: python

   import numpy as np

   def make_rows(n=240, seed=0):
       rng = np.random.RandomState(seed)
       rows = []
       for _ in range(n):
           bursty = rng.rand() < 0.45
           segment = "bursty" if bursty else "steady"
           value = float(rng.normal(18.0 if bursty else 8.0, 2.0 if bursty else 1.0))
           length = int(rng.choice([2, 3, 4]))
           rate = 9.0 if bursty else 3.0
           counts = [int(x) for x in rng.poisson(rate, size=length)]
           rows.append((segment, value, counts))
       return rows

   rows = make_rows()

Build the Model Shape
---------------------

The estimator has the same shape as one observation. A mixture adds a latent
cluster over the whole row.

.. code-block:: python

   from mixle.stats import (
       CategoricalEstimator,
       CompositeEstimator,
       GaussianEstimator,
       MixtureEstimator,
       PoissonEstimator,
       SequenceEstimator,
   )

   def row_component():
       return CompositeEstimator(
           (
               CategoricalEstimator(),
               GaussianEstimator(),
               SequenceEstimator(
                   PoissonEstimator(),
                   len_estimator=CategoricalEstimator(),
               ),
           )
       )

   estimator = MixtureEstimator([row_component(), row_component()])

Read this literally:

* ``CategoricalEstimator`` fits ``segment``.
* ``GaussianEstimator`` fits ``value``.
* ``SequenceEstimator(PoissonEstimator())`` fits a variable-length list of
  counts.
* ``CompositeEstimator`` joins those fields into one row-level model.
* ``MixtureEstimator`` learns two latent row types.

Fit
---

.. code-block:: python

   from mixle.inference import optimize

   model = optimize(rows, estimator, max_its=80, out=None)

The result is a fitted distribution. It can score rows, sample rows, and expose
capabilities through ``mixle.describe``.

Score Rows
----------

.. code-block:: python

   ordinary = rows[0]
   unusual = ("steady", 24.0, [18, 21, 19])

   print(model.log_density(ordinary))
   print(model.log_density(unusual))

Both scores are log probabilities. The unusual row should score poorly because
its label says ``steady`` while its value and counts look more like the
``bursty`` cluster.

Sample
------

.. code-block:: python

   samples = model.sampler(seed=0).sample(3)
   print(samples)

Sampling is part of the distribution contract. Use an explicit seed whenever a
notebook, test, or article needs reproducible output.

Inspect Capabilities
--------------------

.. code-block:: python

   import mixle

   print(mixle.describe(model))
   print(mixle.capabilities(model))

This habit matters. Not every fitted object can enumerate, condition,
marginalize, expose exact densities, run on every backend, or produce latent
posteriors. Capability inspection is the honest way to find out.

Use a Prototype Distribution
----------------------------

When you know the model family, pass a prototype distribution. Mixle derives
the matching estimator from that shape.

.. code-block:: python

   from mixle.stats import GaussianDistribution, MixtureDistribution

   values = [row[1] for row in rows]
   proto = MixtureDistribution(
       [GaussianDistribution(8.0, 1.0), GaussianDistribution(18.0, 4.0)],
       [0.5, 0.5],
   )

   value_model = optimize(values, proto, prev_estimate=proto, out=None)

Passing ``proto`` as the second argument tells ``optimize`` what to fit.
Passing it as ``prev_estimate`` also uses those parameter values as the
starting estimate, which is usually what you want for a mixture example.

Infer a First Estimator
-----------------------

For exploratory work, Mixle can propose an estimator shape from data:

.. code-block:: python

   from mixle.utils.automatic import get_estimator

   inferred = get_estimator(rows)
   inferred_model = optimize(rows, inferred, max_its=40, out=None)

Treat the inferred shape as a starting point. Inspect it, compare it on
held-out data, and replace pieces when the automatic guess does not match the
domain.

Optional: Neural Event Leaf
---------------------------

``mixle.models`` includes incubating neural leaves. Use them when a neural
likelihood is genuinely part of the model, not as the default first path.

For example, an event row can combine a Transformer next-token leaf with a
Gamma timing model:

.. code-block:: python

   from mixle.models import TransformerLMEstimator
   from mixle.stats import CompositeEstimator, GammaEstimator

   # each event: ((history_window, next_event_type), seconds_since_previous)
   event_estimator = CompositeEstimator(
       (
           TransformerLMEstimator(vocab=500, d_model=128, n_layer=4, block=64),
           GammaEstimator(),
       )
   )

This pattern is useful for anomaly scoring in event streams, but it needs the
``torch`` extra and the usual neural-model discipline: fixed seeds, held-out
data, monitored training loss, and reproducible artifacts. See
:doc:`neural-llm` only after the core distribution workflow above is clear.

What To Read Next
-----------------

* :doc:`maturity` explains which namespaces are stable, active, or
  experimental.
* :doc:`concepts` explains the distribution and estimator contract.
* :doc:`hmms-latent` covers mixtures, HMMs, and structured transitions.
* :doc:`neural-llm` covers Transformer leaves, shared embeddings, and DPO.
* :doc:`representation` covers text, signal, graph, image, and scientific
  object encoders.
* :doc:`task-distillation` covers LLM teachers, local students, cascades, and
  active labeling.
* :doc:`uncertainty` and :doc:`reasoning-systems` cover semantic entropy,
  claim reliability, graph-producing LLMs, and cross-modal evidence fusion.
* :doc:`doe` and :doc:`evolution` cover active design, Bayesian optimization,
  sensitivity analysis, and verify-gated model improvement.
* :doc:`automatic-inference` covers inferred estimators and model
  recommendation.
