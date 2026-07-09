Quickstart
==========

This quickstart fits a heterogeneous record model with ``mixle.stats`` and
``mixle.inference.optimize``. It scores new rows, samples from the fitted
distribution, and inspects the fitted object's capabilities.

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
posteriors. Capability inspection is the supported way to find out.

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

Validate Before Interpreting
----------------------------

The quickstart model is intentionally small, but the validation habit should be
the same as a larger workflow:

* split data before comparing model families;
* use several random starts for mixtures or other latent models;
* inspect posterior responsibilities before naming clusters;
* preserve ``NaN`` or missing-input semantics defined by the data pipeline
  instead of silently rewriting inputs during fitting;
* check capabilities before relying on enumeration, conditioning,
  marginalization, or posterior queries; and
* record the estimator shape, initialization policy, seed, and held-out score
  when the fitted object leaves the notebook.

These checks prevent a convenient example from turning into an unsupported
claim about the fitted latent structure.

Create a Certified Artifact
---------------------------

When the fit should carry an explicit certificate and optional post-fit checks,
use ``create`` after the core workflow is clear:

.. code-block:: python

   from mixle.inference import create

   artifact = create(rows, calibrate=0.2, quantify_uq=True, seed=0)

   print(artifact.guarantee)
   print(artifact.is_calibrated())

``create`` still delegates to the same fitting machinery. It adds an artifact
boundary: estimation certificate, optional calibration, optional UQ, and
provenance including row counts and exchangeability diagnostics.

Optional: Neural and Task Layers
--------------------------------

Use the neural and LLM-facing layers when the task genuinely calls for them:
shared embeddings across language-model experts, exact neural density leaves,
local students distilled from expensive teachers, or calibrated LLM abstention.
Those workflows carry extra dependency and validation requirements, so keep the
first model in ``mixle.stats`` unless the data shape demands otherwise. See
:doc:`neural-llm`, :doc:`task-distillation`, and :doc:`uncertainty` after the
core distribution workflow above is clear.
