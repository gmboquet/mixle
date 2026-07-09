Core Concepts
=============

The central idea in ``mixle`` is not "many distributions." It is that model
structure, data structure, and inference structure are the same object viewed
from three angles.

One Observation Is a Python Value
---------------------------------

``mixle`` does not require a flat matrix. One observation can be a scalar, a
tuple, a dictionary, a sequence, a graph-like object, or a neural training pair.

.. list-table::
   :header-rows: 1

   * - Observation
     - Natural model shape
   * - ``3.2``
     - scalar continuous distribution
   * - ``"clicked"``
     - categorical distribution
   * - ``("us", 42.0)``
     - composite tuple distribution
   * - ``{"country": "us", "age": 42.0}``
     - record distribution
   * - ``[3, 4, 5]``
     - sequence distribution
   * - ``([12, 44, 91, 7], 18)``
     - next-token distribution
   * - ``[0.2, 1.8, 1.5, ...]``
     - emission sequence with latent HMM states

The fitted model scores that whole value with ``log_density(x)``. If the model
can sample, it samples values with the same shape.

Keep the observation shape stable. If a field is sometimes absent, sometimes a
list, or sometimes a scalar, represent that policy explicitly with a structured
family rather than relying on preprocessing side effects.

Estimator Shape Mirrors Data Shape
----------------------------------

The most important modeling move is to make the estimator look like one row of
data.

.. code-block:: python

   from mixle.stats import CompositeEstimator, GammaEstimator, PoissonEstimator

   # Observation: (count, wait)
   est = CompositeEstimator((PoissonEstimator(), GammaEstimator()))

Combinators can be nested:

.. code-block:: python

   from mixle.stats import CategoricalEstimator, MixtureEstimator, SequenceEstimator

   row = CompositeEstimator(
       (
           CategoricalEstimator(),
           SequenceEstimator(PoissonEstimator(), len_estimator=CategoricalEstimator()),
       )
   )
   clustered_rows = MixtureEstimator([row, row, row])

The estimator says three things at once:

* what the observation looks like;
* which distribution family fits each part;
* which inference route is needed when the structure contains latents, neural
  leaves, priors, or constraints.

The Five Pieces
---------------

Each full distribution family is built from five cooperating pieces:

.. list-table::
   :header-rows: 1

   * - Piece
     - Role
   * - ``...Distribution``
     - Stores parameters and implements ``log_density(x)``.
   * - ``...Sampler``
     - Draws observations from a fitted distribution.
   * - ``...Estimator``
     - Declares the model shape and performs estimation.
   * - ``...Accumulator``
     - Collects mergeable sufficient statistics or training telemetry.
   * - ``...DataEncoder``
     - Packs Python values into vectorized encoded data.

That contract is what makes scale-out possible. Encoded data can be folded
locally, on a multiprocessing backend, on Spark/Dask/MPI, or through a device
engine while the model code remains the same.

What Happens During ``optimize``
--------------------------------

``optimize(data, estimator)`` runs this outer loop:

1. choose an encoder;
2. encode raw Python observations;
3. initialize a candidate distribution;
4. accumulate evidence under the current distribution;
5. ask the estimator for an updated distribution;
6. repeat until convergence or ``max_its``.

Different structures specialize the same loop:

.. list-table::
   :header-rows: 1

   * - Structure
     - What the loop means
   * - Closed-form leaves
     - one or more maximum-likelihood or conjugate sufficient-statistic updates
   * - Mixtures
     - E-step responsibilities, M-step component updates
   * - HMMs
     - forward-backward expectations, emission and transition updates
   * - Neural leaves
     - gradient M-step against weighted or streamed batches
   * - PPL expressions
     - lowered estimator or target, then route selected by ``how=``
   * - Task cascades
     - fit local model, calibrate, decide answer versus escalation

This is why mixle can fit heterogeneous records and latent structures without a
new training loop for every combination. The child estimators do different work,
but they present the same outer shape to the parent composite or latent model.

Because the same loop serves many structures, validation should check both the
scalar path and the encoded/vectorized path when a family or backend changes.
They should agree on ordinary observations, impossible observations, and
documented missing-data behavior.

Distributions Are Query Objects
-------------------------------

After fitting, stay on the distribution surface:

.. code-block:: python

   score = model.log_density(x)
   samples = model.sampler(seed=0).sample(10)
   encoder = model.dist_to_encoder()

Latent models add posterior queries:

.. code-block:: python

   responsibilities = mixture.posterior(rows)
   path = hmm.viterbi(sequence)

Discrete and structured models may also support enumeration:

.. code-block:: python

   enum = dist.enumerator()
   top = enum.top_k(10)

Capabilities, Not Class Checks
------------------------------

Ask what an object supports instead of guessing from its class:

.. code-block:: python

   import mixle

   print(mixle.describe(model))
   print(mixle.capabilities(model))
   print(mixle.supports(model, mixle.capability.Enumerable))

Capabilities cover exact density, finite support, enumeration, ranking,
conditioning, marginalization, latent posterior behavior, backend scoring,
conjugate updates, and more.

Capability reports are part of release evidence. If a guide relies on
enumeration, posterior queries, backend scoring, or exact density, record the
capability check near the workflow that uses it.

See :doc:`capabilities-contracts` for the full capability catalog and the
contracts used by distribution, estimator, accumulator, and encoder objects.
See :doc:`compute-layer` for the encoded-data and kernel machinery beneath the
public distribution families.

Public Surfaces
---------------

.. list-table::
   :header-rows: 1

   * - Namespace
     - Use it for
   * - ``mixle.stats``
     - distribution families, estimators, combinators, latent models
   * - ``mixle.inference``
     - fit loops, EM, priors, calibration, diagnostics, model comparison
   * - ``mixle.ppl``
     - compact expression language that lowers to stats/inference objects
   * - ``mixle.models``
     - incubating applied helpers: neural leaves, GPs, random forests, graphs
   * - ``mixle.task``
     - LLM labeling, distillation, active learning, cascades, artifacts
   * - ``mixle.reason``
     - LLM uncertainty and cross-modal latent evidence fusion
   * - ``mixle.enumeration``
     - top-k, rank, seek, nucleus, and structured support traversal
   * - ``mixle.engines``
     - NumPy, Torch, JAX, symbolic, and precision-aware compute engines
   * - ``mixle.data``
     - schemas, data sources, validation, hashes, encoded IO
   * - ``mixle.doe``
     - design of experiments, active labeling, optimization, sensitivity
   * - ``mixle.stats.compute``
     - low-level contracts, encoded data, generated kernels, backend scoring
   * - ``mixle.utils``
     - automatic typing, serialization, optional dependencies, metrics,
       parallel runtime helpers

The shortest practical advice is: start with ``mixle.stats`` when you know the
model, ``mixle.task.recommend_model`` when you want help choosing one,
``mixle.ppl`` when the formula is clearer than the estimator tree, and
``mixle.describe`` whenever you are unsure what the fitted object can do.

Release Evidence
----------------

For core Mixle workflows, preserve:

* the observation shape and estimator tree;
* fitting route, seed, restart policy, and validation split;
* scalar/vectorized parity evidence for new families or backends;
* capability checks for downstream operations;
* missing-data and impossible-observation policy; and
* artifact provenance when the fitted model leaves the notebook.
