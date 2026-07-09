HMMs and Latent Structure
=========================

Latent models in ``mixle`` are distributions that wrap other distributions.
They add hidden variables without changing the outer query surface:

* score observations with ``log_density``;
* fit with ``optimize``;
* inspect posteriors when supported;
* sample or enumerate when the model has the capability.

The two most common latent wrappers are mixtures and HMMs.

Mixtures
--------

A mixture adds one latent component assignment per observation.

.. code-block:: python

   from mixle.inference import best_of, optimize
   from mixle.stats import GaussianEstimator, MixtureEstimator

   est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
   model = optimize(data, est, max_its=100, out=None)
   responsibilities = model.posterior(data)

Use ``best_of`` when local optima matter:

.. code-block:: python

   import numpy as np

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

The component can be any estimator with a compatible shape: a scalar
distribution, a record, a sequence model, or a neural leaf.

Interpret mixture components only after checking restart stability and
held-out behavior. Component identities can swap across runs, and a component
that appears in one local optimum should not be treated as a stable label.

Latent-Model Numerical Contract
--------------------------------

Latent wrappers should preserve the same numerical contract as their child
distributions:

* scalar and vectorized scoring routes should agree up to ordinary floating
  point tolerance;
* impossible observations or impossible latent paths should score as
  ``-inf``, not ``NaN``;
* posterior responsibilities should be finite and normalized whenever the
  observation has nonzero probability under the model;
* caller-owned input data should not be rewritten to hide missing or non-finite
  values; and
* ambiguous latent labels should be reported with uncertainty instead of being
  treated as observed classes.

For release work, validate a latent model against a simpler baseline and keep
the initialization, restart, missing-data, and decoding policies with the model
record.

Mixture of Heterogeneous Records
--------------------------------

.. code-block:: python

   from mixle.stats import (
       CategoricalEstimator,
       CompositeEstimator,
       GammaEstimator,
       MixtureEstimator,
       PoissonEstimator,
   )

   component = CompositeEstimator(
       (
           CategoricalEstimator(),  # event type
           GammaEstimator(),        # wait time
           PoissonEstimator(),      # count
       )
   )
   est = MixtureEstimator([component, component, component])

The latent component clusters whole records. Each component owns its own child
distributions.

For heterogeneous records, inspect field-level likelihoods inside components.
A cluster can be driven by one malformed or high-variance field rather than a
meaningful row-level regime.

PPL Markov Models
-----------------

For compact HMMs, ``mixle.ppl`` is often the clearest surface:

.. code-block:: python

   from mixle.ppl import Markov, Normal, free

   hmm = Markov(Normal(free, free), states=3).fit(sequences, how="auto")
   post = hmm.posterior(sequences)

``Markov`` lowers to the same latent estimator machinery as the explicit stats
surface.

Inspect the lowered route when using PPL for HMMs. The compact expression is a
model declaration; the release evidence should still name the fitted
distribution, transition structure, missing-data policy, and posterior query
surface.

Default HMM Execution
---------------------

When Numba is installed, HMM distributions now default to the Numba encoder and
Baum-Welch path, matching the estimator default. This matters for the common
workflow where an initialized HMM is passed as ``prev_estimate``: the encoder
comes from the distribution, so the distribution and estimator must agree on
the fast path.

Explicit settings still win. Pass ``use_numba=False`` on the HMM family when
you need the pure NumPy path for debugging, parity checks, or an environment
where compiled kernels are not desirable.

When Numba is used as release evidence, record the package version and parity
check against the NumPy route on a small sequence. Fast-path differences should
be treated as execution evidence, not as a change in the model.

HMM Diagnostics
---------------

.. list-table::
   :header-rows: 1

   * - Check
     - Expected result
   * - Scalar/vectorized scoring
     - ``log_density`` and ``seq_log_density`` agree on the same sequences.
   * - Impossible observation
     - The sequence scores ``-inf`` and does not produce ``NaN``.
   * - Posterior query
     - State probabilities sum to one on valid observations.
   * - Viterbi query
     - Returned paths have one state per emitted observation.
   * - Fast path
     - Numba and NumPy routes match on a small parity fixture when both are used.
   * - Length model
     - Empty, short, and long sequences follow the documented length contract.

Structured HMMs
---------------

``StructuredHMM`` separates the HMM algorithm from the transition
representation. A transition operator supplies forward products, backward
products, and expected-mass updates. The same forward-backward and EM code can
then use dense, low-rank, sparse, Kronecker, duration, or input-output
structure.

.. code-block:: python

   import numpy as np
   import mixle.stats as S
   from mixle.inference import optimize
   from mixle.stats.latent.structured_hmm import (
       LowRankTransition,
       StructuredHMM,
       _row_normalize,
   )

   rng = np.random.RandomState(0)
   k, rank = 8, 2

   transition = LowRankTransition(
       _row_normalize(rng.rand(k, rank)),
       _row_normalize(rng.rand(rank, k)),
   )
   init = StructuredHMM(
       [S.GaussianDistribution(float(i), 1.0) for i in range(k)],
       np.ones(k) / k,
       transition,
   )

   model = optimize(sequences, init.estimator(), prev_estimate=init, max_its=40, out=None)

Why use a structured transition?

.. list-table::
   :header-rows: 1

   * - Transition
     - Use when
   * - Dense
     - every state can move to every other state
   * - Low-rank
     - many states but transition structure has fewer degrees of freedom
   * - Sparse
     - left-to-right, skip-limited, or graph-constrained motion
   * - Kronecker
     - factorial states such as ``(speaker_state, topic_state)``
   * - Sticky
     - segmentation should prefer staying in the same state
   * - Explicit-duration
     - state durations are not geometric
   * - Input-output
     - an exogenous input controls which transition applies
   * - Terminal states
     - absorbing states determine sequence length as a stopping time

Decoding
--------

HMMs are useful because they expose latent paths, not just likelihoods.

.. code-block:: python

   path = model.viterbi(sequence)
   segments = model.viterbi_segments(sequence)  # explicit-duration models
   state_posteriors = model.posterior(sequence)

Exact method names vary by HMM family; use ``mixle.describe(model)`` to see
which latent queries are available.

Decoded paths are explanations under the fitted model, not observed truth.
When the path drives a decision, keep state posterior uncertainty or top-path
margins with the decision record.

Missing and Impossible Sequence Behavior
----------------------------------------

HMMs do not treat a ``NaN`` in caller data as something to repair implicitly.
If missing values are valid observations for the application, model them with an
emission distribution or wrapper that has an explicit missing-data contract. If
the emission family rejects the value, the HMM should surface that rejection
through the likelihood route instead of silently changing the input.

When every latent path is impossible for a sequence, the correct log-density is
``-inf``. Downstream posterior and decoding calls should be interpreted only for
sequences with positive probability under the fitted model.

Enumeration
-----------

When the support is discrete and the model advertises enumeration, you can ask
for top sequences or paths:

.. code-block:: python

   enum = model.enumerator()
   top = enum.top_k(5)
   nucleus = enum.nucleus_size(0.9)

For decomposable supports, ranking and seek can be exact. For hard latent
marginals, mixle reports bounded or approximate routes rather than pretending
they are exact.

Record whether enumeration returned state paths, observation sequences, or
marginal value rankings. Those are different questions and should not be mixed
in release evidence.

HMMs with Neural or Heterogeneous Emissions
-------------------------------------------

An HMM emission follows the ordinary distribution contract. The emission can be:

* a Gaussian;
* a categorical token model;
* a composite record;
* a sequence model;
* a Transformer or other neural leaf, where supported by the estimator shape.

The HMM parent supplies expected state responsibilities. Each child emission
uses those responsibilities in its own M-step.

Run the Structured HMM Tour
---------------------------

.. code-block:: sh

   python examples/structured_hmm_example.py
   python examples/lookback_hmm_example.py

The structured tour demonstrates low-rank transitions, factorial Kronecker
transitions, sparse left-to-right transitions, sticky priors, decoding,
enumeration, terminal states, explicit-duration HMMs, and input-output HMMs.

Treat those tours as validation examples only when they execute against the
release wheel with their optional dependencies recorded. A source-checkout run
is useful during development but is not enough release evidence.

Release Evidence
----------------

For HMMs and other latent sequence models, preserve:

* initialization and restart policy;
* validation score against a simpler baseline;
* transition structure and optional fast-path settings;
* posterior state uncertainty, top-path margins, or responsibility summaries;
* missing-data and impossible-transition behavior; and
* example or notebook execution status for any documented workflow.

API Map
-------

.. list-table::
   :header-rows: 1

   * - Import
     - Purpose
   * - ``MixtureEstimator``
     - latent clusters over observations
   * - ``best_of``
     - restart latent fitting and select by validation score
   * - ``mixle.ppl.Markov``
     - compact HMM expression surface
   * - ``StructuredHMM``
     - HMM with pluggable transition operator
   * - ``DenseTransition``
     - ordinary dense transition matrix
   * - ``LowRankTransition``
     - factorized transition matrix
   * - ``SparseTransition``
     - edge-constrained transition graph
   * - ``KroneckerTransition``
     - factorial state-space transition
   * - ``ExplicitDurationHMM``
     - HSMM with duration distributions
   * - ``InputOutputHMM``
     - transition chosen by exogenous input
