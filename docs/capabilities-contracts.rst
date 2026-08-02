Capabilities and Contracts
==========================

Mixle models are meant to compose across distribution families, latent
structures, neural leaves, and runtime backends. That only works if code asks
what an object can do instead of hard-coding class names.

Capabilities describe behavior. Contracts describe the methods and data shapes
that make the behavior usable by inference, enumeration, engines, and task
systems.

Capability Inspection
---------------------

The public inspection helpers are available from the top-level package:

.. code-block:: python

   import mixle

   print(mixle.describe(model))
   print(mixle.capabilities(model))
   print(mixle.supports(model, mixle.capability.Enumerable))

Use these calls before relying on optional behavior. A model may be a valid
probability distribution without being enumerable, conditionable, conjugate,
rankable, or backend-resident.

Put capability checks near the code that depends on them. A capability observed
for one fitted object does not automatically apply to a transformed,
quantized, projected, or wrapped object.

Method-Presence Capabilities
----------------------------

Some capabilities are detected by method surface:

.. list-table::
   :header-rows: 1

   * - Capability
     - Meaning
   * - ``Conditionable``
     - The object can return an exact conditional distribution.
   * - ``Marginalizable``
     - The object can return an exact marginal distribution.
   * - ``LatentStructured``
     - The object exposes hidden assignments, paths, or responsibilities.
   * - ``PosteriorPredictive``
     - The object can answer posterior predictive queries.
   * - ``EngineResidentEStep``
     - The E-step can run with model state resident on a compute engine.
   * - ``Transform``
     - The object represents a distributional transform.
   * - ``SupportsBackendScoring``
     - The object has a backend scoring implementation.
   * - ``SupportsBackendComponentScoring``
     - Component-level backend scores are available.
   * - ``SupportsStackedBackend``
     - Several related distributions can be scored in a stacked representation.
   * - ``TemporalPointProcess``
     - The object represents an event-time point process.

Method-presence capabilities are useful for extension because they let a new
family participate by implementing the relevant methods, not by registering
itself in every caller.

Method presence is only the first gate. The method must also preserve the
documented semantics, such as normalization after conditioning or parity
between scalar and vectorized scoring.

Predicate Capabilities
----------------------

Other capabilities are predicates over semantics or metadata:

.. list-table::
   :header-rows: 1

   * - Capability
     - Meaning
   * - ``Enumerable``
     - Support can be traversed in probability order.
   * - ``FiniteSupport``
     - Support size is finite and known.
   * - ``RankableByIndex``
     - The object can rank, unrank, or seek into structural support.
   * - ``Shardable``
     - The object can participate in sharded estimation or scoring.
   * - ``ExponentialFamily``
     - The object exposes a canonical exponential-family form.
   * - ``ConjugateUpdatable``
     - Closed-form conjugate updates are available.
   * - ``ExactDensity``
     - ``log_density`` is the true log-density or log-mass, not a bound or
       approximation.
   * - ``SetValued``
     - Observations are set-valued.
   * - ``HasCDF``
     - A cumulative distribution function is available.
   * - ``HasMoments``
     - Mean, variance, or related moment summaries are available.
   * - ``HasEntropy``
     - Entropy can be computed.
   * - ``Discrete``
     - The model is discrete-valued.
   * - ``Continuous``
     - The model is continuous-valued.
   * - ``Fittable``
     - An estimator is available.
   * - ``Optimizable``
     - The object can be optimized by Mixle inference routes.
   * - ``Neutral``
     - The object is structurally neutral for certain capability intersections.

Predicate capabilities keep important distinctions visible. For example, a
variational topic model may be scoreable but not ``ExactDensity`` if the score
is an ELBO. A continuous model may support moments but not enumeration.

When a workflow depends on exactness, require the exact capability explicitly.
A finite numeric score is not proof that the score has exact-density semantics.

Helper Functions
----------------

``mixle.capability`` exposes programmatic helpers:

``supports(obj, capability)``
    Return ``True`` when an object supports the capability.

``capabilities(obj)``
    Return capability names as a frozen set.

``require(obj, capability, op=None)``
    Raise a clear capability error before running an operation that depends on
    a missing behavior.

``intersect_capabilities(children)``
    Compute the capability set preserved by a combinator's children.

``describe(obj)``
    Render a human-readable report of behavior, density semantics, and
    capability surface.

``summarize(obj)``
    Return numeric or categorical summary information suitable for reports.

``catalog()`` / ``what_supports`` / ``render_catalog_markdown``
    Inspect the global capability catalog and compare objects.

These functions make assumptions explicit in reports, tests, and extension
work.

Core Probability Contracts
--------------------------

The base contracts live in ``mixle.stats.compute.pdist``:

``ProbabilityDistribution``
    Scalar observation interface. Implements ``log_density(x)``, ``density``,
    ``sampler()``, ``estimator()``, JSON serialization helpers, Fisher and
    exponential-family access, and density-semantics reporting.

``SequenceEncodableProbabilityDistribution``
    Distribution that can score encoded batches through ``seq_log_density`` and
    produce a ``DataSequenceEncoder``.

``DistributionSampler``
    Seeded sampling interface.

``ParameterEstimator``
    Declares a family to fit and creates accumulators.

``StatisticAccumulator`` and ``SequenceEncodableStatisticAccumulator``
    Mergeable sufficient-statistic or telemetry containers used by local and
    distributed estimation.

``StatisticAccumulatorFactory``
    Creates accumulators and encoders for an estimator.

``DataSequenceEncoder``
    Encodes raw Python observations into vectorized payloads.

``DistributionEnumerator``
    Traverses support where enumeration is meaningful.

The most important extension rule is that scalar and sequence paths must agree.
For a new family, ``log_density(x)`` and ``seq_log_density(encoder.seq_encode([x]))``
should report the same value up to numerical tolerance.

That parity check should include malformed or impossible observations when the
family has restricted support. The scalar and encoded paths should agree on
``-inf`` and supported missing-data behavior, not only on ordinary rows.

Density Semantics
-----------------

``DensitySemantics`` records whether a score is exact or approximate:

.. list-table::
   :header-rows: 1

   * - Value
     - Meaning
   * - ``EXACT``
     - The score is the true log-density or log-mass.
   * - ``LOWER_BOUND``
     - The score is a lower bound, such as a variational ELBO.
   * - ``UPPER_BOUND``
     - The score is an upper bound.
   * - ``ESTIMATE``
     - The score is an approximation without a guaranteed direction.
   * - ``LIKELIHOOD_FACTOR``
     - The score is an unnormalized factor. It is comparable across values of
       the same object, but it is not a density, does not integrate to one, and
       must not be exponentiated into a probability.

Combinators combine child semantics with ``join_density_semantics``. Callers
that require true likelihoods should use ``require(obj, ExactDensity)`` rather
than trusting a numeric value blindly.

Release artifacts should record when a score is a bound or estimate. This is
especially important when score tables are later used for model comparison,
enumeration, or promotion gates.

Estimators May Return a Factor
------------------------------

An estimator is not required to return a normalized law. When the data it was
given does not determine one, it may return an object whose
``density_semantics()`` is ``LIKELIHOOD_FACTOR``. This is part of the contract,
not a degenerate result to be worked around, and it is deliberate: the
alternative -- raising -- was tried and reverted, because a learned segment
fitted on all-empty sequences, a gated mixture whose evidence is impossible for
one component, and hidden association all legitimately reach this state, and
refusing broke all three.

The worked case is ``CategoricalEstimator`` with nothing accumulated -- not even
a label at zero weight. There is no set of labels to place a distribution over,
so it returns the zero-support categorical: it scores ``-inf`` everywhere,
reports ``LIKELIHOOD_FACTOR``, and a mixture built over it is a factor too.

What such an object owes the caller is honesty about what it is, and that is
enforced rather than documented:

* ``density_semantics()`` reports ``LIKELIHOOD_FACTOR``, so
  ``join_density_semantics`` propagates it through every combinator that
  contains it -- an exact law mixed with a factor is a factor.
* ``is_normalized_probability`` is ``False``, and ``sampler()`` refuses on that
  basis with the reason. A factor cannot be drawn from, because there is no
  distribution to draw from; ``MixtureDistribution.sampler()`` refuses for the
  same reason when any component is a factor.

So a factor-returning estimator cannot silently become a generative model. Code
that requires a law should check ``is_normalized_probability`` or call
``sampler()`` and let it refuse -- both fail at the boundary, with the reason,
rather than returning numbers that look like probabilities.

Subsystem Contracts
-------------------

``mixle.contracts`` gathers lazy subsystem contracts so extension code can
import contract names without importing every implementation package eagerly.
It is useful when a subsystem needs to refer to PPL, task, DOE, analysis, or
inference contracts while avoiding circular imports.

The important practical point is that contracts are owned by behavior, not by
where the class happens to live. A task artifact, a distribution, a PPL target,
and a backend handle can all participate in the broader system if they expose
the contract expected by the caller.

Extension Guidance
------------------

When adding a new family:

* implement the probability, sampler, estimator, accumulator, and encoder
  pieces first;
* expose optional behavior only when it is mathematically correct;
* add declaration metadata when the family is exponential-family or should use
  generated kernels;
* use ``mixle.describe`` in tests to confirm advertised capabilities;
* add scalar/vectorized parity tests and estimator recovery tests;
* document capability limitations explicitly.

Common mistakes:

* advertising enumeration for a support that cannot be traversed exactly;
* reporting exact density when the value is a bound;
* using class checks in callers instead of ``supports`` or ``require``;
* forgetting that combinators must preserve or drop child capabilities
  deliberately;
* adding a backend method that changes model semantics compared with the scalar
  path.

Release Evidence
----------------

For new or documented capabilities, preserve:

* the object returned by ``mixle.describe``;
* the required capabilities checked by downstream operations;
* scalar/vectorized parity evidence for probability families;
* exact-density or bound semantics when scores are compared;
* backend parity evidence when engine-resident paths are advertised; and
* negative tests showing that unsupported operations fail clearly.

API Reference
-------------

* :doc:`api/mixle.capability`
* :doc:`api/mixle.contracts`
* :doc:`api/mixle.stats.compute.pdist`
