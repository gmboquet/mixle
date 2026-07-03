Architecture Notes
==================

``mixle`` is organized around a small set of contracts and a larger set of
capability facets. The concrete modules are numerous, but the mental model stays
compact:

* Objects implement contracts: distributions, samplers, estimators, encoders,
  accumulators, enumerators, relations, engines, and data handles.
* Capabilities describe what those objects can do: enumerate, rank by index,
  condition, marginalize, expose latent posteriors, run on an engine, or update
  conjugately.
* Concerns own algorithms: inference fits models, enumeration ranks supports,
  operations transform distributions, engines execute array math, and data
  sources feed encoders.

Contract stack
--------------

The core distribution cast lives in :mod:`mixle.stats.compute.pdist`:

``ProbabilityDistribution``
    Scalar scoring, sampler creation, estimator creation, and optional support
    queries.

``SequenceEncodableProbabilityDistribution``
    Vectorized scoring over encoded batches, with optional engine support.

``DistributionSampler`` and ``ConditionalSampler``
    Seeded draw surfaces for unconditional and conditional sampling.

``DistributionEnumerator``
    Descending-probability support iteration.

``StatisticAccumulator`` and ``ParameterEstimator``
    Mergeable sufficient statistics and M-step estimation.

Capability layer
----------------

The capability helpers in :mod:`mixle.capability` make behavior inspectable at
runtime. ``mixle.describe(x)`` is the front door for users; ``supports`` and
``require`` are the front door for implementation code.

The most important capability groups are:

* support queries: ``Enumerable``, ``FiniteSupport``, ``RankableByIndex``;
* statistical form: ``ExponentialFamily``, ``ConjugateUpdatable``;
* transformations: ``Conditionable``, ``Marginalizable``, ``Transform``;
* latent models: ``LatentStructured``, ``PosteriorPredictive``;
* backend execution: ``SupportsBackendScoring``, ``EngineResidentEStep``.

Concern modules
---------------

``mixle.inference``
    Owns fitting, EM strategies, objective optimization, posterior objects,
    diagnostics, model comparison, and production-facing inference utilities.

``mixle.enumeration``
    Owns k-best search, quantized indexes, structural count DPs, rank/seek
    queries, and HMM path enumeration.

``mixle.ops``
    Owns operations that transform model capability sets, such as quantize,
    project, condition, marginalize, mixture, transform, and tilt.

``mixle.engines``
    Owns backend-neutral computation, precision tools, generated kernels, and
    symbolic export.

``mixle.data``
    Owns typed schemas, sources, validation, hashing, and encoded-data IO.

Object modules
--------------

Distribution families stay under :mod:`mixle.stats` and its support-oriented
subpackages. Top-level aliases such as :mod:`mixle.dist` and
:mod:`mixle.process` provide discoverable object namespaces without changing
serialization type IDs for existing models.

The architecture favors additive shims and re-exports over breaking moves:
stable import paths matter because serialized models store fully-qualified class
names.

Interface catalog
-----------------

The repository still contains detailed Markdown interface notes under
``docs/``. They are excluded from the Sphinx build because they are internal
cataloging material and include Markdown link patterns that are not portable
across Sphinx/MyST versions. This Sphinx page is the stable public summary; the
generated API reference remains the source-level detail.

