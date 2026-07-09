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

This separation is the main design rule for the package. Distribution objects
should not know every inference algorithm. Inference code should not special
case every family when a capability can describe the behavior. Engines should
not change statistical semantics. Documentation should preserve those
boundaries so users can tell whether a new feature is an object, an inference
route, a transformation, a backend, or a workflow layer.

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

The estimator contract is deliberately plain. Estimators consume encoded or raw
observations, accumulate sufficient statistics, and return a fitted object with
the same distribution semantics users expect from hand-constructed models. EM,
gradient fitting, conjugate updates, and PPL lowering are routing choices around
that contract rather than separate modeling systems.

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

Capabilities are descriptive, not decorative. A capability should mean the
object can actually perform the named operation with the documented shape and
failure behavior. When an operation is approximate, backend-specific, or valid
only for a subset of inputs, the capability documentation or method docstring
should say so directly. This keeps automatic routing and user-facing inspection
from drifting apart.

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

``mixle.ppl``
    Owns the symbolic user-facing probabilistic-programming surface and lowers
    expressions into existing distribution, estimator, state-space, or sampler
    routes.

``mixle.doe``
    Owns experiment-design utilities: candidate generation, active selection,
    multi-fidelity designs, distillation selectors, and verifiable-oracle
    loops.

``mixle.task``
    Owns teacher/student replacement workflows, calibration, cascades,
    structured-output checks, routing economics, and agentic trace utilities.

``mixle.reason`` and ``mixle.substrate``
    Own reasoning-oriented evidence structures, cross-modal checks, substrate
    assembly, answer receipts, and local reasoning harnesses.

Workflow layers should continue to call into the core contracts rather than
forking their own hidden model interfaces. A task cascade, DOE loop, or
reasoning harness is easier to validate when its model, score, sampler,
artifact, and evidence objects are still ordinary Mixle objects.

Object modules
--------------

Distribution families stay under :mod:`mixle.stats` and its support-oriented
subpackages. Top-level aliases such as :mod:`mixle.dist` and
:mod:`mixle.process` provide discoverable object namespaces without changing
serialization type IDs for existing models.

The architecture favors additive shims and re-exports over breaking moves:
stable import paths matter because serialized models store fully-qualified class
names.

Extension guidance
------------------

When adding a model family, start by choosing the smallest stable contract it
must satisfy. A finite discrete family may need enumeration and indexed ranking;
a continuous family may need scoring, sampling, and estimation; a latent family
may need posterior summaries and guarded EM behavior. Add capabilities only
when callers can rely on them.

When adding an inference route, keep the route explicit. Users should be able to
tell whether they are running direct estimation, EM, MAP, variational inference,
MCMC, state-space fitting, or a task-specific calibration loop. Silent fallback
is appropriate only when the fallback preserves semantics and leaves an
inspectable record.

When adding a workflow module, document the boundary between proposal,
verification, fitting, and deployment. DOE and task modules often combine those
steps, but the public API should make clear which component proposes candidates,
which component supplies labels or scores, which model is fit, and which
receipt proves the result.

Compatibility guidance
----------------------

Public import paths, serialized class names, and estimator/distribution method
shapes are compatibility surfaces. Prefer adding a new helper, adapter, or
capability over changing an established object shape in place. If a rename is
unavoidable, keep a documented compatibility shim for at least one release
cycle and mention the migration in the changelog.

Generated API pages should reflect that contract. Private helpers may appear in
source code, but the narrative docs should steer users toward stable modules,
capabilities, and constructors. If a public guide must mention a lower-level
helper, it should also explain why the helper is safe to call directly.
