Compute Layer
=============

Most users should start with ``mixle.stats`` and ``mixle.inference``. The
``mixle.stats.compute`` package is the lower-level machinery that lets those
public APIs scale from scalar Python values to encoded batches, engines,
generated kernels, sharded data, and model-parallel estimation.

This layer matters when you are adding a distribution family, optimizing a hot
path, implementing a backend, or debugging why a model does not expose a
capability.

Core Protocol
-------------

The compute layer is built on the contracts in ``pdist``:

.. list-table::
   :header-rows: 1

   * - Contract
     - Role
   * - ``ProbabilityDistribution``
     - Scalar scoring, sampling, estimator creation, serialization, density
       semantics, and optional Fisher/exponential-family views.
   * - ``SequenceEncodableProbabilityDistribution``
     - Vectorized scoring over encoded data.
   * - ``DataSequenceEncoder``
     - Converts raw observations into encoded payloads.
   * - ``ParameterEstimator``
     - Declares the family and creates accumulator factories.
   * - ``StatisticAccumulator``
     - Collects mergeable sufficient statistics.
   * - ``StatisticAccumulatorFactory``
     - Creates accumulators and encoders for an estimator.

The protocol is intentionally old-fashioned: clear method contracts, explicit
encoded payloads, and mergeable statistics. That is what lets ordinary
distributions, latent models, neural leaves, and distributed backends share the
same outer inference loop.

The reference scalar path remains the semantic anchor. Encoded, generated, or
backend paths should be treated as accelerators until parity evidence proves
they preserve the same scores and updates.

Encoded Data
------------

``mixle.stats.compute.encoded`` provides typed containers for encoded payloads:

``EncodedData``
    Stores the chunked ``[(count, payload)]`` shape used by sequence drivers.

``ResidentEncodedPayload``
    Records payloads that have been moved to a compute engine or resident
    backend.

``as_encoded_data``
    Normalizes local encoded sequences and backend handles into a common
    representation.

``move_encoded_payload``
    Moves encoded payloads to an engine.

``encoded_nbytes``
    Estimates memory use for encoded payloads.

Encoded data is the boundary between Python-shaped observations and vectorized
work. A distribution family should be explicit about what its encoder emits and
what its ``seq_log_density`` expects.

Document encoded payload shapes for new families. Shape ambiguity is one of the
fastest ways for a backend implementation to pass small tests while failing on
real heterogeneous records.

Sequence Drivers
----------------

``mixle.stats.compute.sequence`` contains the vectorized drivers used by
inference:

.. list-table::
   :header-rows: 1

   * - Function
     - Use
   * - ``seq_encode``
     - Encode raw data with an encoder, estimator, or model.
   * - ``seq_log_density``
     - Return per-observation log-density arrays over encoded chunks.
   * - ``seq_log_density_sum``
     - Return total count and summed log-density.
   * - ``log_density`` / ``density``
     - Convenience wrappers for raw data.
   * - ``seq_initialize`` / ``initialize``
     - Build an initial estimate.
   * - ``seq_estimate`` / ``estimate``
     - Run one estimation update over encoded or raw data.

The same functions accept local lists, Spark RDDs, data-source objects, and
parallel encoded-data handles when the relevant backend is available.

Sequence drivers should agree with scalar calls on counts, score sums, and
non-finite behavior. Test ordinary rows, impossible rows, and any supported
missing-data representation.

Declarations
------------

``mixle.stats.compute.declarations`` records family metadata that can be used by
engines and generated kernels:

``DistributionDeclaration``
    Describes parameters, sufficient statistics, support constraints, and
    optional exponential-family structure.

``ParameterSpec`` and ``StatisticSpec``
    Describe parameter and statistic layouts.

``ExponentialFamilySpec``
    Provides the canonical natural-parameter and sufficient-statistic form.

``register_declaration`` / ``declaration_for``
    Register and retrieve declaration metadata.

``validate_declaration`` and diagnostics
    Check that declarations are internally consistent and compatible with
    generated scoring.

Declarations are how a family becomes visible to symbolic backends, generated
Numba kernels, stacked mixture paths, Fisher views, and capability predicates
without adding special cases to central inference code.

A declaration is release-ready only when validation checks both metadata and
behavior. Parameter shapes, support constraints, sufficient statistics, and
generated score parity should all be tested.

Generated Kernels
-----------------

``mixle.stats.compute.kernel`` chooses scoring kernels for a model and engine:

``Kernel``
    Runtime object with scoring and sufficient-statistic methods.

``KernelFactory``
    Produces kernels for compatible distribution types.

``GenericKernelFactory``
    Uses the family-provided vectorized methods.

``NumbaKernelFactory`` and ``GeneratedNumbaKernelFactory``
    Use handwritten or declaration-generated Numba paths where available.

``StackedMixtureKernelFactory``
    Scores many related mixture components in a stacked representation.

``kernel_for`` and ``register_kernel_factory``
    Select or register a kernel path.

The kernel layer should preserve scalar semantics. Faster code is only useful
when it returns the same density and sufficient statistics as the reference
path.

Benchmark results should be recorded after correctness evidence, not before.
Performance numbers without parity evidence can make an invalid path look more
mature than it is.

Backend Scoring
---------------

``mixle.stats.compute.backend`` exposes backend scoring helpers:

``backend_seq_log_density``
    Score encoded data on a compute engine or backend.

``backend_seq_component_log_density``
    Score mixture or component structures when component scores are required.

``backend_log_density_sum``
    Return aggregate count and log-density.

``BackendScoringError``
    Clear failure when a requested backend path is not supported.

Backends should fail loudly when they cannot preserve semantics. Silent
fallbacks are only acceptable when the caller explicitly requested an automatic
route and the reported result still records what happened.

Backend errors are part of the public contract. Test missing optional
dependencies, unsupported engine/model combinations, and fallback reporting.

Stacked and Fused Mixtures
--------------------------

Mixtures are a common performance bottleneck. The compute layer includes
special paths for them:

``mixle.stats.compute.stacked``
    Stacks component parameters, scores component log-densities, computes
    component sufficient statistics, and unpacks component estimates.

``mixle.stats.compute.fused_kernels`` and ``fused_nested``
    Provide fused scoring and accumulation for nested structures.

``mixle.stats.compute.torch_mixture``
    Keeps mixture scoring resident in Torch when the model and engine support
    it.

These paths are implementation details of a public goal: a composed mixture
should still look like a distribution, while the runtime avoids doing expensive
per-component Python work when it can.

Stacked and fused paths need stress checks for component weights,
near-impossible observations, and responsibility normalization. They should not
produce ``NaN`` where the reference path returns a finite value or ``-inf``.

Posterior, Gradient, and Decomposition Metadata
-----------------------------------------------

Additional compute modules support specialized inference:

``posterior``
    Posterior helper logic for latent-variable models.

``gradient``
    Gradient fit state and prior conversion helpers for differentiable updates.

``decomposition``
    Declares axes and reduction operations for sharding model work.

``capabilities``
    Runtime capability metadata associated with compute implementations.

``sampling_api``
    Dispatch surface for sampling paths.

These modules are not usually imported by application code, but they are
important when extending the system.

Maintenance Checklist
---------------------

When adding or changing compute behavior:

* keep scalar and encoded scoring in parity;
* validate declarations before relying on generated code;
* make density semantics explicit;
* add capability tests for optional behavior;
* test local and chunked encoded data;
* test backend errors as well as backend success;
* preserve estimator accumulator merge behavior;
* benchmark only after the reference path is correct.

Release Evidence
----------------

For compute-layer changes, preserve:

* scalar versus encoded score parity;
* accumulator merge parity and sufficient-statistic shape checks;
* declaration validation output for generated routes;
* backend success and backend failure behavior;
* non-finite and impossible-observation parity;
* optional dependency guard behavior; and
* benchmark context only after correctness gates pass.

API Reference
-------------

* :doc:`api/mixle.stats.compute.pdist`
* :doc:`api/mixle.stats.compute.encoded`
* :doc:`api/mixle.stats.compute.sequence`
* :doc:`api/mixle.stats.compute.declarations`
* :doc:`api/mixle.stats.compute.kernel`
* :doc:`api/mixle.stats.compute.backend`
* :doc:`api/mixle.stats.compute.stacked`
* :doc:`api/mixle.stats.compute.gradient`
* :doc:`api/mixle.stats.compute.decomposition`
