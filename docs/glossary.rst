Glossary
========

Accumulator
    The object that collects sufficient statistics, posterior expectations, or
    neural training telemetry during fitting. Accumulators are designed to be
    mergeable across encoded-data chunks.

Backend
    The execution location for encoded data, such as local, multiprocessing,
    Spark, Dask, or MPI.

Capability
    A declared behavior that a model supports, such as exact density,
    enumeration, finite support, latent posteriors, backend scoring, or
    conjugate updates. Prefer ``mixle.describe`` over class checks.

Composite
    A distribution or estimator over tuple-shaped observations. Each tuple
    field has its own child distribution or estimator.

Conformal Calibration
    A finite-sample calibration method used in mixle task models to form label
    sets and decide whether to answer locally or escalate.

Distribution
    A fitted probabilistic object with parameters and scoring behavior,
    normally through ``log_density`` and vectorized encoded scoring.

Encoder
    The object that converts Python observations into vectorized encoded data
    consumed by accumulators and scoring kernels.

Engine
    The array, precision, or device layer used for local math, such as NumPy,
    Torch, JAX, or symbolic engines.

Estimator
    The object that declares a model shape and knows how to estimate a fitted
    distribution from accumulated evidence.

Evolution Loop
    The ``mixle.evolve`` measure-propose-verify-promote workflow used to
    improve a model while preserving an auditable anti-regression gate.

HMM
    Hidden Markov model: a sequence model with a latent state path and emission
    distributions.

Latent Model
    A model with hidden variables, such as mixture component assignments or HMM
    state paths.

LLMUncertainty
    The ``mixle.reason`` wrapper that samples an LLM-like callable, clusters
    answers by meaning, computes semantic entropy, and can calibrate
    answer-or-abstain behavior.

Prototype Distribution
    A distribution object passed to ``optimize`` as the desired model shape.
    Mixle derives the matching estimator from it. If the distribution's
    parameter values should seed the fit, pass it as ``prev_estimate`` as well.

Operation
    A transformation over distributions, such as quantization, conditioning,
    marginalization, projection, truncation, mixture construction, or
    product-of-experts pooling.

Process Model
    A distribution over event histories, trajectories, partitions, or temporal
    arrivals, such as a Hawkes process, renewal process, birth-death process,
    or Chinese restaurant process.

Record
    A named-field observation, usually represented as a dictionary or
    schema-backed record.

Relation
    A ranked feasible set over structured objects, such as assignments, paths,
    edit-distance neighborhoods, spanning trees, feature subsets, or graph
    decisions.

Representation Layer
    The ``mixle.represent`` subsystem that separates segmentation, embedding,
    heterogeneous encoding, and optional vector quantization.

Semantic Entropy
    Entropy over answer meaning clusters rather than surface strings. Used as
    an uncertainty signal for LLM answers.

Task Model
    A durable local model from ``mixle.task`` that can be loaded in a fresh
    process and called as a plain function.

Transformer Leaf
    A neural next-token distribution used as a child in a larger mixle model,
    typically through ``TransformerLMEstimator`` or ``StreamingTransformer``.
