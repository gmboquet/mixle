Glossary
========

Accumulator
    The object that collects sufficient statistics, posterior expectations, or
    neural training telemetry during fitting. Accumulators are designed to be
    mergeable across encoded-data chunks.

Backend
    The execution location for encoded data, such as local, multiprocessing,
    Spark, Dask, or MPI.

Calibration
    A procedure that aligns confidence values with observed error rates.
    Calibration does not prove a model is accurate; it makes uncertainty
    statements easier to interpret and route.

Capability
    A declared behavior that a model supports, such as exact density,
    enumeration, finite support, latent posteriors, backend scoring, or
    conjugate updates. Prefer ``mixle.describe`` over class checks.

Certified
    A fitted block whose estimation method carries a proof-level guarantee --
    closed-form MLE or a conjugate update -- recorded in the estimation
    certificate. Distinct from *exact* (enumeration/support traversal) and
    *approximate* (gradient descent, variational, MCMC): "certified" is about
    how the parameters were solved, not about scoring the support.

Checkpoint
    A *training* checkpoint stores everything needed to resume training --
    optimizer, scheduler, step, RNG, and data-loader position. Distinct from an
    inference *Artifact*, which stores only architecture config plus learned
    weights and is ready for scoring, not for resuming training (see
    ``LM.save`` / :doc:`support-policy`).

Composite
    A distribution or estimator over tuple-shaped observations. Each tuple
    field has its own child distribution or estimator.

Conformal Calibration
    A finite-sample calibration method used in mixle task models to form label
    sets and decide whether to answer locally or escalate.

Distribution
    A fitted probabilistic object with parameters and scoring behavior,
    normally through ``log_density`` and vectorized encoded scoring.

DOE
    Design of experiments: utilities for choosing which candidate, input,
    label, task, modality pair, or oracle call should be evaluated next.

Encoder
    The object that converts Python observations into vectorized encoded data
    consumed by accumulators and scoring kernels.

Engine
    The array, precision, or device layer used for local math, such as NumPy,
    Torch, JAX, or symbolic engines.

Estimator
    The object that declares a model shape and knows how to estimate a fitted
    distribution from accumulated evidence.

Evidence
    A record, observation, score, label, receipt, or artifact used to update a
    model or justify a decision. In release documentation, evidence should name
    its source and validation status.

Evolution Loop
    The ``mixle.evolve`` measure-propose-verify-promote workflow used to
    improve a model while preserving an auditable anti-regression gate.

Experimental
    A maturity tier (see :mod:`mixle.maturity`): no compatibility guarantee.
    Only ``mixle.experimental`` and the frontier-training prototypes are
    experimental. Distinct from *stable* (covered by the compatibility policy)
    and *provisional* (usable, may change within a minor release).

HMM
    Hidden Markov model: a sequence model with a latent state path and emission
    distributions.

Latent Model
    A model with hidden variables, such as mixture component assignments or HMM
    state paths.

Missing Data
    Observations that are absent or undefined. In Mixle statistical code,
    ``NaN`` should remain semantically visible unless a documented model
    explicitly marginalizes, masks, or transforms it.

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

PPL
    Probabilistic-programming layer. ``mixle.ppl`` lets users declare symbolic
    model expressions that lower back into ordinary Mixle distributions,
    estimators, and inference routes.

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

Release Evidence
    The recorded commands, artifacts, versions, commits, validation results,
    and skipped gates that support a release claim.

Representation Layer
    The ``mixle.represent`` subsystem that separates segmentation, embedding,
    heterogeneous encoding, and optional vector quantization.

Semantic Entropy
    Entropy over answer meaning clusters rather than surface strings. Used as
    an uncertainty signal for LLM answers.

Task Model
    A durable local model from ``mixle.task`` that can be loaded in a fresh
    process and called as a plain function.

Train
    Colloquial for fitting a neural leaf by gradient descent. mixle's uniform
    verb is *fit* (via ``optimize``) regardless of family -- "train" is used
    only where the family is a neural network, and "optimize" names the single
    inference entry point that chooses the method from the model's structure.

Transformer Leaf
    A neural next-token distribution used as a child in a larger mixle model,
    typically through ``TransformerLMEstimator`` or ``StreamingTransformer``.

Verifiable Oracle
    A callable that assigns a score or label with enough provenance for a DOE,
    task, or improvement loop to trust the result. Surrogate students may
    propose candidates, but the oracle supplies the accepted verification
    signal.

Wheel
    The built Python package artifact users install. Release validation should
    include checks against the wheel, not only the source checkout.
