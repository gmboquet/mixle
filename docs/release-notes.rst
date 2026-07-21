Release Notes
=============

Mixle 0.8.0 is the credibility, correctness, and evidence release. It retains
the library's broad probabilistic-modeling surface while making maturity,
compatibility, numerical behavior, release claims, and operational limits
explicit and machine-checkable.

This page describes the unreleased 0.8.0 branch. It is not a publication
claim. Final artifact hashes, exact-tip CI results, independent review, and
post-publication verification remain release gates.

Highlights
----------

Evidence-backed product boundary
    Public APIs and maturity tiers have machine-readable manifests and drift
    tests. Stable, provisional, and experimental surfaces now carry different
    compatibility expectations. Architecture, ownership, security,
    scientific-validity, migration, and release documents define the supported
    boundary without presenting research prototypes as production evidence.

Numerical and statistical hardening
    The release includes a broad correctness pass over fused EM, HMM decoding
    and smoothing, probabilistic-programming transforms, posterior summaries,
    uncertainty methods, automatic structure learning, categorical support,
    entropy implementations, and impossible-observation handling. Regression
    tests preserve each repaired behavior.

Reproducible artifacts and execution
    Safe serialization, schema manifests, provenance receipts, deterministic
    hashing, checkpoint metadata, base-install import sweeps, minimum-version
    jobs, platform lanes, and artifact fingerprinting make it possible to tie a
    claim to the code and artifact that produced it.

Distributed training and estimation
    Existing Spark, Dask, Ray, MPI, and multiprocessing estimation paths have
    explicit support levels. Packed language-model training gains typed
    parallel plans, native PyTorch execution where supported, checkpointable
    training state, and adapters for external distributed-training systems.
    Multi-GPU performance remains unverified until retained hardware receipts
    exist; unsupported topology combinations fail before launch.

Scientific workflow building blocks
    The provisional task, reasoning, DOE, causal, calibration, posterior, and
    lifecycle layers now expose clearer typed contracts and receipts. They can
    support model selection, distillation, sequential design, evidence-aware
    decisions, and heterogeneous-data workflows, but do not turn Mixle into a
    frontier-model trainer or a safety guarantee.

Compatibility and dependency changes
------------------------------------

* Python 3.11 is now the minimum supported runtime. Hosted fast lanes cover
  Python 3.11 and 3.12 on Linux x86_64 and macOS arm64; the full lane uses
  Python 3.12.
* ``mpmath`` is no longer installed by the base package. Install
  ``mixle[highprec]`` for the mpmath arbitrary-precision fallback, or the
  separate ``gmpy2`` extra for its supported high-precision paths.
* Stable deprecations warn through the shared deprecation helper and remain
  available for at least two minor releases. Removed or renamed surfaces must
  have migration documentation and compatibility tests.
* Public API additions and removals require a reviewed manifest diff. Names
  under ``mixle.experimental`` remain explicitly outside the compatibility
  guarantee.

Correctness fixes in the final integration
------------------------------------------

The final stabilization includes fixes for array-backed TreeHMM encoder
equality, runtime-resolvable annotations, callable arity probing that could
invoke user code twice or mask its exception, deterministic reductions and
corpus shingles, explicit optimization failures in place of removable
assertions, and early rejection of detailed mixture scores where a scalar
objective is required.

Validation and remaining gates
------------------------------

Every pull request runs lint, minimum-version, Linux, macOS, clean-wheel, full,
documentation, and applicable security checks. The active release branch also
runs tests, strict documentation validation, and security auditing on its exact
tip after integration.

Before publication, the release checklist still requires an immutable candidate
artifact, clean-wheel and resolver evidence tied to that candidate, realistic
backend and performance receipts for retained claims, independent statistical
and systems review, external clean-install reproduction, final sign-off, and
post-publication verification. See :doc:`release-readiness`,
:doc:`claim-evidence-ledger`, and the tracked 0.8.0 checklist for the current
state.
