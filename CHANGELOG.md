# Changelog

All notable changes to mixle are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.6.3] — 2026-07-09

Workstream: generic AI-capability platform pieces on top of the core estimation engine (task
decomposition, cross-modal reasoning, self-improvement loops, a system facade), plus a hardening
pass across the automatic-inference and design-of-experiments subsystems.

### Added

- New model families: `PINNRegression` (physics-informed neural network leaf), `HamiltonianNet`
  (conservation-law-preserving dynamics), `make_deep_set` (permutation-equivariant networks),
  monotonic MLPs and input-convex energy networks, `build_product_energy_net` (energy-based product
  of experts), `CopulaDistribution` (arbitrary marginals + a Sklar dependence core),
  `GatedMixtureDistribution` (input-dependent mixture-of-experts weights).
- Task-decomposition and agent-facing workstreams: plan models fit as Markov chains over agent
  traces, an outcome-trained decomposer, a minimal orchestrator loop, tabular Q-learning and
  maximum-entropy inverse RL, a local model registry, `ExecutionTrace` with bit-identical replay, the
  `Receipt` object (ledger + trace + calibration + provenance, offline re-verifiable), and the
  `System` facade (`answer`/`ingest`/`improve`).
- Cross-modal reasoning: `ModalityView`, per-edge conditional-transport premise checks, belief walks
  across chains of verified transports, cycle-consistency as a self-supervised abstention signal,
  task-sufficient projection, information-gain retrieval, and the workstream-F flagship harness.
- Self-improvement / knowledge-accumulation loops: the collapse monitor shared across amplification
  loops, the composition operator, `DesignModel`'s cross-round what-works prior, diagnosis-directed
  correction, a knowledge-accumulation flywheel measurement, degradation-policy handling for fault
  modes, and cost-aware routing threshold selection.
- `doe`: `VerifiableOracle` + the design-test-learn loop, noise-robust incumbent selection for
  Bayesian optimization, and a budgeted propose-verify-retrain loop over a discrete design space.

### Fixed

- `mixle.utils.automatic`: crashes on empty/degenerate input (`ZeroDivisionError` in the Poisson/
  Gaussian/log-normal estimator builders), every distribution detector silently dropping its own
  already-computed fit and the caller's `pseudo_count`, an `IndexError` on always-empty sequence
  fields, a modality-fingerprint diagnostic that could contradict the actual estimator built, and the
  model-suggestion logic ignoring its own held-out validation signal when it disagreed with the
  in-sample BIC pick.
- `mixle.data`: `Boolean.coerce` silently inverting string-typed values (`bool("False") == True`),
  `Schema.conform_record` silently truncating mismatched records via an unchecked `zip`, an
  inconsistent tuple-vs-list adjacency coercion in the graph data source, and `MaterializedSource`
  silently accepting non-reiterable one-shot iterators.
- `mixle.doe`: an unguarded Cholesky decomposition crashing on singular/near-singular input
  covariance, a Morris-screening `ZeroDivisionError` on a degenerate grid, a silent-`NaN` Gaussian
  Process surrogate when fit with zero observations, an infinite loop given a zero-cost fidelity in
  multi-fidelity optimization, several proposal functions crashing instead of validating
  `n_candidates`, TuRBO overshooting its evaluation budget on trust-region restart, silent
  batch-truncation/duplication under an oversized batch request, and `BayesianOptimizer.ask()`
  re-dispensing duplicate initial-design points in async/parallel ask-before-tell campaigns.
- A circular import that broke `mixle.inference` entirely; a mixture-correction term error in
  `explain()`'s decision-margin ledger; a stale `capacity.py` embedding-head rung mismatch; CI
  flakiness in EM's log-likelihood computation and a de-flaked mixture-of-trees test; Python
  3.10-specific abstention timing in the oracle-timeout path.

### Changed

- `pinn_leaf.py` renamed to `pinn.py`; the "leaf" suffix dropped from PINN naming throughout.
- Default `pytest` worker count capped (`-n 4` instead of `-n auto`) to stop individual test runs
  from oversubscribing shared CI/dev machines when many run concurrently.
- Several `mixle/doe` tests re-marked `slow` (heavy Monte Carlo / neural-density fits) so the default
  fast test gate stays fast; a real duplicate-training bug (an estimator refit once per test instead
  of once per class) fixed alongside the re-marking.

## [0.6.2]

- Test-suite hardening: skip embedder-path `Budget` cases when torch is absent.

## [0.6.1]

- Maintenance release.

## [0.6.0] — First release under the `mixle` name

Renamed from `pysparkplug`. Highlights since 0.5.2:

- PPL language core: deterministic-expression slots, `potential()` custom factors,
  `.each(by=)` indexed-flat hierarchical models, data-indexed latents (`theta[Field]`), non-Normal
  GLMMs, R-hat/ESS in `summary()`.
- Automatic-inference `fit` API: prototype/data coercion, `fit` forwards all `optimize` kwargs.
- Categorical/Dirichlet free-dimension inference.
- Streaming-estimator unification.

## [0.5.2]

- Maintenance release.

## [0.5.1]

- Added lower bounds to every optional-dependency extra (not just the always-installed core), so a
  user selecting an extra with a too-old version gets a clear resolver error instead of a broken
  connector: `tbb>=2021.6`, `pandas>=2.0`, `pyarrow>=14`, `sqlalchemy>=2.0`, `pymongo>=4.3`,
  `fsspec>=2023.1.0`, `networkx>=3.0`, `gmpy2>=2.1`, `pyspark>=3.4`, `dask`/`distributed>=2023.5.0`,
  `torch>=2.0`, `mpi4py>=3.1`.

## [0.5.0]

- Added lower-bound pins for the always-installed core dependencies (`numpy>=1.26`, `scipy>=1.11`,
  `mpmath>=1.3`) so users on older releases get a clear resolver error instead of obscure runtime
  breakage.

[Unreleased]: https://github.com/gmboquet/mixle/compare/v0.6.2...HEAD
[0.6.2]: https://github.com/gmboquet/mixle/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/gmboquet/mixle/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/gmboquet/mixle/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/gmboquet/mixle/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/gmboquet/mixle/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/gmboquet/mixle/releases/tag/v0.5.0
