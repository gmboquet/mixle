# Changelog

All notable changes to mixle are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.8.0] — Unreleased

The credibility, stability, and proof release: turning mixle from a broad, fast-moving research
package into one whose supported core, performance, artifacts, and public claims can be independently
trusted. Tracked by the release checklist in `release-checklists/0.8.0.md`. New capability is deferred
to post-0.8 or kept under `mixle.experimental` per the feature freeze.

### Added

- Stable ``mixle.semantics`` contracts for value roles, units, transforms and
  Jacobians, constraints, priors, observations, posterior/predictive identity,
  uncertainty, calibration, decisions, capability extensions, and trace sinks;
  plus a packaged cross-project Bayesian-inversion fixture whose semantic
  identity excludes backend, job, and storage-location metadata.
- A machine-readable ownership decision for every top-level public module,
  with drift coverage and migration destinations that preserve compatibility
  until replacement and deprecation gates are satisfied.
- A public-API manifest (`api_manifest.json`) and a drift gate so any change to the exported surface
  is a reviewed diff.
- Release-engineering gates: a weighted-estimation contract test, a base-install optional-import guard,
  a tracked benchmark harness, a pull-request template, and the 0.8.0 release checklist.
- `CompiledEM` as a reusable fused full-mixture strategy, automatically selected by `optimize()` for
  eligible partially fusible heterogeneous mixtures; recursive SQUAREM packing for nested
  mixtures/composites; and function-preserving shared-trunk/residual-expert MoE upcycling.

### Fixed

- Block scheduling now prices density, responsibility, and parameter-update work together instead of
  treating density time as the whole block cost; learned controllers receive the same measured cost.
- Dirichlet-prior block and freeze/roll-up updates now use the exact MAP objective and carry the
  posterior weight prior; nested homogeneous mixtures preserve heterogeneous encoding depth.
- `task.regress`'s internal MLP student (`solve_regression`, `RegressionSolution.improve`, and any
  downstream distillation built on it, e.g. `mixle_pde.surrogate.distill_forward`) now builds its
  network at an explicit float32 instead of following `torch.get_default_dtype()`. A caller that had
  changed the process-global default dtype (mixle-pde's PDE code routinely does, for numerical
  precision) left the student's own weights at that ambient dtype while its inputs stayed explicitly
  float32, crashing with "mat1 and mat2 must have the same dtype, but got Float and Double".

### Fixed

Findings from the 2026-07-13 full-tree code review (IDs reference its audit ledger; every fix ships
with a regression test that fails on the unfixed code):

- Fused numba kernels no longer compile with `fastmath=True` (`ninf`/`nnan` miscompiled -inf
  out-of-support scores into positive log-densities, reachable through `optimize()` auto-fusion), the
  fused E-step and nested emitters guard all-impossible rows, nested fusion declines
  min/max-statistic leaves, and the fused Pareto E-step tracks per-component support minima
  (D-1..D-5, D-7; #428).
- HMM `viterbi`/`seq_viterbi` perform real backpointer backtracking; the `taus` parameterization
  scores correctly in both scalar and vectorized paths; `seq_posterior` returns smoothing (not
  filtered) marginals; heterogeneous-emission models work through every read-out API;
  hierarchical-mixture EM is monotone on variable-length corpora; LDA handles empty trailing
  documents; structured/IO-HMM guard zero-mass observations and `fit(fast=True)` honors
  `final_states` (L-1..L-12; #435).
- `ops.quantize` brackets with the spatial quantile (was: half of every symmetric distribution
  silently discarded); `relations.ViterbiPath` delegates to the admissible k-best HMM search;
  split-conformal certification abstains honestly at small calibration sizes and certifies every
  class head; registry/relations/fault edge cases raise clear errors (C-1..C-11; #425).
- ppl: VI applies the non-centered transform (was: z-space values returned as posteriors);
  mixed-model EM converges on all parameters, with GLS standard errors; MAP point estimates drop the
  transform Jacobian (flat-prior MAP equals the MLE again); composite slot names no longer collide in
  summaries; half-normal log-density constant, IRLS coefficient-prior scaling, Kalman-EM
  initial-state timing, the NIG sigma summary, and `vi_fit(seed=)` (P-1..P-11; #431).
- inference: `optimize(schedule="auto")` can no longer accept a zero-support model (impossible rows
  score -inf, matching the mixture contract); keyed (tied) parameters survive the posterior-transform
  strategies and the heterogeneous executor; HMM conditioning answers past the evidence horizon with
  a true forward-algorithm joint; spatial-block folds clamp the max edge; Vuong pretests degenerate
  variance (I-1, I-3, I-6..I-9, I-11; #432).
- UQ/statistics: `brunner_munzel` one-sided p-values un-inverted; Wilcoxon `pratt` zero corrections;
  the particle filter carries SIS weights when not resampling; jackknife+/CV+ use the finite-sample
  order statistics with unbounded small-n endpoints; ESS uses Geyer's initial-monotone truncation per
  component; `nuts_numba` thinning returns the requested draw count; frailty-Cox hazard lookup,
  rank-normalization plotting position, Efron reported likelihood, canonical-link labels, m-out-of-n
  rescaling, and exact permutation p-values (U-1..U-10, I-4, I-5; #430).
- Neural leaves hold `eval()` during scoring and `train()` during fitting (dropout/batchnorm modules
  scored stochastically and mutated running statistics on mere `log_density` calls); multi-field
  accumulator fan-in no longer drops fields; module dtype follows the engine precision (G-1, G-2,
  G-5; #426).
- API consistency: `optimize(seed=)`; mixture/categorical constructor validation (negative
  categorical probabilities were accepted and returned negative densities); empty-data `ValueError`s
  and the `raise Exception` -> `ValueError` narrowing; ppl dialect aliases (`log_density`,
  `sample(size=)`, `GaussianObs`, HMM `components=`, pair-copula `log_density`); scalar
  `pseudo_count` broadcast; GP `fit` returns the model (S-1..S-4, S-6..S-10, S-13, S-14; #433).
- Completeness: structured/IO-HMM `TransitionOperator`s serialize (fitted models round-trip);
  `InputOutputHMM` gains sampler/viterbi/posterior-decode/state-posteriors; BetaBinomial honors the
  finite-support contract; LogSeries/Skellam/DirichletMultinomial enumerate; closed-form moments and
  entropies filled across the univariate catalog; `mixle.ppl` exports `waic`/`loo`;
  `ScheduledHMM.estimator()` restores the prototype convention (F-1, F-2, F-4, F-6, F-7, F-9, F-11,
  F-12; #434).
- Completeness follow-up: #434's "entropies filled across the univariate catalog" covered 5 of the
  11 families F-9 actually names. `SkewNormal`, `NegativeBinomial`, `Rician`, `Nakagami`, `Skellam`,
  and `LogSeries` now have `entropy()` too -- a closed form for Nakagami (via the Gamma-entropy
  identity under `X = sqrt(Y)`), a closed-form reduction plus one adaptively-quadrated term for
  SkewNormal, adaptive quadrature for Rician, and exact series summation (truncated at each
  distribution's own quantile, not a fixed-width heuristic) for NegativeBinomial/Skellam/LogSeries.
  Verified against independent numerical integration/Monte Carlo, not only scipy: scipy's own
  generic `entropy()` silently returns a wrong value ("sum did not converge") for NegativeBinomial
  and LogSeries at strongly over-dispersed parameters (F-9; #512).

### Changed

- Performance (exactness-preserving, parity-tested): multivariate-Gaussian scoring precomputes the
  inverse Cholesky factor for single-gemm scoring (1.3-2.3x kernel, 1.59x end-to-end on the benchmark
  GMM config, identical likelihoods); the torch objective fitters evaluate one forward per Adam
  iteration instead of two, with NaN-aware best-state tracking; `seq_encode` chunking uses stride
  slices (~76x on ndarray inputs) (E-1, E-2/G-3, E-3, I-2; #436).

## [0.7.0] — 2026-07-09

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
- Release-verification pass (found via a fresh, non-editable venv install of the built wheel --
  never caught by the dev environment, which has every optional extra installed): `import mixle`
  was completely broken (a missing-import `NameError` at class-definition time in
  `mixle.models.dpo_leaf` cascaded through `mixle.models.__init__`'s eager import chain into nearly
  every module), plus the same missing-import/stale-duplicate-method pattern recurring across 7
  sibling model files and `pinn.py`; 8 further modules importing torch-gated names unconditionally
  at module level (undermining their own already-correct optional-torch guards); a real
  `DPOAccumulator.value()` bug (weights returned as a list, not an array); a data-shape bug in
  `zero_shot_bootstrap`'s generic neural-density fallback (a 24-dim row was split into 24 scalar
  fields instead of one vector field); a stale test fixture double-wrapping `Registry.tier_stack`'s
  frontier callable; a layering violation (`mixle.experimental.long_context_eval` importing upward
  from `mixle.ppl`); and 2 more test files with the same unguarded-torch-import bug. Also: `numba`'s
  `tbb` dependency floor made `pip install mixle[numba]`/`mixle[all]` uninstallable on Apple Silicon
  (no arm64 wheels) -- now platform-gated; `ray` and `lightning` were used by real, documented
  optional backends with no corresponding `pip install mixle[...]` extra -- both added.

### Changed

- `pinn_leaf.py` renamed to `pinn.py`; the "leaf" suffix dropped from PINN naming throughout.
- Several `mixle/doe` tests re-marked `slow` (heavy Monte Carlo / neural-density fits) so the default
  fast test gate stays fast; a real duplicate-training bug (an estimator refit once per test instead
  of once per class) fixed alongside the re-marking.

## [0.6.2] — 2026-07-05

Workstream: the "frontier ecosystem" reasoning/knowledge stack (substrate, retrieval, the `Reasoner`
facade, `Harness` products, factuality receipts, governance/trust) built out across roughly twenty
parallel workstreams, plus a hardening pass on weighted accumulators, the model registry, MVN/HMM
numerics, and Torch DTensor sharding.

### Added

- Knowledge substrate and reasoning stack: typed/provenanced/scoped storage, multi-hop retrieval,
  `answer_from_substrate` with abstain-and-cite, the full `investigate()` action space
  (retrieve/compute/simulate/create/delegate), the `Reasoner` facade (`answer`/`ingest`/`improve`),
  and the `Harness` product with domain templates and a registry.
- Factuality receipts and a knowledge-graph/ontology stack: constrained decoding against an ontology,
  KG-RAG retrieval, estimation certificates and planner, calibration folded in as a post-condition,
  telemetry dashboards, learned pool/reasoner routing policies, governance/sharing controls, and a
  trust/audit trail.
- Cross-modal reasoning graph nodes, file connectors, exchangeability preconditions, and a
  context-packet/compression layer; four flagship example applications plus vision/edge-distillation
  demos; the "Scientist" laptop product.
- Neural-density families made directly constructible via kwargs (e.g. `VAE(dim=8, latent=2)`)
  instead of `build_*` factories; broadened distillation task support (response/multi-teacher/hint/
  attention/relational/sequence); pickle + `to_dict`/`to_json` serialization for `StreamingTransformer`
  and DPO leaves; optional percentile clipping in `quantize` to bound int4 outlier collapse.

### Fixed

- `mixle.models`: `DPOAccumulator` and `StreamingTransformerAccumulator` silently ignored per-sample/
  per-token weights -- `update`/`seq_update` dropped the weight and `value()`/`estimate()` computed an
  unweighted mean loss, so weighted EM, mixture responsibilities, streaming decay, or explicit sample
  weighting had no effect on DPO or streaming-transformer fits (bit-identical output regardless of
  weight). Weight now carried through the full accumulate/value/M-step path.
- `mixle.inference.production.registry`: `header()`/`metadata()` raised a bare `IndexError` and
  `get()` leaked a raw `FileNotFoundError` with the store path on an unregistered name/missing
  version -- unified behind a single `_resolve_version` guard that raises a consistent `KeyError`.
  Also fixed an unsanitized name/version/alias join onto the store root (a path-traversal vector if
  names are ever API-supplied).
- `mixle.inference.structure`: `_clone` cloned an estimator template via `eval(str(estimator))`;
  since most estimators use the default `<object at 0x...>` repr, the eval always raised
  `SyntaxError` and silently fell back to returning the same shared object rather than a copy --
  correct only by luck for stateless estimators. Replaced with `copy.deepcopy`.
- `mixle.task`: a saved `Solution` with `qhat=inf` reloaded as `None` and broke every subsequent call;
  `inf` is now persisted explicitly. `batch([])`/empty-input handling now returns `[]` uniformly.
- Neural leaves (`NeuralGaussian`, `softmax_leaf`, `mixture_density`, `energy`, `neural_density`):
  accumulators appended one ndarray per row and `np.stack`-ed the entire dataset every EM iteration
  (profiled as a major blowup at scale); rewritten to concatenate once at `value()`. Also fixed an
  array-truthiness bug (`if not xs` raised `ValueError` on an ndarray instead of detecting empty
  input), streamed `LM.nll` via chunking instead of one large `np.stack`, and made `make_mlp` raise
  on non-positive dims instead of silently building a degenerate constant net.
- `mixle.stats` MVN: `_robust_cho_factor` -- float32 (MPS/CUDA) MVN mixture EM crashed with "leading
  minor not positive definite" at higher dims from catastrophic cancellation; now symmetrizes and
  adds trace-scaled jitter only on failure (float64 path unchanged). Separately fixed an
  `(N,K,dim,dim)` memory blowup that OOM'd GPU MVN-mixture fits.
- `mixle.engines` (Torch/DTensor): component-sharding raised `ImportError` on torch 2.0-2.4 even
  though DTensor was reachable via a private module path pre-2.5; fixed with a public-then-private
  import fallback, and the sharded EM fit itself is now explicitly gated to torch >= 2.5 with an
  actionable error instead of crashing on an unsupported `logsumexp`/`isinf` sharding strategy.
- `mixle.inference.glm`: IRLS crashed on rank-deficient/collinear designs (e.g. correlated
  feature-vector parents in cross-modal graph fits); `_solve_psd` now falls back to minimum-norm
  `lstsq`/`pinv` at all three IRLS solve sites, bit-unchanged at full rank.
- `mixle.stats` HMM: the HMM distribution defaulted `use_numba=False` while the estimator defaulted
  it to `HAS_NUMBA`; since `optimize(prev_estimate=init)` encodes data through the distribution's
  encoder, the common "pass an init" HMM fit silently never used numba (~90x slower than expected).
  Distribution default now matches the estimator.

### Changed

- `mixle.stats` MVN covariance accumulation switched from `np.einsum` to a BLAS `matmul` -- 2.7x
  faster end-to-end MVN mixture fits, byte-exact.
- `mixle.stats` HMM distribution's `use_numba` default changed from `False` to `HAS_NUMBA`;
  behavior-preserving (bit-identical) but changes default performance characteristics, and an
  explicit `use_numba=False` is still respected.
- Neural-density model construction moved from `build_*` factory functions to direct constructible
  construction -- an API-shape change for consumers of the old factories.
- `mixle.utils.builder` removed as dead code; example/benchmark harnesses moved out of tracked
  `examples/` into gitignored `benchmarks/`.

## [0.6.1] — 2026-07-04

Workstream: the `mixle.task`/`solve()` lifecycle facade (rigid function to deployed, monitored
model), exact/approximate enumeration engines, neural-density adapters wired into the PPL, a
structured HMM/HSMM/Bayesian-network family, cross-modal fusion and LLM uncertainty quantification,
and a precision/distributed-compute engine push (LNS integer arithmetic, JAX/XLA jitted EM,
FSDP2/Spark/MPI transports).

### Added

- `mixle.task`/`solve()` lifecycle facade: `solve()` closing the loop from a rigid function to a
  deployed model with a reliability/OOD gate, `solve(synthesize=N)` generative dataset creation,
  `Solution` save/load/verification records across regression/multi-label/structured tasks,
  `Solution.health()` live-traffic conformal monitoring, `Cascade` serving with realized savings and
  self-improving harvest, cost-economics route recommendation, calibrated N-tier `Router`, generative
  and grammar-constrained plan decoding, distillation planners/tool-callers, edge distillation with
  int8/int4 quantization and device-budget search, and structured-record extraction tasks
  (`HashedRecord`, active labeling, `recommend_model`).
- Enumeration engines for exact/approximate ranking: `LatticeEnvelopeIndex`, `RescoredIndex`
  speculative enumeration, certified `branch_cap` pruning, `HMMPathIndex` quantized count-DP,
  `AREnvelopeIndex` for LLM deep enumeration, a persistent `SeekIndex`, a numpy/batched fast path
  (~24x on gpt2), and quantized-inference certificates (`logit_error_bucket_slack`).
- Neural-density adapters wired into the PPL as first-class constructors (`NeuralDensity`,
  `NeuralConditionalDensity` wrapping VAE/MAF/MDN/autoregressive/flow torch models), `EnergyModel`
  (NCE + Langevin), `fisher_merge` closed-form Fisher-weighted parameter merge, closed-form
  variational GMM collapse/Runnalls KL mixture reduction, new conjugacy pairs (NIG, Gamma-rate,
  Categorical-Dirichlet, NegBinomial-Beta), and an `explain_fit`/`describe`/`how='laplace'`
  escalation ladder.
- Structured HMM/Bayesian-network family: `StructuredHMM` with low-rank/Kronecker/block-diagonal
  transitions and streaming/parallel Baum-Welch, `ExplicitDurationHMM`/HSMM with segment decoding,
  `InputOutputHMM`, scheduled (length/position-conditional) HMMs, and Bayesian networks with
  heterogeneous regression/GLM/linear-Gaussian edges, mixture-of-DAGs, and `counterfactual()`
  do/abduction-action-prediction.
- Cross-modal fusion and uncertainty: `ProductOfExpertsFusion`, `StructuredFusionClassifier`,
  `CrossModalModel` PoE-VAE with conformal intervals, `CrossModalStore` cross-modal RAG,
  `LLMUncertainty` semantic-entropy plus conformal abstain, claim-level UQ, `BeliefState`
  epistemic-aleatoric decomposition, `DiscreteAnswer.decide`, and the `mixle.reason` front door.
- Compute engines: LNS (logarithmic number system) integer arithmetic with an integer log-sum-exp
  kernel (14x on LM cross-entropy), packed binary/ternary/sub-byte precision kernels, an MPFR
  arbitrary-precision tail, a precision-spectrum planner with data-aware `optimize(precision=)`,
  torch/GPU scoring rolled out across roughly twenty distribution families, distributed EM transports
  (MPI, Spark, FSDP2/CUDA bf16 with DCP sharded checkpoints), and JAX/XLA jitted EM verified ~21x on
  Apple M4.
- `mixle.evolve` Phase 1: typed search space, bandit-population meta-search, and structure operators;
  declarative `LM`/streaming-transformer/DPO leaves with SFT loss-masking and CPT+EWC.

### Fixed

- `mixle` package `__all__`: `ExplicitDurationHMM` was listed but its import line had been dropped by
  a linter re-sort, so `from mixle import *` raised `AttributeError` and broke roughly 70 test
  collections.
- `mixle.inference.streaming`: a circular import through the `mixle.stats` package surface broke
  `mixle.inference` on import.
- `hmm_engine_forward_backward`: read a tensor's shape via `np.asarray` after it had already moved
  onto the torch/MPS engine, crashing GPU forward-backward with a device-conversion error; a related
  `max()` bug was also blocking tweedie torch/GPU scoring.
- `should_auto_fuse`: auto-fusion policy checked fusibility/workload but not numba availability, so a
  numba-free install crashed with `ModuleNotFoundError` mid-fit instead of falling back gracefully.
- PPL conjugate-bridge routing: an unsound route for a binary (logit) GLMM fit by PQL; `how='auto'`
  now warns instead of silently returning a MAP point estimate when the prior has no closed-form
  posterior.
- Constrained plan decoding: an earlier grammar constraint guaranteed output form but not content,
  making an undertrained model more confidently wrong (a silent correctness regression) -- fixed with
  a calibrated confidence floor.
- Enumeration: an earlier "NTT loses to Kronecker" benchmark conclusion was an implementation
  artifact rather than a real limitation; exact multi-prime NTT convolution now lands correctly.
- The README hero example referenced an unused/mislabeled estimator and didn't run; replaced with a
  correct, self-contained hierarchical-mixture topic model, and all runnable README blocks now
  execute standalone.
- Several flaky/order-dependent tests that were masking shared-state bugs: an unrestored
  `set_default_dtype(float64)` leaking across co-scheduled tests, neural-density adapter tests
  depending on global RNG state, and a wall-clock timing assertion invertible under load.

### Changed

- **Breaking**: `optimize(data)`/`fit(data)` now perform automatic dependency-structure discovery by
  default (previously opt-in); text fields also now join the dependency graph.
- Neural leaf classes renamed off the tree-position "...Leaf" suffix, with back-compat aliases kept
  (e.g. `NeuralDensityLeaf` -> `NeuralDensity`, `StreamingTransformerLeaf` -> `TransformerLMEstimator`).
- `mixle.program` (the closure-taking declarative optimization surface) demoted to
  `mixle.experimental.program` as not yet mature; no deletions.
- `benchmarks/` removed from the repo and gitignored; Sphinx `docs/` un-ignored and published to
  GitHub Pages instead of the ad hoc README-embedded examples.

## [0.6.0] — First mixle release

- PPL language core: deterministic-expression slots, `potential()` custom factors,
  `.each(by=)` indexed-flat hierarchical models, data-indexed latents (`theta[Field]`), non-Normal
  GLMMs, R-hat/ESS in `summary()`.
- Automatic-inference `fit` API: prototype/data coercion, `fit` forwards all `optimize` kwargs.
- Categorical/Dirichlet free-dimension inference.
- Streaming-estimator unification.
- Lower-bound version pins across the always-installed core and every optional-dependency extra, so
  users on too-old dependencies get a clear resolver error instead of obscure runtime breakage.

[Unreleased]: https://github.com/gmboquet/mixle/compare/v0.6.2...HEAD
[0.6.2]: https://github.com/gmboquet/mixle/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/gmboquet/mixle/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/gmboquet/mixle/releases/tag/v0.6.0
