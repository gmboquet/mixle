# 06 — Cross-cutting / supporting interfaces (`pysp/utils/*`, `pysp/arithmetic.py`)

These are the *driver* and *supporting* contracts that sit around the core distribution/estimator
ABCs (defined in `pysp/stats/compute/pdist.py`). They consume the core contracts
(`ParameterEstimator`, `SequenceEncodableProbabilityDistribution`, `DataSequenceEncoder`,
`StatisticAccumulator`) rather than defining new distribution casts — most of the contracts here are
**de-facto / duck-typed** (followed by convention), with a few formal ABCs/dataclasses.

The three orthogonal axes from `_SPEC.md` show up as:
- **Contract** — these modules mostly orchestrate the existing distribution/estimator casts.
- **Capability facet** — they realize the *enumerable*, *rank-from-index*, *Fisher/exp-family*,
  *engine-resident*, *streaming/foldable*, *MAP/prior*, *MCMC-sampleable* facets.
- **Composition depth** — the fit/EM drivers and the parallel fold work uniformly across leaf →
  combinator → latent-state → Bayesian models because they only touch the core ABC surface.

---

## A. Fitting / estimation drivers

### `optimize(...)` / `fit(...)` / `best_of(...)`  — de-facto fit-loop entry points
- **Role:** The EM driver surface. `optimize` runs EM to a frequentist point estimate; `fit` is the
  posterior-returning (MAP/VB) counterpart; `best_of` runs many random restarts and keeps the best
  validation-LL model.
- **Formalized in:** `pysp/utils/estimation.py`.
- **Methods (free functions):**
    optimize(data, estimator: ParameterEstimator, max_its=10, delta=1e-9, init_estimator=None,
        init_p=0.1, rng=None, prev_estimate=None, vdata=None, enc_data=None, enc_vdata=None,
        out=sys.stdout, print_iter=1, num_chunks=1, engine=None, precision=None, fields=None,
        resources=None, placement=None, sub_chunks=1, chunk_size=None, backend="local",
        num_workers=None, client=None, comm=None, root=0, root_only=False, strategy=None,
        reuse_estep_ll=True, objective="auto") -> SequenceEncodableProbabilityDistribution
    fit(data, estimator, max_its=10, delta=1e-6, ..., objective="auto")
        -> SequenceEncodableProbabilityDistribution   # returns posterior-bearing model; get_prior() carried fwd
    best_of(data, vdata, est, trials, max_its, init_p, delta, rng, ...) -> (float, model)
- **Fit-loop contract (what it expects of estimator/model):** the loop only calls the *core ABC*
  surface — `seq_initialize(enc_data, estimator, rng, p)`, `seq_estimate(enc_data, estimator, model)`,
  `seq_log_density_sum(enc_data, model)` — plus, for the engine path, `_engine_seq_estimate` /
  `_engine_seq_log_density_sum`. `objective="auto"` is the single Bayesian switch: VB if the model
  exposes `seq_local_elbo`, MAP if the estimator carries a parameter prior (`model_log_density`),
  else MLE. `reuse_estep_ll=True` reuses the E-step normalizer for convergence (latent models).
- **Facets:** engine-resident (optional engine= path), distributed (backend= mp/mpi/dask via the
  encoded-data fold), streaming-adjacent.
- **Notes:** distributed *placement* is deferred to the planner layer; only `objective="mle"` is
  compatible with the fused-E-step shortcut.

### `iterate(...)` / schedule helpers / `BayesianStreamingEstimator`  — streaming-fit surface
- **Role:** Stochastic / streaming EM over batches with decay schedules.
- **Formalized in:** `pysp/utils/estimation.py`.
- **Methods:**
    constant(rho) -> schedule(t)->float ; harmonic(alpha, offset=1.0) ; forgetting(rho) ; posterior_carry()
    BayesianStreamingEstimator(estimator, schedule=None, model=None, ...).update(data=None, enc_data=None) -> model
    BayesianStreamingEstimator.reset()
- **Facets:** streaming/foldable.

### `EMStep` strategy protocol  — de-facto contract (recommend formalizing as Protocol)
- **Role:** Pluggable EM-family step object handed to `optimize(strategy=...)` / `run_em`.
- **Formalized in:** `pysp/utils/em.py` (implicit — duck-typed `step(...)` returning `EMStepResult`).
- **Methods:**
    step(self, enc_data, estimator: ParameterEstimator, model, engine=None,
         objective: Callable[[Any], float] | None = None) -> EMStepResult
    # EMStepResult(model, objective: float|None, accepted: bool=True, metadata: dict|None)
- **Implemented by:** `StandardEM`, `PosteriorTransformEM`, `HardEM`, `AnnealedEM`, `GeneralizedEM`,
  `MonotonicEM`, `ConditionalMaximizationEM`, `MonteCarloEM`, `VariationalEM`, `OnlineEM`,
  `IncrementalEM`, `AcceleratedEM`, `RestartEM`. Driver: `run_em(enc_data, estimator, initial_model,
  strategy=None, max_its=10, delta=1e-9, engine=None, objective=None, max_iter=None) -> model` and
  `observed_log_likelihood(enc_data, engine=None) -> Callable[[model], float]`.
- **Notes:** strategies are orchestration-only — they move encoded data through existing
  estimators/kernels and contain **no** distribution-specific likelihood math. **Recommend a formal
  `EMStrategy` Protocol** (`step(...) -> EMStepResult`).

### Objective interface  — `ObjectiveCallable` + optimizer loop
- **Role:** Generic differentiable-objective optimization (the escape hatch for non-iid losses: GP
  marginal likelihood, NN losses, variational projections).
- **Formalized in:** `pysp/utils/objectives.py`.
- **Type alias / protocol:** `ObjectiveCallable = Callable[[model, enc, engine], tensor]`;
  `ParameterObjectiveCallable = Callable[[Mapping[str,Any], enc, engine], tensor]`.
- **Methods:**
    fit_objective(enc, model, objective: ObjectiveCallable, engine=None, max_its=500, lr=0.05,
        optimizer="adam", tol=1e-7, maximize=True, restore_best=True, return_result=False) -> (model, value)|ObjectiveFitResult
    optimize_torch_objective(parameters: Iterable[tensor], objective: Callable[[], tensor], ...,
        maximize=True, restore_best=True) -> ... # shared gradient loop, reused by fit.py
    fit_parameter_objective(...) ; variational_projection(...) ; projection_samples(...)
- **Objective callables (classes):** `ExpectedLogDensity(weights=None, normalize=False)`,
  `ObjectiveSum(*objectives)`, `UnnormalizedLogLikelihood(...)`, `CallableObjective(fn, name)`;
  parameter packs `ObjectiveParameter`, `ObjectiveParameterSet`; result `ObjectiveFitResult`.
- **Facets:** engine-resident (torch autograd), exp-family-aware (UnnormalizedLogLikelihood log-partition).

### Gradient MLE/MAP drivers  — `fit_mle` / `fit_map`
- **Role:** Autograd point/MAP fitting through a Torch engine (constraint reparameterization +
  declaration-backed priors) — the gradient counterpart of the EM `optimize`.
- **Formalized in:** `pysp/utils/fit.py`.
- **Methods:**
    fit_mle(enc, model, engine=None, max_its=500, lr=0.05, optimizer="adam", tol=1e-7,
        precision=None, return_result=False) -> (model, value)|GradientFitResult
    fit_map(enc, model, engine=None, prior_strength=1.0, priors=None, ...) -> (model, value)|GradientFitResult
    # GradientFitResult(model, value, iterations, history, converged, log_likelihood, log_prior,
    #   prior_strength, tag, best_value/iteration, final_gradient_norm; .as_tuple(), .prior_sensitivity)
- **Implicit per-distribution hook contract (recommend formalizing):** a structured family may own
  its reparameterization by providing `gradient_fit_state(engine, torch, leaves, raw_state, tensor_param)`,
  a state object with `.shadow/.score/.build/.log_prior` methods, plus `gradient_log_prior(...)`. The
  default path reads `declaration_for(dist)` (the `ExponentialFamily`/declaration facet) and
  `backend_seq_log_density`.
- **Facets:** engine-resident, exp-family/declaration-backed, MAP/prior. Constraints handled: positive
  (`log`), unit_interval (`logit`), simplex (`logits`/softmax), ordered-bound (`greater_than:` /
  `less_than:` coupled log-delta).

### `empirical_kl_divergence` + data partitioning  — held-out evaluation
- **Role:** Validation/CV helpers (not a polymorphic interface; thin functions over the core score).
- **Formalized in:** `pysp/utils/evaluation.py`.
- **Methods:**
    empirical_kl_divergence(dist1, dist2, enc_data) -> (kl, n_bad1, n_bad2)
    k_fold_split_index(sz, k, rng) -> np.ndarray
    partition_data_index(sz, pvec, rng) -> list[np.ndarray] ; partition_data(data, pvec, rng) -> list[list]

---

## B. Fisher / geometry

### `FisherView`  — de-facto contract; `to_fisher` dispatch
- **Role:** Accumulator-backed Fisher-geometry view over *any* distribution: exposes
  posterior-expected complete-data sufficient statistics, an empirical/observed Fisher metric, and
  whitened Fisher vectors, without per-family plumbing.
- **Formalized in:** `pysp/utils/fisher.py` (de-facto base class `FisherView`; subclass `FixedFisherView`).
- **Entry point:** `to_fisher(dist, **kwargs) -> FisherView` — dispatches to `dist.to_fisher(**kwargs)`
  if present (the **to_fisher hook** facet), else `FisherView(dist)`.
- **Methods (FisherView):**
    structured_statistics(x, estimate=None, weight=1.0) -> Any          # expected complete-data suff stats
    sufficient_statistics(x, estimate=None, vectorizer=None) -> np.ndarray
    seq_structured_statistics(enc_data, estimate=None) -> list
    statistics_matrix(data=None, enc_data=None, estimate=None, vectorizer=None, fit=True) -> np.ndarray
    fisher_information(stats=None, diagonal=False, ridge=1e-8, **kw) -> np.ndarray   # empirical Fisher
    fisher_vectors(stats=None, metric="diagonal"|"identity"|"full", center=None, fisher=None, ridge=1e-8) -> np.ndarray
    observed_fisher_information(...) / observed_fisher_vectors(...)      # latent (observed-data) metric
    fisher_vector(x, estimate=None, metric="diagonal", ...) -> np.ndarray
    natural_parameters() -> Any                                          # NotImplementedError in generic view
    # FixedFisherView adds canonical coords: mean_statistics(...), _model_mean()/_model_fisher() abstract
- **Helper:** `SufficientStatisticVectorizer` (`fit/partial_fit/transform/fit_transform/label_strings`)
  flattens nested suff-stat structures into numeric vectors.
- **Implemented by (per-family views):** `CountFisherView`, `CompositeFisherView`, `MixtureFisherView`,
  `SequenceFisherView`, `MultinomialFisherView`, `OptionalFisherView`, `WeightedFisherView`,
  `SelectFisherView`, `HeterogeneousPCFGFisherView`, `HiddenMarkovFisherView`,
  `JointMixtureFisherView`, `EmpiricalMetricFixedFisherView`.
- **Facets:** exp-family / sufficient-statistic, latent-structured (observed Fisher), engine-agnostic
  (numpy). Used by `hvis` Fisher-vector affinity.
- **Notes:** **recommend formalizing `to_fisher` as a Protocol method** on the distribution ABC (it is
  currently a duck-typed hook).

### Density-rank / cumulative-probability functions  — `density_rank` & count-DP family
- **Role:** Rank an observation in descending-probability order and bound the cumulative mass; exact
  enumeration + Monte-Carlo fallback, or quantization-approximate structural count-DP.
- **Formalized in:** `pysp/utils/density_rank.py` (functions + result dataclasses, not a class cast).
- **Methods:**
    density_rank(dist, value, max_exact=100_000, n_samples=20_000, seed=0, tol=1e-9) -> DensityRankResult
    truncated_sum_bound(dist, k) -> TruncatedSumBound
    cumulative_probability(dist, value, oversample=64, bin_width_bits=1.0, smear=None) -> float
    count_dp_rank(dist, value, ...) -> CountDPRankResult
    count_dp_seek(dist, index, ...) -> CountDPSeekResult            # inverse of rank
    count_dp_top_p(dist, p, ...) -> CountDPTopPResult               # nucleus size bracket
    mixture_cross_rank(mixture, value, ...) -> int                 # true-marginal rank via K-dim count hist
- **Facets:** enumerable, rank-from-index. Relies on the distribution exposing
  `enumerator()`/`log_density()`/`sampler()`/`support_size()`/`quantized_count_index()`.

---

## C. Enumeration / quantization

### `supports_enumeration` + best-first enumeration core
- **Role:** Lazy, probability-ordered enumeration of structured supports; combinator merge/product of
  sorted child streams; bounded indexable views.
- **Formalized in:** `pysp/utils/enumeration.py`.
- **Capability probe:** `supports_enumeration(dist) -> bool` (can `dist.enumerator()` be built without
  `EnumerationError`).
- **Stream / index classes:**
    BufferedStream(...).get(i) -> (value, log_prob)|None            # random access by rank into lazy sorted stream
    ProductEnumerator(streams, combine=tuple, offset=0.0)          # best-first Cartesian product (min-coord, dup-free)
    LengthFrontierMerge(len_stream, make_stream)                   # per-length lazy frontier (sequences)
    merge_enumerators(streams, offsets) / best_first_union(streams, log_offsets, exact_log_density, tol)
    best_first_union_max(...)                                      # max-bound variant (HMM decoders)
    sound_top_k(dist, k, start=0, budget_bits=40.0, ...) -> list[(value, log_prob)]  # exact top-k, mass certificate
- **Quantized index (rank-from-index facet):**
    QuantizedEnumerationIndex: from_enumerator(enum, max_bits, bin_width_bits=1.0) / from_items(...)
        bin_for_index(index) -> (bin_id, offset) ; get(index) -> (value, log_prob) ; slice/iter_from/bin_items/summary
        @staticmethod bin_for_log_prob(log_prob, bin_width_bits=1.0) -> int
    LazyQuantizedEnumerationIndex(counts, bin_width_bits, max_bits, truncated, getter):  # counts precomputed,
        get(index)/bin_for_index(index) unrank lazily via getter(bin_id, offset)         # items lazily unranked
    QuantizedCrossIndex(items, max_bits, bin_width_bits, truncated)  # aligned support rows across components (mixtures)
    bounded_best_first_union_index(streams, log_offsets, exact_log_density, max_bits, ...) -> QuantizedEnumerationIndex
    quantized_index(enum, max_bits, bin_width_bits=1.0)            # convenience wrapper
    freeze(x) -> Hashable                                          # canonical dedup key
- **Facets:** enumerable, rank-from-index. Realized by leaf/combinator/latent distributions via their
  `enumerator()` and (when quantized) by the count-budget index below.

### `quantized_best_first_decode` + generic best-first  — arbitrary-model decoding
- **Role:** Best-first / A* enumeration of autoregressive (transformer/HMM) sequences in descending
  total log-prob; exact, beam, and nucleus-pruned variants.
- **Formalized in:** `pysp/utils/model_enumeration.py` (pure callable-driven; no pysp dependency).
- **Methods:**
    best_first(start, successors, is_goal, score, heuristic=None, max_results=None) -> Iterator[(state, score)]
    best_first_decode(next_logprobs, eos=None, max_len=None, start=(), heuristic=None, max_results=None)
        -> Iterator[(seq, total_logprob)]                         # exact (each step logprob <= 0)
    beam_search(next_logprobs, beam_width, eos=None, max_len=None, ...) -> list[(seq, logprob)]   # approximate
    top_k_scored(candidates, score, k=None) -> list[(cand, score)]
    quantized_best_first_decode(next_logprobs=None, eos=None, max_len=None, top_k=None, top_p=None,
        bucket_bits=12, batch_next_logprobs=None, batch_size=64, start=(), max_results=None,
        min_mass=None) -> Iterator[(seq, total_logprob)]          # nucleus prune + bucketed PQ + batched scoring
- **Facets:** enumerable, rank-from-index (bucketed).

### Count semiring + count-budget seek index  — quantization subpackage
- **Role:** Structural count-DP over a 2^budget bit window: a witness-retaining count semiring whose
  product is histogram convolution, with seek-by-index unranking for decomposable families
  (Sequence / Composite / MarkovChain).
- **Formalized in:** `pysp/utils/quantization/{core,semiring,parallel}.py`.
- **Semiring contract (ABC):** `DecomposableSemiring` in `quantization/semiring.py`:
    zero() -> E ; one() -> E ; leaf(value, log_prob, quantizer) -> E
    plus(a, b) -> E                                               # pool mutually-exclusive alternatives
    times(a, b, quantizer, max_fine_bucket) -> E                 # compose independent factors (convolution)
    product(elements, quantizer, max_fine_bucket) -> E
  Concrete `CountSemiring` adds `from_enumerator`, `scale`, `map_values`, `power_prefix`; bridges
  `enumerate_and_bin` (stream→index), `ordered_stream_from_count_index` (index→stream),
  `bounded_dedup_stream`.
- **Core value/index types:** `Quantizer(bin_width_bits=1.0, oversample=8, executor=None)` with
  `bits/fine_bucket/coarse_bin/convolve`; `CountHistogram` (semiring value: `shift/add/convolve/
  truncate/count_at/total/max_bucket`); `CountIndex(hist, getter)` with `get_in_bucket(fine_bucket,
  offset) -> (value, log_prob)`.
- **Driver:** `count_budget_index(dist, budget_bits, bin_width_bits=1.0, oversample=8,
  max_depth_bits=4096.0, num_workers=None) -> LazyQuantizedEnumerationIndex`;
  `distinct_budget_stream(dist, budget_bits, ..., dedup="canonical") -> Iterator[(value, log_prob)]`;
  `leaf_count_index(...)`, `convolve_indices(children, quantizer, max_fine_bucket)`,
  `build_budget_index(...)`.
- **Parallel:** `ConvolutionExecutor(num_workers=None, min_parallel_width=2048)` (context-managed
  process pool for big-int histogram convolutions); `resolve_workers(...)`;
  `distributed_unrank(dist, budget_bits, start=0, count=None, ..., backend="local", spark_context=None)`.
- **Facets:** enumerable, rank-from-index, distributed (the real win is distributed unranking).
- **Notes:** approximation semantics (bit-budget truncation); HMM/Mixture deferred (see count-budget
  quantization memory).

---

## D. Priors

### Prior-spec interface  — `as_prior_dict` + frozen dataclasses
- **Role:** Lightweight, serializable, dependency-free descriptions of MAP priors; fitting code
  lowers them to backend tensors at objective time.
- **Formalized in:** `pysp/utils/priors.py`.
- **Contract:** every prior exposes `as_dict() -> dict` with a `"family"` tag;
  `as_prior_dict(prior) -> Any` normalizes a spec / Mapping / nested structure to plain Python.
- **Leaf priors:** `NormalGammaPrior(mu0, kappa, alpha, beta)`, `DirichletPrior(alpha)`,
  `BetaPrior(alpha, beta, parameter=None)`, `GammaPrior(shape, rate, parameter=None)`.
- **Structural priors (mirror the combinator tree):** `CompositePrior(children)`,
  `ConditionalPrior(conditions, default, given)`, `MixturePrior(components, weights)`,
  `MarkovChainPrior(initial, transitions, length)`, `OptionalPrior(observed, missing)`,
  `RecordPrior(fields)`. Factory functions: `normal_gamma/dirichlet/beta/gamma/composite/
  conditional/mixture/markov_chain/optional/record`.
- **Consumed by:** `fit_map` / `_gradient_log_prior_state` in `fit.py` (gamma/beta/dirichlet families
  matched per declaration constraint), and the `objective="map"` path in `fit`.

---

## E. Serialization / streaming / parallel / automatic

### Serialization registry  — closed tagged-JSON codec
- **Role:** Safe JSON (de)serialization of pysp models/estimators via a *closed* class registry (no
  arbitrary code execution).
- **Formalized in:** `pysp/utils/serialization.py`.
- **Methods:**
    register_serializable_class(cls, type_id=None) -> cls ; register_serializable_callable(fn, callable_id=None)
    ensure_pysp_serialization_registry() ; serializable_class_ids() -> set[str]
    to_serializable(value) -> JSON-tagged ; from_serializable(payload) -> value
    to_json(value, **kw) -> str ; from_json(text) -> value
    # SerializationError(ValueError)
- **Implicit contract:** a serializable class round-trips through a tagged dict of its state (registered).

### Streaming estimators  — decay/incremental fold
- **Role:** Online and Neal-Hinton incremental EM over batches/chunks of sufficient statistics.
- **Formalized in:** `pysp/utils/streaming.py`.
- **Methods:**
    StreamingEstimator(estimator, schedule=None, model=None, init_estimator=None, init_p=0.1,
        rng=None, encoder=None, num_chunks=1).update(data=None, enc_data=None) -> model ; .value() ; .reset()
    IncrementalEstimator(estimator, ...).update(chunk_id, data=None, enc_data=None) -> model
        ; .chunk_value(chunk_id) ; .value() ; .reset()
    streaming_accumulate(enc_data, estimator, model) -> (count, accumulator)   # one globally-tied batch fold
- **Facets:** streaming/foldable.

### Encoded-data fold backends  — `EncodedDataHandle` de-facto protocol (recommend formalizing)
- **Role:** Distributed sharded encoded data that exposes a uniform map-reduce fold so the EM driver
  is backend-agnostic.
- **Formalized in:** `pysp/utils/parallel/{multiprocessing,mpi,ray_data,torchrun,lightning_data}.py`
  (all subclass an `EncodedDataHandle` base; the four fold methods are the contract).
- **Fold contract (every backend implements):**
    pysp_seq_log_density_sum(estimate) -> (count: float, log_density_sum: float)
    pysp_seq_estimate(estimator, prev_estimate) -> model              # one distributed EM step
    pysp_seq_initialize(estimator, rng, p) -> model                  # distributed randomized init
    pysp_stream_accumulate(estimator, model) -> (count, accumulator) # globally-folded batch suff stats
    __len__() -> int ; close()
- **Implemented by:** `MPEncodedData` (local process pool, sub_chunks), `MPIEncodedData` (SPMD
  allreduce, root fold), `RayEncodedData` (object-store partitions), `TorchRunEncodedData`
  (torch.distributed collectives), `LightningEncodedData` (mini-batch DataModule; full-EM delegates
  to LocalEncodedData, adds `minibatches()` / `stochastic_em(...)`). Helpers: `mpi_out`,
  `torchrun_out`. **Recommend formalizing `EncodedDataHandle` as a Protocol/ABC.**
- **Facets:** distributed, streaming/foldable, engine-resident (per-shard engines).

### Automatic model selection / profiling
- **Role:** Build estimators from auto-typed data; profile structure (BIC-scored marginals + pairwise
  dependency hints) and recommend a composite estimator.
- **Formalized in:** `pysp/utils/automatic/{factories,profiling}.py`.
- **Methods:**
    factories: get_<family>_estimator(vdict, pseudo_count=None, emp_suff_stat=True, use_bstats=False)
        for categorical/integer_categorical/poisson/gaussian/lognormal/gamma/student_t/
        gaussian_mixture/multivariate_gaussian; plus get_optional/length/sequence/set/ignored/
        composite/dict_record estimators; get_dpm_mixture(data, ...) -> fitted DPM
    profiling: analyze_structure(data, pairwise=True, ...) -> StructureProfile
        StructureProfile.recommend() -> ParameterEstimator ; .summary() ; .explain()
        get_estimator(data, pseudo_count=1.0, ...) -> composite estimator   # via DatumNode profiler
        format_path(path) -> str
        # dataclasses: MarginalFieldProfile (.model_weights()/.summary()), PairwiseDependencyHint, StructureProfile
- **Facets:** produces `ParameterEstimator` casts; conjugate-prior attachment for Bayesian EM
  (`use_bstats`/`pseudo_count`).

---

## F. MCMC

### `Proposal`  — Metropolis-Hastings kernel base (de-facto/base class)
- **Role:** MH proposal kernel: draw a candidate, give its Hastings-correction log density, optionally
  adapt.
- **Formalized in:** `pysp/utils/mcmc/proposals.py` (base class `Proposal`).
- **Methods:**
    sample(self, current, rng: RandomState) -> proposed
    log_density(self, proposed, current) -> float        # log q(proposed|current); default 0.0 (symmetric)
    adapt(self, current, proposed, accepted, step, in_burn_in) -> None   # optional adaptation hook
- **Implemented by:** `RandomWalkProposal`, `AdaptiveRandomWalkProposal` (Robbins-Monro scale),
  `AdaptiveCovarianceProposal` (Welford covariance), `IndependentProposal`, `MixtureProposal`,
  `BlockProposal` (dict-field block), `LangevinProposal` (MALA, needs `grad_log_target`).

### MCMC samplers + diagnostics  — `LogTarget` driven
- **Role:** Generic MCMC drivers over an unnormalized `LogTarget = Callable[[Any], float]` and a
  `Proposal`; plus gradient samplers and multi-chain diagnostics.
- **Formalized in:** `pysp/utils/mcmc/samplers.py`.
- **Methods:**
    distribution_log_target(dist, evidence=None) -> LogTarget       # dist.log_density(x) + evidence(x)
    metropolis_hastings(log_target, initial, proposal: Proposal, num_samples, burn_in=0, thin=1, rng=None) -> MCMCResult
    metropolis_within_gibbs(log_target, initial, proposals, ...) -> MCMCResult     # labelled kernel cycle
    affine_invariant_ensemble(log_target, p0, num_samples, ..., a=2.0) -> MCMCResult
    hamiltonian_monte_carlo(log_target, grad_log_target, initial, num_samples, step_size, num_steps, mass=1.0, ...) -> MCMCResult
    nuts(log_target=None, grad_log_target=None, initial=None, num_samples=0, warmup=1000, ...,
        value_and_grad=None, adapt_mass=False) -> MCMCResult
    sample_distribution(dist, initial, proposal, num_samples, ..., evidence=None) -> MCMCResult
    posterior_predictive(samples, sampler, rng=None, size=None) -> list
    gelman_rubin(chains) -> float|np.ndarray
    run_chains(sampler, num_chains, initials, rng=None, **sampler_kwargs) -> (list[MCMCResult], rhat)
    # MCMCResult(samples, log_probs, accepted, transition_labels): .acceptance_rate,
    #   .acceptance_rate_by_label, .sample_array(), .effective_sample_size(max_lag=None), .summary()
- **Facets:** MCMC-sampleable; latent/gradient targets.

### Conjugate posterior / parameter bridge / gradient bridges
- **Role:** Closed-form conjugate sampling, parameter reparameterization, and autodiff gradient bridges.
- **Formalized in:** `pysp/utils/mcmc/{conjugate,parameter_bridge,gradients,nuts_numba,nuts_torch}.py`.
- **Methods:**
    conjugate: sample_conjugate_posterior(dist, data, draws=1000, seed=None, return_distributions=False) -> MCMCResult
        # exact iid posterior draws for Gaussian/Poisson/Exponential/Bernoulli/Binomial/Geometric
    parameter_bridge: ParameterBridge(dim, to_unconstrained, from_unconstrained, log_abs_det_jacobian,
        build, param_names, initial_theta)   # frozen dataclass: theta<->phi reparameterization
        build_parameter_bridge(prototype) -> ParameterBridge
        sample_parameter_posterior(prototype_dist, data, prior=None, sampler="mh"|"hmc"|"nuts", steps=2000, ...) -> MCMCResult
    gradients: torch_available() -> bool ; torch_gradient(log_target_torch, dtype="float64") -> grad_fn ;
        value_and_torch_gradient(...) -> (value, grad) fn
    nuts_numba(value_and_grad, initial, num_samples=1000, warmup=1000, ...) -> MCMCResult   # njit analytic gradient
    nuts_torch(logp, initial, ..., compile=True, dtype=None, device=None) -> MCMCResult     # on-device autograd
- **Facets:** conjugate, MCMC-sampleable, engine-resident (numba/torch).

---

## G. Arithmetic engine

### `pysp.arithmetic`  — engine-dispatched op surface + active-engine seam
- **Role:** Backend-dispatched array ops and engine-provided scalar constants. Ops dispatch on their
  arguments' engine (`engine_of`); constants resolve from the *active* engine.
- **Formalized in:** `pysp/arithmetic.py` (thin dispatch over `pysp.engines.ComputeEngine`).
- **Active-engine seam:**
    get_default_engine() -> ComputeEngine
    set_default_engine(engine: ComputeEngine|"numpy"|"symbolic") -> ComputeEngine   # returns previous
    using_engine(engine) -> contextmanager
    constant(value) -> active engine's scalar repr
- **Dispatched ops (engine method per arg):** `asarray, zeros, empty, arange, to_numpy, log, exp,
  sqrt, abs, where, maximum, clip, floor, isnan, isinf, dot, matmul, cumsum, logsumexp, stack,
  bincount, index_add, unique, searchsorted, gammaln, digamma, betaln, erf` (+ `sum`; `max` is
  scalar-aware: builtin `max` for numeric varargs, else engine reduction).
- **Engine-provided constants (PEP-562 `__getattr__`):** `pi, e, euler_gamma, one, zero, two, half,
  inf` (resolve from active engine; symbolic engine keeps them exact). Engine-independent limits:
  `maxint, maxrandint, eps`.
- **Notes:** `sum`/`max` are importable by name but deliberately **kept out of `__all__`** so
  `from pysp.arithmetic import *` doesn't shadow the builtins inside numba `nopython` kernels (see
  arithmetic-engine-constants + computation-audit memories).

---

## H. hvis (visualization)  — brief

### Model-based embedding interface
- **Role:** Hierarchical, model-based t-SNE / UMAP: per-field posterior affinities → sparse/dense
  probability matrix → embedding.
- **Formalized in:** `pysp/utils/hvis/{affinity,neighbors,tsne,embed}.py`.
- **Top-level entry points (`embed.py`):**
    htsne(data, emb_dim=2, alpha=1.0, max_components=50, ..., affinity="auto", evidence_cap=1.0,
        fisher_metric="diagonal", fisher_information="observed", method="auto") -> np.ndarray
    humap(data, emb_dim=2, n_neighbors=15, min_dist=0.1, ..., affinity="auto") -> np.ndarray
    dpmsne(P=None, emb_dim=2, ...) -> np.ndarray            # embed a precomputed affinity matrix
- **Affinity facets (`affinity.py`):** `model_log_affinity`, `get_pmat`, `balanced_factors`,
  `local_factors`, `fisher_factors` (uses `FisherView`), `conditional_pmat`.
- **Neighbors / kernels:** `neighbors.py` (`sparse_model_distances`, `approx_sparse_model_distances`
  via RP-trees, `model_knn`); `tsne.py` (`t_kernel`, `update_embed`, `update_alpha`, `tsne_exact`,
  `tsne_barnes_hut`).
- **Facets:** consumes the score / Fisher facets; not a distribution cast itself.

---

## Coverage checklist (every module in scope)

`pysp/arithmetic.py` — engine-dispatched op surface + active-engine seam (`using_engine`/`set_default_engine`/constants).

`pysp/utils/__init__.py` — package namespace (`__all__` of subpackages/modules).
`pysp/utils/aliasing.py` — backward-compat kwarg aliasing (`coalesce_alias`, `require`, `MISSING` sentinel); supporting helper, no interface cast.
`pysp/utils/assignment.py` — k-best assignment enumeration (`best_assignment`, `k_best_assignments` — Murty); combinatorial helper used by matching distributions.
`pysp/utils/builder.py` — indexed-CSV / Spark RDD builders (`read_index_csv`, `get_indexed_rdd_pne`); external/app helper (kept per "unreferenced is not dead").
`pysp/utils/density_rank.py` — **Fisher/geometry §B**: density-rank + cumulative-probability + count-DP rank/seek/top-p (enumerable, rank-from-index facets).
`pysp/utils/em.py` — **fit driver §A**: `EMStep` strategy protocol (`step -> EMStepResult`), 13 strategies, `run_em`, `observed_log_likelihood`.
`pysp/utils/enumeration.py` — **enumeration §C**: best-first core, `supports_enumeration`, `QuantizedEnumerationIndex`/`LazyQuantizedEnumerationIndex`/`QuantizedCrossIndex` (seek `get(index)`/`bin_for_index`), product/length/union merges, `sound_top_k`.
`pysp/utils/estimation.py` — **fit driver §A**: `optimize`/`fit`/`best_of` EM entry points, schedules, `BayesianStreamingEstimator`, `iterate` (the fit-loop interface).
`pysp/utils/evaluation.py` — **fit driver §A**: `empirical_kl_divergence`, `k_fold_split_index`, `partition_data(_index)` (held-out evaluation helpers).
`pysp/utils/fisher.py` — **Fisher/geometry §B**: `FisherView`/`FixedFisherView` contract, `to_fisher` dispatch, `SufficientStatisticVectorizer`, per-family views.
`pysp/utils/fit.py` — **fit driver §A**: autograd `fit_mle`/`fit_map`, `GradientFitResult`, declaration-backed reparameterization + MAP prior lowering, `gradient_fit_state` hook.
`pysp/utils/metrics.py` — classification/ROC helpers (`classify`, `roc_curve`, `auc`, `roc_auc`, `ranking_depth`); evaluation helper, no interface cast.
`pysp/utils/model_enumeration.py` — **enumeration §C**: `best_first`, `best_first_decode`, `beam_search`, `top_k_scored`, `quantized_best_first_decode` (callable-driven decoding).
`pysp/utils/objectives.py` — **fit driver §A**: `ObjectiveCallable` protocol, `fit_objective`/`optimize_torch_objective`/`fit_parameter_objective`, objective classes, `variational_projection`.
`pysp/utils/optional_deps.py` — optional-dependency shims (`numba`, `HAS_NUMBA`, `pyspark`, `HAS_PYSPARK`, `RDD_TYPES`, `require`); packaging helper.
`pysp/utils/optsutil.py` — generic collection utilities (`map_to_integers`, `reduce_by_key`, `group_by`, `count_by_value`, `flat_map`, …); supporting helper.
`pysp/utils/priors.py` — **priors §D**: `as_prior_dict` + frozen prior dataclasses (leaf + structural) — the MAP prior-spec interface.
`pysp/utils/pvalues.py` — composite-of-binomials log-density histogram (`binomial_rank`); statistical helper.
`pysp/utils/serialization.py` — **serialization §E**: closed-registry tagged-JSON codec (`register_serializable_class`, `to/from_json`, `to/from_serializable`).
`pysp/utils/spanning.py` — k-best spanning-tree enumeration (`k_best_spanning_trees` — Gabow); combinatorial helper used by tree distributions.
`pysp/utils/special.py` — special functions (`log_erfcx`, `stirling2`, `logpdet`, `trigamma`, `digammainv`); numeric helper.
`pysp/utils/streaming.py` — **streaming §E**: `StreamingEstimator`, `IncrementalEstimator`, `streaming_accumulate` (decay/incremental fold).
`pysp/utils/vector.py` — array/linear-algebra helpers (`gammaln`, `sorted_merge`, `make`, `make_pdf`, `mat_inv`, `dot`, `outer`); numeric helper.

### Subpackage: `pysp/utils/automatic/*`
`automatic/__init__.py` — re-exports factories + profiling (flat-API preservation).
`automatic/factories.py` — **automatic §E**: `get_<family>_estimator` builders, `get_dpm_mixture` (auto-typed-data estimator factory).
`automatic/profiling.py` — **automatic §E**: `analyze_structure`/`StructureProfile`/`get_estimator`/`DatumNode` (BIC structure profiler + pairwise hints).

### Subpackage: `pysp/utils/hvis/*`
`hvis/__init__.py` — re-exports the embedding API.
`hvis/affinity.py` — **hvis §H**: model affinity factors (`model_log_affinity`, `get_pmat`, `balanced/local/fisher_factors`, `conditional_pmat`).
`hvis/embed.py` — **hvis §H**: top-level `htsne`/`humap`/`dpmsne` orchestrators.
`hvis/neighbors.py` — **hvis §H**: sparse/approx model-distance graphs, `model_knn`.
`hvis/tsne.py` — **hvis §H**: t-SNE kernels/optimizers (`tsne_exact`, `tsne_barnes_hut`, `t_kernel`, `update_embed`).

### Subpackage: `pysp/utils/mcmc/*`
`mcmc/__init__.py` — re-exports proposals/samplers/bridges/gradients.
`mcmc/proposals.py` — **MCMC §F**: `Proposal` kernel base + 8 proposals (random-walk, adaptive, covariance, independent, mixture, block, Langevin).
`mcmc/samplers.py` — **MCMC §F**: `MCMCResult`, MH/Gibbs/ensemble/HMC/NUTS drivers, `gelman_rubin`, `run_chains`, `distribution_log_target`.
`mcmc/conjugate.py` — **MCMC §F**: `sample_conjugate_posterior` (exact conjugate draws).
`mcmc/parameter_bridge.py` — **MCMC §F**: `ParameterBridge`, `build_parameter_bridge`, `sample_parameter_posterior` (reparameterization + posterior driver).
`mcmc/gradients.py` — **MCMC §F**: torch autodiff bridges (`torch_gradient`, `value_and_torch_gradient`, `torch_available`).
`mcmc/nuts_numba.py` — **MCMC §F**: `nuts_numba` (njit analytic-gradient NUTS).
`mcmc/nuts_torch.py` — **MCMC §F**: `nuts_torch` (on-device compiled-autograd NUTS).

### Subpackage: `pysp/utils/parallel/*`
`parallel/__init__.py` — backend exports.
`parallel/multiprocessing.py` — **parallel fold §E**: `MPEncodedData` (`EncodedDataHandle` fold over a process pool).
`parallel/mpi.py` — **parallel fold §E**: `MPIEncodedData` (SPMD allreduce fold), `mpi_out`.
`parallel/ray_data.py` — **parallel fold §E**: `RayEncodedData` (Ray object-store partition fold).
`parallel/torchrun.py` — **parallel fold §E**: `TorchRunEncodedData` (torch.distributed collective fold), `torchrun_out`.
`parallel/lightning_data.py` — **parallel fold §E**: `LightningEncodedData` (mini-batch DataModule, `stochastic_em`; full-EM delegates to LocalEncodedData).

### Subpackage: `pysp/utils/quantization/*`
`quantization/__init__.py` — exports `Quantizer`/`CountHistogram`/`CountIndex`/`DecomposableSemiring`/`CountSemiring`/`ConvolutionExecutor` + drivers.
`quantization/core.py` — **quantization §C**: `Quantizer`, `CountHistogram`, `CountIndex`, `count_budget_index`, `distinct_budget_stream`, `build_budget_index` (count-budget seek index).
`quantization/semiring.py` — **quantization §C**: `DecomposableSemiring` ABC, `CountSemiring`, Axis-A/B stream↔index bridges.
`quantization/parallel.py` — **quantization §C**: `ConvolutionExecutor`, `resolve_workers`, `distributed_unrank` (parallel/distributed unranking).

---

## Interfaces recommended for formalization (Protocol/ABC)
- **`EncodedDataHandle`** (parallel fold) — the four `pysp_seq_*` fold methods are a real contract;
  formalize as a Protocol/ABC.
- **`EMStrategy`** (`em.py`) — `step(enc, estimator, model, engine, objective) -> EMStepResult` is
  duck-typed across 13 classes.
- **`to_fisher` hook** — a Protocol method on the distribution ABC (currently a duck-typed dispatch).
- **`gradient_fit_state` hook** (`fit.py`) — the structured-family reparameterization hook
  (`shadow/score/build/log_prior`) is implicit; worth a Protocol.
- **`ObjectiveCallable`** is already a typed alias; `DecomposableSemiring` and `Proposal` are already
  formal (ABC / base class).
