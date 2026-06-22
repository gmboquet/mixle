# pysparkplug — Interface Design Outline

A complete collection of the type-interfaces in `pysp`, covering every module. The library has
~1700 concrete classes but only a handful of formal ABCs — most contracts are **de-facto** (followed
by convention). This document names them, gives their method surface, and maps every package onto
them. Exhaustive per-module catalogs are in [`docs/interfaces/sections/`](interfaces/sections/)
(one per package group); this file is the unifying design.

> Branch note: the catalog reflects the `main` line. A few items below are marked
> *(formalized on `consistency-fixes`)* — the engine op-surface contract, the combinator single-child
> base, the ppl `PosteriorResult` protocol, and the `self.keys` merge-key normalization were built
> there and are pending merge.

---

## 1. The design model — one contract, orthogonal facets, composition depth

Every object in `pysp` is placed by **three orthogonal axes**:

1. **Contract** — what it *is*: the fixed cast of collaborators (distribution · encoder · accumulator
   · factory · estimator · sampler · enumerator).
2. **Capability facet** — what it *can do*: score · sample · estimate · enumerate · rank-from-index ·
   engine-reside · exponential-family · conjugate · condition · latent. Facets are **orthogonal** — a
   categorical HMM holds several at once.
3. **Composition depth** — how it is *built*: leaf → combinator → latent-state (HMM/Markov/PCFG) →
   Bayesian/nonparametric.

Interfaces are therefore of three shapes: **contract ABCs** (§2), **capability protocols** (§3, mostly
implicit today), and **role interfaces** per layer (§4–§8).

---

## 2. The core contract (the cast)

Defined in [`pysp/stats/compute/pdist.py`](../pysp/stats/compute/pdist.py). These are `@abstractmethod`
stubs on plain classes (not `abc.ABC`) — **recommended to promote to `Protocol`/`ABC`** so the contract
is enforced rather than conventional.

### `ProbabilityDistribution` — `pdist.py:64`
The root. What every distribution must provide, plus optional capability queries.
```
log_density(self, x) -> float                       # required; the one true scorer
density(self, x) -> float                            # default exp(log_density)
sampler(self, seed=None) -> DistributionSampler      # required
estimator(self, pseudo_count=None) -> ParameterEstimator   # required; pseudo_count = conjugate/MAP knob
# --- optional capability surface (raise/Null by default) ---
enumerator(self) -> DistributionEnumerator
support_size(self) -> int | None                     # finite support => an int
density_enumeration(self, ...) / density_quantile(self, q, ...)
count_budget_index(self, budget_bits) -> LazyQuantizedEnumerationIndex   # rank/unrank by index
count_budget_distinct(self, ...) ; is_canonical_copy(self) ; structural_fine_bucket(self)
```

### `SequenceEncodableProbabilityDistribution(ProbabilityDistribution)` — `pdist.py:476`
Adds vectorized, engine-resident scoring.
```
seq_log_density(self, x) -> np.ndarray               # required; vectorized over an encoded batch
dist_to_encoder(self) -> DataSequenceEncoder         # required
supports_engine(self, engine) -> bool ; supported_engines(self) -> tuple[str,...]
seq_log_density_lambda(self) -> list[Callable]       # composition hook used by Sequence
```

### `DataSequenceEncoder` — `pdist.py:820`
Turns raw observations into the columnar payload `seq_log_density` consumes.
```
seq_encode(self, x) -> Any                           # required
__eq__(self, other) -> bool                          # required; equality drives batch interchange
```

### `StatisticAccumulator[SS]` → `SequenceEncodableStatisticAccumulator[SS]` — `pdist.py:651, 692`
Weighted sufficient-statistic accumulation; the E-step sink.
```
update(self, x, weight, estimate) -> None            # one observation (estimate = prev model for E-step)
initialize(self, x, weight, rng) -> None
seq_update(self, enc, weights, estimate) -> None     # vectorized
seq_initialize(self, enc, weights, rng) -> None
combine(self, suff_stat: SS) -> Self                 # merge partitions
value(self) -> SS ; from_value(self, x: SS) -> Self  # the suff-stat round-trip
scale(self, c) -> Self                               # weight rescale (override if value() has non-linear metadata)
key_merge(self, stats_dict) / key_replace(self, stats_dict)   # parameter-tying via self.keys
acc_to_encoder(self) -> DataSequenceEncoder
# de-facto engine-resident hook (getattr-checked, ~58 call sites):
seq_update_engine(self, enc, weights, estimate, engine) -> None
```

### `StatisticAccumulatorFactory` — `pdist.py:736`
`make(self) -> SequenceEncodableStatisticAccumulator` — one fresh accumulator per worker.

### `ParameterEstimator[SS]` — `pdist.py:743`
```
accumulator_factory(self) -> StatisticAccumulatorFactory
estimate(self, nobs, suff_stat: SS) -> SequenceEncodableProbabilityDistribution   # the M-step
resident_accumulation_supported(self) -> bool        # may the engine-resident E-step be used?
```

### `DistributionSampler` / `ConditionalSampler` — `pdist.py:530, 644`
`sample(self, size=None, *, batched=True) -> Any` · conditional adds `sample_given(self, given) -> Any`.

### `DistributionEnumerator` — `pdist.py:568`
`__next__(self) -> tuple[value, log_prob]` — lazy iteration in **descending probability** (k-best).

---

## 3. Capability facets (orthogonal protocols)

These are the "kinds" from the type hierarchy. Most are **implicit** today; each is a strong
candidate to formalize as a `runtime_checkable Protocol` so the fit/enumeration/inference layers can
dispatch on capability instead of `getattr`/`isinstance`.

| Facet | Surface | Held by | Status |
|---|---|---|---|
| **EngineResident** | `seq_log_density`, `supports_engine`, `backend_*` hooks, `seq_update_engine` | numpy/torch/symbolic-ready families | implicit; promote `seq_update_engine` into the accumulator ABC |
| **ExponentialFamily** | `compute_declaration() -> DistributionDeclaration` with an `ExponentialFamilySpec` | 29 leaves + most multivariate | declared via spec; see §4 |
| **Enumerable** | `enumerator()`, `supports_enumeration()`, descending-prob `(value, logp)` | finite + countable discrete, rankings, trees, graphs (~48 modules) | ABC exists (`DistributionEnumerator`); promote to documented Protocol |
| **Rankable-by-index** | `count_budget_index(budget_bits)` → `get(index)` / `bin_for_index(index)` | finite-support discrete + structured (via the count-DP semiring) | implicit; the strongest form of enumerable |
| **Conditionable / Marginalizable** | `condition(observed) -> Self`, `marginal(keep) -> Self` | MVN, diagonal_gaussian, multivariate_student_t, Mixture | implicit; exactly the elliptical + mixture families |
| **ConjugateUpdatable** | `set_prior(...)`, `estimator(pseudo_count=...)` → MAP/Bayes | beta/gamma/normal-gamma/dirichlet-conjugate leaves | implicit; per-leaf convention |
| **LatentStructured** | `latent_posterior(x) -> LatentPosterior`, `posterior_predictive(...)` | Mixture, LDA, HMM (3 of ~20 expose it) | implicit; see §5 |
| **TemporalPointProcess** | event-time realization encoding, `intensity(t, history)`, compensator, Ogata sampler | hawkes_process, power_law_hawkes, multivariate_hawkes, inhomogeneous_poisson, birth_death | implicit; inconsistently named — formalize |
| **SetValued / EditDistribution** | per-element membership log-prob, `required` forced members, set→set edit rates | bernoulli_set, integer_bernoulli_set/_edit/_step_edit | implicit |
| **Transform** | `forward`, `inverse`, `log_abs_det_inverse_jacobian`, `invalid_inverse_value` | combinator/transform.py (Identity/Affine/Exp/Log/Logit) | near-formal; promote & reuse |

**Implication edges** (the lattice): finite-support ⟹ Enumerable ⟹ Rankable-by-index · ExponentialFamily
⟹ ConjugateUpdatable + EngineResident-stacked-kernel · a combinator over Enumerable children is
Enumerable; over ExponentialFamily children it is generally *not* ExponentialFamily.

---

## 4. The compute & engine layer

[`pysp/stats/compute/`](../pysp/stats/compute/) + [`pysp/engines/`](../pysp/engines/).

### `ComputeEngine` — `engines/base.py` (the only formal `abc.ABC` in the core)
Backend-neutral array ops behind numpy / torch / symbolic. 6 ops are abstract
(`asarray/zeros/empty/arange/to_numpy/stack`); a much larger op surface
(`log/exp/where/sum/logsumexp/gammaln/digamma/index_add/…`, ~30) is **de-facto, unenforced**.
Dispatch flags: `supports_numba`, `resident_estep`, `supports_autograd`, `accumulator_dtype`.
*(formalized on `consistency-fixes`: a `REQUIRED_OPS` contract enforced at `__init_subclass__` + an
op-parity test across all three engines.)*

### `Kernel` / `KernelFactory` — `compute/kernel.py` (formal ABCs)
`score(enc) · component_scores(enc) · accumulate(enc, weights) · refresh(dist)`. Concrete kernels:
`GenericKernel`, `NumbaKernel`, `GeneratedNumbaKernel` (declaration-generated). The bridge between a
distribution's declaration and the active engine.

### Declaration / metadata (the spec dataclasses) — `compute/declarations.py`, `capabilities.py`
- `DistributionDeclaration` — `name`, `parameters`, `statistics`, `support`, `exponential_family`,
  `children`, `child_roles`.
- `ExponentialFamilySpec` — the callable EF pieces (`sufficient_statistics`, `natural_parameters`,
  `log_partition`, `base_measure_from_params`, `legacy_sufficient_statistics`) + flags `fixed_base`,
  `runtime_scoring`.
- `ParameterSpec(constraint, differentiable)` · `StatisticSpec(kind, additive, scales)` ·
  `DistributionCapabilities(engine_ready, kernel_status, numpy_only_reason)`.
- **Backend-hook convention** (load-bearing, zero formal declaration — recommend `Protocol`s):
  `compute_capabilities`, `compute_declaration`, `backend_seq_log_density`,
  `backend_seq_component_log_density`, `backend_log_density_from_params`, the `exp_family_*` callables.

---

## 5. Composition interfaces

### `Combinator` (de-facto) — [`pysp/stats/combinator/`](../pysp/stats/combinator/)
Every combinator wraps child distribution(s) and composes their contract: child
`log_density`/`seq_encode`/accumulator/sampler/enumerator; `compute_capabilities =
intersect_engine_ready(children)`; priors factor over children; facets are preserved. Sub-roles:
- **structural product** — sequence, composite, record, conditional, select, ignored, null_dist
- **latent weighting** — mixture (the bridge to §5-latent), weighted, optional
- **support surgery / renorm** — truncated, censored, survival, hurdle, zero_inflated
- **change of measure** — transform (Jacobian), exponential_tilt, finite_stochastic_transform

`SingleChildCombinator` / `SingleChildAccumulator` + `MaskedBaseEncoder` *(built on `consistency-fixes`)*
factor the shared single-child delegation the renorm/transform combinators duplicate.

### `LatentPosterior` (ABC) — [`pysp/stats/latent_posterior.py`](../pysp/stats/latent_posterior.py)
The q(z|x) spine: `marginals · sample · mode · entropy`. Realizations:
`CategoricalLatentPosterior` (mixture, exact), `MarkovChainLatentPosterior` (HMM, forward-backward +
FFBS + Viterbi), `MeanFieldLDAPosterior` (LDA, mean-field).

### Latent-state model facets (implicit — recommend formalizing)
- **ResponsibilityModel** (finite/exchangeable latent): `component_log_density`, `posterior`,
  `seq_posterior`, `expected_log_density`, `latent_posterior`, `posterior_predictive`, `conditional`.
  → the 11 mixture/topic modules (mixture, gaussian_mixture, hierarchical/heterogeneous/joint/
  semi_supervised/spatial mixtures, probabilistic_pca, lda, labeled_lda, iPLSI).
- **SequentialLatent** (the hidden finite-state automaton): forward-backward / Baum-Welch,
  `seq_posterior`, `viterbi`/`seq_viterbi`, terminal (absorbing) states, FFBS sampling. → the 8
  HMM/association modules + the shared `_hidden_markov_numba_kernels`.

### Bayesian inference interfaces — [`pysp/stats/bayes/`](../pysp/stats/bayes/)
- `ConjugatePosterior` (+ `ConjugatePosteriorSampler`, `MixtureConjugatePosterior`) — consumes an
  ExponentialFamily likelihood, returns a posterior: `update / mean / sample / log-evidence /
  cross_entropy / entropy`. Factories `conjugate_posterior` (registry over 19 likelihood families),
  `mixture_conjugate_posterior` (Diaconis–Ylvisaker).
- **Prior-over-parameters families** (distributions *and* `VariationalPrior` `cross_entropy`/`entropy`):
  dirichlet, dict_dirichlet, symmetric_dirichlet, normal_gamma, multivariate_normal_gamma, normal_wishart.
- **Bayesian-nonparametric mixtures**: dirichlet_process_mixture, hierarchical_dpm, pitman_yor
  (stick-breaking / CRP / EPPF).

---

## 6. Structured-support families — [`pysp/stats/graph/`](../pysp/stats/graph/), [`sets/`](../pysp/stats/sets/)

All realize the core contract over combinatorial supports; most expose `DistributionEnumerator`
(rankable). Sub-families: finite-state sequence (markov_chain, integer_markov_chain, markov_transform,
sparse_markov_transform) · ranking/permutation (mallows, plackett_luce, spearman_rho, matching) ·
tree/spanning (chow_liu_tree, integer_chow_liu_tree, spanning_tree) · random graphs (erdos_renyi,
stochastic_block, random_dot_product, knowledge_graph) · grammar · sets (bernoulli_set + integer
variants, edit-distance models).

---

## 7. Cross-cutting service interfaces — [`pysp/utils/`](../pysp/utils/)

- **Fitting drivers** — `optimize`/`fit`/`best_of` (`estimation.py`); `EMStrategy.step() -> EMStepResult`
  + `run_em` (`em.py`, 13 strategies); `ObjectiveCallable` + `fit_objective` (`objectives.py`);
  `fit_mle`/`fit_map` autograd + `gradient_fit_state` hook + `GradientFitResult` (`fit.py`). The fit
  loop's contract is the §2 accumulator/estimator surface.
- **Fisher / geometry** — `FisherView`/`FixedFisherView` + `to_fisher` dispatch (`fisher.py`).
- **Enumeration / quantization** — best-first core + `QuantizedEnumerationIndex` /
  `LazyQuantizedEnumerationIndex` (`get(index)`/`bin_for_index`) (`enumeration.py`);
  `DecomposableSemiring` ABC + `CountSemiring` + `count_budget_index` (`quantization/*`);
  `quantized_best_first_decode` (`model_enumeration.py`).
- **Parallel** — `EncodedDataHandle` (4 `pysp_seq_*` fold methods) over MP/MPI/Ray/torchrun/Lightning
  (`parallel/*`) — the cleanest implicit contract in the repo; **formalize as a Protocol**.
- **Streaming / serialization / priors / MCMC** — `StreamingEstimator`/`IncrementalEstimator`
  (`streaming.py`); closed-registry JSON codec (`serialization.py`); `as_prior_dict` + frozen prior
  dataclasses (`priors.py`); `Proposal` kernel ABC + `LogTarget`-driven samplers + `MCMCResult`
  (`mcmc/*`).
- **Arithmetic engine** — engine-dispatched ops + `using_engine`/`set_default_engine` (`arithmetic.py`).

---

## 8. Higher-layer interfaces

### PPL — [`pysp/ppl/`](../pysp/ppl/)
- `RandomVariable` (`core.py:798`) — the central object; a 10-tag `_kind` algebra
  (sample/bound/param/apply/sum/prod/pow/select/given/joint), `free`/`value`/dist-in-slot
  construction, `lower()`, `sample`, `log_prob`, and `fit(how=...)` dispatch (map/laplace/mcmc/hmc/vi/…).
- `PosteriorResult` — the duck-typed `.result` surface (`summary/samples/mean/predictive/
  pointwise_log_likelihood`). *(formalized as a `Protocol` on `consistency-fixes`.)*
- `Fitter` — the implicit `(rv, data, **kw) -> bound RV` contract; recommend a registry replacing the
  `how==` ladder.
- `Proxy` (`field.py`) — field-inference likelihood model (`params/loglik/residual`).
- `DynamicsOperator` (`dynamics.py`, true ABC) + registry · the PDE forward-operator/adjoint contract
  (`pde_solve.sparse_solve` + assemblers, fronted by `inverse.Differential`).

### Optimization-as-distribution — [`pysp/relations.py`](../pysp/relations.py)
`Relation` (real `abc.ABC`) + `RelationSampler` — `enumerator/solve/top/sampler`. The dual of the
core enumerate/sample/score triple over a constrained combinatorial space.

### Apps — models / doe / uq / infer / data / planner
- `InferenceBackend` (frozen dataclass contract, `infer/backends.py`) + `register_inference_backend`;
  diagnostics (R-hat/ESS).
- doe: acquisition `fn(mean,std,best,*,maximize,**p)` + `register_acquisition`; optimal-design criterion
  `fn(info,*,ref)` + `register_criterion`; GP-surrogate duck contract (`fit`+`predict(...,return_cov)`);
  `OptimizationResult` base *(on `consistency-fixes`)*.
- uq: propagate (MC/unscented), sensitivity (Sobol/Morris), calibration (Kennedy–O'Hagan).
- planner: `EncodedDataHandle` + `register_encoded_data_backend`; data: dataframe/graph-data adapters.
- models: a **heterogeneous** layer — only `RandomForest*` fully honors Distribution+Estimator; GP/
  SparseGP/Neural/KnowledgeGraph are sklearn-style `fit/predict`; a generic `FitResult[ModelT]` unifies
  the result dataclasses *(on `consistency-fixes`)*.

### Registries ("register, don't branch")
encoded-data backends · acquisitions · optimal-design criteria · inference backends · conjugate-posterior
families · dynamics operators. The recommended pattern for the remaining `how==`/`method==` ladders
(ppl fitters, uq propagators).

---

## 9. Module-coverage map

Every package maps to the interfaces above; exhaustive per-module checklists are in the section files.

| Package | Primary interfaces | Section |
|---|---|---|
| `stats/compute`, `engines` | core contract ABCs, `ComputeEngine`, `Kernel`, declaration specs | [01](interfaces/sections/01-core-compute-engine.md) |
| `stats/leaf` (45) | core contract + ExponentialFamily / Enumerable / TemporalPointProcess / Conjugate facets | [02](interfaces/sections/02-leaf.md) |
| `stats/multivariate` (11), `stats/combinator` (18) | Conditionable/Marginalizable; Combinator + Transform | [03](interfaces/sections/03-multivariate-combinator.md) |
| `stats/latent` (24), `stats/sets` (5) | LatentPosterior, ResponsibilityModel, SequentialLatent, SetValued | [04](interfaces/sections/04-latent-sets.md) |
| `stats/graph` (17), `stats/bayes` (11) | Enumerable/Rankable over structured supports; ConjugatePosterior, VariationalPrior, BNP | [05](interfaces/sections/05-graph-bayes.md) |
| `utils` (+5 subpackages), `arithmetic` | EMStrategy, EncodedDataHandle, FisherView, semiring, Proposal, codec | [06](interfaces/sections/06-utils.md) |
| `ppl` (25) | RandomVariable, PosteriorResult, Fitter, Proxy, DynamicsOperator, ForwardOperator | [07](interfaces/sections/07-ppl.md) |
| `models`, `doe`, `uq`, `infer`, `data`, `relations`, `planner`, stats top-level | Relation, InferenceBackend, acquisition/criterion, model wrappers, the registries | [08](interfaces/sections/08-apps-toplevel.md) |

---

## 10. Recommended formalizations (implicit → `Protocol`/`ABC`)

The single highest-leverage design move is to turn the load-bearing **de-facto** contracts into
`runtime_checkable Protocol`s (or ABCs), so dispatch is by declared capability rather than
`getattr`/`isinstance`. In priority order:

1. **The 10 core `pdist.py` contracts** → real `Protocol`/`ABC` (currently unenforced abstractmethod stubs).
2. **`ComputeEngine` full op surface** + a cross-engine parity test *(done on `consistency-fixes`)*.
3. **Backend-hook protocols** — `SupportsBackendScoring`, `SupportsExpFamily`, `SupportsEngineResidentEstep`
   (the `backend_*`/`exp_family_*`/`seq_update_engine` convention).
4. **Capability protocols** (§3) — `Enumerable`, `RankableByIndex`, `Conditionable`/`Marginalizable`,
   `ConjugateUpdatable`, `LatentStructured`, `TemporalPointProcess`, `SetValued`, `Transform`.
5. **`EncodedDataHandle`** (the parallel fold) and **`EMStrategy`** — the two cleanest cross-cutting
   contracts, already obeyed by 5 backends / 13 strategies respectively.
6. **Higher-layer**: `PosteriorResult` *(done)*, a `Fitter` registry, a `ForwardOperator` protocol, a
   shared `FitResult`/`OptimizationResult` *(done)* across the models layer.

These nine families of interfaces — **core contract, engine, declaration, capability facets, combinator,
latent, Bayesian, cross-cutting services, higher-layer roles** — are the complete interface surface of
pysparkplug.
