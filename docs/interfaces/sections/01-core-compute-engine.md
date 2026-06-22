# 01 — Foundational compute + engine interfaces

The contract spine every distribution, estimator, and backend in pysp.stats is built on.
Two layers:

1. **Core contract** (`pysp/stats/compute/pdist.py`) — the de-facto ABCs (`@abstractmethod` on
   plain classes, *not* `abc.ABC`) every distribution/estimator/sampler/encoder realizes.
2. **Compute layer** (`pysp/engines/*`, the rest of `pysp/stats/compute/*`) — the engine op surface,
   evaluation kernels, declarative metadata, and the backend-hook dispatch that lets one set of
   estimators run on numpy / torch / numba / symbolic.

A recurring pattern across this whole area: contracts are **register-don't-branch**. New families
opt in by attaching a class hook (`compute_declaration`, `compute_capabilities`, `backend_*`,
`exp_family_*`) or registering a factory; core dispatch never type-switches on concrete classes.

---

## Part A — Core contract (`pdist.py`)

> All classes below are **de-facto contracts**: they carry `@abstractmethod` markers but do *not*
> inherit `abc.ABC`, so they are not enforced at instantiation — they are followed by convention and
> by the abstractmethod stubs raising at call time. Every one **should be a `typing.Protocol` or a
> real `ABC`**. `SS = TypeVar("SS")` is the sufficient-statistic payload type threaded through the
> accumulator/estimator pair.

### `ProbabilityDistribution`  — [de-facto contract / base class]
- **Role:** evaluate (log-)density of one observation; mint a sampler and an estimator; optionally
  enumerate/rank a discrete support; carry a conjugate prior; serialize.
- **Formalized in:** `pysp/stats/compute/pdist.py:64`.
- **Methods (core):**
    log_density(self, x: Any) -> float                       # pdist.py:120 ABSTRACT — log-mass/density of one obs
    density(self, x: Any) -> float                           # pdist.py:111 default exp(log_density)
    sampler(self, seed: int|None=None) -> DistributionSampler          # pdist.py:125 ABSTRACT
    estimator(self, pseudo_count: float|None=None) -> ParameterEstimator  # pdist.py:130 ABSTRACT
- **Serialization:** `to_dict`/`from_dict` (79/85), `to_json`/`from_json` (95/101) — via
  `pysp.utils.serialization`.
- **Fisher / exp-family views:**
    to_fisher(self, **kwargs)                                # pdist.py:134 accumulator-backed Fisher geometry
    to_exponential_family(self, engine=None) -> ExponentialFamilyForm|None  # pdist.py:146 reads declaration.exponential_family
- **Bayesian / variational facet:**
    get_prior(self) -> ProbabilityDistribution|None          # pdist.py:169 reads .prior (None => MLE)
    set_prior(self, prior) -> None                           # pdist.py:179
    expected_log_density(self, x) -> float                   # pdist.py:188 E_q[log p]; degenerates to log_density
- **Capability-query / enumeration surface (the "facets" asked about):**
    enumerator(self) -> DistributionEnumerator               # pdist.py:198 default raises EnumerationError
    support_size(self) -> int|None                           # pdist.py:206 cardinality (None=infinite/continuous)
    support_is_finite(self) -> bool                          # pdist.py:218
    density_quantile(self, q, n_samples=20000, seed=None) -> Any   # pdist.py:238 value at descending-density index q (MC default)
    density_enumeration(self, num_points, n_samples=20000, seed=None) -> list[(val,logp)]  # pdist.py:265 MC continuous analogue of enumerator
    quantized_index(self, max_bits, bin_width_bits=1.0)      # pdist.py:222 bit-bucketed seek index (wraps enumerator)
    quantized_count_index(self, quantizer, max_fine_bucket)  # pdist.py:304 count-semiring structural index
    count_budget_index(self, budget_bits, bin_width_bits=1.0, oversample=8, num_workers=None)   # pdist.py:327 top-2**budget seek
    count_budget_distinct(self, budget_bits, ..., dedup="canonical", start=0, stop=None, max_entries=1<<16, num_workers=None)  # pdist.py:352 distinct (value,logp) stream
    is_canonical_copy(self, value, coarse_bin, quantizer) -> bool  # pdist.py:402 stateless dedup hook (default True)
    structural_fine_bucket(self, value, quantizer) -> int    # pdist.py:412 bucket the count index actually used
    quantized_cross_index / quantized_multi_cross_index(...) # pdist.py:471/425 aligned cross-bin views vs other dists
- **NOT a method here:** `supports_engine` / `supported_engines` live on the *sequence-encodable*
  subclass (below).
- **Implemented by:** every concrete distribution in `pysp/stats/**` (≈1700 classes) via the
  subclass below.
- **Facets:** Score (`log_density`), Sample (`sampler`), Estimate (`estimator`), Enumerable
  (`enumerator`/`support_size`), Rank-from-index (`quantized_*` / `count_budget_*`), ExponentialFamily
  (`to_exponential_family`), Conjugate/Bayesian (`get_prior`/`expected_log_density`),
  Fisher (`to_fisher`).
- **Notes:** Enumeration + count-budget + cross-index methods are the "Rankable-by-index" capability
  facet; leaves give exact closed forms, combinators compose them structurally, uncountable supports
  fall back to Monte-Carlo (`density_quantile`/`density_enumeration`). `EnumerationError` (pdist.py:20)
  and `child_enumerator` (pdist.py:50) thread a child *path* through nested combinator failures.

### `SequenceEncodableProbabilityDistribution(ProbabilityDistribution)`  — [de-facto contract]
- **Role:** add vectorized log-density over *encoded* iid sequences and engine awareness. This is the
  class virtually all real distributions actually subclass.
- **Formalized in:** `pysp/stats/compute/pdist.py:476`.
- **Class attr:** `engine_ready = ("numpy",)` — default engine allow-list.
- **Methods:**
    dist_to_encoder(self) -> DataSequenceEncoder             # pdist.py:524 ABSTRACT — the matching encoder
    seq_log_density(self, x) -> np.ndarray                   # pdist.py:502 vectorized over encoded batch (default: loop log_density)
    seq_expected_log_density(self, x) -> np.ndarray          # pdist.py:506 vectorized E_q[log p]
    seq_log_density_lambda(self) -> list                     # pdist.py:514
    seq_ld_lambda(self)                                      # pdist.py:498 (legacy hook)
    supported_engines(self) -> tuple[str,...]                # pdist.py:486 -> capabilities_for(self).engine_ready
    supports_engine(self, engine) -> bool                    # pdist.py:492 -> capabilities_for(self).supports_engine
    kernel(self, engine=None, estimator=None) -> Kernel      # pdist.py:518 -> kernel_for(...)
- **Implemented by:** essentially all leaves + combinators in `pysp/stats/**`.
- **Facets:** adds EngineResident (`supports_engine`, `kernel`) on top of `ProbabilityDistribution`.
- **Notes:** `supports_engine`/`supported_engines`/`kernel` are the EngineResident facet entry points;
  they delegate to the capabilities + kernel registries (Part C/D), so a family becomes engine-resident
  by declaring metadata, not by overriding these.

### `DistributionSampler`  — [de-facto contract]
- **Role:** draw iid observations from a fitted distribution with a seeded / shared `RandomState`.
- **Formalized in:** `pysp/stats/compute/pdist.py:530`.
- **Methods:**
    __init__(self, dist, seed=None, *, rng: np.random.RandomState|None=None)   # pdist.py:537 keyword-only rng shares a stream
    sample(self, size: int|None=None, *, batched: bool=True) -> Any            # pdist.py:553 ABSTRACT — one obs or length-n collection
    new_seed(self) -> int                                                       # pdist.py:549 fresh child seed
- **Implemented by:** every distribution's nested `*Sampler` class.
- **Facets:** Sample. `batched=True` vectorizes combinator child streams; `batched=False` is the
  guaranteed-identical per-draw reference.

### `ConditionalSampler`  — [de-facto contract / mixin]
- **Role:** sampler mixin for conditional draws `P(. | x)`.
- **Formalized in:** `pysp/stats/compute/pdist.py:644`.
- **Methods:** `sample_given(self, x)` — pdist.py:647 ABSTRACT.
- **Implemented by:** conditional/regression-style samplers (e.g. conditional mixtures, HMM emissions).
- **Facets:** Conditionable.

### `DistributionEnumerator`  — [de-facto contract / iterator]
- **Role:** lazy iterator over a discrete support in **non-increasing probability order**, yielding
  `(value, log_prob)` exactly once per support point.
- **Formalized in:** `pysp/stats/compute/pdist.py:568`.
- **Methods:**
    __next__(self) -> tuple[Any, float]                      # pdist.py:586 ABSTRACT
    __iter__(self) -> DistributionEnumerator                 # pdist.py:583
    top_k(self, k) -> list[(val,logp)]                       # pdist.py:589
    top_p(self, p, max_items=None) -> list[(val,logp)]       # pdist.py:593 nucleus/minimal-coverage prefix
    quantized_index(self, max_bits, bin_width_bits=1.0)      # pdist.py:624 build bit-bucketed index (consumes enumerator)
- **Contract:** dedup is the enumerator's job; `log_prob == dist.log_density(value)` to ~1e-10;
  log_probs non-increasing; zero-prob values skipped; ties broken by insertion order.
- **Implemented by:** discrete leaves (Categorical, IntegerRange, Poisson, …) and combinator
  enumerators (Composite/Sequence/Mixture/MarkovChain).
- **Facets:** Enumerable, Rankable-by-index.

### `StatisticAccumulator[SS]`  — [de-facto contract, `Generic[SS]`]
- **Role:** accumulate weighted sufficient statistics of type `SS`; merge across partitions/keys.
- **Formalized in:** `pysp/stats/compute/pdist.py:651`.
- **Methods:**
    update(self, x, weight: float, estimate) -> None         # pdist.py:661 add one obs (estimate=prev model for E-step posteriors; None on init)
    initialize(self, x, weight, rng) -> None                 # pdist.py:663 default -> update(x,weight,None)
    combine(self, suff_stat: SS) -> StatisticAccumulator     # pdist.py:666 ABSTRACT merge partition stats
    value(self) -> SS                                        # pdist.py:669 ABSTRACT export payload
    from_value(self, x: SS) -> SeqEncStatisticAccumulator    # pdist.py:672 ABSTRACT load payload
    scale(self, c: float) -> StatisticAccumulator            # pdist.py:675 scale linear stats (structural default; override for support metadata)
    key_merge(self, stats_dict: dict[str,Any]) -> None       # pdist.py:685 ABSTRACT pool keyed/tied sites
    key_replace(self, stats_dict: dict[str,Any]) -> None     # pdist.py:688 ABSTRACT
- **Implemented by:** every distribution's nested accumulator.
- **Facets:** Estimate (E-step half). `key`/`key_merge`/`key_replace` realize the **parameter-tying**
  capability; `KeyValidationError` (pdist.py:39) + `validate_estimator_keys` (pdist.py:1021) guard
  incompatible keyed sites.

### `SequenceEncodableStatisticAccumulator(StatisticAccumulator[SS])`  — [de-facto contract]
- **Role:** add vectorized accumulation over the encoded sequence form.
- **Formalized in:** `pysp/stats/compute/pdist.py:692`.
- **Methods:**
    seq_update(self, x, weights: np.ndarray, estimate) -> None        # pdist.py:703 ABSTRACT vectorized E-step
    seq_initialize(self, x, weights: np.ndarray, rng) -> None         # pdist.py:706 ABSTRACT vectorized init
    acc_to_encoder(self) -> DataSequenceEncoder                       # pdist.py:709 ABSTRACT
    get_seq_lambda(self)                                             # pdist.py:700 (legacy hook)
- **Extra (de-facto, optional) engine hook:** `seq_update_engine(self, enc, weights, estimate, engine)` —
  not declared here but recognized across the compute layer (kernel.py:135, backend.py:26); accumulators
  that implement it accumulate **engine-resident** instead of round-tripping through host numpy. 58 call
  sites. **Should be promoted into this ABC** as the engine-resident E-step.
- **Implemented by:** all real accumulators.
- **Facets:** Estimate + EngineResident.

### `StatisticAccumulatorFactory`  — [de-facto contract]
- **Role:** mint fresh zeroed accumulators (one per estimator).
- **Formalized in:** `pysp/stats/compute/pdist.py:736`.
- **Methods:** `make(self) -> SequenceEncodableStatisticAccumulator` — pdist.py:739 ABSTRACT.

### `ParameterEstimator[SS]`  — [de-facto contract, `Generic[SS]`]
- **Role:** map accumulated sufficient statistics (+ optional prior regularization) to a new
  distribution; the M-step half.
- **Formalized in:** `pysp/stats/compute/pdist.py:743`.
- **Methods:**
    estimate(self, nobs: float|None, suff_stat: SS) -> SeqEncProbabilityDistribution   # pdist.py:783 ABSTRACT M-step
    accumulator_factory(self) -> StatisticAccumulatorFactory                            # pdist.py:786 ABSTRACT
    resident_accumulation_supported(self) -> bool                                       # pdist.py:789 fixed-width stats suffice? (default True)
    get_prior(self) -> ProbabilityDistribution|None                                     # pdist.py:800 reads .prior (None=>MLE)
    model_log_density(self, model) -> float                                             # pdist.py:810 prior log-density (ELBO global term, default 0.0)
- **Serialization:** `to_dict`/`from_dict` (751/757), `to_json`/`from_json` (767/773).
- **Implemented by:** every distribution's nested `*Estimator`.
- **Facets:** Estimate (M-step), Conjugate/Bayesian (`get_prior`/`model_log_density`).
- **Notes:** `resident_accumulation_supported=False` is the opt-out that forces generated/stacked
  kernels back to the host accumulator (e.g. NegativeBinomial dispersion needs the full histogram),
  keeping every backend's fixed point identical.

### `DataSequenceEncoder`  — [de-facto contract]
- **Role:** transform a raw iid observation sequence into the vectorized payload consumed by all
  `seq_*` methods; interchangeable when `__eq__`-equal.
- **Formalized in:** `pysp/stats/compute/pdist.py:820`.
- **Methods:**
    seq_encode(self, x) -> Any                               # pdist.py:832 encode the iid batch
    __eq__(self, other) -> bool                              # pdist.py:840 ABSTRACT (interchangeability)
    nbytes(self, x) -> int                                   # pdist.py:836 encoded byte size (-> encoded_nbytes)
    __str__(self) -> str                                     # pdist.py:829
- **Implemented by:** every distribution's nested `*Encoder`.
- **Facets:** EngineResident (encoded payloads are what kernels score).

**Module-level helpers in pdist.py (not interfaces, but part of the contract):**
`scale_suff_stat` (713) recursive numeric scaling; `encoded_nbytes` (844) payload sizing;
the key registry `validate_estimator_keys` (1021) / `validate_accumulator_keys` (1037) enforcing the
`KeyValidationError` contract over the `_KEY_ATTRS` (`key, keys, weight_key, comp_key, init_key,
trans_key, state_key`).

---

## Part B — Engine op surface (`pysp/engines/`)

### `ComputeEngine`  — [formal `abc.ABC`]
- **Role:** the small array-backend op surface (array library + device + dtype + optional compile)
  that backend-neutral kernels depend on. The one true ABC in this area.
- **Formalized in:** `pysp/engines/base.py:11`.
- **Class-level policy attrs:** `name="base"`, `supports_autograd=False`, `dtype=None`,
  `device="cpu"`; **dispatch capability flags** `supports_numba=False` (host-numpy: numba/pure-numpy
  kernels + numpy `seq_log_density` fallback apply) and `resident_estep=True` (prefer engine-resident
  `seq_update_engine` over host round-trip). Mathematical constants `pi,e,euler_gamma,inf,zero,one,
  two,half` are engine-owned (numeric engines return floats; symbolic returns symbolic nodes) and are
  read by `pysp.arithmetic`.
- **Abstract ops (the REQUIRED_OPS contract — implicit, no literal `REQUIRED_OPS` constant):**
    asarray(self, x, dtype=None) -> Any        # base.py:61 ABSTRACT
    zeros(self, shape, dtype=None) -> Any       # base.py:66 ABSTRACT
    empty(self, shape, dtype=None) -> Any       # base.py:71 ABSTRACT
    arange(self, *args, **kwargs) -> Any        # base.py:76 ABSTRACT
    to_numpy(self, x) -> Any                     # base.py:81 ABSTRACT (host boundary)
    stack(self, arrays, axis=0) -> Any           # base.py:86 ABSTRACT
- **Concrete defaults / extension points:**
    constant(value)                  # base.py:37 scalar repr (identity for numeric engines)
    precision (property)             # base.py:50 -> stable dtype name
    with_precision(precision)        # base.py:57 returns a re-dtyped engine (default raises)
    requires_grad(x) -> bool         # base.py:91 (default False)
    compile(fn) -> fn                # base.py:95 (default identity)
    replicate(x) / place_component_axis(x, axis=0)   # base.py:99/103 (DTensor placement hooks)
- **Implemented by:** `NumpyEngine`, `TorchEngine`, `SymbolicEngine`.
- **Notes:** The **de-facto op surface is much larger than the 6 abstract methods** — concrete engines
  also expose `log, exp, sqrt, abs, where, maximum, clip, floor, isnan, isinf, sum, max, dot, matmul,
  cumsum, logsumexp, bincount, unique, searchsorted, gammaln, digamma, betaln, erf, index_add` (see
  numpy_engine.py:72–114). These are the real "REQUIRED_OPS" that `exp_family_*` / `backend_*` math
  relies on but are **not declared abstract** — *should be formalized* (Protocol or expanded ABC) so a
  new backend knows the full surface.

### `NumpyEngine(ComputeEngine)`  — [concrete]
- **Formalized in:** `pysp/engines/numpy_engine.py:14`. `name="numpy"`, `supports_numba=True`,
  `resident_estep=False`. Ops are NumPy/SciPy staticmethods; `accumulator_dtype` (always float64) and
  a precision-promoting `sum` keep float32 fits from drifting. The default local execution path
  (`NUMPY_ENGINE` singleton).

### `TorchEngine(ComputeEngine)`  — [concrete]
- **Formalized in:** `pysp/engines/torch_engine.py:20`. `name="torch"`, `supports_autograd=True`,
  device/dtype-parameterized, `resident_estep=True` (inherited). Adds `replicate` /
  `place_component_axis` over `torch.distributed.tensor.DTensor` (component-axis / replicated
  placement). Ops map to `torch.*`; `requires_grad` and `compile` (`torch.compile`) are live.

### `SymbolicEngine(ComputeEngine)`  — [concrete] + `SymbolicExpression`
- **Formalized in:** `pysp/engines/symbolic_engine.py:160` (engine) / `:24` (expression node).
  `name="symbolic"`. Builds a `SymbolicExpression` op-tree instead of numbers; constants are exact
  (`half = 1/2`, `pi` symbolic). Adds inspection ops: `evaluate(x, values)`, `symbols(x)`,
  `op_counts(x)`, `diagnostics(x)`, and export `to_sympy/to_sage/to_latex`. `SymbolicExpression`
  overloads every arithmetic/comparison/logical dunder so the *same* `exp_family_*`/`backend_*` math
  traces to a formula. This is what powers `generated_log_density_diagnostics` and the
  symbolic→numba lowering in declarations.py.

### `symbolic_export.py`  — [free functions]
- `to_sympy(expr)` / `to_latex(expr)` / `to_sage(expr)` — lower a `SymbolicExpression` tree to SymPy,
  LaTeX, or Sage (passagemath) via per-op tables. Not interfaces; the export side of the symbolic facet.

### `precision.py`  — [free functions]
- `precision_name`, `normalize_numpy_dtype`, `normalize_torch_dtype`, `engine_with_precision`,
  `auto_precision(data, *, engine, sample_size)`. The dtype-policy helpers behind
  `ComputeEngine.with_precision`/`.precision`. `auto_precision` recommends float32 only for a
  well-conditioned-data GPU torch engine.

### `engines/__init__.py`  — [registry + dispatch]
- Re-exports the engines + `NUMPY_ENGINE`/`SYMBOLIC_ENGINE` singletons. `register_array_type(type,
  engine)` + `engine_of(x, default=NUMPY_ENGINE)` resolve the owning engine of any (possibly nested)
  array/encoded payload (symbolic-object-array → symbolic; torch tensor → device/dtype-matched
  `TorchEngine`; mixing engine classes raises). `to_numpy(x)` is the explicit host boundary.

---

## Part C — Declarative metadata (`declarations.py`)

> Frozen dataclasses describing a family's parameters / sufficient statistics / exp-family form so
> generated kernels (scalar, stacked, numba, symbolic) can be **emitted** rather than hand-written.
> These are data records, not method-bearing interfaces — but they ARE the contract a family fills in
> via its `compute_declaration` hook.

### `ParameterSpec`  — [frozen dataclass] — declarations.py:16
- **Fields:** `name: str`, `constraint: str = "real"`, `differentiable: bool = True`.
- `constraint` is interpreted by generated/scoring utilities; the validated vocabulary is
  `_KNOWN_PARAMETER_CONSTRAINTS` (declarations.py:1092): `real`, `real_vector`, `positive`,
  `positive_vector`, `positive_matrix`, `unit_interval`, `simplex[_vector|_map]`,
  `row_simplex_matrix|map`, `column_simplex_matrix`, `integer[_vector|_matrix]`,
  `positive_integer`, `non_negative_integer`, `optional_integer`, `fixed`, `metadata`,
  `log_probability_tables`, `log_unit_interval_vector`, `optional_log_unit_interval_vector`, plus the
  coupled-ordered form `greater_than:<param>`. `_SHARED_STACKED_PARAMETER_CONSTRAINTS` marks the ones
  that must be identical across stacked components.

### `StatisticSpec`  — [frozen dataclass] — declarations.py:32
- **Fields:** `name: str`, `kind: str = "moment"`, `additive: bool = True`, `scales: bool = True`.
- `kind` drives layout/aggregation: `moment` | `histogram` | `child_stat` | `choice_child_stats` |
  `tuple` | `mapping` (validated in `_statistic_value_issues`, declarations.py:1224).

### `ExponentialFamilySpec`  — [frozen dataclass] — declarations.py:42
- The canonical `p(x)=h(x)·exp(<eta,T(x)> − A(eta))` pieces as callables `(enc/params, engine) -> ...`:
  `sufficient_statistics` (T), `natural_parameters` (eta), `log_partition` (A), optional
  `base_measure` (h), `sufficient_statistics_from_params`, `base_measure_from_params`,
  `legacy_sufficient_statistics`.
- **The two flags asked about:**
  - `fixed_base: bool = True` — base `h(x)` is component-independent, so the **stacked** fixed-base
    loop (one `(n,)` base broadcast over `k` components) is valid. Families whose base varies per
    component (NegativeBinomial's `lgamma(x+r)`) set `False` → keep scalar/exp-family view but route
    stacked scoring through `backend_*`.
  - `runtime_scoring: bool = True` — the `<eta,T(x)>` dot form is numerically safe as a runtime
    scorer. Set `False` when valid as a *map* but unsafe at runtime (Categorical's `eta=log(p)` has
    `-inf` → `0·-inf = NaN`); `to_exponential_family` still exposes the canonical map, scoring stays
    on `backend_*`. Read via `_exp_family_runtime_scoring` / `_exp_family_stacked_scoring`.

### `DistributionDeclaration`  — [frozen dataclass] — declarations.py:74
- **Fields:** `name`, `distribution_type: type`, `parameters: tuple[ParameterSpec,...]`,
  `statistics: tuple[StatisticSpec,...]`, `support: str`, `children`/`child_roles` (nested combinator
  decls), `differentiable`, `exponential_family: ExponentialFamilySpec|None`,
  `legacy_sufficient_statistics`.
- **Methods:** `parameter_values(dist)`, `statistic_values(suff_stat)`, properties
  `parameter_names`, `statistic_names`, `has_exponential_family`.
- `support` is a string tag (e.g. `*_vector` triggers vector-encoded handling in diagnostics/numba
  lowering).

### Declaration registry + generation API (module functions)
- **Registry:** `register_declaration(decl)`, `declaration_for(x)` (checks the `compute_declaration`
  hook first, then class MRO), `declared_distribution_types()`.
- **Validation:** `declaration_issues` / `validate_declaration`, `statistic_layout_issues` /
  `validate_statistic_layout`.
- **Capability queries:** `generated_stacked_available/preferred/strategy`,
  `generated_numba_log_density_available`, `generated_numba_stacked_available`,
  `generated_sufficient_statistics_available`, `generated_stacked_sufficient_statistics_available`.
- **Generation:** `generated_log_density`, `generated_stacked_params/_log_density`,
  `generated_sufficient_statistics`, `generated_stacked_sufficient_statistics`,
  `generated_numba_log_density`, `generated_numba_stacked_log_density`,
  `generated_log_density_diagnostics` (symbolic trace).
- **Symbolic→numba compiler:** `_build_generic_numba_kernel` / `_lower_symbolic_to_numba` lower a
  family's `backend_log_density_from_params` formula to a nopython scalar loop for non-exp-family
  leaves; exp-family leaves use the `_numba_*_exp_family_log_density` kernels.

---

## Part D — Kernels, backend hooks, stacked/fused paths

### `Kernel`  — [formal `abc.ABC`] — `pysp/stats/compute/kernel.py:40`
- **Role:** the engine-aware evaluation object for a fitted distribution (score / accumulate / refresh).
- **Methods:**
    score(self, enc) -> Any                          # kernel.py:44 ABSTRACT per-row log densities
    accumulate(self, enc, weights) -> Any            # kernel.py:53 ABSTRACT legacy-format sufficient stats
    refresh(self, dist) -> None                      # kernel.py:58 ABSTRACT swap params after M-step, keep structure
    component_scores(self, enc) -> Any               # kernel.py:48 default raises (mixture facet)
- **Implemented by:** `GenericKernel` (81, backend-hook/seq_* fallback), `NumbaKernel` (166, fused
  `CompiledMixture`), `GeneratedNumbaKernel` (219, declaration-generated), `StackedMixtureKernel`
  (stacked.py:331).

### `KernelFactory`  — [formal `abc.ABC`] — `kernel.py:63`
- **Role:** build a `Kernel` for `(dist, engine, estimator?)`.
- **Methods:** `build(self, dist, engine: ComputeEngine, estimator=None) -> Kernel` — kernel.py:67 ABSTRACT.
- **Implemented by:** `GenericKernelFactory` (147), `NumbaKernelFactory` (328),
  `GeneratedNumbaKernelFactory` (349, the default-safe one).
- **Registry/dispatch:** `register_kernel_factory(dist_type, factory)` + `kernel_for(dist, engine,
  estimator)` (kernel.py:650) walk the MRO; `_DEFAULT_FACTORY = GeneratedNumbaKernelFactory()`.
  `EngineNotSupportedError` (kernel.py:19) when no kernel can run on the engine.

### Backend-hook naming convention  — [implicit contract — class methods on distributions]
This is the de-facto **EngineResident** contract a family opts into; dispatched generically in
`backend.py` and `declarations.py`, never type-switched. Counts are live usage across `pysp/stats`:
- **`backend_seq_log_density(enc, engine)`** (198 uses) — vectorized engine-neutral score; primary
  hook tried by `backend_seq_log_density(dist, enc, engine)` (backend.py:35) before generated/seq fallback.
- **`backend_seq_component_log_density(enc, engine)`** (17) — per-component scores (mixture facet).
- **`backend_log_density_from_params(data, *params, engine)`** (104) — per-row formula keyed by
  *declared parameters*; drives generated scalar/stacked/numba/symbolic kernels. Signature **must end
  with `engine`**; leading args are encoded data, trailing args are declared params.
- **`exp_family_*`** (the `ExponentialFamilySpec` callable names, attached per family):
  `exp_family_sufficient_statistics` (44), `exp_family_natural_parameters` (45),
  `exp_family_log_partition` (44), `exp_family_base_measure` (18), `exp_family_legacy_sufficient_statistics`
  (20), `exp_family_base_measure_from_params` (12), `exp_family_sufficient_statistics_from_params` (9),
  `exp_family_from_natural` (7), … — the per-family pieces a module wires into its
  `ExponentialFamilySpec`.
- **`compute_declaration()`** (70) — class/instance hook returning a `DistributionDeclaration`
  (read first by `declaration_for`).
- **`compute_capabilities()`** (86) — class/instance hook returning `DistributionCapabilities`
  (read by `capabilities_for`).
- **`seq_update_engine(enc, weights, estimate, engine)`** (58) — the engine-resident E-step on
  accumulators (Part A note); routed by `child_seq_update` (backend.py:18) and the kernels.
- **Helpers:** `BackendScoringError` (backend.py:12), `backend_log_density_sum` (backend.py:60),
  `child_seq_update` (backend.py:18, pushes residency down a model tree).
- **Recommendation:** these hook *names* are a contract with no formal declaration — **should be a
  `Protocol`** (`SupportsBackendScoring`, `SupportsExpFamily`, `SupportsEngineResidentEstep`) so
  families and tooling can type-check participation.

### `DistributionCapabilities`  — [frozen dataclass] — `capabilities.py:9`
- **Fields:** `engine_ready: tuple[str,...] = ("numpy",)`, `kernel_status: str = "generic"`,
  `numpy_only_reason: str|None = None`.
- **Methods:** `supports_engine(engine)`, property `is_permanently_numpy_only`.
- **Registry/dispatch:** `register_capabilities`, `capabilities_for(x)` (class `engine_ready` →
  `compute_capabilities` hook → registry → MRO), `intersect_engine_ready(children, preferred_order)`
  (combinators take the intersection of children), `numpy_only_distribution_types`,
  `supported_engines`. This backs `SequenceEncodableProbabilityDistribution.supports_engine`.

### Stacked / fused / resident plumbing
- **`stacked.py`** — `StackedMixtureKernel` (Kernel for homogeneous mixtures with stacked component
  params) + value objects `StackedComponentParams`, `StackedMixtureResidentStats`,
  `StackedMixtureShardEstimate`, `StackedEstimatorView`, and the functional route
  `stacked_component_params` / `stacked_component_log_density` /
  `stacked_component_sufficient_statistics` / `unstack_component_stats` / `tie_component_shard_values`.
  The distributed M-step shard path for mixtures.
- **`fused_kernels.py`** — `CompiledMixture` + per-leaf numba `_*_ld` / `_*_acc` kernels and `_*B`
  `_LeafBuilder`s (Gaussian, Categorical, IntRange, Poisson, Exponential, Geometric, NegativeBinomial,
  StudentT, Logistic, Weibull, Rayleigh, Pareto, Uniform, Gamma, Binomial, DiagGaussian, …). The
  hand-fused columnar mixture path behind `NumbaKernel`. Not an interface — a kernel implementation.
- **`torch_mixture.py`** — `TorchMixture`, a thin compatibility adapter over `ComputeEngine` kernels
  (`encode`/`seq_log_density`/`seq_component_log_density`/`posteriors`/`weighted_suff_stats`/`em_step`/
  `fit`/`fit_mle`/`fit_map`). Legacy surface, now delegating to the kernel layer.
- **`gradient.py`** — `*GradientFitState` objects (`CategoricalGradientFitState`,
  `OptionalGradientFitState`, …) + prior-decomposition helpers (`mixture_priors`,
  `markov_chain_priors`, `composite_child_priors`, …). The autograd-fitting state the torch path owns;
  `GradientFitError` when a family can't be generically gradient-fit. The LatentStructured / gradient
  facet's plumbing.
- **`encoded.py`** — `EncodedData` (one-chunk payload + planner metadata: count, engine, nbytes;
  `from_data`/`from_payload`/`as_seq_chunk`) and `ResidentEncodedPayload` (host + resident engine
  encoding pair); `as_encoded_data`, `move_encoded_payload(payload, engine)`. The
  `enc.engine_payload` / `enc.host_payload` attributes the kernels unwrap come from here.

---

## Coverage checklist

- `pysp/stats/compute/pdist.py` — the core de-facto ABCs: `ProbabilityDistribution`,
  `SequenceEncodableProbabilityDistribution`, `DistributionSampler`, `ConditionalSampler`,
  `DistributionEnumerator`, `StatisticAccumulator`, `SequenceEncodableStatisticAccumulator`,
  `StatisticAccumulatorFactory`, `ParameterEstimator`, `DataSequenceEncoder`; capability-query +
  enumeration/count-budget surface; key-tying validation; `EnumerationError`/`KeyValidationError`.
- `pysp/stats/compute/declarations.py` — declarative metadata: `ParameterSpec`, `StatisticSpec`,
  `ExponentialFamilySpec` (`fixed_base`/`runtime_scoring`), `DistributionDeclaration`; registry +
  validation + the kernel-generation/symbolic-numba-lowering functions.
- `pysp/stats/compute/capabilities.py` — `DistributionCapabilities` + `capabilities_for`/registry +
  `intersect_engine_ready`; backs `supports_engine`.
- `pysp/stats/compute/kernel.py` — `Kernel` (ABC), `KernelFactory` (ABC), concrete kernels/factories,
  `kernel_for` dispatch, `EngineNotSupportedError`.
- `pysp/stats/compute/backend.py` — backend-hook dispatch (`backend_seq_log_density`,
  `backend_seq_component_log_density`, `backend_log_density_sum`, `child_seq_update`,
  `BackendScoringError`); the `backend_*`/`exp_family_*`/`seq_update_engine` naming convention.
- `pysp/stats/compute/stacked.py` — `StackedMixtureKernel` + stacked component param/stat/shard route
  (distributed mixture M-step).
- `pysp/stats/compute/encoded.py` — `EncodedData` / `ResidentEncodedPayload` planner-metadata wrappers
  (`engine_payload`/`host_payload`).
- `pysp/stats/compute/fused_kernels.py` — `CompiledMixture` + per-leaf fused numba kernels/builders
  (implementation behind `NumbaKernel`; no new interface).
- `pysp/stats/compute/gradient.py` — `*GradientFitState` autograd-state objects + prior-decomposition
  helpers + `GradientFitError` (torch gradient-fit facet plumbing).
- `pysp/stats/compute/torch_mixture.py` — `TorchMixture` compatibility adapter over the kernel layer.
- `pysp/engines/base.py` — `ComputeEngine` (the one formal ABC); 6 abstract ops + the larger de-facto
  op surface + `supports_numba`/`resident_estep`/`supports_autograd` dispatch flags + engine constants.
- `pysp/engines/numpy_engine.py` — `NumpyEngine` (host default, `NUMPY_ENGINE`); full op set,
  `accumulator_dtype`, precision-promoting `sum`.
- `pysp/engines/torch_engine.py` — `TorchEngine` (autograd + DTensor placement, `compile`).
- `pysp/engines/symbolic_engine.py` — `SymbolicEngine` + `SymbolicExpression` (op-tree, dunder
  overloads, `diagnostics`/`evaluate`/`symbols`/`op_counts`, `is_symbolic_payload`).
- `pysp/engines/symbolic_export.py` — `to_sympy`/`to_latex`/`to_sage` lowering (symbolic facet export).
- `pysp/engines/precision.py` — dtype-policy helpers (`precision_name`, `normalize_*_dtype`,
  `engine_with_precision`, `auto_precision`).
- `pysp/engines/__init__.py` — engine singletons + `register_array_type`/`engine_of`/`to_numpy`
  array→engine resolution & host boundary.
