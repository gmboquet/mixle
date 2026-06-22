# Engine-neutral compute core — computation ledger

Scope: `pysp/stats/compute/{pdist,declarations,kernel,stacked,backend,encoded,capabilities}.py`.
Focus: the exponential-family form `base(x) + T(x)·eta(theta) - A(theta)`, its stacked/numba
lowerings, the `generated_sufficient_statistics` / `generated_stacked_sufficient_statistics`
reductions (`_weighted_row_sum` / `_weighted_component_sum` / `_weighted_histogram`), arity/kind
alignment, log-partition signs, base-measure handling, float32 accumulation policy, and the pdist
ABC contract.

All probes run with `.venv/bin/python`. Summary up front: **no correctness or stability defect found
in the compute core.** Every generated path I could exercise matches the host accumulator / host
`seq_log_density` to machine tolerance, on numpy and on torch float64/float32.

---

## Module: pysp/stats/compute/pdist.py  (the ABC contract / API standard)

### ProbabilityDistribution / SequenceEncodableProbabilityDistribution contract (`pdist.py:64,473`)
- **Computes:** the canonical leaf API — `density`/`log_density` (scalar), `seq_log_density`
  (vectorized), `to_exponential_family` (canonical EF view), `dist_to_encoder`, `estimator`,
  `sampler`, `enumerator`, `kernel`, capability hooks.
- **How:** abstract methods plus concrete defaults; `to_exponential_family` (`pdist.py:143`) reads
  `declaration_for(self).exponential_family` and wraps it in `ExponentialFamilyForm` (no type
  switch — adding a family = providing a spec). `supported_engines`/`supports_engine` delegate to
  `capabilities_for`.
- **Why correct:** matches reference leaves; the EF view is purely metadata-driven, so engine-swap
  cannot diverge from the scalar math.
- **Numerical stability:** `density` default is `exp(log_density)` (log-space first). `density_quantile`
  / `density_enumeration` guard `log(0)` via `np.errstate(divide="ignore")` and drop `-inf` rows.
- **Engine-swap:** neutral. `to_exponential_family` defaults to `NUMPY_ENGINE` but accepts an engine.
- **Verdict:** OK. **Ambiguities (LOW, documentation-only):**
  - `density` (`pdist.py:111`) and `log_density` (`:116`) are both `@abstractmethod`, yet `density`
    ships a concrete body (`math.exp(self.log_density(x))`). The decorator says "must override" but
    the body says "you may inherit." Reference leaves do override; the mixed signal is a contract
    ambiguity, not a bug.
  - `StatisticAccumulator.update` (`:658`) is **not** abstract and has an empty body, while
    `initialize` (`:660`) calls `self.update(...)`. A subclass that forgets `update` silently no-ops
    instead of raising. Contract relies on convention.
  - `DataSequenceEncoder.seq_encode` (`:829`) is a concrete identity default but `__eq__` (`:837`) is
    abstract — an encoder author can get a working `seq_encode` while forgetting the equality contract
    that batching depends on. Documented in the class docstring but not enforced.

### scale_suff_stat / StatisticAccumulator.scale (`pdist.py:710,672`)
- **Computes:** linear scaling of nested sufficient-statistic payloads by `c`.
- **How:** structural recursion over ndarray/dict/tuple/list/np.generic/python-number; bool arrays and
  non-numeric leaves are copied unscaled.
- **Why correct:** weighted sums are linear in the weight; dicts (e.g. NegativeBinomial histogram of
  weighted counts) scale their float values, which is right because the histogram is additive,
  scalable weighted counts. Verified NegativeBinomial does not override `scale`, so the default path is
  the live one and is correct.
- **Numerical stability:** n/a.
- **Engine-swap:** host-only (operates on materialized values), which is correct — scaling happens on
  legacy payloads.
- **Verdict:** OK.

### Keyed-site validation (`pdist.py:886-1037`)
- **Computes:** structural compatibility of keyed accumulator/estimator sites before EM pools stats.
- **How:** best-effort signature hashing; raises `KeyValidationError` on incompatible shared keys.
- **Verdict:** OK (protocol-level, explicitly best-effort).

---

## Module: pysp/stats/compute/declarations.py

### ExponentialFamilySpec / DistributionDeclaration schema (`declarations.py:42,74`)
- **Computes:** the metadata that drives all generated scoring/accumulation:
  `sufficient_statistics`, `natural_parameters`, `log_partition`, optional `base_measure[_from_params]`,
  `legacy_sufficient_statistics`, `fixed_base`, `runtime_scoring`.
- **Why correct:** `runtime_scoring=False` (`:54`) cleanly removes a valid-but-unsafe EF from the
  scalar/stacked/numba *runtime* path while keeping the `to_exponential_family` map — motivated by
  categorical `eta=log(p)` having `-inf` entries (`0 * -inf = NaN`). `fixed_base=False` (`:64`) keeps a
  component-dependent base off the broadcast stacked loop (NegativeBinomial `lgamma(x+r)`).
- **Verdict:** OK. Verified live: categorical reports `runtime_scoring=False`, `numba_avail=False`, and
  `generated_log_density` raises (no backend hook) so the generic kernel uses its own safe indexing path.

### `_generated_exp_family_scalar_expression` / `_generated_exp_family_log_density` (`:759,730`)
- **Computes:** `log p(x) = base(x) + Σ_j T_j(x)·eta_j(theta) - A(theta)` (scalar and stacked `(n,k)`).
- **How:** pulls `T` from `sufficient_statistics[_from_params]`, `eta` from `natural_parameters`, base
  from `base_measure[_from_params]` or a generated zero base; checks `len(T)==len(eta)`; subtracts
  `A`. Stacked base broadcast: `(n,)` base → `base[:,None]` over k; `(n,k)` base kept (`:753`).
- **Why correct:** log-partition is **subtracted** (`rv - spec.log_partition`) — correct sign for the
  EF normalizer. Verified numerically vs `seq_log_density`: Poisson/Gaussian/NegBin/Geometric scalar
  generated == reference to 1e-9 (incl. `-inf` boundary rows).
- **Numerical stability:** evaluates in log-space throughout; base-measure `-inf` for invalid support
  is preserved (NegBin maps non-integer/negative counts to `-inf` in `exp_family_base_measure_from_params`).
- **Engine-swap:** neutral — all ops go through `engine`. The stacked base broadcast only fires for
  `fixed_base=True` families (the only ones routed here), so the `(n,k)`-base branch is unreachable but
  harmless; consistent with the numba twin.
- **Verdict:** OK.

### Numba lowerings: `_numba_exp_family_log_density` / `_numba_stacked_exp_family_log_density` (`:895,907`)
- **Computes:** the same EF dot form, lowered to nopython scalar loops; `value = base + Σ_j stat·eta - A`.
- **Why correct:** subtracts `log_partition` (per-row and per-component). Verified Poisson/Gaussian/
  NegBin/Geometric numba-generated == reference to 1e-9. `generated_numba_stacked_log_density`
  broadcasts a `(n,)` base across k and rejects mismatched `(n,k)` bases (`:698-701`) — only reached for
  `fixed_base=True`, so it never collides with NegBin's component-dependent base.
- **Numerical stability:** float64 `out` arrays regardless of nominal engine dtype.
- **Engine-swap:** host-only by construction (numba is a numpy-host accelerator); guarded by
  `supports_numba`.
- **Verdict:** OK.

### Generic symbolic→numba compiler `_build_generic_numba_kernel` / `_lower_symbolic_to_numba` (`:580,530`)
- **Computes:** a nopython scalar loop for non-EF leaves (Laplace, Logistic, StudentT, Weibull, …) by
  tracing `backend_log_density_from_params` to a SymbolicExpression and emitting Python source.
- **How:** maps symbolic ops to `math.*`; special-cases `betaln`→lgamma sum, `clip`, `where`, `max`;
  `_INF`/`_NAN` constants; rejects vector encoded data / non-scalar params / unsupported ops by
  returning `None` (cached).
- **Why correct:** verified StudentT/VonMises/Gumbel/Logistic/Weibull generated suff-stats match host;
  generated scoring for these leaves is exercised by the catalog parity test (22 dists, 440 subtests,
  green).
- **Numerical stability:** `betaln`/`gammaln`→`math.lgamma` (log-space). `pow` is lowered to `**`;
  no domain guard beyond what the source formula encodes (acceptable — formula owns its guards).
- **Engine-swap:** host-only accelerator; fail-closed to `None`.
- **Verdict:** OK.

### `generated_sufficient_statistics` + reductions (`:404`, `_weighted_row_sum` `:1435`, `_weighted_histogram` `:1450`)
- **Computes:** per-row `T(x)` from `legacy_sufficient_statistics`, then weighted-reduces over rows to
  the legacy estimator payload. Dispatch on `spec.kind`: `histogram` → `_weighted_histogram`
  (dict `{int(value): Σw}`), else `_weighted_row_sum` → `_host_legacy_value`.
- **Why correct (the recent NegativeBinomial fix class):** the declared `statistics` arity, each
  `kind`, the per-row stat that `backend_legacy_sufficient_statistics` emits, and the numpy
  accumulator `value()` must all line up. I probed this exhaustively (see below). NegBin emits
  `(count, sum, histogram-rows)`; `_weighted_histogram` folds the third row stat into the exact
  `{x: weight}` dict that `NegativeBinomialAccumulator.value()` returns. **No other leaf has the
  histogram class of mismatch** — only NegBin declares `kind="histogram"`, and its data is integer
  counts so `np.rint` is exact.
- **Numerical stability:** the over-rows reduction uses `dtype=engine.accumulator_dtype` (=float64
  even for a float32 torch engine) so large-N fits don't drift; `_weighted_histogram` folds in numpy
  float64 unconditionally.
- **Engine-swap:** neutral. `_weighted_histogram` and `_host_legacy_value` cross back via
  `engine.to_numpy`, which is the correct boundary for legacy estimator payloads.
- **Verdict:** OK. Verified host==generated for **19 leaves** (Poisson, Gaussian, Bernoulli,
  Exponential, Gamma, Geometric, NegativeBinomial, Beta, InverseGaussian, LogGaussian, Rayleigh,
  HalfNormal, InverseGamma, LogSeries, Weibull, Gumbel, Logistic, StudentT, VonMises) and **3
  multivariate** (DiagonalGaussian, MultivariateGaussian, VonMisesFisher) — all OK.

### `generated_stacked_sufficient_statistics` + `_weighted_component_sum` (`:707,1414`)
- **Computes:** component-stacked legacy stats from `(n,k)` posteriors. For `vector_moment`/
  `matrix_moment` kinds, `Σ_n w[n,k,…] * arr[n,…]` with broadcast over the moment axes.
- **Why correct:** the `extra_axes` index `weights[(slice,slice)+(None,)*]` × `arr[:,None,...]` yields
  the right `(k,…)` reduction; verified live for DiagonalGaussian (vector), MultivariateGaussian
  (vector+matrix), VonMisesFisher (vector). `accumulator_dtype` float64 accumulation here too.
- **Engine-swap:** neutral.
- **Verdict:** OK.

### `generated_stacked_params` / placement (`:297`)
- **Computes:** stacks homogeneous component parameters into `(k,…)` arrays; shares fixed/integer/
  metadata params across components, errors if a "shared" param differs.
- **Verdict:** OK (conservative rank caps, homogeneity enforced).

### Declaration schema validation (`_declaration_issues` `:1135`, statistic-layout `:1204`)
- **Computes:** schema-level checks (names, known constraints, ordered-constraint anchors, child-role
  arity, callable EF pieces).
- **Verdict:** OK. Pure metadata validation, no concrete-distribution imports.

---

## Module: pysp/stats/compute/kernel.py

### GenericKernel.score / accumulate (`:94,119`)
- **Computes:** engine-aware scoring (backend hook → generated → host `seq_log_density`) and
  accumulation (generated suff-stats → resident `seq_update_engine` → host `seq_update`).
- **Why correct:** the host `seq_log_density` fallback is gated on `supports_numba` (host numpy) so a
  non-numpy engine surfaces the failure instead of silently returning host numpy arrays (`:104`).
  Accumulate prefers `generated_sufficient_statistics` when available, else resident, else host.
- **Engine-swap:** neutral with an explicit host gate.
- **Verdict:** OK.

### GeneratedNumbaKernel.score / posteriors / accumulate (`:241,300,262`)
- **Computes:** generated numba leaf/mixture scoring and posteriors, with host-accumulator fallbacks.
- **Why correct:** the mixture row-logsumexp (`:247-252`) is the stable max-shift form and assigns
  `-inf` (not NaN) to all-`-inf` rows; `posteriors` (`:300`) falls back to prior weights for
  unsupported rows, matching `StackedMixtureKernel.posteriors`. Accumulate respects
  `_estimator_resident_supported` (`:25`) so a NegBin-style estimator that needs the full histogram
  falls back to the host accumulator, keeping every backend's fixed point identical.
- **Numerical stability:** explicit max-shift logsumexp; NaN-guarded.
- **Engine-swap:** numba kernels require `supports_numba`; otherwise factory defers to generic.
- **Verdict:** OK.

### kernel_for / factory dispatch (`:650`, `GeneratedNumbaKernelFactory` `:349`)
- **Computes:** MRO-based factory lookup; default is `GeneratedNumbaKernelFactory` (numba on numpy,
  generic elsewhere, never raises, never selects fused columnar adapter).
- **Verdict:** OK (guaranteed fallback chain).

---

## Module: pysp/stats/compute/stacked.py

### StackedMixtureKernel score/posteriors/accumulate (`:331`)
- **Computes:** homogeneous mixture `logsumexp(component_scores + log_w)`, posteriors, and legacy
  suff-stats; component math owned by the leaf via `backend_stacked_*` or the generated route.
- **Why correct:** uses `engine.logsumexp`; `posteriors` (`:368`) detects `-inf` denominators
  (`isinf(denom) & denom<0`) and falls back to prior weights, avoiding `exp(-inf-(-inf))=NaN`. The
  zero-weight component mask (`zw`, `:359`) forces those components to `-inf` before logsumexp.
- **Numerical stability:** log-space logsumexp; NaN-guarded posteriors.
- **Engine-swap:** neutral; factory only selects this on the torch engine for stackable mixtures
  (`:511`), else defers.
- **Verdict:** OK.

### Component-shard M-step (`estimate_component_shard_value` `:104`, `tie_component_shard_values` `:162`)
- **Computes:** model-parallel mixture M-step on a `[start,stop)` component shard: per-component
  estimate, plus mixture-weight update from the scalar global count (with pseudo-count/prior variants).
- **Why correct:** weight formulas — `counts/global_total`; with pseudo-count `(counts+p)/(total+pc)`
  where `p=pc/K`; with prior `(counts+prior·pc)/(total+pc)`; `global_total==0` → uniform. These match
  the standard Dirichlet-smoothed mixture-weight MLE. Shard bounds validated against `num_components`.
- **Numerical stability:** n/a (counts are non-negative; division guarded by the `==0` branch).
- **Verdict:** OK.

### `_unstack_component_stats` / `_take_component` (`:630,655`)
- **Computes:** splits a `(k,…)` stacked payload back into per-component legacy stats; 0-d→float.
- **Verdict:** OK (verified indirectly by the parity tests on stacked mixtures).

---

## Module: pysp/stats/compute/backend.py

### backend_seq_log_density / child_seq_update (`:35,18`)
- **Computes:** dispatch to `backend_seq_log_density` hook → `generated_log_density` → `BackendScoringError`;
  `child_seq_update` recurses engine-residency into child accumulators.
- **Why correct:** unwraps `engine_payload`; raises a typed error the GenericKernel catches. Residency
  recursion gated on `resident_estep` and a callable `seq_update_engine`.
- **Engine-swap:** neutral.
- **Verdict:** OK.

---

## Module: pysp/stats/compute/encoded.py

### EncodedData / move_encoded_payload (`:15,74`)
- **Computes:** wraps an encoded payload with count/engine/nbytes metadata; moves numeric arrays onto
  an engine while leaving object/string arrays and Python metadata on the host.
- **Why correct:** `dtype.kind in ("O","U","S")` arrays stay host (labels/maps), numeric arrays move —
  the correct boundary for engine residency. Recurses tuples/lists/dicts.
- **Engine-swap:** neutral (this *is* the host→engine transfer).
- **Verdict:** OK.

---

## Module: pysp/stats/compute/capabilities.py

### DistributionCapabilities / capabilities_for / intersect_engine_ready (`:9,54,88`)
- **Computes:** per-family engine-readiness metadata; `capabilities_for` resolves via class attr →
  instance/class hook → registry → MRO → default `("numpy",)`. `intersect_engine_ready` ANDs children.
- **Why correct:** class-dict `engine_ready` short-circuit (`:57`) lets a leaf declare readiness without
  a registry entry; combinator readiness is the child intersection in a preferred order.
- **Verdict:** OK.

---

## Cross-engine accumulate parity probes (evidence)

- `engine_accumulate_parity_test.py`: **22 passed, 440 subtests** (numpy + torch float64), green.
- Host-vs-`generated_sufficient_statistics` (numpy): **19 univariate + 3 multivariate leaves all OK**.
- Host-vs-generated on **torch float64 and float32**: Poisson / Gaussian / NegativeBinomial all OK —
  float32 fit does not drift because `accumulator_dtype` is float64 on both numpy and a float32 torch
  engine.
- Generated scalar EF, generic numba, and EF numba scoring vs `seq_log_density`: Poisson / Gaussian /
  NegativeBinomial / Geometric match to 1e-9 (including `-inf` support-boundary rows).
- Categorical (`runtime_scoring=False`): correctly excluded from generated scalar/numba scoring; the
  `0 * -inf = NaN` trap is avoided.

**Net:** the exp-family algebra, sufficient-statistic reductions, and kernel dispatch are correct and
engine-neutral. The NegativeBinomial histogram fix is sound and is the only `kind="histogram"` site;
no sibling leaf carries the same numpy-vs-generated mismatch. Only LOW, documentation-level ABC
ambiguities were found (no behavioral defect).
