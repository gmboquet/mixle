# pysparkplug — Computation Correctness & Numerical-Stability Audit

**Date:** 2026-06-22 · **Branch:** `speed-lda-gamma-cap` · **Scope:** the full `pysp` package
(~92 K LOC in `pysp/stats` plus engines, utils, doe/uq/infer).

This is the entry-point ledger. It consolidates the findings and the cross-cutting analysis.
The **per-computation reference ledger** — every meaningful formula, how it is computed, and why
it is correct — lives in the per-domain section files under [`audit/sections/`](sections/), one per
audit domain. Each section also records the numeric cross-checks that were run.

## How this audit was performed

The codebase was partitioned into 15 coherent computational domains and each was audited
independently against four axes (in priority order): **(1) computational correctness**,
**(2) numerical stability**, **(3) engine-swap safety** (numpy ↔ torch ↔ symbolic parity),
**(4) API-standard conformance** to the real `pysp` Distribution contract in
[`pysp/stats/compute/pdist.py`](../pysp/stats/compute/pdist.py).

Findings are **evidence-based**: every nontrivial formula was either cross-checked against
`scipy.stats`/`scipy.special`/`mpmath`, validated against a brute-force reference
(double-sums for Hawkes, path-enumeration for HMM forward, Monte-Carlo for Fisher/normalizers),
or finite-differenced (gradients). The verification numbers are quoted in each section file.

| Domain | Section | Result |
|---|---|---|
| Core numerics (`special`, `vector`, `arithmetic`, numpy engine) | [01](sections/01-core-numerics.md) | clean; special fns exact vs scipy/mpmath |
| Symbolic + torch engines | [02](sections/02-symbolic-torch-engines.md) | engine-swap parity gaps (latent paths) |
| Compute core (declarations, kernels, stacked, ABC) | [03](sections/03-compute-core.md) | clean; 22-leaf accumulate parity green |
| Fused numba kernels + gradients + torch mixture | [04](sections/04-compute-fused-gradient.md) | **1 HIGH** (NaN posteriors), gradients exact |
| Count leaves | [05](sections/05-leaf-count.md) | densities exact vs scipy; 2 minor |
| Categorical / multinomial leaves | [06](sections/06-leaf-categorical.md) | normalizers exact; mutation + NaN edges |
| Continuous leaves A (location-scale) | [07](sections/07-leaf-continuous-a.md) | densities exact; variance-cancellation |
| Continuous leaves B (positive-support, EVT) | [08](sections/08-leaf-continuous-b.md) | densities exact; Tweedie series cap |
| Directional + point processes | [09](sections/09-leaf-directional-process.md) | **clean — no findings** |
| Multivariate | [10](sections/10-multivariate.md) | Cholesky path stable; nobs==0 NaN mean |
| Latent (mixtures, HMM, LDA, PCA) + EM | [11](sections/11-latent.md) | **1 HIGH** (init crash), unguarded `w` |
| Combinator | [12](sections/12-combinator.md) | composition correct; tail underflow |
| Bayes / graph / sets | [13](sections/13-bayes-graph-sets.md) · [13a](sections/13a-graph.md) · [13b](sections/13b-sets.md) | **1 HIGH** (beta mode); graph clean |
| Estimation / fit / Fisher | [14](sections/14-estimation-fisher.md) | Fisher MC-verified; minor boundary |
| DoE / UQ / inference | [15](sections/15-doe-uq-infer.md) | EI/R-hat/Sobol formula-correct |

**Headline:** the mathematical core is sound. Densities, log-partitions, sufficient statistics,
samplers, conjugate updates, EM/Baum-Welch/forward-backward, Fisher information, and the DoE/UQ
estimators all reproduce their canonical references to machine precision. There are **0 CRITICAL**
and **3 HIGH** defects (each a real wrong-result or crash in a reachable path), plus a set of
MEDIUM numerical-stability and engine-swap items that share a few recurring root causes.

---

## Consolidated findings ledger

Severity key: **CRITICAL** = wrong results in default use, no guard · **HIGH** = wrong results or a
crash in a reachable (non-pathological) path · **MEDIUM** = stability/precision/edge-case or an
engine-swap break in a non-default path · **LOW** = style / docstring / dead code / API polish.

### HIGH

| # | Location | Defect | Verified | Suggested fix |
|---|---|---|---|---|
| H1 | [`latent/hierarchical_mixture.py:683`](../pysp/stats/latent/hierarchical_mixture.py) | `seq_initialize` does `comp_counts[:,i] = np.bincount(idx1[idx], w)` — **overwrites** (scalar path uses `+=`) and omits `minlength`; the default vectorized init raises `ValueError` (shape mismatch) on a normal corpus, or silently corrupts counts in the length-1 case | yes (reproduced) | `comp_counts[:,i] += np.bincount(idx1[idx], w, minlength=self.num_mixtures)` |
| H2 | [`compute/fused_kernels.py:1267-1275`](../pysp/stats/compute/fused_kernels.py) | `CompiledMixture.posteriors` returns **NaN** responsibilities for a row where every component log-density is `-inf` (out-of-support observation); legacy path returns uniform `1/K`. NaN flows through `gamma.sum(axis=0)` into the fused EM mixture weights and diverges the fit. numba path only — torch path is correct. | yes (counts `[nan,nan]`) | add the finite-max guard `seq_log_density` already has: mask non-finite row max, assign uniform/zero responsibility before the softmax |
| H3 | [`sets/bernoulli_set.py:887-900`](../pysp/stats/sets/bernoulli_set.py) | `_beta_posterior_mode` is wrong: the `a==b` case falls through to `return 1.0` (true mode `0.5`) and the `b>a` branch returns `(a-1)/(a+b-2)` instead of `a/(a+b)`; feeds the fitted `pmap` via `_estimate_conjugate`, so the conjugate point estimate is wrong whenever an element favors exclusion | yes (`Beta(2,2)+5/10 → 1.0`) | `mode = a/(a+b)` for `a,b>0`; keep `0/0.5/1` only for degenerate `a≤0`/`b≤0` |

### MEDIUM — numerical stability

| # | Location | Defect | Verified | Suggested fix |
|---|---|---|---|---|
| M1 | [`leaf/gaussian.py:725`](../pysp/stats/leaf/gaussian.py) (+ `log_gaussian.py:735`, `student_t.py:280`, `logistic.py:255`) | M-step variance via `E[x²]−E[x]²` → catastrophic cancellation | yes: `N(1e8,1)` → σ²=4.0 (true ≈1), a 4× error in float64 | centered / Welford accumulation, or track `Σ(x−x̄)²` |
| M2 | [`leaf/skew_normal.py:200`](../pysp/stats/leaf/skew_normal.py) (+ `exgaussian.py:287`) | MoM central 2nd/3rd moments from raw power sums cancel when `|loc|≫scale` | yes: `loc=1e6` → recovered shape −2031 (true 4) | accumulate centered moments |
| M3 | [`leaf/tweedie.py:59`](../pysp/stats/leaf/tweedie.py) | compound-Poisson-Gamma series caps `n_max=20000`; large λ (large μ / small φ) truncates the series and silently under-counts the density | yes: ~6.8e5 log-error at μ=1e3,φ=0.01 | bound `n_max` by the true series mode (~√y) and widen adaptively to a tail tolerance |
| M4 | [`multivariate/multivariate_gaussian.py:883`](../pysp/stats/multivariate/multivariate_gaussian.py) & [`diagonal_gaussian.py:804`](../pysp/stats/multivariate/diagonal_gaussian.py) | `mu = sum_x/nobs` gives **NaN mean** when `nobs==0` (zero-responsibility EM component); covariance is floored but the mean is not. This is the source of the `RuntimeWarning: invalid value in divide` seen in the suite | yes (2 warnings) | short-circuit `nobs<=0` to a safe fallback before the divide (mirror the t-dist `count<=0` path) |
| M5 | [`latent/hidden_markov.py:2680`](../pysp/stats/latent/hidden_markov.py) (+ `lookback_hidden_markov_model.py:1225`, `tree_hidden_markov_model.py:1697`) | initial-state `w = init_counts/init_counts.sum()` has **no zero-sum guard** (the transition rows right below it *are* guarded) → `[nan,nan,nan]` on empty/zero-mass data | yes | uniform fallback when `sum<=0`, mirroring the transition-row guard and the segmental/semi-sup HMMs |
| M6 | [`combinator/censored.py:73-82`](../pysp/stats/combinator/censored.py) & [`survival.py:62`](../pysp/stats/combinator/survival.py) | interval mass `log(F(b)−F(a))` / `log1p(−cdf(t))` underflow to `-inf` in the tails — the normal survival/reliability use case (deep right-censoring) is silently zeroed | yes: Gaussian `logS(40)=-inf` (true −804.6) | route through `base.logsf`/`logcdf` with a `logsubexp` difference; linear form only in the bulk |
| M7 | [`combinator/truncated.py:66`](../pysp/stats/combinator/truncated.py) | forbidden-set renormalizer `Z = 1 − Σexp(logp(f))` catastrophically cancels when forbidden mass ≈ 1 | yes (retained mass = rounding noise) | accumulate retained mass in log space via `logsumexp` over the complement |
| M8 | [`bayes/conjugate.py:463`](../pysp/stats/bayes/conjugate.py) | NIG scatter `ss = sx2 − n·x̄²` is the cancellation-prone form; can dip slightly negative for offset data under a weak prior (note, not a wrong result in normal use) | — | centered scatter `Σ(x−x̄)²` |
| M9 | [`utils/fit.py:573,582`](../pysp/utils/fit.py) | beta/dirichlet MAP log-prior `log1p(-v)` / `log(v)` → `-inf` when the constrained value saturates to exactly 1/0 (sigmoid/softmax tail, raw ≳37, float32 *and* float64); iterate is rejected (no corrupt result) but Adam can stall at the boundary and the `log_prior` diagnostic is polluted | yes | clamp the constrained value into `(eps, 1-eps)` before the prior terms |

### MEDIUM — engine-swap parity

| # | Location | Defect | Verified | Suggested fix |
|---|---|---|---|---|
| E1 | [`engines/symbolic_engine.py:502`](../pysp/engines/symbolic_engine.py) | `_EVAL_OPS` has no `digamma` entry, yet the engine emits `digamma` nodes (LDA / labeled-LDA M-steps); `expr.evaluate(...)` raises `KeyError('digamma')` | yes | add a numeric ψ to `_EVAL_OPS` |
| E2 | [`engines/symbolic_export.py:214`](../pysp/engines/symbolic_export.py) | sage `where(cond,a,b)` lowered as `a·SR(cond)+b·(1−SR(cond))` — `SR(relation)` is not a 0/1 indicator in sage's ring, so the export is algebraically wrong (sympy path correctly uses `Piecewise`) | by reading (sage not installed) | encode a genuine boolean→0/1 indicator / `piecewise` |
| E3 | [`engines/symbolic_engine.py:394`](../pysp/engines/symbolic_engine.py) | symbolic `logsumexp` is naive `log(Σexp)` with no max-shift → `OverflowError` where numpy/torch return the stabilized value | yes (`[1000,1000]`) | shift-and-add-back the running max |
| E4 | [`engines/torch_engine.py:190`](../pysp/engines/torch_engine.py) | `cumsum` (no axis) and tuple-axis `max` don't translate `axis→dim`; torch raises `TypeError` where numpy works. No current callers (latent) | yes | add the `axis→dim` shim used by `sum`/`logsumexp` |
| E5 | [`engines/numpy_engine.py:83`](../pysp/engines/numpy_engine.py) | `engine.sum = staticmethod(np.sum)`: a float32 input reduces in float32 and drifts on large N; correctness currently relies on every accumulator passing `dtype=accumulator_dtype` (float64). Verified the stats reducers *do* pass it, so no live drift — but the engine itself does not enforce it | yes (rel err 9.2e-8 raw) | default host-engine float reductions to `accumulator_dtype` |

### MEDIUM — correctness edge cases & protocol

| # | Location | Defect | Verified | Suggested fix |
|---|---|---|---|---|
| C1 | [`leaf/integer_multinomial.py:246`](../pysp/stats/leaf/integer_multinomial.py) | scalar `log_density` of an out-of-support value with count 0 computes `(-inf)*0 = NaN` (seq path is safe) | yes | mask out-of-range terms / skip `cnt==0` |
| C2 | [`leaf/categorical.py:939-942`](../pysp/stats/leaf/categorical.py) | pure-`pseudo_count` `estimate` **mutates the caller's `suff_stat` dict in place**, overwriting counts with probabilities | yes | build a fresh `p_map` |
| C3 | [`leaf/integer_uniform_spike.py:719,739,757`](../pysp/stats/leaf/integer_uniform_spike.py) | estimator `pseudo_count` branches mutate the caller's `count_vec` in place | yes | copy before adding pseudo_count |
| C4 | [`graph/integer_markov_chain.py:~1251`](../pysp/stats/graph/integer_markov_chain.py) | `estimate` row-normalizes with no pseudo_count → `0/0` NaN transition rows for any never-observed lagged state (cond_mat also float32) | yes | clamp zero rows to uniform; consider float64 |
| C5 | [`sets/integer_bernoulli_set.py:101-132`](../pysp/stats/sets/integer_bernoulli_set.py) | degenerate `p_k=1` yields `log_density([0])=+inf`; reachable via the estimator at `min_prob=0` (default `1e-128` is safe) | yes | treat `p_k=1` as required membership, or raise |
| C6 | [`sets/integer_bernoulli_edit.py:609`](../pysp/stats/sets/integer_bernoulli_edit.py) & [`integer_step_bernoulli_edit.py:512`](../pysp/stats/sets/integer_step_bernoulli_edit.py) | `seq_update` calls `estimate.init_dist` without the `None if estimate is None` guard the scalar `update` has → `AttributeError` on `seq_update(enc,w,None)` | yes | add the same guard |
| C7 | [`doe/optimal.py:67,82`](../pysp/doe/optimal.py) | A/I-optimality call `np.linalg.inv(info)` rather than a solve; near-singular `M` less stable (mitigated by `LinAlgError→-inf` and `n≥p`) | by reading | use `np.linalg.solve`/`lstsq` |
| C8 | [`uq/propagate.py:81`](../pysp/uq/propagate.py) | unscented transform does `cholesky((d+λ)Σ)` with no guard; a user `kappa` making `d+λ<0` crashes (defaults are safe) | by reading | guard `d+λ>0` or fall back to eigendecomposition |

### LOW (style / docstring / dead code / API)

`geometric.py:708` clips `p` to inclusive `1.0` → spurious `log1p(-1)` divide-by-zero warning on all-ones
data (model still correct) · `integer_uniform_spike.py:98,703` `num_vals==1` → `log(0)` NaN warning ·
docstring formula errors in `integer_multinomial.py`, `categorical_multinomial.py`, `gaussian.py:7`,
`multivariate_gaussian.py:14`, `diagonal_gaussian.py:10`, `multivariate_gaussian.py:286` (writes
`det` for `log|Σ|`) · `inverse_gamma.py:395` uses a local finite-difference `_trigamma` instead of the
exact `pysp.utils.special.trigamma` · `binomial.py:1004` infers `n=max−min` from data (documented
support behavior) · `pdist.py:111/658/829` ABC ambiguities (concrete `density` body under
`@abstractmethod`; non-abstract empty `update`; concrete `seq_encode` default vs abstract `__eq__`) ·
`torch_mixture.py:62-90` bare `except Exception` fallbacks swallow real errors ·
`fused_kernels.py:761-768` Binomial `min_val` suff-stat differs from legacy (benign) ·
`optional.py:88` dead `self.log1_p = log1p(self.p)` (looks like a typo for `log1p(-p)`) ·
`special.py:140` unused `polygamma_loc(0,·)` returns `inf` not `digamma` · `special.py:164`
`digammainv` ignores `out=` · `vector.py:21` inconsistent scalar return type · `arithmetic.py:144`
`sum`/`max` missing from `__all__` · `estimation.py:705` MAP-vs-MLE auto-detect via `==0.0` ·
`objectives.py:601-604` `nanargmax` raises on all-NaN history · `latent` LOW items
(`segmental` uniform-fill on zero-prob seqs, `joint_mixture` / `heterogeneous_mixture` degenerate paths).

---

## Cross-cutting themes (root causes worth a systemic fix)

These patterns recur across modules — fixing them at the source removes whole classes of the table above.

1. **`E[x²] − E[x]²` variance cancellation.** The single most common stability hazard: M1
   (Gaussian/LogGaussian/StudentT/Logistic), M2 (SkewNormal/ExGaussian), M8 (NIG scatter), and the
   diagonal/MVN covariance steps all accumulate raw power sums and subtract. Float64 hides it until
   the data mean is large relative to the spread (verified failures at `loc≈1e6`–`1e8`). A shared
   centered/Welford accumulation helper for the moment-form M-steps would retire all of them and is
   the highest-value cleanup. *None of these produce wrong results on well-centered data — the suite
   passes — but they are latent on real offset data.*

2. **Unguarded normalize-by-count.** `x / nobs` or `counts / counts.sum()` with no zero guard →
   NaN: M4 (MVN/diagonal-Gaussian mean), M5 (three HMM initial-weight steps), C4 (integer Markov
   rows). The fix pattern already exists elsewhere in the same files (transition rows, t-dist
   `count<=0`); these are the spots that didn't adopt it. A zero-sum→uniform/fallback guard is the
   uniform remedy.

3. **`-inf · 0 = NaN` and `log(0)` on out-of-support / degenerate params.** C1 (integer multinomial),
   C5 (bernoulli set `p=1`), the `num_vals==1` and all-ones spike/geometric warnings. Scalar paths
   are the usual offenders (seq paths mask correctly). Guard before the multiply / special-case the
   degenerate parameter.

4. **CDF-difference / survival underflow in the tails.** M6 (censored interval, survival) and M7
   (truncated forbidden mass) all compute `log(F(b)−F(a))` or `1−Σp` in probability space and lose
   the tail. Routing through `logcdf`/`logsf` + `logsubexp` is the standard fix and matters because
   tail-censoring is the *normal* use case for those combinators.

5. **In-place mutation of caller-owned sufficient statistics.** C2, C3 — estimators that overwrite
   the input `suff_stat`/`count_vec`. Harmless until a caller reuses the stats (e.g. multi-estimator
   or warm-start). Copy-on-write.

6. **Engine-swap parity is good on the hot paths, thin on the cold ones.** The numpy↔torch accumulate
   parity is solid (22-leaf parity test green; float32 doesn't drift because `accumulator_dtype` is
   float64 on both engines — E5). The gaps (E1–E4) are in **symbolic** evaluation and **uncommon torch
   ops** (`digamma` evaluate, sage `where`, naive `logsumexp`, `cumsum` axis) — paths not yet exercised
   by the default numeric flows but exactly what would break first when actually swapping a model onto
   the symbolic or a non-float64 torch engine. These are the items to close before relying on
   engine-swapping for LDA-class models.

---

## Engine-swap correctness assessment (the user's explicit concern)

- **numpy ↔ torch, scoring & accumulation:** verified equivalent to ≤1e-15 (float64) across 22 leaves
  + composites + mixtures via the parity test and direct probes; float32 stays accurate because both
  engines reduce sufficient statistics in `accumulator_dtype=float64`. The exp-family lowering
  (`base + T·η − A`), its stacked and numba variants, and the histogram suff-stat reduction all agree
  with the host accumulator `value()`. **Safe to swap numpy↔torch for the covered families.**
- **→ symbolic:** the op set lowers with correct sign/argument order, but three evaluation gaps (E1
  `digamma`, E3 `logsumexp` overflow, E2 sage `where`) will surface for any model whose M-step or score
  touches them — notably the LDA family. **Close E1–E3 before relying on symbolic for those.**
- **Algorithm nesting:** mixtures/HMMs/combinators compose child kernels through log-space rules that
  were verified against brute-force references (HMM forward vs path enumeration, mixture posteriors vs
  logsumexp, composite/sequence factorization). The nesting itself is correct; the only nesting-level
  defect is H2 (fused numba mixture posteriors NaN on an all-`-inf` row) and H1 (hierarchical init).

---

## Recommended remediation order

1. **H1, H2, H3** — real reachable wrong-result/crash bugs; small, local fixes. (H1 crashes the default
   hierarchical-mixture init; worth doing first.)
2. **M4 + M5 + C4** — the NaN-on-zero-count cluster; one guard pattern, removes the suite's
   `RuntimeWarning`s.
3. **M1/M2/M8** — shared centered-moment helper for the variance M-steps (highest-value stability win).
4. **M6/M7** — `logsubexp` for the survival/censored/truncated normalizers.
5. **E1–E3** — symbolic-engine evaluation gaps, to make engine-swapping real for LDA-class models.
6. **C1–C8, then LOW** — edge-case guards, copy-on-write, docstrings, dead code.

None of H1–H3 or the MEDIUM items are exercised by the current green test suite, which is itself a
finding: the suite covers well-centered, fully-populated, default-parameter cases. Targeted
regression tests for **offset data** (large mean), **empty/zero-responsibility components**,
**tail censoring**, and **out-of-support scalar `log_density`** would catch this whole class.

---

*Detailed per-computation ledgers (formula → implementation → why-correct → stability → engine notes)
are in [`audit/sections/`](sections/). The numeric verification values quoted above are reproduced in
each section file.*
