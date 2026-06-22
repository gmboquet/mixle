# Fused numba kernels, gradients, torch mixture E-step — computation ledger

Scope: `pysp/stats/compute/fused_kernels.py`, `gradient.py`, `torch_mixture.py`.
All numeric checks run with `.venv/bin/python`; comparisons are against each
distribution's own `seq_log_density` / `seq_posterior` / accumulator `value()`
(the legacy seq_ path), which is the parity contract the module docstring claims
("results agree with the legacy seq path to floating-point tolerance").

Headline result: every leaf/composite/sequence/optional kernel log-density and
every weighted sufficient statistic matches the legacy seq path to machine
epsilon (verified, see per-row below). Analytic gradient/prior formulas in
gradient.py are correct (finite-difference and autograd-reference confirmed).
**One real correctness break:** `CompiledMixture.posteriors` produces NaN
responsibilities for rows where every component log-density is `-inf`, whereas
the legacy `seq_posterior` returns a uniform distribution; this poisons the
fused EM M-step (NaN component counts).

## Module: pysp/stats/compute/fused_kernels.py

### Leaf log-density kernels (`fused_kernels.py:97-365`)
- **Computes:** scalar log-density per (row i, component k) for Gaussian,
  LogGaussian, Gamma, Categorical, IntegerCategorical, Bernoulli, Poisson,
  Exponential, Geometric, NegativeBinomial, StudentT, Logistic, Weibull,
  Rayleigh, Pareto, Uniform, Binomial, DiagonalGaussian, Ignored.
- **How:** closed-form log-pdf/pmf evaluated from per-component param arrays plus
  precomputed encode-time columns (lgamma(x+1), log x, x², log-binom coef).
- **Why correct:** each formula matches the leaf's `seq_log_density`. Verified
  numerically — kernel-vs-seq max abs diff per family:
  Gaussian/Gamma/Rayleigh/Weibull/Weibull+0/Pareto/StudentT/Poisson/Geometric/
  Binomial/Exponential/Bernoulli/Categorical/IntCategorical = 0.0;
  Logistic/LogGaussian = 2.2e-16; NegBinomial = 1.8e-15; DiagGaussian = 8.9e-16.
- **Numerical stability:** log-space throughout; Logistic uses the sign-split
  `-log1p(exp(-|z|))` form (overflow-safe); Weibull handles x=0 by shape
  (returns +inf/-inf/-logscale per shape<1/>1/=1, matches leaf); positive-support
  leaves return `_NEG_INF` outside support; lgamma via `math.lgamma`.
- **Engine-swap:** numba/numpy-only by construction — this is the numpy fused
  path; not used on torch/symbolic. OK (documented).
- **Verdict:** OK

### Accumulator kernels & `stats_to_ss` (`fused_kernels.py:106-369, 405-1106`)
- **Computes:** weighted sufficient statistics per component, then maps the flat
  stat buffers into the legacy accumulator `value()` tuple shape.
- **How:** `acc(i,k,w,...)` adds `w`-weighted moments into per-component stat
  arrays; `stats_to_ss` reshapes to the legacy tuple/dict order.
- **Why correct:** verified on a 3-component composite (Gaussian+Gamma+
  Categorical) mixture and on Binomial/Gaussian mixtures: fused per-component ss
  equal legacy `acc.value()` after the same `gamma[:,i]` weights (no diffs at
  1e-7/1e-8). counts (`gamma.sum(0)`) max diff 0.0.
- **Numerical stability:** plain weighted sums; Pareto/Uniform track min/max via
  comparison (additive-safe across thread chunks because only sums are merged,
  min/max recomputed). n/a otherwise.
- **Engine-swap:** numpy-only fused path. OK.
- **Verdict:** OK (see Binomial min/max note below — benign)

### Binomial suff-stat min/max recovery (`fused_kernels.py:327-334, 761-768`)
- **Computes:** data-wide (min,max) recovered as `round(stats[2or3]/count)`.
- **How:** accumulates `w*cols[2][0]` / `w*cols[2][1]` (constant data-wide
  min/max) so the value survives threaded chunk merges, divides out by count.
- **Why correct:** the recovered value is the true data min/max. Verified: with a
  sample whose true min is 1, fused returns `min_val=1` (correct) while the
  legacy accumulator returns `min_val=0` — but only because the *legacy factory*
  seeds `min_val=0`. End-to-end EM is identical: the default `BinomialEstimator`
  takes `min(suff_stat_min, self.min_val=0)=0` in both paths, so the fitted
  `(p,n,min_val)` are bit-identical (verified: comp0/comp1 p match to 1e-10).
- **Verdict:** OK / LOW — raw `value()[2]` (min_val) differs from legacy
  (fused=true data min, legacy=factory-seeded 0). Benign with the default
  estimator; could differ only with a non-default `min_val` estimator that
  defers to the suff-stat. Noted as F3 (LOW).

### Composite fusion (`fused_kernels.py:803-884`)
- **Computes:** sum of child log-densities; child suff-stats nested as a binary
  tree of 2-tuples.
- **How:** `_pair_kernels`/`_pair_accs` fold children into a balanced binary
  tree; `_tree`/`_untree` convert between flat slot lists and the nested shape.
- **Why correct:** `_untree` is the exact inverse of `_tree` (same shape
  recursion). Verified on the composite mixture above (suff-stats match legacy).
- **Engine-swap:** numpy/numba-only. OK.
- **Verdict:** OK

### Sequence kernel (`fused_kernels.py:887-976`)
- **Computes:** Σ_t inner(t) over the offset range + optional length-model term.
- **How:** offset array `off[i]:off[i+1]` indexes flattened token columns; length
  kernel scored once per row when `has_len`.
- **Why correct:** verified Sequence(Poisson tokens, len=Poisson) vs leaf
  `seq_log_density` — max diff 0.0. `len_normalized` is rejected at build time
  (raises), matching the documented exclusion.
- **Numerical stability:** sum of log terms; n/a.
- **Verdict:** OK

### Optional kernel (`fused_kernels.py:979-1062`)
- **Computes:** missing rows → `log_p` (miss mass); present rows →
  `log_pn + inner(child)`.
- **How:** presence-index column routes present rows into compacted child cols;
  missing/present mass split per `has_p`.
- **Why correct:** verified Optional(Gaussian, p=0.3, sentinel missing) vs leaf
  `seq_log_density` — max diff 0.0. (NaN-missing path not separately run because
  the leaf encoder rejects NaN, but the kernel `_is_missing` mirrors the leaf.)
- **Verdict:** OK

### `seq_log_density` mixture LSE (`fused_kernels.py:1255-1265`)
- **Computes:** log Σ_k w_k exp(ll_k) via max-shift log-sum-exp.
- **How:** `mx = ll.max(1)`; rows with non-finite max get `-inf`, others
  `log(Σ exp(ll-mx)) + mx`.
- **Why correct:** verified vs `mix.seq_log_density` — max diff 3.6e-15. The
  `good = isfinite(mx)` guard correctly returns `-inf` for all-`-inf` rows
  (matches legacy, which returns `-inf` log-density for impossible obs).
- **Numerical stability:** correct max-shift LSE; impossible-row guard present.
- **Verdict:** OK

### `posteriors` softmax over components (`fused_kernels.py:1267-1275`)
- **Computes:** responsibilities = softmax_k(ll_k + log_w_k).
- **How:** `ll -= ll.max(1)`; `exp`; divide by row sum.
- **Why correct (normal rows):** verified vs `mix.seq_posterior` — max diff
  4.4e-16 on a normal composite mixture.
- **BUG:** for a row where every component log-density is `-inf`
  (observation impossible under all components — e.g. mixture of Uniforms with x
  outside every support), `ll.max(1) = -inf`, so `ll - (-inf) = NaN`,
  `exp(NaN)=NaN`, and the row sum is NaN → the entire responsibility row is NaN.
  Legacy `seq_posterior` returns uniform `[1/K,...,1/K]` for the same row
  (verified: legacy gives `[0.5,0.5]`, fused gives `[nan,nan]`). Unlike
  `seq_log_density`, `posteriors` has NO finite-max guard.
- **Downstream blast radius:** `em_step` (`:1316-1323`) calls `posteriors` then
  `weighted_suff_stats`. The acc driver guards `w>0.0` (NaN>0 is False, so
  per-component stats skip the NaN row), but `weighted_suff_stats` returns
  `gamma.sum(axis=0)` as the mixture weights (`:1313`) — that sum is NaN, so the
  re-estimated mixture weights are NaN and the whole EM diverges. A *single*
  impossible observation poisons the fit. Verified: one out-of-support row →
  `counts = [nan, nan]`. `torch_mixture.posteriors` does NOT have this bug
  (it falls back to logsumexp / `seq_posterior`, returns `[0.5,0.5]`).
- **Numerical stability:** missing the all-`-inf` guard that `seq_log_density`
  already has.
- **Verdict:** FINDING(F1) — HIGH.

### `weighted_suff_stats` threaded merge (`fused_kernels.py:1279-1314`)
- **Computes:** chunked parallel accumulation merged additively.
- **How:** per-chunk stat buffers, `_merge_stats` recursive add.
- **Why correct:** verified serial-vs-8-thread suff-stats max diff 2.5e-9 (FP
  reassociation only) on 300k rows; component log-density thread parity 0.0.
- **Numerical stability:** float reassociation across chunks gives ~1e-9 drift;
  acceptable. n/a.
- **Verdict:** OK

### `initialize` / `fit` (`fused_kernels.py:1325-1364`)
- **Computes:** sparse Dirichlet-random responsibility init then EM to delta.
- **How:** `rng.rand`/`rng.dirichlet` host numpy; loops `em_step`.
- **Why correct:** standard EM; fit converged (final ll -1980.76 on the test
  mixture, monotone). Inherits the F1 risk only if init or any iterate yields an
  all-`-inf` row.
- **Engine-swap:** host numpy RNG — fine, this is the numpy fused path.
- **Verdict:** OK

## Module: pysp/stats/compute/gradient.py

### `normal_gamma_log_prior` (`gradient.py:197-208`)
- **Computes:** Normal-Gamma log prior kernel
  `(α-1)logτ - βτ + ½logτ - ½κτ(μ-μ0)²`, τ=1/σ².
- **Why correct:** autograd of this expression matches a finite-difference of the
  reference kernel — grad_μ = -0.09411765 (both), grad_σ² = -0.72110727 (both).
- **Numerical stability:** `log(tau)` with τ=1/σ²; fine for σ²>0.
- **Verdict:** OK

### `CategoricalGradientFitState.log_prior` (`gradient.py:243-253`)
- **Computes:** Dirichlet `Σ(α-1)·log_softmax(logits)`; weak fallback
  α = 1 + strength/numel.
- **Why correct:** verified both branches against the closed form — diff 0.0 for
  the explicit Dirichlet(α=2) and for the weak fallback (α=1.5).
- **Numerical stability:** uses `log_softmax` (stable). OK.
- **Verdict:** OK

### `OptionalGradientFitState.log_prior` / `.score` (`gradient.py:276-323`)
- **Computes:** missing→log p, present→child+log1p(-p); beta missingness prior
  `(a-1)log p + (b-1)log1p(-p)`.
- **Why correct:** beta-prior term verified vs closed form — diff 0.0. Score uses
  `sigmoid(logit_p)` and `log`/`log1p` consistently (matches the fused
  `_optional_kernel` semantics).
- **Numerical stability:** `log(sigmoid)` could underflow for very negative
  logit_p, but the weak/beta priors keep p away from {0,1}; acceptable.
- **Verdict:** OK

### `MixtureGradientFitState.score` / `.log_prior` (`gradient.py:598-622`)
- **Computes:** `logsumexp(comp + log_softmax(w_logits))`; Dirichlet/weak weight
  prior over `log_softmax`.
- **Why correct:** score matches the reference logsumexp — diff 0.0; Dirichlet
  weight prior (α=[2,1.5,1]) matches closed form — diff 0.0.
- **Numerical stability:** `engine.logsumexp` + `log_softmax` (stable). OK.
- **Verdict:** OK

### Prior-routing helpers (`gradient.py:24-180`)
- **Computes:** normalize user `priors` (dict/list/family-tagged) into per-child
  structures for composite/conditional/record/select/mixture/sequence/markov.
- **How:** pure dispatch on `family`/keys; pad/truncate via `prior_sequence`.
- **Why correct:** structural plumbing, no math; behavior is the documented
  family routing. Not separately fuzzed (out of numeric scope).
- **Engine-swap:** engine-neutral (operates on Python containers + `torch`/
  `engine` passed in).
- **Verdict:** OK

### Other FitState `score` methods (Conditional/Select/Sequence/Transform/
Composite/Record) (`gradient.py:341-556`)
- **Computes:** product/sum/index-add aggregation of child scores; Sequence
  applies `len_normalized` scaling; Transform adds Jacobian + `-inf` on invalid.
- **Why correct:** these mirror the corresponding distributions' seq math; they
  use only engine-neutral ops (`engine.zeros/asarray/index_add/where`,
  `torch.*`). No host-only numpy leak observed. Not separately finite-differenced
  (no novel closed form beyond the child scores), but logic reviewed.
- **Engine-swap:** neutral (engine + torch threaded through). OK.
- **Verdict:** OK

## Module: pysp/stats/compute/torch_mixture.py

### `posteriors` (`torch_mixture.py:74-90`)
- **Computes:** softmax over `component_scores + log_w` via logsumexp, or kernel
  `posteriors`; falls back to `seq_posterior` on any exception.
- **Why correct:** float64 max diff vs `seq_posterior` = 8.9e-16. For the
  all-`-inf` impossible-row case it returns `[0.5,0.5]` (matches legacy) — this
  path is NOT affected by F1.
- **Numerical stability:** logsumexp-based; stable. float32 drift ~4.5e-7
  (expected, acceptable).
- **Engine-swap:** torch path; numpy conversion only at the boundary. OK.
- **Verdict:** OK

### `seq_log_density` / `_score` (`torch_mixture.py:54-72, 232-237`)
- **Computes:** per-row model log-density via `model.kernel(engine).score`,
  fallback to `seq_log_density`.
- **Why correct:** float64 max diff vs legacy = 8.9e-16.
- **Verdict:** OK

### `weighted_suff_stats` (`torch_mixture.py:92-110`)
- **Computes:** per-component legacy suff-stats by running the host accumulator
  `acc.seq_update(enc, gamma[:,i])`.
- **Why correct:** delegates to the real accumulators; verified suff-stats equal
  legacy `acc.value()` exactly (it IS the same accumulator). counts =
  `gamma.sum(0)`.
- **Numerical stability:** host float64. OK.
- **Engine-swap:** converts gamma to numpy then uses numpy accumulators — host
  bookkeeping by design (the M-step lives in numpy). Acceptable; note this is a
  host-only op but intentional and isolated.
- **Verdict:** OK

### `em_step` / `fit` (`torch_mixture.py:112-168`)
- **Computes:** kernel accumulate → `estimator.estimate`, EM loop to delta.
- **Why correct:** standard; broad `except Exception` fallback to host accumulate.
- **Numerical stability:** OK.
- **Verdict:** OK — LOW note: the bare `except Exception` in `_score`,
  `posteriors`, `seq_component_log_density`, `em_step` silently masks real kernel
  bugs (a kernel that raises for a correctness reason is swallowed and the slow
  path is used with no signal). Not a math bug; noted F2 (LOW).

## Findings summary
- **F1 (HIGH):** `fused_kernels.py:1267-1275` `CompiledMixture.posteriors`
  returns NaN responsibilities for all-`-inf` rows; legacy returns uniform.
  Poisons fused EM weights (`gamma.sum(0)`). Add the same finite-max guard used
  by `seq_log_density`.
- **F2 (LOW):** `torch_mixture.py` bare `except Exception` fallbacks mask kernel
  errors silently.
- **F3 (LOW):** `fused_kernels.py:761-768` Binomial `value()` min_val differs
  from legacy (fused=true data min, legacy=factory-seeded 0); benign for default
  estimator (EM result bit-identical).
