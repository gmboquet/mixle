# Combinator distributions — computation ledger

Domain: structural composition over child distributions, `pysp/stats/combinator/`.
Verification venv: `.venv/bin/python`. All numeric checks below reproduced.

---

## Module: combinator/truncated.py
### Renormalizer `log Z` (`truncated.py:62-69`)
- **Computes:** `Z = sum_{a in allowed} p_base(a)` or `Z = 1 - sum_{f in forbidden} p_base(f)`; `log_density = log p_base(x) - log Z` for in-support `x`, else `-inf`.
- **How:** explicit finite-set sums of `exp(base.log_density(v))`.
- **Why correct:** truncation conditions the base on a support set: `p(x)=p_base(x)/Z`. Verified numerically: truncated Poisson(2) on `{1..5}` renorm densities sum to 1.0; `forbidden={0}` gives `logZ = log1p(-p0)` exactly.
- **Numerical stability:** allowed-set path stable. **Forbidden-set path `1 - sum(exp(...))` catastrophically cancels** when forbidden mass ≈ 1: `forbidden=range(0,30)` on Poisson(2) yields `logZ=-36.7` (retained ≈1.1e-16, pure rounding noise) vs true tail ≈1e-33. FINDING(C1).
- **Engine-swap:** `seq_log_density` is host numpy (`rv[~allowed_mask]=-inf`); a numeric allowed-mask is fine but `freeze()`-based membership is host-only. Acceptable (combinator is a numpy-side wrapper).
- **Verdict:** FINDING(C1) (forbidden cancellation, edge case)

### seq_log_density (`truncated.py:103-108`)
- **Computes:** per-row `base.seq_log_density - log_z`, `-inf` off-support.
- **Why correct:** `rv` is a fresh array (subtraction), in-place `-inf` assignment safe.
- **Verdict:** OK

---

## Module: combinator/censored.py
### Interval log-mass `log(F(b)-F(a))` (`censored.py:73-82`)
- **Computes:** censored contribution `log P(a<=X<=b) = log(F(b)-F(a))`; right-cens `b=inf`→`F(b)=1`, left `a=-inf`→`F(a)=0`.
- **How:** naive `mass = fb - fa; log(mass)` with `mass<=0 -> -inf`.
- **Why correct:** the interval-censoring likelihood. Direction handled (`b<a` swap).
- **Numerical stability:** **`F(b)-F(a)` underflows to 0 in the tails → `-inf` for genuine mass.** Verified: Gaussian(0,1) `(40, inf)` returns `-inf`; true `logsf(40) ≈ -804.6`. Narrow tail `(10, 10.0001)` returns `-inf`; true ≈ `-60.1`. Heavy right-censoring at large `t` (standard in survival/reliability) silently zeroes the likelihood. FINDING(C2).
- **Suggested fix:** use base `logcdf`/`logsf` with a `logsumexp`-style difference (`logsubexp`) when available; fall back to the linear diff only in the bulk.
- **Verdict:** FINDING(C2)

### Accumulator / estimator (`censored.py:143-234`)
- **Computes:** fits base on **exact** observations only; censored rows skipped. Documented non-MLE.
- **Why correct:** as documented (fixed-bounds, not the coupled censored MLE). Weights/idx scatter correct.
- **Verdict:** OK (documented approximation)

---

## Module: combinator/survival.py
### Log-survival `log S(t)=log1p(-F(t))` (`survival.py:57-72`)
- **Computes:** `log f(t)` for events, `log(1-F(t))` for right-censored rows.
- **Numerical stability:** **same tail underflow as censored.** Verified: Gaussian survival `logS(40)=-inf` (true -804.6), `logS(10)=-inf` (true -53.2), emits a divide-by-zero warning. FINDING(C3) (same class as C2; a `logsf` path fixes both).
- **Engine-swap:** `seq_log_density` recomputes base density for censored rows then overwrites — wasteful but correct.
- **Verdict:** FINDING(C3)

### Conditional-quantile imputation EM (`survival.py:101-127`)
- **Computes:** censored `(c,w)` → `n_impute` points at `F^{-1}(F(c)+q(1-F(c)))`, weight `w/K`, midpoint grid `q=(i+0.5)/K`.
- **Why correct:** midpoint-rule quadrature of the conditional tail expectation; re-imputed each EM step → right-censored MLE. Sound.
- **Verdict:** OK

---

## Module: combinator/hurdle.py
### log_density two-part rule (`hurdle.py:62-93`)
- **Computes:** `P(0)=pi`; `P(k>0)=(1-pi) p_base(k)/(1-p_base(0))`. `log_pi`, `log1mpi=log1p(-pi)`, `log_renorm=log1p(-p0)`.
- **Why correct:** zero-truncated base for positives, point mass at 0. `log1p` used for both `1-pi` and `1-p0` (stable). Mixture/normalizer math verified by inspection.
- **Numerical stability:** good (`log1p`). `pi=0` → `log_pi=-inf` only used on the `x==0` branch, fine.
- **Verdict:** OK

### Zero-truncated MLE EM (`hurdle.py:261-277`)
- **Computes:** imputes `N_missing = N_pos·p0/(1-p0)` pseudo-zeros, refits, iterates to the zero-truncated MLE.
- **Why correct:** standard EM for truncation; converges (monotone p0). Closed-form `pi` = zero rate. Correct two-part independence.
- **Verdict:** OK

---

## Module: combinator/zero_inflated.py
### log_density mixture (`zero_inflated.py:77-91`)
- **Computes:** `P(0)=pi+(1-pi)p_base(0)`, `P(k>0)=(1-pi)p_base(k)`. Uses `logaddexp(log_pi, log1mpi+lb)` at 0.
- **Why correct:** standard ZI mixture. `logaddexp` is the right log-space combine; `pi=0`→`log_pi=-inf`, `logaddexp(-inf,·)` safe. `log1mpi=log1p(-pi)`.
- **Verdict:** OK

### EM responsibility (`zero_inflated.py:142-178`)
- **Computes:** `r = pi/(pi+(1-pi)p_base(0))` constant over all zeros; structural-zero count and down-weighted base update; `pi = E[structural]/N`.
- **Why correct:** exact E-step for the latent zero source; `r` correctly computed once per batch. Init heuristic (half the zeros structural) is benign.
- **Numerical stability:** `p0=exp(base.log_density(0))` in linear space — fine for count bases (p0 not tiny); `denom>0` guarded.
- **Verdict:** OK

---

## Module: combinator/exponential_tilt.py
### Analytic tilt registry (`exponential_tilt.py:81-112`)
- **Computes:** identity-statistic CGFs. Gaussian `logZ=θμ+½θ²σ²`, tilted `N(μ+θσ²,σ²)`; Poisson `logZ=λ(e^θ-1)`, `Poisson(λe^θ)`; Gamma `logZ=-k·log1p(-θ·scale)`, `Gamma(k,scale/(1-θ·scale))`; Exponential analogous.
- **Why correct:** these are the exact MGFs / Esscher transforms. Verified: Poisson(2) tilt θ=0.5 → `logZ=1.2974` (=`2(e^0.5-1)`), closed-form `λ=3.297`, tilted vs closed-form log-density max err 7e-15. Domain guards (`θ·scale>=1 → inf`) correct.
- **Numerical stability:** `log1p`/`expm1` used. `Z` non-finite → raises (good).
- **Verdict:** OK

### Enumerated/SIR normalizer (`exponential_tilt.py:144-156`)
- **Computes:** `logZ = logsumexp(lp + θ·T)` over the enumerated base.
- **Why correct:** exact discrete `Z`. `logsumexp` stable.
- **Verdict:** OK

### θ MLE score solve (`exponential_tilt.py:446-494`)
- **Computes:** solve `A'(θ)=mean(T)` via central-diff CGF gradient + bracketing bisection; `A` convex so `A'` monotone.
- **Why correct:** exp-family score equation `E_θ[T]=mean(T)`. Bracket direction matches `A'` increasing. Finite-difference `h=1e-5` adequate. Vector-θ raises NotImplemented (documented).
- **Verdict:** OK

---

## Module: combinator/transform.py
### Change-of-variables log-density (`transform.py:235-252`, transforms `:50-170`)
- **Computes:** `log p_Y(y) = log p_X(T^{-1}(y)) + log|d T^{-1}/dy|`.
- **Why correct:** Jacobians verified per transform — Affine `-log|scale|`, Exp `-log y`, Log `y`, Logit `-log y - log1p(-y)`. Numerically: Exp(Gaussian)=lognormal max abs err 5.5e-17 & integral 1.0; Logit-normal integral 1.0; Affine matches `N(3,4)`. `density_correction` off for discrete children (enumerable → mass, no Jacobian).
- **Numerical stability:** Logit `forward` branches on sign of x to avoid `exp` overflow; `inverse` uses `log1p`. Invalid inverses → `valid=False` → `-inf`. Good.
- **Engine-swap:** has `backend_seq_log_density` / stacked params / gradient state; log_jac precomputed at encode time (host) then added on-engine — neutral.
- **Verdict:** OK

---

## Module: combinator/finite_stochastic_transform.py
### Output marginal (`finite_stochastic_transform.py:78-108`)
- **Computes:** `log P(Y=y) = logsumexp_x[log P(X=x) + log kernel[x,y]]`.
- **Why correct:** exact marginalization over the noisy channel. `logsumexp` over axis 0 stable; row-stochastic kernel validated (`_row_stochastic`). Log-kernel clipped to 1e-300 in the accumulator to avoid `log 0`.
- **Verdict:** OK

---

## Module: combinator/weighted.py
### Weighted wrapper (`weighted.py:91-116, 257-308`)
- **Computes:** `P((x,w)) = p_base(x)` (weight excluded from likelihood); weight scales suff-stat contribution `accumulator.update(x, weight*w)`.
- **Why correct:** matches the documented contract; sampler emits neutral weight 1.0. Backend stacked path scales posterior weights by the obs weight (`weights * x[1][:,None]`) consistently.
- **Verdict:** OK

---

## Module: combinator/sequence.py
### iid product + length (`sequence.py:220-279, 1182-1193`)
- **Computes:** `log P(x) = Σ_i log p_base(x_i) + log p_len(len(x))`; optional `len_normalized` divides the entry sum by `len`.
- **Why correct:** iid product times length factor. Numpy/scalar parity verified: encoder stores `icnt=1/len` (`:1186`) so seq `ll_sum*icnt` equals scalar `rv/len`. `-inf` from any entry propagates correctly (summation).
- **Verdict:** OK

---

## Module: combinator/composite.py
### Independent-component product (`composite.py:171-210`)
- **Computes:** `log P(x) = Σ_k log p_k(x_k)`.
- **Why correct:** independent product. `-inf` propagates. Note `seq_log_density` mutates `dists[0]`'s returned array in-place via `+=` (`:205-208`) — relies on children returning fresh arrays (pysp convention; holds). LOW/none.
- **Verdict:** OK

---

## Module: combinator/record.py
### Named-field product (`record.py:123-140`)
- **Computes:** `log P(x) = Σ_field log p_field(x[source])`; independent fields keyed by record source.
- **Why correct:** same product rule as composite, keyed access. OK.
- **Verdict:** OK

---

## Module: combinator/conditional.py
### Joint factorization (`conditional.py:273-333`)
- **Computes:** `log P(x) = log P_cond(x1|x0) + log P_given(x0)`; missing key w/o default → `-inf`.
- **Why correct:** conditional×marginal = joint. `-inf` handling correct (`x0` not in dmap, no default).
- **Verdict:** OK

---

## Module: combinator/select.py
### Deterministic routing (`select.py:111-171`)
- **Computes:** route `x` to `dists[choice(x)]`, score by that child only.
- **Why correct:** choice is a deterministic partition of the sample space, so no mixture normalization is needed; each region scored by its own child. Grouped seq path scatters by original index. OK.
- **Verdict:** OK

---

## Module: combinator/optional.py
### Missing-value mixture (`optional.py:201-250`)
- **Computes:** missing→`log(p)`, observed→`log(1-p)+log p_base(x)`. `log_pn=log1p(-p)`; `p=1`→`log_pn=-inf` (degenerate, only hit on observed branch).
- **Why correct:** point mass `p` at missing + scaled base. Conjugate-Beta `expected_log_density` uses digamma terms correctly. NaN-missing branch handled.
- **Numerical stability:** `log1p` used. **Dead variable `self.log1_p = np.log1p(self.p)` (`:88`) = log(1+p)**, assigned, never read; likely a typo for `log_pn`. FINDING(C4) (LOW, dead code).
- **Verdict:** FINDING(C4)

---

## Module: combinator/ignored.py
### Pass-through (`ignored.py:95-116`)
- **Computes:** delegates `log_density` to wrapped dist (the wrapper marks the field ignored elsewhere).
- **Verdict:** OK

---

## Module: combinator/null_dist.py
### Neutral element (`null_dist.py:82-106`)
- **Computes:** `log_density ≡ 0` (density 1) — multiplicative identity for product composition (used as the `len_dist`/`given_dist`/default sentinel).
- **Why correct:** correct neutral element; `seq` returns a length-matched zero vector.
- **Verdict:** OK

---

## Findings summary
- **C1** truncated.py:66 — forbidden-set `Z=1-Σexp` cancels when forbidden mass≈1 (MEDIUM).
- **C2** censored.py:73-82 — `F(b)-F(a)` tail underflow → spurious `-inf` (MEDIUM).
- **C3** survival.py:62 — `log1p(-cdf)` tail underflow → spurious `-inf` (MEDIUM).
- **C4** optional.py:88 — dead `log1_p` variable, wrong formula (LOW).
