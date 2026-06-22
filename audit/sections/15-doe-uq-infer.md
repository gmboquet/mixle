# Design-of-experiments, UQ & inference backends — computation ledger

Scope: `pysp/doe/{designs,optimal,bayesopt,constrained,multiobjective,optimizer}.py`,
`pysp/uq/{propagate,sensitivity,calibration}.py`, `pysp/infer/{backends,diagnostics}.py`.

All four headline formulas were numerically cross-checked with `.venv/bin/python`:
- **EI** vs `scipy.stats.norm` closed form *and* 2M-sample Monte Carlo (match to 4 sig figs).
- **R-hat** vs hand-computed Gelman-Rubin (`0.99981252` identical); ~1.0 for independent chains, `5.16` for deliberately unmixed chains.
- **ESS** iid ≈ 4858/5000, AR(1) ρ=0.8 ≈ 2165 vs analytic 2222.
- **Sobol S1/ST** vs the Ishigami analytic indices (match to 4 sig figs incl. the S3=0 / ST3≈0.244 interaction term).
- **Unscented transform** exact (to machine precision) for an affine map's mean and covariance.

No CRITICAL/HIGH issues found. A handful of MEDIUM/LOW edge-case and stability notes below.

---

## Module: pysp/doe/designs.py

### Latin hypercube (`designs.py:56`)
- **Computes:** stratified LHS, `unit[:,j] = (perm + offset)/n`, offset 0.5 (center) or U(0,1).
- **Why correct:** one sample per stratum per axis with independent per-axis permutations — the defining LHS property. OK.
- **Numerical stability:** n/a. **Engine-swap:** host numpy DoE utility, not on any engine path. **Verdict:** OK

### maximin LHS (`designs.py:80`)
- **Computes:** best-of-`trials` LHS by max-min squared Euclidean distance on bound-normalized coords.
- **Why correct:** `np.min(sq[triu])` is the min pairwise squared distance; maximizing it is maximin. Squared vs unsquared does not change the argmax. OK.
- **Edge:** `n<2` short-circuits to the first design (no pairwise distance) — correct. **Verdict:** OK

### Sobol / Halton (`designs.py:125`, `:142`)
- **Computes:** scipy `qmc.Sobol`/`Halton` with Owen scrambling, scaled into bounds.
- **Why correct:** delegates to scipy qmc; seeds the engine from an int drawn off the RandomState for reproducibility. OK.
- **Note (LOW):** Sobol with non-power-of-two `n` triggers scipy's balance UserWarning; documented in the docstring, not silenced. `optimal_design` rounds its pool up to a power of two, but `sobol_design` itself does not. Acceptable.
- **Verdict:** OK

### full_factorial (`designs.py:159`) — OK (linspace grid, midpoint for 1-level axes, row-major meshgrid).

---

## Module: pysp/doe/optimal.py

### d/a/i criteria (`optimal.py:59`, `:65`, `:74`)
- **Computes:** D = `log det M` via `slogdet` (−inf if `sign<=0`); A = `−trace(M⁻¹)`; I = `−mean(diag(ref M⁻¹ refᵀ))` via `einsum("ij,jk,ik->i", ref, inv, ref)`.
- **Why correct:** standard alphabetic-optimality merits; `slogdet` is the stable log-det. The I-optimal einsum computes `gᵀ M⁻¹ g` per reference row — verified index pattern. OK.
- **Numerical stability (MEDIUM):** A- and I-optimality call `np.linalg.inv(info)` directly. For near-singular `M` this is less stable than solving, but a `LinAlgError` is caught → −inf, and D-optimality's `slogdet<=0` guard already rejects singular subsets. The Fedorov loop also enforces `n >= p`. Practically safe; a `solve`/pseudo-inverse would be marginally more robust. **Verdict:** OK (note)

### Fedorov exchange (`optimal.py:129`)
- **Computes:** greedy single in/out swap maximizing the criterion, `best_gain` threshold `1e-10`, multi-restart.
- **Why correct:** classic modified-Fedorov; rebuilds `Mᵢ = Fᵀ F` per trial (O(p²) per candidate but correct). The `1e-10` floor prevents infinite cycling on flat plateaus. OK. **Verdict:** OK

---

## Module: pysp/doe/bayesopt.py

### expected_improvement (`bayesopt.py:30`)
- **Computes:** EI = `I·Φ(z) + σ·φ(z)`, `I = best−mean−ξ` (min) or `mean−best−ξ` (max), `z=I/σ`; `σ→0` ⇒ EI=0; clipped `max(EI,0)`.
- **Why correct:** standard closed-form EI with `Φ=ndtr`, `φ=exp(−z²/2)/√(2π)`. **Numerically verified** against scipy and MC. OK.
- **Numerical stability:** `pos = std > 1e-12` masks the zero-σ divide; `z` pre-zeroed; final `np.maximum(ei,0)` guards round-off negatives. Solid. **Verdict:** OK

### probability_of_improvement (`bayesopt.py:51`)
- **Computes:** `Φ((best−mean−ξ)/σ)` (min) / `Φ((mean−best−ξ)/σ)` (max); deterministic 0/1 at σ=0. OK. **Verdict:** OK

### upper_confidence_bound (`bayesopt.py:71`)
- **Computes:** `mean + κ·std` (max) or `κ·std − mean` (min, = −LCB so larger merit ⇒ lower objective). Sign convention correct for an argmax-merit loop. **Verdict:** OK

### GP posterior std (`bayesopt.py:163`, also constrained `:69`)
- **Computes:** `std = sqrt(clip(diag(cov), 0, None))`.
- **Numerical stability:** the `clip(...,0,None)` is exactly the negative-predictive-variance guard the spec asks for; round-off-negative GP variances can't produce NaN. OK. **Verdict:** OK

### kriging-believer batch (`bayesopt.py:224`) / minimize (`:273`) — OK (fantasize posterior mean, refit, repeat; incumbent = min/max of observed y).

---

## Module: pysp/doe/constrained.py

### probability_of_feasibility (`constrained.py:46`)
- **Computes:** `∏_k Φ(−mean_k/σ_k)`, deterministic `1[mean_k<=0]` at σ_k=0 (Gardner et al. 2014).
- **Why correct:** `P(c_k<=0)=Φ((0−mean)/σ)=Φ(−mean/σ)`. Correct sign. `pos` mask guards σ=0. **Verdict:** OK

### _best_feasible (`constrained.py:73`) — OK (best feasible by masked argmin/argmax; else least-infeasible by `sum(max(c,0))`).

### feasibility-weighted acquisition (`constrained.py:137`) — OK (acq·PF; acq held at 1 until first feasible point).

---

## Module: pysp/doe/multiobjective.py

### pareto_mask (`multiobjective.py:43`)
- **Computes:** non-dominated mask; row i dominated iff some row `<=` on all and `<` on some.
- **Why correct:** standard Pareto dominance. Self-comparison gives `all(<=) & any(<)` = `True & False` = not self-dominated. OK. **Verdict:** OK

### augmented Tchebycheff scalarize (`multiobjective.py:62`)
- **Computes:** `max_m(w·ŷ) + ρ·Σ_m(w·ŷ)` on min-max normalized `ŷ`. ParEGO (Knowles 2006). `span` floored at 1e-12. **Verdict:** OK

### simplex weights (`multiobjective.py:71`) — OK (normalized exponentials = uniform on simplex; `total==0` fallback to uniform).

---

## Module: pysp/uq/propagate.py

### Monte Carlo propagate (`propagate.py:19`) — OK (`multivariate_normal` sample, mean/std/quantiles along axis 0).

### unscented_transform (`propagate.py:63`)
- **Computes:** Van der Merwe scaled UT: `λ=α²(d+κ)−d`, sigma points `mean ± chol((d+λ)Σ)ᵀ`, weights `Wm₀=λ/(d+λ)`, `Wc₀=Wm₀+(1−α²+β)`, others `1/(2(d+λ))`.
- **Why correct:** matches the standard scaled-UT weight set; **verified exact** for an affine map's mean and covariance (machine precision). OK.
- **Numerical stability (MEDIUM, edge):** `chol((d+λ)Σ)`. With the default `α=1e-3, κ=0`, `d+λ = α²(d+κ) = 1e-6·d > 0`, so the scaling stays positive and Cholesky is fine for SPD Σ. But a user passing `κ<0` with small d, or a κ making `d+λ<0`, would make `(d+λ)Σ` non-SPD and crash Cholesky. No guard. Default-safe; flagged as a parameter-misuse edge. **Verdict:** OK (note)
- **Engine-swap:** host numpy UQ helper, not engine-pathed. 

---

## Module: pysp/uq/sensitivity.py

### sobol_indices (`sensitivity.py:35`)
- **Computes:** Saltelli sampling; `var = Var([yA;yB])`; first-order `S1=mean(yB·(yAB−yA))/Var` (Saltelli 2010), total `ST=0.5·mean((yA−yAB)²)/Var` (Jansen). A,B taken as the two halves of a single 2d-dim Sobol block (independence). Clips `S1∈[0,1]`, `ST>=0`.
- **Why correct:** these are the canonical Jansen/Saltelli estimators with correct denominators (total output variance). **Numerically verified** against the Ishigami analytic indices to 4 sig figs. OK.
- **Numerical stability:** `var<=0` (constant output) short-circuits to all-zero indices — avoids 0/0. OK.
- **Note (LOW):** splitting one 2d-Sobol block into A|B is a legitimate way to get two independent low-discrepancy matrices; verified it gives correct indices. **Verdict:** OK

### morris_screening (`sensitivity.py:84`)
- **Computes:** elementary effects on a `levels`-grid, step `Δ = levels/(2(levels−1))`, `μ* = mean|EE|`, `σ = std(EE)`.
- **Why correct:** the `Δ = p/(2(p−1))` step is the standard Morris choice (maps a base level to a distinct level). Base drawn from the lower half `grid[:levels//2+1]` so a `+Δ` step stays in-grid. The `min(x+Δ,1)` clamp plus `if step!=0` guard avoids divide-by-zero on a clamped step. OK.
- **Note (LOW):** clamping at the upper boundary can occasionally yield a step `< Δ` (still a valid finite difference, just smaller); EE is divided by the *actual* step, so the estimate stays unbiased. **Verdict:** OK

---

## Module: pysp/uq/calibration.py

### KO negative log-likelihood (`calibration.py:86`)
- **Computes:** no-discrepancy: `0.5·Σr²/σ² + n·log σ` (Gaussian iid, const dropped). With discrepancy: GP marginal NLL `0.5·rᵀK⁻¹r + Σ log diag(L) + 0.5n·log2π` via Cholesky of `K = RBF + (σ²+1e-8)I`.
- **Why correct:** both are the standard exact forms; `Σ log diag(chol)` = `0.5 log det K`. `LinAlgError`→`1e12` penalty. OK.
- **Numerical stability:** Cholesky-based solve (not explicit inverse); `+1e-8` jitter on the noise diagonal. Lengthscale **fixed** (not fitted) to resolve KO θ/δ identifiability — documented and standard. OK.
- **Note (LOW):** `predict` (`:41`) uses `np.linalg.solve` on `K` (un-jittered beyond `noise²`); fine since noise>0 after fit. **Verdict:** OK

### predict / discrepancy GP (`calibration.py:35`) — OK (η + k_*ᵀ K⁻¹ resid).

---

## Module: pysp/infer/diagnostics.py

### rhat (`diagnostics.py:29`)
- **Computes:** Gelman-Rubin PSRF. `W = mean_chains(var(chain, ddof=1))`; `B = n·var(chain_means, ddof=1)`; `var_hat = (n−1)/n·W + B/n`; `R̂ = sqrt(var_hat/W)`.
- **Why correct:** `B/n = var(chain_means, ddof=1)`, so `var_hat = (n−1)/n·W + var(means)` — the standard estimator. **Numerically verified** identical to a hand computation; ≈1 for mixed chains, `5.16` for unmixed. OK.
- **Numerical stability:** `W==0` (degenerate, all draws equal) returns `R̂=1` instead of 0/0 via the nested `np.where`. `m<2 or n<2` → NaN. Solid. **Verdict:** OK

### ess (`diagnostics.py:56`)
- **Computes:** `ESS = n_total / τ`, `τ = 1 + 2·Σ ρ_lag` truncated at the first non-positive lag (Geyer initial-positive-sequence). Multi-chain: center each chain, pool autocov, `var = mean(centered²)`.
- **Why correct:** standard IPS estimator; verified iid≈N and AR(1) ρ=0.8 matches `N(1−ρ)/(1+ρ)`. `max(1, …)` floor. OK.
- **Numerical stability (LOW):** autocorrelation `ρ_lag = mean(c[:,:-lag]·c[:,lag:]) / var`. The denominator `var` uses the full-length normalization (biased estimator) while the numerator uses `n−lag` terms — this is the standard biased ACF that *under*-weights long lags and improves stability, intentional. The `break` on first non-positive lag is the Geyer truncation. `var==0` dims → `ESS=n_total`. OK.
- **Note (LOW):** when chains have differing means, centering per-chain (not pooled) is correct for ESS (removes between-chain location), consistent with the docstring. **Verdict:** OK

---

## Module: pysp/infer/backends.py

### registry / select_backend (`backends.py:53`, `:72`)
- **Computes:** name→backend registry with availability probes and `target_kind` preference table for `"auto"`.
- **Why correct:** "register, don't branch" dispatcher mirroring `register_kernel_factory`; `auto` resolves by target-kind preference then numpy default. Pure control flow, no numerics. OK.
- **Engine-swap:** this *is* the engine-selection seam; numpy is the dependency-free default, torch/jax/numba gated by `available()`. Correct by design. **Verdict:** OK
- **Note (LOW):** `_returns_torch` always returns `False` (intentionally conservative — won't call a possibly-stateful target); documented. OK

---

## Summary

No wrong acquisition formulas, no negative variances reaching a sqrt/log, no unstable inverse in a normal-use path. EI, PI, UCB, PF, R-hat, ESS, Sobol S1/ST, and the unscented transform are all formula-correct and numerically validated. The only notes are defensive (`inv` vs `solve` in A/I-optimality, unscented Cholesky under adversarial κ) and stylistic; none change results in normal use.
