# Continuous leaf distributions, group B (positive-support & extreme-value) — computation ledger

All densities cross-checked against `scipy.stats` to machine precision (≤ 4.4e-16) and all
estimators verified to recover true parameters on 2e5–3e5 samples. Details inline.

## Module: pysp/stats/leaf/gamma.py
### log_density (`gamma.py:218`, `seq_log_density:241`, `backend:263`)
- **Computes:** log f = -lgamma(k) - k·log θ + (k-1)·log x - x/θ for x>0.
- **How:** `log_const = -(gammaln(k)+k·log θ)` cached; scalar/seq/backend share it.
- **Why correct:** canonical Gamma(shape k, scale θ). scipy diff = 2.2e-16.
- **Numerical stability:** x≤0 / non-finite guarded in scalar, seq, and backend (`np.where`/`engine.where`). `seq_log_density` skips the `(k-1)·log x` term when k==1, avoiding 0·(-inf) at x=0.
- **Engine-swap:** neutral; backend hooks use `engine.*`; stacked params/suff-stats provided.
- **Verdict:** OK

### estimate_shape (`gamma.py:649`)
- **Computes:** MLE shape root of `log k - ψ(k) = s`, `s = log(mean) - mean(log x)`.
- **How:** Robust bracketing/bisection (not digamma/trigamma Newton), with hi-bound doubling, ≤200 iters, clamps `[1e-12, 1e12]`.
- **Why correct:** `s>0` always (Jensen), and `log k - ψ(k)` is monotone decreasing in k, so bisection on `f(k)=log k-ψ(k)-s` is well-posed. Recovered k=3.007 (true 3).
- **Numerical stability:** s≤0 → max shape; degenerate counts → Gamma(1,1) fallback; scale floored at tiny.
- **Verdict:** OK (uses bisection, not the spec's Newton; mathematically equivalent and robust)

## Module: pysp/stats/leaf/exponential.py
### log_density / seq / backend (`exponential.py:189,206,225`)
- **Computes:** log f = -log β - x/β, x≥0.
- **Why correct:** canonical. x<0 → -inf guarded in all three paths.
- **Numerical stability:** out-of-place arithmetic to keep torch autograd graph (`:219`).
- **Conjugate path:** Gamma prior on rate 1/β; `expected_log_density` (`:157`) uses E[η]=-kθ, E[-log(-η)]=ψ(k)+log θ. `_estimate_conjugate` (`:562`) returns Gamma(n,1/s) posterior-mode rate (n-1)/s. Standard Gamma–Exponential conjugacy. OK.
- **Verdict:** OK

## Module: pysp/stats/leaf/beta.py
### log_density / seq / backend (`beta.py:169,179,185`)
- **Computes:** log f = (a-1)log x + (b-1)log(1-x) - B(a,b), x∈(0,1).
- **How:** `log_const = lgamma a + lgamma b - lgamma(a+b)`; `math.log1p(-x)` for log(1-x).
- **Why correct:** scipy diff = 2.2e-16. `log1p` avoids cancellation as x→1.
- **Numerical stability:** scalar guards (0,1); **`seq_log_density` does NOT mask off-support** — it relies on `BetaDataEncoder.seq_encode` rejecting x≤0 or x≥1 (`:417`). Consistent with the encoder contract but the seq path itself is unguarded.
- **Engine-swap:** neutral; uses `engine.betaln`.
- **Verdict:** OK

### estimate (`beta.py:385`)
- **Computes:** MLE shapes solving ψ(a)-ψ(a+b)=mean(log x), ψ(b)-ψ(a+b)=mean(log(1-x)).
- **How:** delegates to `dirichlet_param_solve` (Newton on digamma) with a moment-matched initializer `_moment_initial` (`:372`).
- **Why correct:** Beta is Dirichlet on 2 cells; the log-moment equations are the Dirichlet ones. Recovered a,b = 2.009, 5.026 (true 2, 5).
- **Numerical stability:** `_moment_initial` uses `var = sum_x2/count - mean²` (cancellation-prone) but only as an *initializer*; clipped mean and var≥0 floor; final α floored at 1e-12.
- **Verdict:** OK

## Module: pysp/stats/leaf/inverse_gamma.py
### log_density / seq / backend (`inverse_gamma.py:177,187,112`)
- **Computes:** log f = α log β - lgamma α - (α+1)log x - β/x, x>0.
- **Why correct:** law of 1/Y, Y~Gamma(α,1/β). scipy diff = 2.2e-16.
- **Numerical stability:** scalar guards x≤0; seq path unguarded (encoder validates, `:412`); encoder uses `errstate(divide='ignore')` for log/recip.
- **Verdict:** OK

### _estimate_shape (`inverse_gamma.py:357`)
- **Computes:** Gamma shape on reciprocals y=1/x via Newton on `log k - ψ(k) = s`, Minka initializer, then β = α/E[1/x].
- **Why correct:** if X~InvGamma(α,β) then 1/X~Gamma(α,1/β); fit Gamma to 1/x, map back. Recovered α,β = 3.979, 2.984 (true 4, 3).
- **Numerical stability / FINDING(IG-1):** Newton uses a **local finite-difference `_trigamma`** (`:395`, h=1e-5) instead of the exact `trigamma`, even though `trigamma` is importable from `pysp.utils.special` (the module already imports `digamma, gammaln` from there). Measured rel-err of the FD trigamma ≈ 1e-6 at α=0.01, 1e-8–1e-10 elsewhere — adequate for the Newton step but unnecessarily imprecise and a latent catastrophic-cancellation risk for very small α (subtracting two nearly equal digammas). LOW.
- **Verdict:** FINDING(IG-1) (precision/dead-import nit; result correct)

## Module: pysp/stats/leaf/inverse_gaussian.py
### log_density / seq / backend (`inverse_gaussian.py:160,173,193`)
- **Computes:** log f = 0.5(log λ - log 2π - 3 log x) - λ(x-μ)²/(2μ²x), x>0. seq form expands (x-μ)²/x = x - 2μ + μ²/x.
- **Why correct:** canonical Wald. scipy `invgauss(mu/lam, scale=lam)` diff = 2.2e-16. seq expansion algebraically identical and verified equal to scalar.
- **Numerical stability:** x≤0/non-finite guarded (scalar, seq, backend).
- **Verdict:** OK

### estimate (`inverse_gaussian.py:400`)
- **Computes:** closed-form MLE μ = mean(x), 1/λ = mean(1/x) - 1/μ.
- **Why correct:** exact IG MLE. Recovered μ,λ = 1.498, 2.007 (true 1.5, 2).
- **Numerical stability:** harmonic gap `mean(1/x)-1/μ` floored; non-finite/≤0 → max λ. OK.
- **Verdict:** OK

## Module: pysp/stats/leaf/weibull.py
### log_density / seq / backend (`weibull.py:107,120,136`)
- **Computes:** log f = log k - log s + (k-1)log(x/s) - (x/s)^k, x≥0.
- **Why correct:** canonical weibull_min. scipy diff = 0.0. The x==0 limit is handled explicitly per shape regime (k<1→+inf, k>1→-inf, k=1→-log s) in scalar, seq, backend, and stacked paths; verified seq x=0 with k=0.7 → +inf matching scalar.
- **Numerical stability:** seq uses `errstate(divide,invalid='ignore')` then masks x<0 and x==0; backend guards x==0 only when `not requires_grad(shape)` (torch grad path keeps the finite branch). OK.
- **Verdict:** OK

### estimate / _shape_from_moments (`weibull.py:322,29`)
- **Computes:** moment (CV²-matching) estimator: solve var/mean² = Γ(1+2/k)/Γ(1+1/k)² - 1 for k by bisection, then s = mean/Γ(1+1/k).
- **Why correct:** matches the coefficient of variation; `_weibull_cv2` monotone in k. Recovered k,s = 1.694, 1.997 (true 1.7, 2). (Method of moments, not MLE — a deliberate design choice; documented.)
- **Numerical stability:** `_weibull_cv2` guards overflow via `lgamma` and a `log(finfo.max)` cap (`:24`); shape clamped `[1e-3,1e3]`, scale floored.
- **Verdict:** OK

## Module: pysp/stats/leaf/rayleigh.py
### log_density / seq / backend (`rayleigh.py:101,109,116`)
- **Computes:** log f = log x - log σ² - x²/(2σ²), x≥0.
- **Why correct:** canonical Rayleigh(scale σ). scipy diff = 2.2e-16. exp-family declaration A=log σ², η=-0.5/σ², T=x², base=log x — consistent.
- **Numerical stability:** x<0 → -inf; x==0 → -inf (scalar). seq masks x<0 only (x==0 gives -inf naturally via log 0 → encoder allows 0, log(0)=-inf, fine). OK.
- **Verdict:** OK

### estimate (`rayleigh.py:272`)
- **Computes:** closed-form MLE σ² = mean(x²)/2.
- **Why correct:** exact. Recovered σ = 1.399 (true 1.4). pseudo-count prior adds 2·σ₀² per unit (E[x²]=2σ²) — consistent. OK.
- **Verdict:** OK

## Module: pysp/stats/leaf/pareto.py
### log_density / seq / backend (`pareto.py:110,120,127`)
- **Computes:** log f = log α + α log xm - (α+1)log x, x≥xm.
- **Why correct:** canonical Pareto-I. scipy `pareto(α, scale=xm)` diff = -4.4e-16. Support `x≥xm` masked in all paths.
- **Numerical stability:** base measure h(x)=1/x is scale-dependent (`fixed_base=False` correctly declared, `:59`), so stacked scoring uses backend hooks. OK.
- **Verdict:** OK

### estimate (`pareto.py:326`)
- **Computes:** MLE xm = min(x), α = n / (Σ log x - n log xm).
- **Why correct:** standard Pareto MLE. Recovered xm,α = 1.200, 3.002 (true 1.2, 3). denom floored at min_denom. OK.
- **Verdict:** OK

## Module: pysp/stats/leaf/gumbel.py
### log_density / seq / backend (`gumbel.py:102,111,66`)
- **Computes:** log f = -log β - z - e^{-z}, z=(x-μ)/β, all reals.
- **Why correct:** canonical gumbel_r. scipy diff = 0.0.
- **Numerical stability:** n/a (no support boundary). `e^{-z}` can overflow for very negative z (far-left tail) → +inf → log f = -inf, acceptable.
- **Verdict:** OK

### estimate (`gumbel.py:269`)
- **Computes:** moment estimator β = √(6 var)/π, μ = mean - β·γ_Euler.
- **Why correct:** inverts the Gumbel mean μ+βγ and var (π²/6)β². Recovered μ,β = 0.497, 1.297 (true 0.5, 1.3).
- **Numerical stability:** var = sum2/count - mean² (cancellation-prone but standard), floored ≥0; scale floored. OK.
- **Verdict:** OK

## Module: pysp/stats/leaf/generalized_pareto.py
### log_density / seq (`generalized_pareto.py:69,81`)
- **Computes:** log f = -log σ - (1/ξ+1)log(1+ξy/σ), y=x-μ≥0 (ξ≠0); -log σ - y/σ (ξ≈0). Upper endpoint μ-σ/ξ for ξ<0.
- **Why correct:** canonical GPD. scipy `genpareto(ξ, loc, scale)` diff = 4.4e-16 (ξ>0) and 0.0 (ξ<0, finite upper endpoint correctly excluded).
- **Numerical stability:** `_XI_TOL=1e-8` exponential limit; t≤0 masked; seq uses `errstate(divide,invalid='ignore')`. OK.
- **Verdict:** OK

### sampler (`generalized_pareto.py:125`)
- **Computes:** inverse-CDF y = (σ/ξ)(U^{-ξ}-1), using U (since 1-U is uniform).
- **Why correct:** F^{-1}(1-U) with U~Unif. Correct.
- **Verdict:** OK

### estimate (`generalized_pareto.py:225`)
- **Computes:** method-of-moments ξ = (1 - m²/v)/2, σ = m(1-ξ), m=exceedance mean, v=var.
- **Why correct:** GPD m=σ/(1-ξ), v=σ²/((1-ξ)²(1-2ξ)) ⇒ m²/v=1-2ξ. Recovered σ,ξ = 1.489, 0.205 (true 1.5, 0.2). ξ clamped <1/2 (finite-variance requirement). OK.
- **Verdict:** OK

## Module: pysp/stats/leaf/generalized_extreme_value.py
### log_density / seq (`generalized_extreme_value.py:91,101`)
- **Computes:** log f = -log σ - (1/ξ+1)log s - s^{-1/ξ}, s=1+ξz>0 (ξ≠0); Gumbel form (ξ≈0). EVT sign convention (scipy c=-ξ).
- **Why correct:** scipy `genextreme(-ξ, loc, scale)` diff = -2.2e-16 (ξ>0) and 0.0 (ξ=0). s≤0 masked.
- **Numerical stability:** `errstate` guards; `_XI_TOL` Gumbel branch.
- **Verdict:** OK

### estimate / skewness inversion (`generalized_extreme_value.py:247,39,47`)
- **Computes:** MoM: ξ from monotone skewness(ξ) by bisection, then σ=√var·|ξ|/√(g2-g1²), μ=mean-σ(g1-1)/ξ, g_k=Γ(1-kξ).
- **Why correct:** standard GEV moment formulas; `_gev_skewness` defined for ξ<1/3 (third moment finite), monotone. Recovered μ,σ,ξ = 0.493, 1.223, 0.194 (true 0.5, 1.2, 0.2). ξ clamped [-1, 1/3-1e-4].
- **Numerical stability:** central third moment `sum3/n - 3 mean(sum2/n) + 2 mean³` (cancellation-prone but standard MoM); var floored. OK.
- **Verdict:** OK

## Module: pysp/stats/leaf/tweedie.py
### log_density / series (`tweedie.py:112,49`)
- **Computes:** point mass log P(Y=0)=-λ; for y>0 the compound Poisson-Gamma series
  f(y)=Σ_{n≥1} Pois(n;λ)·Gamma(y; nα, θ) with λ=μ^{2-p}/(φ(2-p)), α=(2-p)/(p-1), θ=φ(p-1)μ^{p-1}.
- **How:** windowed log-sum-exp over n∈[1, n_max] (`:65`), term derivation verified algebraically.
- **Why correct:** series and term algebra verified; scipy-free reference (explicit Pois×Gamma sum) diff ≤ machine precision for y∈{0.1,1,3,7} and matches the *full* (n up to 3e5) reference at y up to 1e6 for default params. Estimators recover μ,φ = 1.998, 0.701 (true 2, 0.7).
- **Numerical stability / FINDING(TW-1):** the upper window bound caps at **20000** (`:59`): `n_max = min(2·n_peak_max + 10√(n_peak_max+1) + 50, 20000)` where `n_peak_max = max(y)/(a·θ)` *over-estimates* the true series mode (true mode grows ~√y, not ~y, so for ordinary λ the cap is harmless and the window always brackets the peak — confirmed). **But** when λ is large (large μ with small φ) the *true* mode can exceed 20000 and the series silently truncates: e.g. μ=1000, φ=0.01, p=1.5, y=1e7 gives true mode n≈2e5 and the code returns log f = -6.307e7 vs the true -6.239e7, an error of ~6.8e5 in log-space. So with heavy mass / small dispersion the density is materially wrong. MEDIUM (edge case; default/moderate-λ use is correct).
- **Numerical stability (overflow):** the log-sum-exp subtracts the per-row max (`:68`), so no `exp` overflow even at y=1e6; values stay finite. Good.
- **Engine-swap:** numpy-only (no backend hooks / declaration). seq path is host numpy. Acceptable for a non-exp-family leaf but note it has no torch/symbolic scoring (unlike the other group-B leaves).
- **Verdict:** FINDING(TW-1)

### estimate (`tweedie.py:268`)
- **Computes:** MoM μ=mean, φ=var/μ^p at fixed p.
- **Why correct:** E[Y]=μ, Var[Y]=φμ^p exact. Recovered correctly. var via sum2/n-mean² (cancellation), φ floored. OK.
- **Verdict:** OK
