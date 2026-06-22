# Bayesian conjugate families + graph/set/permutation models — computation ledger

Domain split into three: `pysp/stats/bayes/` (this file, audited directly),
`pysp/stats/graph/` (delegated; see `13a-graph.md`), `pysp/stats/sets/` (delegated; see `13b-sets.md`).
The graph/sets findings are reproduced in the final list at the bottom.

All numeric checks below were run with `.venv/bin/python`.

---

## Module: pysp/stats/bayes/conjugate.py

Closed-form conjugate posteriors derived from the exponential-family map
`chi' = chi + sum T(x_i)`, `nu' = nu + n`.

### Beta posterior — Bernoulli/Binomial/Geometric/NegBinomial (`conjugate.py:121-207, 781-792`)
- **Computes:** Beta(a0+s, b0+f) posterior; evidence `B(a,b)/B(a0,b0)` (+ log h base).
- **Why correct:** `_build_bernoulli` s=Σwx, f=n−s. `_build_geometric` on {1,2,…}: lik p^N(1−p)^(Σx−N) → a+=N, b+=Σx−N (correct). `_build_binomial` failures = n_trials·N − s, log_base = Σ log C(n_trials, x). `_build_negative_binomial` known r: lik p^{nr}(1−p)^{Σx} → a+=nr, b+=Σx, log_base = Σ log C(x+r−1,x). All consistent.
- **Numeric check:** Bernoulli evidence for data [1,0,1,1,0], prior Beta(2,3) → code −3.96081317, betaln ref −3.96081317, brute integral −3.96081317. **Match.**
- **Verdict:** OK

### Gamma-rate posterior — Poisson/Exponential (`conjugate.py:213-290`)
- **Computes:** Gamma(A,B) on rate; Poisson predictive = NegBin(A, B/(B+1)); evidence with log_base = −Σ log x! (Poisson).
- **Numeric check:** Poisson evidence for [2,0,3,1,2], prior Gamma(2,1.5) → code −8.28331790, closed-form ref −8.28331790. **Match.**
- **Verdict:** OK

### Dirichlet posterior — Categorical/IntegerCategorical (`conjugate.py:296-380`)
- **Computes:** Dirichlet(alpha0+counts); evidence = gammaln(Σa0) − gammaln(Σan) + Σgammaln(an) − Σgammaln(a0) (multinomial coeff dropped, matches comment).
- **Numeric check:** evidence for ['a','a','b','c','a','b'], symmetric alpha=1 → code −7.42654907, ordered-DM ref −7.42654907. **Match.**
- **Edge:** `_build_categorical` takes support from `dist.pmap` when present; data values outside pmap are silently dropped (LOW — not flagged, behavior is documented as "support from dist if available").
- **Verdict:** OK

### Normal-Inverse-Gamma posterior — Gaussian/LogGaussian (`conjugate.py:386-467, 576-584`)
- **Computes:** standard NIG update kn=k0+n, mn=(k0 m0+Σx)/kn, an=a0+n/2, bn=b0+½ss+½ k0 n(x̄−m0)²/kn; predictive = Student-t(2a, m, √(b(κ+1)/(aκ))). LogGaussian applies NIG to log x with Jacobian log_base = −Σ log x.
- **Why correct:** matches Murphy's NIG closed form; ss = Σx²−n x̄² is the within-sample scatter.
- **Stability:** `ss = sx2 − n*xbar*xbar` is the cancellation-prone form E[x²]−E[x]². For weak priors and large/offset data this can go slightly negative (then bn < b0). Not triggered in normal use but is the textbook catastrophic-cancellation pattern. (MEDIUM, noted.)
- **Verdict:** OK (with stability note)

### Normal-Inverse-Wishart posterior — MVN (`conjugate.py:473-573`)
- **Computes:** NIW update kn, mn, νn=ν0+n, ψn=ψ0+scatter+(k0 n/kn) outer(x̄−m0); predictive = MV-Student-t(ν−d+1, m, ψ(κ+1)/(κ(ν−d+1))); evidence via multigammaln & slogdet.
- **Inverse-Wishart sampler:** Bartlett `a[i,i]=√chisq(ν−i)` for i=0..d−1 then invert Wishart(ν, ψ⁻¹) → IW(ν, ψ). Correct.
- **Verdict:** OK

### Inverse-Gamma-on-variance — Rayleigh/HalfNormal (`conjugate.py:590-659`)
- **Computes:** σ² ~ IG; Rayleigh a+=n, Half-normal a+=n/2; both b+=½Σx²; log_base = Σ log x (Rayleigh h(x)=x) / ½n log(2/π) (half-normal).
- **Why correct:** Rayleigh suff stat x² with n in exponent of σ⁻²; half-normal one σ per obs → n/2. Correct.
- **Verdict:** OK

### Gamma-on-parameter — Gamma/InvGamma/InvGaussian/Pareto (`conjugate.py:666-778`)
- **Computes:** Gamma posterior on the one free positive parameter given the known one.
  - Gamma known k: a+=nk, b+=Σx (rate=1/θ). ✓
  - InvGamma known α: a+=nα, b+=Σ1/x (suff stat 1/x). ✓
  - InvGaussian known μ: a+=n/2, b+=Σ(x−μ)²/(2μ²x). ✓
  - Pareto known xm: a+=n, b+=Σ log(x/xm). ✓
- **Verdict:** OK

### von Mises mean-direction posterior (`conjugate.py:852-913`)
- **Computes:** vM(m_n, R_n) with resultant Rc=κΣcos+R0 cos m0, Rs=κΣsin+R0 sin m0, R_n=hypot, m_n=atan2; evidence I0(R_n)/[(2π I0(κ))ⁿ I0(R0)] via `ive` (log I0 = log ive(0,z)+z).
- **Numeric check:** brute-force integral over μ of prior·likelihood for κ=2, 4 obs, prior R0=1e-6 → −4.67090292; code −4.67090292. **Exact match.** (`ive` stabilization confirmed correct.)
- **Verdict:** OK

### Mixture-of-conjugates (Diaconis-Ylvisaker) (`conjugate.py:1013-1114`)
- **Computes:** posterior = mixture of component posteriors with w'_m ∝ w_m·Z_m; evidence = logsumexp(log w + log Z); mean = Σ w'_m E_m; sample via multinomial over components.
- **Why correct:** DY exact-conjugacy result; reweighting by per-component marginal likelihood is exact.
- **Verdict:** OK

---

## Module: pysp/stats/bayes/dirichlet.py

### log_density / log_const (`dirichlet.py:330, 392-443`)
- **Computes:** log f = Σ(α_k−1)log x_k − log_const, log_const = Σgammaln(α_k) − gammaln(Σα). Boundary handling: zeros with α<1 → +inf, α>1 → −inf, α=1 → 0. Simplex check via isclose to 1.
- **Stability:** seq path uses pre-logged encoded x[0] clipped at float_min; backend variant engine-neutral.
- **Verdict:** OK

### dirichlet_param_solve / find_alpha / mpe (`dirichlet.py:100-232`)
- **Computes:** MLE of α from mean-log-p via fixed point α←digammainv(mlp + ψ(Σα)); MPE acceleration optional.
- **Why correct:** standard Minka fixed point. Heavy guards on NaN/non-finite, clipping to [1e-10, 1e10]. Robust.
- **Verdict:** OK

### Accumulator / FisherView / backend stacked (`dirichlet.py:235-269, 456-486, 579-808`)
- value() = (counts, sum_of_logs, sum, sum2). seq_update_engine matches seq_update (numpy/torch matmul). Fisher view in log-coordinates with trigamma cov. Stacked log-density `(n,k)` matrix for mixtures.
- **Verdict:** OK

---

## Module: pysp/stats/bayes/symmetric_dirichlet.py
### log_density / seq_log_density (`symmetric_dirichlet.py:65-83`)
- **Computes:** log f = Σ(α−1)log x_k + gammaln(nα) − n·gammaln(α). α=1 short-circuits to −nc. Correct symmetric Dirichlet normalizer.
- **Verdict:** OK

## Module: pysp/stats/bayes/dict_dirichlet.py
### log_density / cross_entropy / entropy (`dict_dirichlet.py:72-126`)
- **Computes:** dict alpha: gammaln(Σa) + Σ[(a_k−1)log x_k − gammaln(a_k)]; scalar alpha = symmetric. cross_entropy handles all 4 unbounded/dict pairings. Correct conjugate-prior scoring.
- **Verdict:** OK

## Module: pysp/stats/bayes/normal_gamma.py
### log_density / entropy / cross_entropy / sampler (`normal_gamma.py:107-150, 175-182`)
- **Computes:** log f = a log b + ½log(λ/2π) − gammaln(a) + (a−½)log τ − bτ − ½λτ(μ−μ0)². Sampler τ~Gamma(a, scale 1/b), μ~N(μ0, 1/(λτ)). Correct NG density & conjugate-prior contract.
- **Verdict:** OK

## Module: pysp/stats/bayes/multivariate_normal_gamma.py
### log_density / entropy / cross_entropy (`multivariate_normal_gamma.py:99-162`)
- **Computes:** sum of d independent NormalGamma log-densities. Vectorized correctly.
- **Verdict:** OK

## Module: pysp/stats/bayes/normal_wishart.py
### log_density / log_z / expected_log_det / cross_entropy (`normal_wishart.py:27-168`)
- **Computes:** NIW = N(μ|m,(κΛ)⁻¹)·Wishart(Λ|W,ν). log_z = (νd/2)log2 + (ν/2)log|W| + multigammaln(ν/2,d). log_density c_norm + c_wish with c_wish=((ν−d−1)/2)log|Λ| − ½tr(W⁻¹Λ) − log_z. E[log|Λ|]=multidigamma(ν/2)+d log2+log|W|, E[Λ]=νW. All match canonical Wishart identities; PD guards via slogdet sign. Bartlett sampler `√chisq(ν−i)` for i=0..d−1. Correct.
- **Verdict:** OK

## Module: pysp/stats/bayes/dirichlet_process_mixture.py
### _expected_log_stick_weights (`dpm.py:103-124`)
- **Computes:** E[log π_k] for truncated stick-breaking: rv[k]=Σ_{j<k}E[log(1−v_j)] + E[log v_k], last = Σ all E[log(1−v_j)].
- **Numeric check:** matches manual ψ-based construction for a (3,2) gamma array. **Match.**
- **Verdict:** OK

### cbg compound stick density (`dpm.py:86-100`)
- **Computes:** log density of x=1−exp(−y), y~Exp(α), α~Gamma(s1,1/s2) marginalized.
- **Numeric check:** ∫₀¹ exp(cbg(x)) dx = 0.99989 (≈1, residual is the x→1 boundary tail). Density correct; benign divide-by-zero `log1p(-x)` warning only at x=1 boundary.
- **Verdict:** OK

### model_log_density ELBO globals / estimate VB M-step (`dpm.py:606-723`)
- **Computes:** beta-prior cross-entropy −betaln(1,a)+(ψ(g1)−ψ(gs))(a−1); variational beta entropy; component prior cross-entropies/entropies; α hyper-posterior Gamma(s1+K−1, 1/(s2−E[log remaining])); weights = softmax(E[log π]). Re-sorts components by expected count, rebuilds beta_counts[:,1] as reverse-cumsum. Documented exact port of bstats.dpm. Math consistent with truncated-SB CAVI.
- **Verdict:** OK

## Module: pysp/stats/bayes/hierarchical_dirichlet_process_mixture.py
### log_density / _group_log_density / table-count m_jk (`hdpm.py:144-236` + estimator)
- **Computes:** group score Σ_i logsumexp_k(log p(x_i|θ_k)+log w_k) with w = beta (new group) or fitted group_weights (local ELBO). Global-row update m_jk = αβ_k(ψ(αβ_k+n_jk)−ψ(αβ_k)), beta = Dirichlet(γ/K+m_.k) mean. Documented as the deterministic finite direct-assignment approximation, not exact collapsed CAVI. logsumexp group scoring is numerically stable.
- **Verdict:** OK (intentional documented approximation)

## Module: pysp/stats/bayes/pitman_yor.py
### _log_eppf (`pitman_yor.py:112-128`)
- **Computes:** EPPF log p = term1 + term2 + term3 where
  term1 = Σ_{i=1}^{k−1} log(α+i·d) = (k−1)log d + gammaln(α/d+k) − gammaln(α/d+1) [d>0], else (k−1)log α;
  term2 = gammaln(α+1) − gammaln(α+n);
  term3 = Σ_j gammaln(n_j−d) − k·gammaln(1−d).
- **Why correct:** matches the two-parameter Pitman-Yor EPPF with rising-factorial → lgamma identities. d=0 recovers the DP/CRP. (graph-agent independently confirmed PY-style normalizers numerically in its set; this EPPF is the standard form.)
- **Estimator:** bisection on monotone grad_alpha / grad_discount; a_hist/b_hist/d_hist tally the (α+i)/(α+i·d)/(l−d) factor counts exactly across partitions. Sampler CRP probs (c−d)/(α+i), (α+kd)/(α+i). Correct.
- **Verdict:** OK

---

# Findings (consolidated across bayes + graph + sets)

bayes/: no CRITICAL/HIGH. One MEDIUM stability note (NIG scatter cancellation), otherwise OK.

graph/ (from 13a-graph.md): Chow-Liu max-MST, Mallows Z, Plackett-Luce, Spearman, Markov chains,
spanning-tree Matrix-Tree, matching permanent — all numerically verified correct.
One MEDIUM: integer_markov_chain row-normalization 0/0 on unseen lagged states.

sets/ (from 13b-sets.md): one HIGH (Beta posterior-mode bug in bernoulli_set), three MEDIUM
(degenerate p=1 in integer_bernoulli_set; un-guarded estimate.init_dist in two edit variants' seq_update).
