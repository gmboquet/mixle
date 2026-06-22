# Count / integer leaf distributions — computation ledger

All scalar `log_density` formulas were cross-checked numerically against `scipy.stats`
(poisson, binom, nbinom, bernoulli, geom, betabinom, skellam, logser) and agree to ~1e-15.
The engine-neutral `backend_seq_log_density` paths were checked against the numpy
`seq_log_density` and agree to machine epsilon. Estimators were checked to recover true
parameters from large samples (Poisson/Binomial/NB/Geometric/Skellam/BetaBinomial/LogSeries).

## Module: pysp/stats/leaf/bernoulli.py
### log_density (`bernoulli.py:161`)
- **Computes:** log p for 1/True, log(1-p) for 0/False.
- **How:** `_as_bool` maps {0,1,bool}→bool, returns cached `log_p`/`log_1p`.
- **Why correct:** definitional; matches `scipy.bernoulli.logpmf` (−0.3567 / −1.2040 at p=0.3).
- **Numerical stability:** `log1p(-p)` for log(1-p); cached at init. Guards p∈(0,1).
- **Engine-swap:** `backend_log_density_from_params` uses `x>=0.5` threshold on numeric x — neutral; matches numpy path.
- **Verdict:** OK

### Estimator M-step (`bernoulli.py:371`)
- **Computes:** p̂ = (Σwx + pc·p0)/(Σw + pc); conjugate Beta MAP = posterior mode.
- **Why correct:** Bernoulli MLE is the weighted success rate; Beta(a,b) posterior mode (a'−1)/(a'+b'−2) standard.
- **Numerical stability:** clips p to [1e-12, 1−1e-12]; count==0 → 0.5 fallback.
- **Verdict:** OK

### Sufficient statistics (`bernoulli.py:294`)
- value()=(count, sum) matches declaration statistics (count, sum) and legacy suff-stat row order. **Verdict:** OK

## Module: pysp/stats/leaf/binomial.py
### log_density (`binomial.py:273`)
- **Computes:** log C(n,k) + k·log p + (n−k)·log(1−p), k=x−min_val, support [0,n].
- **How:** gammaln binomial coefficient + cached log_p/log_1p; out-of-support → −inf.
- **Why correct:** matches `scipy.binom.logpmf` exactly; k=11 (>n) → −inf.
- **Numerical stability:** gammaln; log1p(-p). OK.
- **Engine-swap:** `backend_log_density_from_params` recomputes via `engine.gammaln`/`engine.log` — neutral, matches numpy to ~1e-15.
- **Verdict:** OK

### seq_log_density (`binomial.py:297`)
- Uses unique-value compression (ux, ix); evaluates only on `good` mask, scatters −inf elsewhere, then re-expands by ix. Correct and avoids gammaln(negative). **Verdict:** OK

### Estimator M-step (`binomial.py:964`)
- **Computes:** p̂ = (Σx − min·count)/(count·n) with n = max_val − min_val inferred from data bounds; pseudo_count and Beta-conjugate MAP variants.
- **Why correct:** binomial MLE p = mean/n. Numerically recovers p=0.3 (n fixed=10 via estimator max_val).
- **CAVEAT (by design, not a bug):** when constructed via `.estimator()` with no fixed `max_val`, `n` is inferred as the observed `max_val − min_val`. If the true maximum value `n` is never observed, `n` is underestimated (saw n=9 fit on true n=10 when 10 wasn't drawn). This is the documented data-driven-support behavior shared across the integer leaves; flagged LOW.
- **Verdict:** OK (LOW — see L1)

### Engine-resident accumulation (`binomial.py:703` seq_update_engine)
- Reduces count/sum on the active engine, keeps scalar min/max as host bookkeeping; matches seq_update. **Verdict:** OK

## Module: pysp/stats/leaf/beta_binomial.py
### log_density (`beta_binomial.py:58`)
- **Computes:** log C(n,k) + betaln(k+a, n−k+b) − betaln(a,b), k∈{0,…,n}.
- **Why correct:** Beta-binomial compound pmf; matches `scipy.betabinom.logpmf` to ~1e-15.
- **Numerical stability:** scipy `betaln`/`gammaln`; cached `_log_beta_ab`. OK.
- **Verdict:** OK

### seq_log_density (`beta_binomial.py:66`)
- Computes `betaln(k+a, n−k+b)` unconditionally then masks `(k<0)|(k>n)` → −inf via np.where. For k>n, `n−k+b` can be ≤0 making betaln return inf/nan, but np.where discards it; verified no NaN leakage on [−1,0,3,5,6,8]. **Verdict:** OK

### Estimator (method of moments) (`beta_binomial.py:185`)
- **Computes:** π=mean/n (→a/(a+b)); intra-class corr ρ=(var/binom_var−1)/(n−1); s=a+b=1/ρ−1; a=πs, b=(1−π)s.
- **Why correct:** standard beta-binomial MoM (var inflation factor 1+(n−1)ρ). Recovers (a,b)=(2,5) from n=20 sample.
- **Numerical stability:** clamps π to (1e-12,1−1e-12), ρ and concentration s to [min_conc,max_conc]; n=1 or non-overdispersed → binomial limit (s=max_conc). All-zero data → degenerate a≈1e-4/b≈1e8 (no crash).
- **Note:** n is a fixed known parameter, not estimated (documented). **Verdict:** OK

## Module: pysp/stats/leaf/negative_binomial.py
### log_density (`negative_binomial.py:167`)
- **Computes:** lgamma(x+r) − lgamma(r) − lgamma(x+1) + r·log p + x·log(1−p), failures-before-r-successes parameterization.
- **Why correct:** matches `scipy.nbinom.logpmf(x, r, p)` to ~1e-15.
- **Numerical stability:** gammaln; cached log_p/log_1p/log_gamma_r. OK.
- **Engine-swap:** `backend_log_density_from_params` recomputes via engine gammaln/log — neutral, exact match to numpy.
- **Verdict:** OK

### Sufficient statistics / histogram (`negative_binomial.py:296`, `:91`)
- value()=(count, sum, histogram). The dispersion r has no finite suff-stat; the weighted count histogram is accumulated for the r-solve. Declaration declares `StatisticSpec("histogram", kind="histogram")` and `backend_legacy_sufficient_statistics` returns the per-row (1, x, x) triple so the histogram reducer can fold counts. `resident_accumulation_supported()` returns False when estimate_r (the histogram cannot be a fixed-width resident stat). Consistent — this is the recently-fixed path; sanity-checked: encoder rejects negatives, seq_update builds the weighted histogram via np.add.at.
- **Verdict:** OK

### Dispersion solve (`negative_binomial.py:420` estimate_dispersion)
- **Computes:** profiles p=r/(r+xbar), bisects the score g(r)=Σ h(k)[ψ(k+r)−ψ(r)] − N·log(1+xbar/r).
- **Why correct:** standard NB MLE score for r with p profiled out; g strictly decreasing. Recovered r=3.0, p=0.4 from 2e5 samples.
- **Numerical stability:** all-zero data (xbar≤0, r unidentified) → keep r_init; var≤xbar (no overdispersion) → r=_MAX_NB_SHAPE (Poisson limit); bracket-expands hi, 200-iter bisection with relative tol. Clamped to [1e-8, 1e7].
- **Verdict:** OK

### Estimator p closed form (`negative_binomial.py:471`)
- p̂ = r·count/(r·count + Σx); pseudo_count regularizes toward prior_p. Correct given r. **Verdict:** OK

## Module: pysp/stats/leaf/geometric.py
### log_density (`geometric.py:192`)
- **Computes:** (k−1)·log(1−p) + log p, support k≥1; p==1 → 0 at k=1 else −inf.
- **Why correct:** matches `scipy.geom.logpmf` to ~1e-15. p==1 boundary handled explicitly.
- **Numerical stability:** log1p(-p) cached; p==1 special-cased in scalar and seq paths.
- **Engine-swap:** backend path handles p==1 via nested where — neutral, matches numpy.
- **Verdict:** OK

### Estimator M-step (`geometric.py:673`)
- **Computes:** p̂ = count/sum = 1/x̄ (geometric MLE on support {1,2,…}); pseudo_count regularizer; Beta-conjugate MAP.
- **Why correct:** geometric mean is 1/p so p=count/Σx. Recovered p=0.25.
- **Numerical stability:** clips p to [1e-12, 1.0] — **upper bound is 1.0 inclusive**. When all data == 1 (count==sum) p̂=1.0, and `GeometricDistribution.__init__` then evaluates `np.log1p(-1.0)` → −inf raising a `RuntimeWarning: divide by zero`. The resulting model is mathematically correct (degenerate atom at 1, log_density handles p==1), but emits a spurious warning. Flagged MEDIUM (M1).
- **Verdict:** FINDING(M1)

### Conjugate posterior (`geometric.py:651`)
- Beta(a+count, b+Σx−count); posterior mode with boundary clamping. Standard. **Verdict:** OK

## Module: pysp/stats/leaf/skellam.py
### log_density (`skellam.py:93`)
- **Computes:** −(√μ1−√μ2)² + (k/2)·log(μ1/μ2) + log I_{|k|}(2√(μ1μ2)).
- **How:** uses `ive(v,z)=I_v(z)e^{−z}`; `log I_v = log(ive)+z` and the +z cancels into −(μ1+μ2)+z = −(√μ1−√μ2)², which is precomputed. Bessel·0 → −inf.
- **Why correct:** matches `scipy.skellam.logpmf` to ~1e-15 across k∈{−3..5} and large k (400 → −inf).
- **Numerical stability:** exponentially-scaled Bessel avoids overflow for large z; `errstate(divide=ignore)` in seq path turns log(0)→−inf cleanly. Support is all integers, so no spurious −inf for valid ints.
- **Engine-swap:** scipy-only (no backend hooks / no exp-family declaration) — host-only by design; not engine-ready. OK for this family.
- **Verdict:** OK

### Estimator (method of moments) (`skellam.py:238`)
- **Computes:** μ1=(v+m)/2, μ2=(v−m)/2 from sample mean m and var v (E[K]=μ1−μ2, Var=μ1+μ2).
- **Why correct:** exact MoM inversion; recovered (3,1) from 5e4 samples.
- **Numerical stability:** floors var to |m|+ε so both rates stay positive; clamps each μ to ε.
- **Numerical note:** var = Σx²/N − m² is the catastrophic-cancellation form, but for integer Skellam data with moderate rates this is benign; the var<|m| floor masks the worst case. Acceptable.
- **Verdict:** OK

## Module: pysp/stats/leaf/logseries.py
### log_density (`logseries.py:164`)
- **Computes:** k·log p − log k − log(−log(1−p)), k≥1.
- **Why correct:** matches `scipy.logser.logpmf` to ~1e-15.
- **Numerical stability:** normalizer log(−log1p(−p)) cached; log1p(-p) for 1−p. OK.
- **Engine-swap:** linear-in-(k, log k) backend, fixed base measure −log k; neutral, exact numpy match.
- **Verdict:** OK

### Estimator (mean inversion) (`logseries.py:327`, `:44`)
- **Computes:** mean = Σx/Σw, then `_solve_p` bisects the monotone mean(p)=−p/((1−p)log(1−p)).
- **Why correct:** log-series MLE = MoM (mean is a sufficient statistic for the 1-param exp family); recovered p=0.7 from 2e5 samples.
- **Numerical stability:** mean≤1 → p=_MIN_P; 200-iter bisection on (1e-12, 1−1e-12). OK.
- **Verdict:** OK

## Module: pysp/stats/leaf/point_mass.py
### log_density / seq_log_density (`point_mass.py:67`, `:71`)
- **Computes:** 0 if x == fixed atom else −inf.
- **How:** scalar uses `_same_value` (np.array_equal / freeze-hash / ==); seq path encodes data to a **boolean equality mask** and `np.where(mask, 0, −inf)`.
- **Why correct:** the encoder produces a bool mask, so `np.where(x,…)` keys on the mask not the raw value — verified correct even for atom value 0 (encode([0,0,1,5]) → [T,T,F,F] → [0,0,−inf,−inf]) and string atoms.
- **Numerical stability:** n/a.
- **Engine-swap:** `backend_seq_log_density` consumes the same bool flags; stacked broadcast adds zeros — neutral.
- **Verdict:** OK

### Accumulator / estimator (`point_mass.py:158`, `:214`)
- No free parameters: accumulation is a no-op on every engine; estimate returns the fixed atom. Consistent with empty `statistics=()` declaration. **Verdict:** OK

---
## Findings summary
- **M1** — `geometric.py:708` + `:129`: p clipped to upper bound 1.0; all-ones data yields p=1.0 and `np.log1p(-1.0)` emits a `divide by zero` RuntimeWarning at construction. Result is correct; warning is noise.
- **L1** — `binomial.py:1004`: trial count `n` inferred as observed `max_val−min_val`; underestimates `n` when the true maximum is never observed (documented data-driven-support behavior, shared across integer leaves).
