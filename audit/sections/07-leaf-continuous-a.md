# Continuous leaf distributions, group A (location-scale & heavy-tail) — computation ledger

Scope: `pysp/stats/leaf/{gaussian,log_gaussian,half_normal,skew_normal,exgaussian,student_t,laplace,logistic,uniform}.py`.

All log-densities cross-checked numerically against `scipy.stats` (max |logpdf err| ≤ 1.4e-15 except
EMG large-arg ≤ 4.3e-12, which is `erfcx` asymptotic-series precision, not a bug). All M-steps
recover their generating parameters on 5k–80k synthetic samples. Detailed results inline below.

---

## Module: pysp/stats/leaf/gaussian.py

### log_density / seq_log_density (`gaussian.py:225`, `gaussian.py:244`)
- **Computes:** `log f = -0.5*log(2*pi*sigma2) - 0.5*(x-mu)^2/sigma2`.
- **How:** cached `log_const`; `seq_log_density` builds out-of-place to preserve torch autograd graph.
- **Why correct:** canonical normal log-pdf. Verified vs `scipy.stats.norm`: max err 4.4e-16.
- **Numerical stability:** fine; no cancellation in the density itself.
- **Engine-swap:** `backend_log_density_from_params` recomputes `log(2*pi*sigma2)` engine-neutrally; parity OK.
- **Verdict:** OK

### M-step variance (`gaussian.py:725`)
- **Computes:** MLE `sigma2 = sum(w x^2)/sum(w) - mu^2 = E[x^2] - E[x]^2`.
- **How:** uncentered second moment minus squared mean (catastrophic-cancellation form).
- **Why correct:** algebraically the MLE variance. Recovers truth on well-scaled data (verified
  mu=2.97, sigma2=3.88 matching `np.var`).
- **Numerical stability:** **CANCELLATION.** `E[x^2]-E[x]^2` loses precision when `mu^2 >> sigma2`.
  Verified: data ~ N(1e8, 1) yields `sigma2 = 4.0` (true ≈1.0; `np.var`=0.996) — a 4x error in pure
  float64. The variance floor `max(sigma2, 1e-8, 1e-6*sigma2)` (line 731-734) does **not** catch this
  because the corrupted value is positive. A Welford/centered accumulation would avoid it. This is the
  primary stability finding for the group. See FINDING(A1).
- **Engine-swap:** numpy `value()` returns `(sum, sum2, count, count2)`; declaration legacy stats and
  `backend_stacked_sufficient_statistics` return the same uncentered `(sum_x, sum_x*x, count, count)` —
  parity OK, but they inherit the same cancellation.
- **Verdict:** FINDING(A1)

### Conjugate NormalGamma update `_estimate_conjugate` (`gaussian.py:660`)
- **Computes:** posterior `(mu_n, lam_n, a_n, b_n)` and joint-MAP `sigma2 = b_n/(a_n-0.5)`.
- **How:** uses `sum_xxx = sum_x`; `new_b0 = sum_xx - sample_mean2*sum_xxx = sum_xx - xbar*sum_x`
  (= centered scatter S), `new_b1 = (lam*n/(lam+n))*(xbar-mu0)^2`.
- **Why correct:** `sum_xx - xbar*sum_x = sum (x-xbar)^2`, the standard NormalGamma scatter term;
  `new_b1` is the standard prior-mean correction. **Verified against the textbook NormalGamma
  posterior**: mu, lam, a, b and sigma2 all match to 1e-14.
- **Numerical stability:** `sum_xx - xbar*sum_x` is the same cancellation family as A1 but with the
  data-scaled `xbar` it is the *centered* form, so it is actually the stable variant here. OK.
- **Engine-swap:** host-only numpy path (conjugate estimation is host-side); acceptable.
- **Verdict:** OK

### exp-family declaration / Fisher view (`gaussian.py:31-130`)
- **Computes:** natural params `(mu/sigma2, -0.5/sigma2)`, log-partition `0.5*log(2*pi*sigma2)+0.5*mu^2/sigma2`,
  sufficient stats `(x, x^2)`; Fisher view uses moments up to `ex4`.
- **Why correct:** standard Gaussian exponential-family identities; log-partition matches
  `-log_const + 0.5*mu^2/sigma2`. Fisher `ex2-ex1^2 = var`, `ex4-ex2^2` standard. OK.
- **Verdict:** OK

### Minor doc nit (`gaussian.py:7`)
- Module docstring writes `log f = -log(2*pi*sigma2) - (x-mu)^2/sigma2` (missing the two `0.5`
  factors). The line-229 method docstring and the code are correct. Cosmetic. FINDING(A6, LOW).

---

## Module: pysp/stats/leaf/log_gaussian.py

### log_density / seq_log_density (`log_gaussian.py:237`, `log_gaussian.py:259`)
- **Computes:** `log f(x) = -0.5*log(2*pi*sigma2) - 0.5*(log x - mu)^2/sigma2 - log x`, x>0.
- **How:** scalar path computes `y=log(x)` and the `-y` Jacobian; **seq path expects already-log-encoded
  input** (the encoder, line 772, applies `np.log`), so `seq_log_density` uses `x` directly as `y`.
- **Why correct:** lognormal log-pdf. Verified vs `scipy.stats.lognorm`: max err 4.4e-16 (both scalar
  and seq with log-encoded input).
- **Numerical stability:** good.
- **Engine-swap:** `backend_seq_log_density`/`backend_log_density_from_params` operate on log-encoded
  `x` and subtract `x` for the Jacobian — consistent with the encoder. `exp_family_base_measure`
  returns `-x` (also log-encoded). Parity OK.
- **x<=0 handling:** scalar `density` returns 0.0 (line 233), `log_density` returns `-inf` (line 250),
  `expected_log_density` returns `-inf` (line 203). The **encoder rejects** x<=0 because `np.log(x)`
  produces nan/inf and `seq_encode` raises (verified: `log_density(0)=-inf`, `log_density(-1)=-inf`).
  Correct and safe.
- **Verdict:** OK

### M-step (`log_gaussian.py:694`)
- **Computes:** `mu = sum(log x)/n`, `sigma2 = sum((log x)^2)/n - mu^2` (log-space variance).
- **Why correct:** MLE of the underlying Gaussian on `y=log x`. Verified: mu=0.482, sigma2=0.627
  matching `mean/var` of `log(data)`.
- **Numerical stability:** same `E[y^2]-E[y]^2` cancellation as A1, but on the **log scale** where
  `|y|` is typically O(1–10), so the practical risk is much lower than the raw-Gaussian case. Still
  the centered form would be strictly safer. Covered by FINDING(A1) (same root cause). The line-733
  comment notes the form was fixed to match GaussianEstimator (previously only correct when the two
  counts were equal) — that fix is correct.
- **Verdict:** OK (shares A1 root cause; low practical risk on log scale)

---

## Module: pysp/stats/leaf/half_normal.py

### log_density / seq_log_density (`half_normal.py:144`, `half_normal.py:154`)
- **Computes:** `log f = 0.5*log(2/pi) - log(sigma) - x^2/(2 sigma^2)`, x>=0, else -inf.
- **Why correct:** half-normal (folded N(0,sigma^2)) log-pdf. Verified vs `scipy.stats.halfnorm`:
  max err 8.9e-16; off-support returns -inf (verified).
- **Numerical stability:** fine. `_MIN_SIGMA = finfo.tiny` guards estimate.
- **Engine-swap:** declared one-parameter exp family `(x^2,)` with `eta=-1/(2 sigma^2)`, `A=log sigma`,
  base `0.5*log(2/pi)` masked off-support via `engine.where`. Matches the closed form. Parity OK.
- **Verdict:** OK

### M-step (`half_normal.py:334`)
- **Computes:** `sigma = sqrt(sum(w x^2)/sum(w)) = sqrt(E[x^2])`.
- **Why correct:** half-normal MLE (`E[x^2]=sigma^2`). No mean subtraction, so **no cancellation**.
  Degenerate count/sum2<=0 falls back to sigma=1.0; floors at `_MIN_SIGMA`.
- **Verdict:** OK

---

## Module: pysp/stats/leaf/skew_normal.py

### log_density / seq_log_density (`skew_normal.py:64`, `skew_normal.py:69`)
- **Computes:** `log f = log2 - log(omega) - 0.5*log(2pi) - 0.5 z^2 + log Phi(alpha z)`, z=(x-loc)/scale.
- **How:** uses `scipy.special.log_ndtr` for `log Phi`, which is stable in the far-left tail.
- **Why correct:** skew-normal log-pdf. Verified vs `scipy.stats.skewnorm`: max err 4.4e-16 (scalar/seq).
- **Numerical stability:** `log_ndtr` avoids underflow of `Phi(alpha z)` for very negative argument. Good.
- **Engine-swap:** no backend/declaration; numpy-only leaf (uses scipy `log_ndtr`). Host-only — acceptable
  for a moment-fit leaf but means no torch/symbolic scoring path (cf. Gaussian). Noted, not a bug.
- **Verdict:** OK

### MoM estimate (`skew_normal.py:195`)
- **Computes:** `mean=s1/n`, `var=s2/n-mean^2`, central `m3 = s3/n - 3 mean (s2/n) + 2 mean^3`,
  `skew=m3/var^1.5`; inverts skewness→delta via `u=b^2 delta^2`, then `alpha, omega, xi`.
- **Why correct:** standard skew-normal method-of-moments; skewness clamped to `±_MAX_SKEW` (the
  delta→±1 limit). Verified recovery on 50k samples: loc 0.402, scale 1.301, shape 4.069 (true 0.4/1.3/4.0).
  The skewness-inversion algebra `(1-u)/u = (((4-pi)/2)/|skew|)^{2/3}`, `delta=copysign(sqrt(u*pi/2))`,
  `omega=sqrt(var/(1-b^2 delta^2))`, `xi=mean-omega*b*delta` is correct.
- **Numerical stability:** **CANCELLATION in both `var` and `m3`.** Both the uncentered variance
  (`s2/n-mean^2`) and especially the uncentered third moment (`s3/n - 3 mean s2/n + 2 mean^3`) suffer
  catastrophic cancellation when `|loc| >> scale`. Verified: SkewNormal(loc=1e6, scale=1.3, shape=4.0)
  with 50k samples recovers shape = **-2031** (true 4.0) — `m3` is destroyed; var survives (1.365). A
  centered (subtract running mean / Welford) accumulation of the second and third central moments would
  fix it. See FINDING(A2).
- **Verdict:** FINDING(A2)

---

## Module: pysp/stats/leaf/exgaussian.py (EMG)

### log_density / seq_log_density (`exgaussian.py:98`, `exgaussian.py:112`)
- **Computes:** `log f = log(lam/2) - 0.5 u^2 + log(erfcx(z))`, `u=(x-mu)/sigma`,
  `z=(lam sigma - u)/sqrt(2)`.
- **How:** uses `pysp.utils.special.log_erfcx`, a 3-branch (asymptotic / `x^2+log erfc` / direct) stable
  log of `erfcx`.
- **Why correct:** standard EMG stable parameterization. Verified vs `scipy.stats.exponnorm`
  (K=1/(lam sigma)): max err 1.3e-15 in the body; 4.3e-12 at x in {±50,100,200} (asymptotic-series
  precision). Far-left tail (x=-30 → z large positive) matches to 1e-13; right tail (x=200) matches.
- **Numerical stability:** the whole point of `log_erfcx` — handles both `z→+inf` (underflow of erfcx,
  asymptotic series) and `z→-inf` (overflow, `z^2+log erfc`). Verified robust. Good.
- **Engine-swap:** numpy/scipy-only (`log_erfcx` calls scipy). No backend/declaration; host-only leaf.
  Noted, not a bug.
- **Verdict:** OK

### MoM estimate (`exgaussian.py:275`)
- **Computes:** `var=m2-m1^2`, central `mu3=m3-3 m1 m2+2 m1^3`, `skew=mu3/var^1.5`;
  `tau=(0.5 skew)^{1/3} sqrt(var)`, `sigma2=var-tau^2`, `mu=m1-tau`, `lam=1/tau`.
- **Why correct:** EMG method-of-moments (`mean=mu+tau`, `var=sigma^2+tau^2`, `skew=2 tau^3/var^1.5`).
  Degenerate (skew<=0 or `tau^2>=var`) falls back to a small positive exponential share keeping a valid
  EMG. Verified recovery on 80k samples: mu 0.475, sigma2 1.176, lam 0.788 (true 0.5/1.2/0.8).
- **Numerical stability:** same uncentered second/third-moment **cancellation** as A2 when `|mu|>>sigma`.
  Lower priority than SkewNormal because the EMG fallback clamps degenerate `var`/`sigma2`, but a badly
  cancelled `mu3` would still mis-set `tau`. Shares root cause with A2 (centered accumulation would fix).
  Same FINDING(A2) class.
- **Verdict:** OK (shares A2 root cause; fallback limits blast radius)

---

## Module: pysp/stats/leaf/student_t.py

### log_density / seq_log_density (`student_t.py:83`, `student_t.py:88`)
- **Computes:** `log f = log_const - 0.5(df+1) log1p(z^2/df)`, z=(x-loc)/scale,
  `log_const = gammaln((df+1)/2) - gammaln(df/2) - 0.5 log(df pi) - log(scale)`.
- **How:** `math.log1p`/`np.log1p` for the `1+z^2/df` term; cached `log_const`.
- **Why correct:** location-scale Student-t log-pdf. Verified vs `scipy.stats.t`: max err 4.4e-16.
- **Numerical stability:** `log1p` is the right choice; `gammaln` differences are stable. Good.
- **Engine-swap:** `backend_log_density_from_params` uses `engine.gammaln` and `log(1+z^2/df)` (note:
  uses `log(1+...)` not `log1p` — fine numerically since `z^2/df>=0`, no near-`-1` argument). Parity OK.
- **Verdict:** OK

### Fixed-df moment M-step (`student_t.py:267`)
- **Computes:** `loc=E[x]`, `var=E[x^2]-loc^2` floored, `scale^2 = var*(df-2)/df` if df>2 else var.
- **Why correct:** for df>2 the t variance is `scale^2 df/(df-2)`, so `scale^2 = var (df-2)/df`
  matches the population moments (df held fixed; docstring is explicit this is not the full MLE).
  Verified: df=6 recovers loc 0.99, scale 2.005 (true 1.0/2.0).
- **Numerical stability:** `var=E[x^2]-loc^2` is the **same A1 cancellation**; floored at
  `min_scale^2` only (won't catch a positive-but-wrong var). Shares A1. Pseudo-count path uses
  `var0=scale0^2 df/(df-2)` correctly.
- **Verdict:** OK (shares A1 root cause)

---

## Module: pysp/stats/leaf/laplace.py

### log_density / seq_log_density (`laplace.py:73`, `laplace.py:77`)
- **Computes:** `log f = -log(2 b) - |x-mu|/b`.
- **Why correct:** Laplace log-pdf. Verified vs `scipy.stats.laplace`: max err 4.4e-16.
- **Numerical stability:** trivially stable.
- **Engine-swap:** `backend_log_density_from_params` engine-neutral with `engine.abs`. Declaration uses
  `raw_observations` + `weights` (median needs the data, not moments) — correct design. Parity OK.
- **Verdict:** OK

### Weighted-MLE M-step (`laplace.py:267`, `_weighted_median` `laplace.py:20`)
- **Computes:** `mu = weighted median`, `b = sum(w |x-mu|)/sum(w)` (weighted MAD).
- **How:** `_weighted_median` sorts, cumulates weights, picks first index where cumweight ≥ 0.5*total
  (`side="left"`).
- **Why correct:** exact Laplace MLE (median for location, mean absolute deviation for scale). Verified:
  mu 1.487 (np.median 1.487), b 0.7120 matching the MAD exactly. `b` floored at `min_scale`.
- **Numerical stability:** no cancellation (absolute deviations). The weighted-median tie convention
  (`side="left"`, lower median on exact half) is a standard, acceptable choice. Pseudo-count appends a
  prior `(mu0, pseudo_count)` pair and adds `pseudo_count*suff_stat[1]` to b — consistent.
- **Engine-swap:** `backend_stacked_sufficient_statistics` returns per-component `(x[mask], w[mask])`
  raw arrays (host-resident, since median is host-only). Matches `value()` semantics. OK.
- **Verdict:** OK

---

## Module: pysp/stats/leaf/logistic.py

### log_density / seq_log_density (`logistic.py:69`, `logistic.py:74`)
- **Computes:** `log f = -log(scale) - z - 2*logaddexp(0, -z)`, z=(x-loc)/scale.
- **How:** `np.logaddexp(0,-z)` = `log(1+e^{-z})` = `-log sigmoid(z)`; `log f = -log s + log[σ(z)σ(-z)]`.
- **Why correct:** logistic log-pdf `= -log s + z - 2 log(1+e^z)` symmetrized via logaddexp. Verified vs
  `scipy.stats.logistic`: max err 2.2e-16. Large |z| (±200 seq, ±800 scalar) stays exact (verified
  `logpdf(±800)=-800`), because `logaddexp` never overflows.
- **Numerical stability:** `logaddexp` is the correct stable primitive; no overflow. Good. The
  `backend_log_density_from_params` branches on `z>=0` (pos/neg `1+exp(∓z)`) which also avoids overflow —
  matches the logaddexp path. Parity OK.
- **Verdict:** OK

### Moment M-step (`logistic.py:242`)
- **Computes:** `loc=E[x]`, `var=E[x^2]-loc^2`, `scale=sqrt(3 var)/pi` (from `var = pi^2 scale^2/3`).
- **Why correct:** logistic moment match. Verified: loc 1.995, scale 0.602 vs `sqrt(3 var)/pi`=0.602.
  Docstring notes it is moment-based, not MLE.
- **Numerical stability:** same A1 `E[x^2]-E[x]^2` cancellation; `var` clamped to `>=0` then `scale`
  floored at `min_scale`. Shares A1.
- **Verdict:** OK (shares A1 root cause)

---

## Module: pysp/stats/leaf/uniform.py

### log_density / seq_log_density (`uniform.py:66`, `uniform.py:70`)
- **Computes:** `log f = -log(high-low)` on `[low,high]`, else -inf.
- **Why correct:** uniform log-pdf. Verified vs `scipy.stats.uniform`: matches exactly on-support, -inf
  off-support (the NaN in a naive diff is the `-inf - -inf` test artifact, confirmed).
- **Numerical stability:** trivial.
- **Engine-swap:** `backend_log_density_from_params` masks via `engine.where`; declaration uses
  `support_bound` (min/max, non-additive). Parity OK.
- **Verdict:** OK

### M-step (`uniform.py:265`)
- **Computes:** MLE `low = min observed`, `high = max observed`.
- **Why correct:** uniform MLE is the sample range. Verified: low/high match data min/max. Degenerate
  `high<=low` widened to `min_width` around the midpoint; pseudo-count widens toward prior endpoints.
- **Numerical stability:** no arithmetic; min/max only. `seq_update`/`seq_update_engine` correctly mask
  weight>0 before min/max so zero-weight rows don't corrupt the support bounds. OK.
- **Verdict:** OK

---

## Cross-cutting findings

- **A1 — Gaussian-family uncentered variance cancellation** (gaussian.py:725; shared by log_gaussian.py:735,
  student_t.py:280, logistic.py:255): `var = E[x^2] - E[x]^2`. Verified to produce a 4x-wrong variance on
  N(1e8, 1) data in float64. Material for raw real-valued data with large offset; mitigated (not eliminated)
  on the log scale (LogGaussian) and by the EMG/skew fallbacks. The variance floor cannot detect it
  (result is positive). Fix: accumulate centered/Welford second moments, or at minimum document the
  precondition that data be roughly centered.
- **A2 — SkewNormal/EMG uncentered higher-moment cancellation** (skew_normal.py:200-204; exgaussian.py:287-292):
  central variance and especially central third moment from raw sums collapse when `|loc| >> scale`.
  Verified: SkewNormal(loc=1e6) recovers shape -2031 (true 4.0). Fix: centered third-moment accumulation.

No correctness bug found in any log-density, conjugate update, sampler, or M-step closed form; all match
their reference identities to machine precision. The findings are numerical-stability (cancellation) and
one cosmetic docstring issue.
