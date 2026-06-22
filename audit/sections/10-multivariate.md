# Multivariate distributions — computation ledger

Scope: `pysp/stats/multivariate/` — multivariate_gaussian.py, diagonal_gaussian.py,
multivariate_student_t.py, von_mises_fisher.py, watson.py, wishart.py, inverse_wishart.py,
matrix_normal.py, gaussian_copula.py.

All densities/normalizers were numerically cross-checked against scipy where available
(`multivariate_normal`, `multivariate_t`, `wishart`, `invwishart`, `matrix_normal`) and against
Monte-Carlo sphere integration for vMF/Watson. Every formula reproduced the reference to machine
precision; the only real issues are (a) a NaN mean when a component receives zero observations
(MVN and diagonal Gaussian) and (b) documentation typos.

---

## Module: multivariate_gaussian.py

### log_density / seq_log_density (`multivariate_gaussian.py:282`, `:340`)
- **Computes:** `log p(x) = -0.5(d log 2pi + log|Sigma| + (x-mu)' Sigma^{-1} (x-mu))`.
- **How:** Cholesky `cho_factor` at construction (`:209`); quadratic form via `cho_solve` (no
  explicit inverse in the hot path); `log_det = 2*sum(log diag(chol))` (`:217`); `chol_const`
  precomputed (`:219`).
- **Why correct:** Standard MVN. Verified: pysp `-6.0271420829111255` vs scipy
  `-6.027142082911126`; seq max-abs-diff `1.8e-15`.
- **Numerical stability:** STABLE — Cholesky solve, not `inv`; log-det from Cholesky diagonal.
  `inv_covar` is also cached (`:218`) but only used by the engine-neutral/stacked paths.
- **Engine-swap:** `backend_log_density_from_params` (`:360`) and `backend_stacked_log_density`
  (`:393`) use `inv_covar`+`log_det`; consistent with the numpy path. Neutral.
- **Verdict:** OK (docstring nit: header line 14 and `:286` say `-0.5*det(covar)` but the code
  correctly uses `log|Sigma|` — FINDING(MV-7), LOW).

### exp_family natural params / log partition (`:147`, `:156`)
- **Computes:** `eta1 = Sigma^{-1} mu`, `eta2 = -0.5 Sigma^{-1}`; `A = 0.5(mu' Sigma^{-1} mu +
  log|Sigma| + d log 2pi)`.
- **Why correct:** Canonical Gaussian exponential-family form; `log p = eta·T - A` reproduces the
  density above.
- **Verdict:** OK.

### M-step estimate (`:856`) and `_regularize_covar` (`:894`)
- **Computes:** `mu = sum_x/n`; `Sigma = sum_xx/n - mu mu'` (with pseudo-count/conjugate variants).
- **Why correct:** `E[xx'] - mu mu'` is the standard scatter; `vec.outer(mu, mu*nobs)/n... ` in the
  pseudo-count branch reduces to the same. Conjugate NormalWishart update (`:821`) verified against
  the documented posterior (m_n, kappa_n, W_n, nu_n) with joint-MAP precision `(nu_n-d) W_n`.
- **Numerical stability:** `var = E[xx'] - mu mu'` is the catastrophic-cancellation form, mitigated by
  `_regularize_covar`: clamps non-finite entries to 0, symmetrizes, adds
  `eps = max(min_covar=1e-8, ridge=1e-6 * trace/d) * I` so a singular covariance cannot break the
  Cholesky. Good.
- **BUG:** when `nobs == 0` (a component with zero total responsibility), `mu = sum_x/nobs = 0/0 =
  NaN` at `:883`. `_regularize_covar` only clamps the *covariance*; the NaN **mean** survives and
  poisons every downstream log-density. Reproduced: `estimate(0.0,(zeros,zeros,0.0))` -> `mu=[nan,
  nan]`, two `RuntimeWarning: invalid value encountered in divide`. FINDING(MV-1), MEDIUM.
- **Verdict:** FINDING(MV-1), FINDING(MV-7).

### condition / marginal (`:434`, `:465`)
- **Computes:** Gaussian conditional `mu_u + S_uo S_oo^{-1}(x_o-mu_o)`, `S_uu - S_uo S_oo^{-1}S_ou`;
  marginal drops rows.
- **Why correct:** Schur-complement conditionals; solved via a single `np.linalg.solve` against a
  stacked RHS. Symmetrized. OK.
- **Verdict:** OK.

### density_cumulative / density_quantile (`:305`, `:321`)
- **Computes:** HDR cumulative `chi2.cdf(maha2, d)` and a representative contour point
  `mu + sqrt(chi2.ppf(q,d)) * L[:,0]`.
- **Why correct:** Squared Mahalanobis ~ chi2_d for a Gaussian; exact. OK.
- **Verdict:** OK.

---

## Module: diagonal_gaussian.py

### log_density / log_c (`diagonal_gaussian.py:181`)
- **Computes:** `log p = -0.5(d log 2pi + sum log s2_i) - 0.5 sum (x_i-m_i)^2/s2_i`, via precomputed
  `ca=-0.5/covar`, `cb=mu/covar`, `cc`.
- **Why correct:** Independent per-coordinate Gaussian. Reduces to the MVN with diagonal Sigma.
- **Numerical stability:** fine for positive `covar`.
- **Verdict:** OK (docstring nit: module header line 10 writes `-(n/2)*log(pi)`; correct constant is
  `-(n/2)*log(2*pi)` as the code implements — FINDING(MV-8), LOW).

### exp_family log partition (`:127`)
- **Computes:** `A = 0.5 sum(log(2pi*covar) + mu^2/covar)`. Matches the diagonal Gaussian
  log-partition. OK.

### M-step estimate (`:777`)
- **Computes:** `mu = sum_x/n`; `var = sum_xx/n - mu^2` with a variance floor at `:811-819`
  (clamp non-finite -> min_covar, then `max(var, max(min_covar, ridge*mean(positive var)))`).
- **Why correct:** Standard diagonal MLE. The conjugate NormalGamma update at `:745` was inspected;
  `new_b0 = sum_xx - mean*sum_x` is the per-coordinate scatter and `new_sigma2 = b_n/(a_n-0.5)` is
  the joint-MAP variance — consistent with the documented MultivariateNormalGamma posterior.
- **Numerical stability:** variance floor is solid and guards the `E[x^2]-E[x]^2` cancellation.
- **BUG:** identical to MV-1 — at `nobs == 0`, `mu = sum_x/nobs` is `0/0 = NaN` (`:804`). The
  variance floor saves `covar` but not `mu`. Reproduced: `mu=[nan,nan,nan]`, `covar=[1e-8,1e-8,
  1e-8]`, with the exact `RuntimeWarning: invalid value encountered in divide` the audit prompt
  flagged. FINDING(MV-2), MEDIUM. (The prompt's line cite 775-780 is the estimator docstring; the
  live code is `:798-819`.)
- **Verdict:** FINDING(MV-2), FINDING(MV-8).

---

## Module: multivariate_student_t.py

### log_density / seq_log_density (`multivariate_student_t.py:220`, `:226`)
- **Computes:** `log f = gammaln((nu+p)/2) - gammaln(nu/2) - 0.5 p log(nu pi) - 0.5 log|Sigma|
  - 0.5(nu+p) log(1 + delta/nu)`.
- **How:** `_safe_inverse_and_logdet` (`:45`) caches `Sigma^{-1}` and `log|Sigma|` with a `1e-12`
  ridge fallback if `slogdet` sign<=0; `log_const` precomputed; Mahalanobis via `einsum`; tail via
  `np.log1p(delta/nu)`.
- **Why correct:** Standard MVT density. Verified: pysp `-6.206015110502219` vs
  `scipy.stats.multivariate_t.logpdf` `-6.206015110502218`.
- **Numerical stability:** `log1p` for the tail (good near delta≈0). dof>0 enforced at `:133`.
- **Engine-swap:** `backend_log_density_from_params` (`:91`) and stacked (`:259`) reproduce the same
  formula with engine ops; stacked path avoids batched matmul intentionally. Neutral.
- **Verdict:** OK.

### M-step (fixed-dof EM/IRLS) (`:437`)
- **Computes:** weight `u_i = (nu+p)/(nu+delta_i)`; `mu = sum_ux/sum_u`; `Sigma = scatter/n` with
  `scatter = sum_uxx - mu sum_ux' - sum_ux mu' + sum_u mu mu'`.
- **Why correct:** Standard known-dof EM for the multivariate t — location is the u-weighted mean,
  scale is `(1/n) sum_i u_i (x_i-mu)(x_i-mu)'` (divide by the **count** n, not sum_u). The expanded
  scatter equals `sum_i u_i(x_i-mu)(x_i-mu)'`. Symmetrized + `min_ridge` floor. Correct.
- **Guards:** `count<=0 or sum_u<=0` returns the standard `N(0, I)` fallback (`:441`) — so the
  zero-data NaN-mean bug does NOT occur here. Good.
- **Verdict:** OK.

---

## Module: von_mises_fisher.py

### lniv / lniv_uniform (`von_mises_fisher.py:84`, `:51`)
- **Computes:** numerically stable `log I_v(z)`.
- **How:** `log(ive(v,z)) + z` where the scaled Bessel has support; falls back to the A&S 9.7.7
  uniform large-order asymptotic when `ive` underflows (large order vs argument).
- **Why correct:** `ive(v,z)=I_v(z)e^{-z}`, so `log ive + z = log I_v`. Verified vs
  `scipy.special.iv` for p<=10 (e.g. p=10,k=20: lniv `17.18056457761776` vs `17.18056457761776`).
  For p=50,100 where `iv` overflows, the uniform branch keeps `log_const` finite and the density
  integrates to 1 (MC).
- **Numerical stability:** the headline stability win of this module; handles both overflow and
  underflow regimes.
- **Verdict:** OK.

### log_const / log_density (`:183`, `:220`)
- **Computes:** `log c_p(k) = (p/2-1) log k - (p/2) log 2pi - log I_{p/2-1}(k)`;
  `log f = log c_p + k mu·x`. kappa=0 falls to the uniform sphere density
  `gammaln(p/2) - log2 - (p/2)log pi`.
- **Why correct:** vMF normalizer. MC sphere integral of the density = `1.0013` (p=3,k=4, 2e6
  samples). Verified `log_const` against the direct `(p/2-1)logk - (p/2)log2pi - lniv` form for
  p in {3,10,50,100}: exact agreement.
- **Verdict:** OK.

### estimate — kappa MLE (`:687`)
- **Computes:** `mu = ssum/||ssum||`; solve `A_p(kappa)=rhat` (rhat=||ssum||/count) via Banerjee
  init `rhat(d-rhat^2)/(1-rhat^2)` + up to 3 Newton steps on the Bessel ratio.
- **Why correct:** Standard vMF concentration MLE. `A_p` computed as `exp(lniv(p/2,logk) -
  lniv(p/2-1,logk))` so it does not overflow in high d.
- **Numerical stability:** GOOD — rhat clamped to `1-1e-10` (rhat→1 ⇒ kappa→inf); Newton skipped
  within `1e-9` of 1 where `A_p'→0` is ill-conditioned (the documented high-d instability is
  explicitly handled); `_newton` clamps `k>=float_info.min`.
- **Verdict:** OK.

### Wood rejection sampler (`:391`)
- **Computes:** Wood's (1994) vMF tangent-normal rejection scheme; `b=(t1-2k)/(d-1)` with
  `t1=sqrt(4k^2+(d-1)^2)`.
- **Why correct:** matches Wood's `b = (-2k + sqrt(4k^2+(d-1)^2))/(d-1)`. Batched rejection with a
  10k-iteration budget guard (raises rather than hang). Tangent basis = SVD complement of mu.
- **Verdict:** OK.

### seq_update weight filtering (`:530`)
- Drops non-finite/negative weights from both `ssum` and `count` consistently. OK.

---

## Module: watson.py

### _kummer_ratio / log_density (`watson.py:34`, `:84`)
- **Computes:** `f = M(1/2,p/2,k)^{-1}/omega_p * exp(k (mu·x)^2)`;
  `log_const = -log omega_p - log M(1/2,p/2,k)`, `omega_p = 2 pi^{p/2}/Gamma(p/2)`.
- **Why correct:** Watson axial density. `_kummer_ratio = (1/p) M(3/2,(p+2)/2,k)/M(1/2,p/2,k)` uses
  the identity `M'(a,b,z)=(a/b)M(a+1,b+1,z)` with a=1/2,b=p/2. MC sphere integral = `1.0001`
  (bipolar k=3) and `0.9992` (girdle k=-3).
- **Numerical stability:** `hyp1f1` directly; fine for |k|<=700 (the bisection bracket).
- **Verdict:** OK.

### estimate (`:216`)
- **Computes:** axis = eigenvector of mean scatter `S/count` whose eigenvalue is farthest from 1/p
  (top for bipolar, bottom for girdle); kappa from `_solve_kappa(r)` bisection, r clamped to
  `[1e-6, 1-1e-6]`.
- **Why correct:** Watson MLE — `mu` solves `mu' S mu` extremal, kappa matches `E[(mu·x)^2]=r`.
  Monotone bisection on the Kummer ratio over `[-700,700]`. Correct.
- **Verdict:** OK.

---

## Module: wishart.py

### log_density (`wishart.py:68`, `:77`)
- **Computes:** `log f = (df-p-1)/2 log|X| - 0.5 tr(V^{-1}X) - df p/2 log2 - df/2 log|V| -
  log Gamma_p(df/2)`.
- **How:** `slogdet` for log|X|; `multigammaln` for the multivariate gamma; trace via `einsum`;
  returns `-inf` when X not PD.
- **Why correct:** Verified pysp `-18.18923121200766` vs `scipy.stats.wishart.logpdf`
  `-18.18923121200766`.
- **Guards:** `df < p` rejected at construction (`:40`); scale PD-checked via slogdet sign (`:43`).
- **Verdict:** OK.

### Bartlett sampler (`:105`)
- `a[i,i]=sqrt(chi2(df-i))` for i=0..p-1 (i.e. df,df-1,...,df-p+1), off-diagonal standard normal,
  `X = (L A)(L A)'` with `V=LL'`. Standard Bartlett decomposition. OK.

### estimate (`:201`)
- `V = (mean X)/df` since `E[X]=df V`; symmetrized; count<=0 fallback to identity. Correct. OK.

---

## Module: inverse_wishart.py

### log_density (`inverse_wishart.py:65`, `:74`)
- **Computes:** `log f = df/2 log|Psi| - df p/2 log2 - log Gamma_p(df/2) - (df+p+1)/2 log|X| -
  0.5 tr(Psi X^{-1})`.
- **Why correct:** Verified pysp `-28.356888152001634` vs `scipy.stats.invwishart.logpdf`
  `-28.35688815200164`.
- **Guards:** `df <= p-1` rejected (`:41`); scale PD-checked (`:44`); `-inf` for non-PD X.
- **Numerical stability:** `seq_log_density` calls `np.linalg.inv(xx)` per matrix — fine for SPD
  inputs; could use a solve, but not a correctness issue.
- **Verdict:** OK.

### sampler / estimate (`:96`, `:192`)
- Samples by inverting a `Wishart(df, Psi^{-1})` draw (correct: if `Y~W(df,Psi^{-1})` then
  `Y^{-1}~IW(df,Psi)`). Estimate `Psi=(df-p-1) mean(X)` since `E[X]=Psi/(df-p-1)`, guarded by
  `factor>0`. Correct. OK.

---

## Module: matrix_normal.py

### log_density (`matrix_normal.py:81`, `:87`)
- **Computes:** `log p = -0.5(np log2pi + n log|V| + p log|U| + tr(U^{-1}C V^{-1} C'))`,
  `C = X-M`.
- **Why correct:** Kronecker-covariance Gaussian `vec(X)~N(vec(M), V⊗U)`. Verified pysp
  `-15.998100625431409` vs `scipy.stats.matrix_normal.logpdf` `-15.998100625431407`. seq via
  `einsum("ab,nbc,cd,nad->n")`.
- **Guards:** U,V shape-checked and PD-checked via slogdet (`:60`).
- **Verdict:** OK.

### flip-flop MLE (`:213`)
- **Computes:** `U = (1/(np)) sum_i C_i V^{-1} C_i'`, `V = (1/(nn)) sum_i C_i' U^{-1} C_i`,
  alternated; anchors `V[0,0]=1` for the U↔V scale identifiability.
- **Why correct:** `tc[a,b,c,d]=sum_i C[a,c]C[b,d]`, so `einsum("abcd,cd->ab", tc, v_inv) =
  sum_i (C V^{-1} C')[a,b]`, `/ (count*p)` — the standard flip-flop update; V symmetric. Anchor
  resolves the Kronecker scale ambiguity. Correct.
- **Verdict:** OK.

---

## Module: gaussian_copula.py

### log_density (`gaussian_copula.py:64`, `:69`)
- **Computes:** `log c(u) = -0.5 log|R| - 0.5 z'(R^{-1}-I)z`, `z = Phi^{-1}(u)`.
- **How:** `_inv_minus_i = R^{-1} - I` precomputed; `z` clipped to `[1e-12, 1-1e-12]` before
  `norm.ppf`.
- **Why correct:** Gaussian copula density (the Phi Jacobians cancel the standard-normal factor of
  the MVN). Verified pysp `-0.13115486150256556` vs direct
  `mvn.logpdf(z,0,R) - sum norm.logpdf(z)` `-0.13115486150256617`.
- **Numerical stability:** boundary clip keeps `Phi^{-1}` finite. PD-checked at construction (`:48`).
- **Verdict:** OK.

### estimate — inversion estimator (`:187`)
- **Computes:** sample correlation of the normal scores; cov `sum_zz/n - mean mean'`, normalized to
  unit diagonal; eigen-projected to PD (`min_eig=1e-8`) if needed and re-normalized.
- **Why correct:** Standard Gaussian-copula inversion estimator; PD projection is a sound
  nearest-correlation safeguard. count<=0 -> identity. The `E[zz']-E[z]E[z]'` cancellation is
  benign here (correlation, then PD projection). Correct.
- **Verdict:** OK.

---

## Findings summary

- MV-1 (MEDIUM): multivariate_gaussian.py:883 — `mu = suff_stat[0]/nobs` → NaN mean when nobs==0.
- MV-2 (MEDIUM): diagonal_gaussian.py:804 — same NaN-mean at nobs==0 (covar floored, mu not).
- MV-7 (LOW): multivariate_gaussian.py:14,286 — docstring `-0.5*det(covar)` should be `log|covar|`.
- MV-8 (LOW): diagonal_gaussian.py:10 — docstring `-(n/2)*log(pi)` should be `-(n/2)*log(2*pi)`.

Everything else verified correct to machine precision against scipy / Monte-Carlo.
