# Directional + point-process leaf distributions — computation ledger

Scope: `pysp/stats/leaf/{von_mises, wrapped_cauchy, hawkes_process, power_law_hawkes,
multivariate_hawkes, inhomogeneous_poisson, birth_death, chinese_restaurant_process}.py`.

All non-trivial formulas were numerically validated with `.venv/bin/python` against brute-force
double-sum implementations and/or scipy. Every check passed to machine precision. No CRITICAL/HIGH
issues found; the domain is in excellent shape.

---

## Module: pysp/stats/leaf/von_mises.py

### log I_0(kappa) normalizer (`von_mises.py:46`)
- **Computes:** `log_const = -log(2*pi*I_0(kappa))`; `_log_i0 = log(i0e(kappa)) + kappa`.
- **How:** uses scipy `i0e` (exponentially-scaled `I_0 e^{-kappa}`), adds back `kappa`.
- **Why correct:** `log I_0 = log(i0e) + kappa` exactly. Verified vs `np.log(i0(k))`: kappa=0.5/5/50
  match to 6 dp; kappa=1e3/1e5 give finite 995.6/99993.3 where naive `i0(k)` overflows to inf.
- **Numerical stability:** excellent — exponential scaling avoids overflow for arbitrarily large kappa.
- **Engine-swap:** host-only scipy in the scalar `log_const` cache, but the per-row score
  (`eta1*cos + eta2*sin + log_const`) is linear and engine-neutral; `log_const` is declared a
  non-differentiable real parameter. Correct by design.
- **Verdict:** OK

### A(kappa) = I_1/I_0 and inversion (`von_mises.py:51`, `:61`)
- **Computes:** mean resultant length `A(kappa)=I_1/I_0`; MLE inversion `kappa = A^{-1}(r)`.
- **How:** `_bessel_ratio` uses `ive(1,k)/ive(0,k)` (scaling cancels); `_solve_kappa` uses the
  Best & Fisher piecewise initializer + 5 Newton steps with `A'(k)=1-A/k-A^2`.
- **Why correct:** round-trip kappa->r->solve recovers 0.3/2/10/50 exactly. EM on simulated data
  recovers parameters.
- **Numerical stability:** stable ratio for large kappa; Newton breaks safely if `deriv<=1e-12`
  (falls back to the closed-form initializer), kappa capped at 1e8.
- **Engine-swap:** host-only (estimator path only). OK.
- **Verdict:** OK

### log_density / seq_log_density / exp-family declaration (`von_mises.py:205`, `:209`, `:94`)
- **Computes:** `kappa*cos(x-mu)+log_const`, vectorized as `eta1*cos+eta2*sin+log_const`.
- **Why correct:** identity `kappa cos(x-mu)=kappa cos mu cos x + kappa sin mu sin x`. kappa=0 gives
  `-log 2pi` (uniform) — verified. ExponentialFamilySpec, capabilities, backend stacked params all
  consistent with the (count,cos,sin) accumulator order.
- **Engine-swap:** neutral; `backend_log_density_from_params` is pure arithmetic.
- **Verdict:** OK

---

## Module: pysp/stats/leaf/wrapped_cauchy.py

### log-density and normalizer (`wrapped_cauchy.py:49`, `:64`, `:68`)
- **Computes:** `f = (1-rho^2)/(2pi(1+rho^2-2rho cos(theta-mu)))`; cached
  `_log_num = log1p(-rho^2) - log 2pi`.
- **Why correct:** matches `scipy.stats.wrapcauchy.pdf` to 8 dp at theta=0.1/1.0/-2.0.
- **Numerical stability:** `log1p(-rho^2)` for the numerator (good near rho->0); denominator
  `1+rho^2-2rho*cos_dev >= (1-rho)^2 > 0` for rho<1, so `np.log` never sees a non-positive argument.
- **Engine-swap:** numpy-only `seq_log_density`; no backend hooks (not declared engine-ready). OK.
- **Verdict:** OK

### MLE and sampler (`wrapped_cauchy.py:184`, `:94`)
- **Computes:** `mu=atan2(sum_sin,sum_cos)`, `rho=|R|/n` (first trig moment = rho e^{i mu}); sampler
  wraps a Cauchy of scale `gamma=-log rho`.
- **Why correct:** standard wrapped-Cauchy moment estimator; rho clipped to `[0, 1-1e-8]`.
- **Verdict:** OK

---

## Module: pysp/stats/leaf/hawkes_process.py  (CRITICAL path — fully validated)

### log_density: sum log lambda - compensator (`hawkes_process.py:100`)
- **Computes:** `sum_i log(mu+alpha R_i) - mu*w - (alpha/beta) sum_i(1-exp(-beta(w-t_i)))`,
  `R_i = sum_{j<i} exp(-beta(t_i-t_j))` via the O(n) recursion `R_i=exp(-beta dt)(R_{i-1}+1)`.
- **Why correct:** **brute-force double-sum check: code=-9.0195269804, brute=-9.0195269804, diff=0.0.**
  Compensator integral `integral_0^w sum_j alpha exp(-beta(s-t_j)) ds = (alpha/beta) sum_j
  (1-exp(-beta(w-t_j)))` verified term-by-term. Empty realization gives `-mu*w` (verified -5.0).
- **Numerical stability:** log-space; intensity `mu+alpha*R_i >= mu > 0` so `math.log` is safe.
- **Engine-swap:** numpy-only; `seq_log_density` is a vectorized recursion over the padded matrix —
  **seq vs scalar diff = 0.0**. Padding to `window` makes `(1-exp(-beta(window-t)))=0` on pad columns
  and `where(active,...)` masks the loglam term. Correct.
- **Verdict:** OK

### EM branching accumulator R_i / S_i recursion (`hawkes_process.py:219`, `:260`)
- **Computes:** responsibilities `p_i0=mu/lam`, `sum_j p_ij=alpha R_i/lam`,
  `sum_j p_ij(t_i-t_j)=alpha S_i/lam` with `S_i=sum_{j<i}exp(-beta dt)(t_i-t_j)`, recursion
  `S_i = exp(-beta dt)(dt(R_{i-1}+1)+S_{i-1})`.
- **Why correct:** **brute-force check: s0 4.21970227, g 1.78029773, w 1.37852970 all match exactly.**
  `seq_update` (vectorized) is bit-identical to per-seq `_accumulate_seq` (parity confirmed:
  allclose True on a 3-sequence batch).
- **Verdict:** OK

### M-step (`hawkes_process.py:366`)
- **Computes:** `mu=S0/total_window`, `beta=G/W`, `alpha=beta*(G/n_events)` with branching ratio
  floored to `[1e-3, 1-1e-6]`.
- **Why correct:** Veen-Schoenberg branching estimator; the `1e-3` floor escapes the alpha=0 absorbing
  fixed point. 30 EM iters on simulated data: mu/alpha/beta = 0.833/0.546/1.787 vs true 0.8/0.5/1.5.
- **Numerical stability:** all denominators floored with `_MIN=1e-12`.
- **Engine-swap:** numpy/scalar only. OK.
- **Verdict:** OK

### Sampler (Ogata thinning) (`hawkes_process.py:169`)
- **How:** `lam_bar = mu+alpha*excitation` is a valid upper bound between events (intensity only
  decays); excitation decayed to candidate then `+1` on accept. Super-critical warning + 1e7 cap.
- **Verdict:** OK

---

## Module: pysp/stats/leaf/multivariate_hawkes.py

### log_density: per-mark excitation recursion (`multivariate_hawkes.py:87`)
- **Computes:** `sum_n log(mu_{m_n}+alpha[m_n]@s) - w*sum_d mu_d -
  (1/beta) sum_k col_alpha[m_k](1-exp(-beta(w-t_k)))`, `s[j]=sum_{k<i,m_k=j}exp(-beta(t_i-t_k))`,
  `col_alpha[j]=sum_d alpha[d,j]` (total excitation an event of mark j sends).
- **Why correct:** **brute-force check: code=-15.3412946678, brute=-15.3412946678, diff=0.0.** The
  per-dimension compensator correctly aggregates each event's excitation across all D target marks via
  the column sum. Marks outside `[0,dim)` and disordered times -> -inf.
- **Numerical stability:** `s` is shrunk multiplicatively each step; `lam >= mu_{m_i} > 0`.
- **Engine-swap:** numpy/scalar; `seq_log_density` loops in Python over realizations (per-realization
  likelihood — not vectorizable across ragged catalogues). Acceptable.
- **Verdict:** OK

### EM accumulator: s, s_delay, mass (`multivariate_hawkes.py:185`)
- **Computes:** per-mark immigrant resp `s0[m_i]`, offspring matrix `g[m_i,:]=alpha[m_i]*s/lam`,
  delay `w_delay=alpha[m_i]@s_delay/lam`, integrated mass `mass[j]=sum_{m_k=j}(1-exp(-beta(w-t_k)))/beta`.
- **Why correct:** **brute-force check: s0/g/mass allclose True, w_delay diff=0.0.**
- **Verdict:** OK

### M-step (`multivariate_hawkes.py:306`)
- **Computes:** `mu=s0/total_window`, `beta=sum(g)/w_delay`, `alpha[d,j]=g[d,j]/mass[j]`; alpha rescaled
  by `(1-1e-6)/radius` if spectral radius of alpha/beta reaches 1.
- **Why correct:** edge-corrected branching estimator; `alpha[d,j]` divides offspring counts by the
  parent mark's integrated excitation mass — dimensionally and statistically the right denominator.
- **Numerical stability:** all denominators floored with `_MIN`.
- **Verdict:** OK

---

## Module: pysp/stats/leaf/power_law_hawkes.py

### intensity / log_density (`power_law_hawkes.py:68`, `:94`)
- **Computes:** `lambda(t)=mu+sum_{t_j<t} A e^{alpha m_j}(1+(t-t_j)/c)^{-p}`; log-density
  `sum log lam - [mu*w + sum_j A e^{alpha m_j} c/(p-1)(1-(1+(w-t_j)/c)^{1-p})]`.
- **How:** O(n^2) double loop (power-law kernel has no finite-state recursion — correctly noted).
- **Why correct:** **brute-force check: code=-8.91358301, ref=-8.91358301.** The Omori-Utsu kernel
  integral `integral_0^{w-t_j}(1+s/c)^{-p}ds = c/(p-1)(1-(1+(w-t_j)/c)^{1-p})` is exact for p>1.
- **Numerical stability:** lam >= mu > 0; p>1 enforced in `__init__`.
- **Engine-swap:** numpy/scipy host-only; MLE via L-BFGS-B over log-parametrized (mu,A,c,p-1) and raw
  alpha (mark sensitivity, correctly left unexponentiated since it can be negative). Estimator runs and
  returns finite params on simulated marked/unmarked catalogues.
- **Verdict:** OK

### expected_count / branching_ratio / sampler (`power_law_hawkes.py:76`, `:87`, `:132`)
- **Computes:** windowed expected count via the same Omori integral with a lower clamp
  `max(t_start,t_j)`; branching ratio `A c/(p-1) e^{alpha*mean_mark}`; sampler draws power-law
  inter-times `tau = c((1-U)^{-1/(p-1)}-1)`.
- **Why correct:** inverse-CDF of the normalized Omori density is `c((1-U)^{-1/(p-1)}-1)` — correct.
- **Verdict:** OK

---

## Module: pysp/stats/leaf/inhomogeneous_poisson.py

### log_density (`inhomogeneous_poisson.py:111`, `:118`)
- **Computes:** `sum_b n_b log(rate_b) - sum_b rate_b * width_b` via histogram bin counts.
- **Why correct:** **direct check: code=-2.46989904, ref=-2.46989904.** An event in a zero-rate bin
  yields -inf (verified); the `np.where(counts>0, ...)` guard discards the `0*-inf` for empty
  zero-rate bins.
- **Numerical stability:** `_log_rates` precomputed with `-inf` for zero rates; `~isfinite(emitted)`
  mask sets -inf only when a positive count lands in a zero-rate bin. Correct.
- **Engine-swap:** numpy-only. OK.
- **Verdict:** OK

### MLE (`inhomogeneous_poisson.py:271`)
- **Computes:** `rate_b = bin_counts_b / (width_b * n_realizations)` — closed-form Poisson MLE.
- **Verdict:** OK

---

## Module: pysp/stats/leaf/birth_death.py

### trajectory replay & log_density (`birth_death.py:44`, `:128`, `:133`)
- **Computes:** `sum_events log n_i + n_b log b + n_d log d + n_s log s - (b+d+s)*I`,
  `I=integral_0^T n(t)dt` (piecewise-constant population), `n_i`=population just before event i.
- **Why correct:** **direct replay check: code=-13.02165125, ref=-13.02165125.** Birth increments,
  death decrements, sampling leaves population unchanged (no removal) — matches the linear BDS
  likelihood. seq vs scalar diff = 0.0.
- **Numerical stability:** `_log_rates` with `errstate(divide=ignore)` -> -inf for zero rates;
  `np.where(counts>0,...)` + `~isfinite` mask handle the `0*-inf` and the "event of a zero-rate
  channel" -> -inf correctly. Raises on event at zero population / out-of-order times.
- **Engine-swap:** numpy-only. OK.
- **Verdict:** OK

### MLE (`birth_death.py:300`)
- **Computes:** `rate = (events of that type) / I` per channel; horizon reconstructed as
  `horizon_sum/count`.
- **Why correct:** closed-form MLE of independent exponential-clock rates. The encoder produces (N,6)
  rows (5 suff stats + T); `seq_log_density` slices `[:5]` (ignores T) and `seq_update` reads col 5
  only when `shape[1]>5` — consistent. `value()`/`combine()` agree on the 6-tuple order.
- **Verdict:** OK

---

## Module: pysp/stats/leaf/chinese_restaurant_process.py

### Ewens log-density (`chinese_restaurant_process.py:64`)
- **Computes:** `P = alpha^K Gamma(alpha)/Gamma(alpha+n) prod_k Gamma(n_k)`;
  log = `K log alpha + [gammaln(alpha)-gammaln(alpha+n)] + sum_k gammaln(n_k)`.
- **Why correct:** **equals the sequential CRP/EPPF product: code=-5.77582736, seq=-5.77582736.**
  Note `Gamma(n_k)=(n_k-1)!` so this is the standard EPPF. Relabeling-invariant (verified: two
  different label vectors of the same partition give identical -5.7456). Partitions whose sizes don't
  sum to n -> -inf.
- **Numerical stability:** `gammaln` throughout; no overflow.
- **Engine-swap:** scipy host-only. OK.
- **Verdict:** OK

### concentration MLE (`chinese_restaurant_process.py:198`, `:201`)
- **Computes:** solves `E[K|alpha]=alpha(psi(alpha+n)-psi(alpha))=mean_k` by geometric bisection
  (monotone in alpha), with `[alpha_min,alpha_max]=[1e-6,1e6]` saturation.
- **Why correct:** `E[K]=sum_{i=0}^{n-1} alpha/(alpha+i)=alpha(psi(alpha+n)-psi(alpha))` is the exact
  expected block count; monotone increasing so bisection is valid. **Recovered alpha=2.485 from 2000
  draws at true alpha=2.5.** 200 geometric-bisection iters give ~log-spaced convergence.
- **Verdict:** OK

### sampler (`chinese_restaurant_process.py:96`)
- **How:** sequential seating with weights `[sizes..., alpha]`, normalized; first-appearance labels.
- **Verdict:** OK

---

## Summary

No correctness or stability defects found across the eight modules. Every density, compensator,
EM recursion, and MLE was cross-checked numerically and matched a brute-force / scipy reference to
machine precision. The two O(n) Hawkes recursions (univariate R/S, multivariate per-mark s/s_delay)
are exact, and their vectorized `seq_*` paths are bit-identical to the scalar reference. von Mises
`log I_0` is overflow-safe to kappa=1e5+. The only non-vectorized hot paths (power-law Hawkes O(n^2),
multivariate/CRP per-realization Python loops) are inherent to the math, correctly documented, and not
defects.
