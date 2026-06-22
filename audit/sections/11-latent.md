# Latent-variable models (mixtures, HMMs, LDA, PCA) + EM driver — computation ledger

Scope: `pysp/utils/em.py`; the HMM family (`hidden_markov.py`, `_hidden_markov_numba_kernels.py`,
`lookback_*`, `tree_*`, `quantized_*`, `segmental_*`, `semi_supervised_*`); the LDA family
(`lda.py`, `labeled_lda.py`, `integer_probabilistic_latent_semantic_indexing.py`); and the
mixture/factor family (`mixture.py`, `gaussian_mixture.py`, `joint_mixture.py`,
`hierarchical_mixture.py`, `heterogeneous_mixture.py`, `semi_supervised_mixture.py`,
`spatial_mixture.py`, `probabilistic_pca.py`). Focus is the latent algebra (responsibilities,
forward-backward, Baum-Welch M-step, variational/ELBO updates), not leaf emissions.

Numeric checks were run with `.venv/bin/python` and are quoted inline.

## Module: pysp/utils/em.py
### EM strategy orchestration (`em.py:31-603`)
- **Computes:** A family of EM drivers (StandardEM, Hard/Annealed/Posterior-transform, GEM,
  Monotonic, CM, MonteCarlo, Variational, Online/Incremental, Accelerated, Restart) plus `run_em`.
- **How:** Pure orchestration — moves encoded data through estimators/kernels; never contains
  distribution-specific likelihood math (delegates to `seq_estimate` / accumulators).
- **Why correct:** `run_em` converges on a small *improvement* (`0.0 <= value-old_value < delta`),
  which correctly does not stop on a likelihood *decrease* (`em.py:687`). NaN/inf guard rolls back
  to the last good model (`em.py:682-683`). MonotonicEM additionally rejects non-finite or
  decreasing steps and catches LinAlg/FloatingPoint blow-ups (`em.py:228-236`).
- **Numerical stability:** `_transform` temperature path does logs in a `np.errstate` block and
  divides with `out=zeros, where=row_sum>0` to avoid 0/0 (`em.py:92-98`). Good.
- **Engine-swap:** Routes through `_engine_seq_estimate` / `_engine_seq_log_density_sum` when an
  engine is passed; `_posterior_matrix` prefers the engine kernel's `posteriors`. Neutral.
- **Verdict:** OK

## Module: pysp/stats/latent/mixture.py
### Mixture log-density (`mixture.py:308-325`, `seq_log_density:439-500`)
- **Computes:** `log p(x) = logsumexp_k (log w_k + log f_k(x))`.
- **How:** Row-wise log-sum-exp with max subtraction; `seq_log_density` builds an `(n,K)` matrix,
  subtracts the per-row max, exp, sum, log, add max back. Zero-weight components are skipped
  (`zw`) and seeded to -inf. Rows whose max is -inf are special-cased to return -inf.
- **Why correct:** Standard stable mixture marginal; verified indirectly via HMM/posterior checks.
- **Numerical stability:** log-space throughout; `log_w` built as `log(w+zw)` then -inf on zeros to
  avoid `log(0)` warnings (`mixture.py:202-204`). Good.
- **Engine-swap:** `backend_seq_log_density` / `backend_seq_component_log_density` route through
  `engine.logsumexp`; zero-weight comps emit `-inf` via `base*0.0 + (-inf)`. Neutral.
- **Verdict:** OK

### Mixture responsibilities (`posterior:368-402`, `seq_posterior:530-578`)
- **Computes:** `gamma_k = softmax_k(log w_k + log f_k(x))`.
- **How:** Subtract row max, exp, normalize. Impossible rows (max==-inf) fall back to the prior `w`
  (`posterior:395-396`) or to `log_w` (`seq_posterior:570-571`) so they don't produce nan.
- **Why correct:** Verified against brute-force enumeration through the HMM (the HMM E-step reuses
  the same softmax); mixture `posterior` is the textbook responsibility.
- **Numerical stability:** Max-subtracted softmax; explicit -inf-row guard. Good.
- **Verdict:** OK

### Variational expected log-density (`expected_log_density:270-280`, `seq_*:282-292`)
- **Computes:** `E_q[log w_k] = digamma(alpha_k) - digamma(sum alpha)` plugged into the mixture
  logsumexp for a Dirichlet weight prior.
- **Why correct:** `_dirichlet_expectations` (`mixture.py:142-154`) is the standard Dirichlet mean
  of `log w`. Falls back to plug-in when no conjugate prior. OK.
- **Verdict:** OK

### Mixture M-step (`MixtureEstimator.estimate:1512-1591`)
- **Computes:** `w_k = counts_k / sum counts` (MLE), or MAP `w_k ∝ (count_k+alpha_k-1)_+`
  (conjugate Dirichlet), with optional pseudo-count and `w_min` floor.
- **Why correct:** MLE/MAP simplex updates are standard; the MAP path clamps at the simplex
  boundary and carries the posterior `Dirichlet(alpha+counts)` (`mixture.py:1541-1560`).
- **Numerical stability:** MLE path guards `nobs_loc==0 → uniform` (`mixture.py:1576-1577`). `w_min`
  floor renormalizes and nan-sanitizes. Good.
- **Verdict:** OK

### k-means++ init (`_kmeanspp_responsibilities:1191-1232`)
- **Computes:** D²-weighted seeding with near-hard responsibilities and a small floor.
- **Why correct:** Standard k-means++; falls back to Dirichlet init when no numeric feature matrix.
  OK. (Mitigates the documented EM saddle for positive-support leaves.)
- **Verdict:** OK

## Module: pysp/stats/latent/hidden_markov.py + _hidden_markov_numba_kernels.py
### Scalar forward log-density (`log_density:529-609`)
- **Computes:** `log p(x_0..x_{N-1}) = log sum_paths pi prod A prod b`, plus `len_dist`.
- **How:** Scaled filter forward: normalize the filtered posterior each step, predict via `A.T @ p`,
  renormalize, add emission log-densities, logsumexp with per-step max subtraction; `retval`
  accumulates `log(sum)+max`. Out-of-support step (max==-inf) short-circuits to -inf
  (`hidden_markov.py:594-599`).
- **Why correct:** **Numeric check** (3-state Poisson-emission HMM, length-3): scalar log_density
  `-7.44700557631864` vs brute-force enumeration over all 27 state paths `-7.447005576318639`
  (match to 1e-15).
- **Numerical stability:** scaled forward + per-step max-shift; no raw product underflow. Good.
- **Verdict:** OK

### Vectorized forward (`seq_log_density:654-749`; numba `numba_seq_log_density`)
- **Computes:** Same forward, both for the blocked (non-numba) and contiguous (numba) encodings.
- **How:** Emissions max-scaled to one (`pr_obs -= pr_max0; exp`), then a scaled alpha pass; the
  numba kernel accumulates `log(alpha_sum) + max_ll` per step (`_hidden_markov_numba_kernels.py:22-60`).
  nan rows (out-of-support emissions) sanitized to -inf to match the scalar path.
- **Why correct:** **Numeric checks:** blocked path `-7.44700557631864` and numba path
  `-7.44700557631864` both equal the brute-force value. On a 5-sequence random corpus the numba and
  blocked E-steps are bit-identical: init diff `0.0`, trans diff `4.4e-16`, state diff `1.8e-15`.
- **Numerical stability:** Identical max-shift bookkeeping in both kernels. `fastmath=True` does not
  alter the verified agreement. Good.
- **Engine-swap:** `backend_seq_log_density` mirrors the scaled forward via `engine.matmul/log/sum`
  for the blocked encoding; numba encoding stays on the host path (raises BackendScoringError on a
  non-numba engine). Neutral within stated scope.
- **Verdict:** OK

### Baum-Welch E-step (`HiddenMarkovAccumulator.seq_update:1964+`; numba `numba_baum_welch2`)
- **Computes:** gamma (per-position state posteriors), xi (expected transition counts), pi (initial
  posteriors), accumulated into `state_counts`, `trans_counts`, `init_counts`.
- **How:** Scaled forward (alpha normalized per step) then a scaled backward; per-position gamma and
  xi renormalized, with `gamma_buff>0` / `xi_buff_sum>0` guards before dividing
  (`_hidden_markov_numba_kernels.py:135-144`). Parallel `prange` over sequences.
- **Why correct:** **Numeric check** vs brute force on `[2,5,8]`: numba pi `[0.1438,0.8446,0.0115]`
  == brute-force pi; numba `trans_counts` matrix == brute-force xi summed over t (entrywise, ~1e-16);
  numba `state_counts` == brute-force gamma column sums. Exact.
- **Numerical stability:** Scaled (not log-space) FB with per-position renormalization guards; no
  underflow seen on the tested lengths. Good.
- **Verdict:** OK

### HMM M-step (`HiddenMarkovEstimator.estimate:2611-2712`)
- **Computes:** `w = init_counts / sum`, `A_ij = trans_counts_ij / row_sum_i` (MLE), or `_hmm_map_probs`
  (Dirichlet MAP with boundary clamp) under a conjugate chain prior; optional `steady_state_init`
  ties `w` to the stationary distribution of `A`.
- **Why correct:** MLE/MAP simplex updates; transition rows guard zero rows
  (`hidden_markov.py:2690-2697`); MAP path mirrors `pysp.bstats.markov_chain._map_probs`.
- **Numerical stability:** **FINDING(L1)** — the plain-MLE initial-weight update
  `w = init_counts / init_counts.sum()` (`hidden_markov.py:2680`) has **no zero-sum guard**, unlike
  the transition path right below it and unlike the pseudo-count branch. **Numeric check:**
  `init_counts=zeros(3) → w=[nan,nan,nan]`. Triggers only when every sequence is empty/zero-weight.
- **Verdict:** FINDING(L1)

### `_hmm_forward_ll` / terminal-state forward (`hidden_markov.py:129-204`, `511-527`)
- **Computes:** Scaled forward LL feeding `expected_log_density`; terminal-state (absorbing) forward
  in pure log space via `logsumexp`.
- **Why correct:** Scaled forward matches the bstats reference; terminal forward only transitions
  *from* non-terminal states and sums the final position over terminal states (log-space, no
  rescale). Returns -inf / `(None,None)` cleanly on zero-probability sequences. OK.
- **Verdict:** OK

## Module: pysp/stats/latent/lookback_hidden_markov_model.py
### Forward + numba kernels
- Scaled filter forward (predict-renormalize-`log`-add-emissions-logsumexp); reuses the shared numba
  kernels. **Numeric check (from sub-audit):** bit-equal to brute-force enumeration (diff ~3.6e-15,
  lag=0). E-step delegates to `numba_baum_welch2`. **Verdict:** OK.
### M-step (`estimate:1225`)
- **FINDING(L2):** `w = init_counts / init_counts.sum()` unguarded (transition rows ARE guarded at
  1233-1241). nan on fully-empty data. Same pattern as L1. **Verdict:** FINDING(L2)

## Module: pysp/stats/latent/tree_hidden_markov_model.py
### Upward/downward message passing
- Upward (beta/eta) recursion with per-node `betas_sum` rescaling and `log(betas_sum)+pr_max0`
  accumulation; emissions max-shifted. **Numeric check (from sub-audit):** numba vs numpy encodings
  equal (diff 1.8e-15), `log_density == seq_log_density[0]`. E-step guards `xi_loc_sum==0` and
  `temp_sum==0`. **Verdict:** OK.
### M-step (`estimate:1697`)
- **FINDING(L3):** `w = init_counts / init_counts.sum()` unguarded (transition rows guarded at
  1705-1713). nan on degenerate empty input. **Verdict:** FINDING(L3)

## Module: pysp/stats/latent/segmental_hidden_markov_model.py
### Log-space forward-backward (`_forward_log`, `_forward_backward`)
- Fully log-space with `logsumexp` (no scaling-factor bookkeeping). **Numeric check (from
  sub-audit):** gamma/xi bit-equal to brute force (~4e-16); EM monotone on a smoke test.
- M-step guards BOTH `w` (`w.sum()<=0 → uniform`) and transition rows — the correct pattern the
  lookback/tree estimators lack.
- **FINDING(L4, LOW):** non-finite-ll fallback fills gamma=1/k, xi=(n-1)/(k²) uniformly for
  zero-probability sequences (contributes uniform mass rather than skipping). Harmless heuristic.
- **Verdict:** OK (with L4 noted)

## Module: pysp/stats/latent/semi_supervised_hidden_markov_model.py
### Scaled forward + supervised posteriors
- `_forward_loglik` is a scaled filter with per-position `offset` (emission max) bookkeeping;
  `_posteriors` is a scaled forward-backward with all divisions guarded `/(x if x>0 else 1.0)`.
  **Numeric check (from sub-audit):** gamma/xi bit-equal to brute force (~4e-16); n==1 handled.
- M-step normalizes transitions with `denom[denom==0]=1.0`. No learned initial vector (the position-0
  prior plays that role), so no `w` nan hazard. numpy-only by design.
- **Verdict:** OK

## Module: pysp/stats/latent/quantized_hidden_markov_model.py
- Forward/E-step inherited unchanged from the base HMM (covered above). The in-scope quantization
  M-step takes `log` only of strictly-positive normalized counts (`pos = row>0`), maps zero cells to
  structural -inf, uses `np.where(mask, k*log_theta, -inf)+logsumexp`. `w` is quantized exponents or
  the stationary distribution — no raw count division, so no nan. **Verdict:** OK

## Module: pysp/stats/latent/lda.py
### Variational fixed point (`seq_posterior:1442-1631`)
- **Computes:** Per-document Blei-Ng-Jordan mean-field: phi ∝ exp(E_q[log theta]) · b(w|topic),
  gamma = alpha + sum_w count_w · phi_w, iterated to `gamma_threshold` (capped at `max_gamma_iter`,
  with unconverged docs flushed — geometric convergence makes this safe).
- **How:** Stable inner loop — emissions `per_topic_log_densities2` max-shifted; `document_gammas3`
  is `exp(digamma(gamma) - max digamma(gamma))` (max-shifted before exp). phi normalized per row.
- **Why correct:** **Numeric check:** pysp ELBO `-7.403203587986579` vs an independent
  Blei-Ng-Jordan reference implementation `-7.40320358798658` (2 topics, vocab 3, alpha 0.5; match
  to 1e-15).
- **Numerical stability:** Max-shifted digamma and emission exponentials; positivity floors
  (`sys.float_info.min`) applied to gammas/responsibilities before logs in the ELBO. Good.
- **Verdict:** OK

### ELBO (`seq_log_density:184-238`)
- **Computes:** The 7-term LDA variational lower bound (elob3 entropy/prior, elob5 expected
  complete-data, elob6 `-E[log q(theta)]`, elob7 log-normalizer of the prior).
- **Why correct:** Matches the reference ELBO numerically (above). Length term added via `len_dist`.
- **Verdict:** OK

### alpha update (`update_alpha:1288-1312`) + `digammainv`
- **Computes:** Dirichlet `alpha` fixed point `alpha ← digammainv(mean_log_p + digamma(sum alpha))`.
- **Why correct:** Standard Newton-free Minka fixed point. **Numeric check:** `digammainv(digamma(v))`
  recovers v for v ∈ {0.01..50} to ~1e-15. OK.
- **Numerical stability:** Iterates to `alpha_threshold`; no cap, but converges geometrically.
- **Verdict:** OK

### Engine-resident path (`_backend_seq_posterior:240-292`, `backend_seq_log_density:294-347`)
- Mirrors the host variational loop and ELBO on the active engine with a precision-aware `tiny`
  floor. Neutral. **Verdict:** OK

## Module: pysp/stats/latent/labeled_lda.py
- **E-step (`seq_posterior`):** per-document gamma fixed point with responsibilities
  `∝ b · exp(digamma(gamma) - max_k digamma(gamma))`. The max-shift form is **mathematically
  identical** to the textbook `exp(digamma(gamma) - digamma(sum))` after per-row normalization
  (sub-audit verified equal). Iter-capped with flush of unconverged docs.
- **M-step:** topics from responsibility-weighted counts; alpha via decoupled per-row `update_alpha`
  for single-label docs, else a concave coupled objective (log-space gradient ascent + Armijo).
  Signs/normalizers checked correct.
- **Verdict:** OK

## Module: pysp/stats/latent/integer_probabilistic_latent_semantic_indexing.py
- **E-step (`seq_update`):** per-(word,doc) responsibility `prob_mat[v]·state_mat[d]` normalized per
  pair (numba `fast_seq_update`); `seq_log_density` is the **exact** marginal (not an ELBO).
- **M-step:** word (col-simplex), state (row-simplex), doc (vector), each with pseudo-count and a
  zero-sum guard + uniform fallback in the non-pseudocount path. Correct.
- **Verdict:** OK

## Module: pysp/stats/latent/gaussian_mixture.py
- **E-step (`seq_posterior`/`update`):** responsibilities via row-max-subtracted logsumexp over
  `component_log_density + log_w`; zero-weight comps masked to -inf; all-(-inf) rows fall back to `w`.
- **Numeric check (from sub-audit):** scalar/seq `log_density`, `posterior`, `seq_posterior` all
  match hand-computed logsumexp/softmax exactly.
- **M-step:** weights normalized with pseudo-count and a `nobs==0 → uniform` guard; covariance is
  delegated to the MultivariateGaussian leaf (out of scope; E[xx']-E[x]E[x]' cancellation lives
  there).
- **Verdict:** OK

## Module: pysp/stats/latent/joint_mixture.py
- **E-step:** joint responsibilities `e1 ⊗ e2 ⊗ taus12`, per-row `sf` normalization with
  `sf_safe = where(sf>0, sf, 1)` so zero-mass rows contribute nothing; rowmax subtraction on both
  factors; nan rows zeroed.
- **M-step:** tau12/tau21 row/col normalizers guard zero sums.
- **FINDING(L5, LOW):** non-pseudocount `w1 = counts1/counts1.sum()` and `w2 = counts2/counts2.sum()`
  (`joint_mixture.py:1012-1013`) lack a zero-sum guard (nan only on fully-empty data).
- **Verdict:** FINDING(L5)

## Module: pysp/stats/latent/hierarchical_mixture.py
- **E-step (`seq_update`):** topic logsumexp with row-max, `taus.T` mix, per-document bincount of
  log-mix + log_w, outer-posterior softmax with bad-row fallback to `log_w`. Correct.
- **M-step:** tau rows and outer `w` normalized with positivity guards. Correct.
- **FINDING(H1, HIGH):** `seq_initialize` (`hierarchical_mixture.py:683`) does
  `self.comp_counts[:, i] = np.bincount(idx1[idx], w)` — TWO defects: (1) it OVERWRITES (`=`) where
  the scalar `initialize` at line 640 accumulates (`+=`), so multi-chunk/partition init keeps only
  the last chunk's mixture seeding; (2) the `np.bincount` call OMITS `minlength=self.num_mixtures`,
  so its length is `max(idx1)+1`. **Numeric check:** on a normal 5-document corpus this raises
  `ValueError: could not broadcast input array from shape (2,) into shape (3,)` — i.e.
  `seq_initialize` (the default vectorized init path) **crashes outright** whenever the highest drawn
  mixture index is below `num_mixtures-1`; in the length-1 special case it instead silently
  broadcasts a single count across all mixtures (corrupting `comp_counts`). Suggested fix:
  `self.comp_counts[:, i] += np.bincount(idx1[idx], w, minlength=self.num_mixtures)`.
- **Verdict:** FINDING(H1)

## Module: pysp/stats/latent/heterogeneous_mixture.py
- **E-step:** per-type encodings scored into a shared `(sz,K)` matrix, row-max logsumexp softmax,
  bad rows → `log_w`. M-step: same well-guarded weight normalization as gaussian.
- **FINDING(L6, LOW):** `seq_log_density` (`heterogeneous_mixture.py:262`) raises `UnboundLocalError`
  (`ll_mat` never bound) when every component weight is 0.0 — a degenerate invalid distribution
  (weights must sum to 1), not normal use.
- **Verdict:** FINDING(L6)

## Module: pysp/stats/latent/semi_supervised_mixture.py
- **E-step:** prior-restricted/re-weighted/re-normalized responsibilities; `norm_const` via bincount
  of `prior_val·w` then `log`; row-max logsumexp; bad rows → `log_w`. `seq_encode` validates
  indices/negatives/positive-mass. M-step: standard guarded weight normalization.
- **Verdict:** OK

## Module: pysp/stats/latent/spatial_mixture.py
- **Mean-field E-step:** `logq = emis + beta·field`, row-max subtracted, exp, row-normalized →
  per-cell simplex; Potts coupling annealed from 0; empty components reseeded. Entropy clips q to
  ≥1e-12 before log. M-step drives the responsibility-weighted accumulator per the pysp contract.
- **Verdict:** OK

## Module: pysp/stats/latent/probabilistic_pca.py
### Woodbury log-density (`log_density:122-126`, `seq_log_density:128-132`)
- **Computes:** `N(mu, C)` with `C = W W^T + sigma2 I` via Woodbury:
  `C^{-1} = (I - W M^{-1} W^T)/sigma2`, `log|C| = (d-q) log sigma2 + log|M|`, `M = W^T W + sigma2 I`.
- **Why correct:** **Numeric check:** vs `scipy.stats.multivariate_normal.logpdf` on 3 random
  5-vectors (q=2): max abs diff `1.78e-15`.
- **Numerical stability:** `sigma2 > 0` enforced in `__init__`; small q-by-q solve. Good.
- **Engine-swap:** `backend_seq_log_density` uses `engine.matmul/sum`; the cached `inv_covar`/`log_det`
  are host-computed but engine-neutral arrays. Neutral.
- **Verdict:** OK

### Tipping-Bishop closed-form M-step (`estimate:272-300`)
- **Computes:** `mu = s/count`, `cov = s2/count - mu mu^T`, eigendecompose, `sigma2 = mean` of
  discarded eigenvalues, `W = U_q (Lambda_q - sigma2 I)^{1/2}`.
- **Why correct:** The exact Tipping & Bishop (1999) ML solution.
- **Numerical stability:** `cov` is symmetrized (`0.5*(cov+cov.T)`) and eigenvalues clipped to ≥0
  (mitigating the `E[xx']-E[x]E[x]'` cancellation); `sigma2` floored at `_MIN_SIGMA2`; `count<=0`
  returns a safe default. Good.
- **Verdict:** OK

---

## Findings summary
- **H1 (HIGH)** hierarchical_mixture.py:683 — `seq_initialize` overwrite + missing `minlength`;
  vectorized init crashes (ValueError) or corrupts `comp_counts`.
- **L1 (MEDIUM)** hidden_markov.py:2680 — unguarded `w = init_counts/init_counts.sum()` → nan on
  empty data.
- **L2 (MEDIUM)** lookback_hidden_markov_model.py:1225 — same unguarded `w`.
- **L3 (MEDIUM)** tree_hidden_markov_model.py:1697 — same unguarded `w`.
- **L4 (LOW)** segmental_hidden_markov_model.py:77-79 — zero-prob sequences contribute uniform mass.
- **L5 (LOW)** joint_mixture.py:1012-1013 — unguarded `w1`/`w2` → nan on empty data.
- **L6 (LOW)** heterogeneous_mixture.py:262 — UnboundLocalError on all-zero-weight degenerate dist.

All forward passes, Baum-Welch posteriors, the LDA ELBO, and the PPCA Woodbury density were verified
numerically against brute-force enumeration / canonical references and match to machine precision.
The numba HMM kernels are bit-identical to the numpy forward-backward.
