# Sets / Bernoulli-set models — computation ledger

Audited modules:
- `pysp/stats/sets/bernoulli_set.py`
- `pysp/stats/sets/integer_bernoulli_set.py`
- `pysp/stats/sets/integer_bernoulli_edit.py`
- `pysp/stats/sets/integer_step_bernoulli_edit.py`

All numeric checks below were run with `.venv/bin/python` and confirmed against brute-force
reference computations.

---

## Module: pysp/stats/sets/bernoulli_set.py

### `__init__` log-parameter precompute (`bernoulli_set.py:110-141`)
- **Computes:** Per-element `log_dmap[k] = log p_k − log(1−p_k)` and constant `nlog_sum = Σ_k log(1−p_k)`, so `log p(x) = nlog_sum + Σ_{k∈x} log_dmap[k]`.
- **How:** `vv = np.log1p(-v)` for the complement; `log_dmap = log(v) − vv`; degenerate p∈{0,1} handled per-branch (min_prob=0 → required set / −inf; min_prob>0 → floor via min_pv/min_nv).
- **Why correct:** Independent-inclusion factorization `Σ_{in} log p + Σ_{out} log(1−p) = Σ_k log(1−p) + Σ_{in}(log p − log(1−p))`. Verified vs brute force (5 sets, all `np.isclose`).
- **Numerical stability:** Uses `log1p(-v)` for the complement (good near p→0). p=1 with min_prob=0 → `required`, log_dmap=0; p=0 → −inf. min_prob>0 floors both tails.
- **Engine-swap:** Host numpy precompute on construction; fine.
- **Verdict:** OK

### `log_density` (`bernoulli_set.py:251-270`)
- **Computes:** `nlog_sum + Σ_{v∈x} log_dmap[v]`, −inf if a required element is absent.
- **Why correct:** Matches factorization; `required.issubset(x)` gate handles p=1. Verified.
- **Verdict:** OK

### `seq_log_density` (`bernoulli_set.py:272-295`)
- **Computes:** Vectorized log_density over encoded `(sz, idx, val_map_inv, xs)`.
- **How:** `bincount(idx, weights=log_dmap[val_map_inv][xs])` + nlog_sum; required-mask via per-row required count.
- **Why correct:** Scalar/seq parity verified including the required-mask (p=1, min_prob=0): `[-0.693,-0.693,-inf,-inf]` matches scalar.
- **Numerical stability:** n/a beyond construction.
- **Engine-swap:** numpy host; `backend_seq_log_density` (297-315) is the engine-neutral mirror, also OK.
- **Verdict:** OK

### `expected_log_density` / `seq_expected_log_density` (`bernoulli_set.py:200-227`)
- **Computes:** VB term `E_q[log p(x|p)] = Σ_{k∉x} E[log(1−p_k)] + Σ_{k∈x} (E[log p_k] − E[log(1−p_k)])` with `E[log p_k]=ψ(a_k)−ψ(a_k+b_k)`, `E[log(1−p_k)]=ψ(b_k)−ψ(a_k+b_k)`.
- **Why correct:** Standard Beta-conjugate digamma expectations. Scalar/seq parity verified (`np.allclose`).
- **Verdict:** OK

### `BernoulliSetEnumerator` (`bernoulli_set.py:430-476`)
- **Computes:** Subsets in descending log-density via per-element two-choice best-first product, offset nlog_sum.
- **Why correct:** Each inclusion-flag tuple maps to a unique subset; sorted two-choice streams + ProductEnumerator give descending order. Guards p∉[0,1] via EnumerationError.
- **Verdict:** OK

### `BernoulliSetSampler.sample` (`bernoulli_set.py:497-519`)
- **Computes:** Each element included iff `U ≤ p_k`.
- **Why correct:** Standard Bernoulli draw. `<=` boundary inconsequential for continuous U.
- **Verdict:** OK

### Accumulator suff-stats (`bernoulli_set.py:540-585`)
- **Computes:** Per-element weighted inclusion count + total weight.
- **Why correct:** `update`/`seq_update`/`seq_update_engine` parity verified (counts `{a:3,b:2,c:3}, tot 7`). seq path emits `np.str_` keys but they hash/compare equal to `str` (verified), so `combine` merges correctly.
- **Verdict:** OK

### Plain-MLE `estimate` (`bernoulli_set.py:791-826`, non-conjugate branches)
- **Computes:** `p_k = count_k / n` (or pseudo-count smoothed); n=0 → 0.5 fallback.
- **Why correct:** Bernoulli MLE; divide-by-zero guarded (`if suff_stat[1] != 0`). Verified recovery on simulated data.
- **Verdict:** OK

### Conjugate `_estimate_conjugate` + `_beta_posterior_mode` (`bernoulli_set.py:770-789`, `869-900`)
- **Computes:** Posterior Beta(a0+v, b0+tot−v) and its **mode** as the point estimate `p_k`.
- **How:** Shifted counts `a=(a0−1)+v`, `b=(b0−1)−v+tot`; correct mode is `a/(a+b)` for a,b>0.
- **Why WRONG:** Faithfully ports the bstats branch logic, which is buggy:
  - `a>b` branch → `a/(a+b)` ✓
  - `b>a` branch → `(a−1)/(a+b−2)` ✗ (should be `a/(a+b)`)
  - `a==b` → no branch fires → `else: return 1.0` ✗ (should be `a/(a+b)=0.5`)
  - Verified: Beta(2,2), v=5/tot=10 (a==b) returns **1.0** (true mode 0.5); v=3/tot=10 (b>a) returns 0.3 (true mode 0.333); v=1/tot=10 returns 0.1 (true 0.167). Flows into fitted `pmap` (`{'x':0.3,'y':1.0}` for v=3,5). The carried-forward `posteriors` (a0+v,b0+tot−v) are correct; only the point estimate is wrong.
- **Verdict:** FINDING(B1)

---

## Module: pysp/stats/sets/integer_bernoulli_set.py

### `__init__` (`integer_bernoulli_set.py:95-109`)
- **Computes:** `log_dvec = log_pvec − log_nvec`, `log_nsum = Σ log_nvec` (finite entries only); when log_nvec=None it is derived as `log1p(−exp(log_pvec))`.
- **Why partially WRONG (degenerate p=1):** When `p_k=1` (`log_pvec[k]=0`), `log_nvec[k]=log1p(−1)=−inf`. `log_nsum` **masks** this −inf via `[np.isfinite(...)]`, and `log_dvec[k]=0−(−inf)=+inf`. Result:
  - `log_density` of a set containing k → `+inf` (verified: returns `inf`, true value −1.05).
  - `log_density` of a set **excluding** k → finite (verified −1.05) instead of the correct `−inf` (k has inclusion prob 1, so absence is impossible).
  - Unlike `BernoulliSetDistribution`, there is no `required`/forced-membership mechanism, so a p=1 element silently produces nonsense densities. The enumerator (line 222) DOES raise `EnumerationError` for this case, confirming the density form is known-degenerate, but `log_density`/`seq_log_density` do not guard.
  - **Reachable from the estimator** at `min_prob=0`: an element present in *every* observation yields `is_one` → `log_nvec=−inf, log_pvec=0` → same +inf density on essentially all training data (verified end-to-end). Default `min_prob=1e-128>0` floors `nvec` and avoids it (verified safe).
- **Numerical stability:** Emits a `divide by zero encountered in log1p` RuntimeWarning when log_nvec=None and any p=1.
- **Verdict:** FINDING(I1)

### `log_density` / `seq_log_density` / `backend_seq_log_density` (`integer_bernoulli_set.py:121-141`)
- **Computes:** `Σ_{k∈x} log_dvec[k] + log_nsum`.
- **Why correct (non-degenerate):** Independent-inclusion factorization. Brute-force + scalar/seq parity verified on random params. (Degeneracy is the I1 p=1 issue above.)
- **Verdict:** OK (modulo I1)

### Accumulator (`integer_bernoulli_set.py:294-348`)
- **Computes:** Per-integer weighted inclusion count vector + total weight.
- **Why correct:** `bincount` length ≤ num_vals, `pcnt[:n] += agg` safe (trailing zeros / empty handled). seq/scalar parity holds.
- **Verdict:** OK

### `estimate` (`integer_bernoulli_set.py:451-490`)
- **Computes:** `p_k = count_k/n`, `1−p_k = (n−count_k)/n` in log-space, with pseudo-count and min_prob variants.
- **Why correct:** Bernoulli MLE; n=0 → 0.5 fallback (guarded); min_prob>0 floors both; min_prob=0 sets exact −inf for p=0/p=1 tails. Verified recovery on 80k samples. (The p=1 → log_pvec=0 output feeds I1 in the distribution constructor.)
- **Verdict:** OK (estimator arithmetic correct; downstream I1 is in the distribution)

---

## Module: pysp/stats/sets/integer_bernoulli_edit.py

### `__init__` edit-matrix precompute (`integer_bernoulli_edit.py:99-121`)
- **Computes:** 4-col `log_edit_pmat = [log p(miss|miss), log p(miss|pres), log p(pres|miss), log p(pres|pres)]`; from 2-col input fills complements via `log1p(−exp(·))`. `log_nsum = Σ log p(miss|miss)`; `log_dvec = log_edit_pmat[:,1:] − log_edit_pmat[:,0,None]` (cols = miss|pres, pres|miss, pres|pres relative to miss|miss).
- **Why correct:** Conditional-independence transition factorization; empty-set baseline = all-missing → miss|miss product. Verified.
- **Numerical stability:** `log1p` complement; `log_nsum` masks −inf miss|miss entries (only matters for degenerate transition probs).
- **Verdict:** OK

### `log_density` / `seq_log_density` / `backend_seq_log_density` (`integer_bernoulli_edit.py:144-206`)
- **Computes:** `log p(x1|x0) = log_nsum + Σ_kept log_dvec[·,2] + Σ_added log_dvec[·,1] + Σ_removed log_dvec[·,0]`, plus `init_dist.log_density(x0)`.
- **How:** `in10=isin(x1,x0)` (kept), `~in10` (added), `in01=isin(x0,x1,invert=True)` (removed).
- **Why correct:** Edit-type→column mapping matches encoder (type 0/1/2 ↔ cols 0/1/2 ↔ removed/added/kept). Brute-force over all 4 transition cases + scalar/seq parity verified (4 test pairs, all `np.isclose`).
- **Verdict:** OK

### Sampler (`integer_bernoulli_edit.py:466-511`)
- **Computes:** next-set inclusion: `U≤p(pres|miss)` default, overridden to `U≤p(pres|pres)` for prev members.
- **Why correct:** Correct conditional Bernoulli; prev from init_dist. `sample_given` mirrors it.
- **Verdict:** OK

### Enumerator (`integer_bernoulli_edit.py:338-440`)
- **Computes:** (prev,next) pairs in descending joint log-density via prev-stream × conditional next-product, merged with a heap frontier.
- **Why correct:** Best-first merge; `_valid_prev` filters non-integer-set prevs; conditional next streams sorted and offset by lp_prev.
- **Verdict:** OK

### Accumulator edit counts (`integer_bernoulli_edit.py:544-644`)
- **Computes:** `pcnt[k] = (removed, added, kept)` weighted counts; total weight; init child stats.
- **Why correct:** scalar `update` matches `seq_update` aggregation (col0/1/2). Recovery verified end-to-end.
- **BUG:** `seq_update` (line 609) calls `estimate.init_dist` **without** the `None if estimate is None else …` guard that scalar `update` (line 564) and `seq_update_engine` (line 643) both use. Verified: `seq_update(enc, w, None)` raises `AttributeError: 'NoneType' object has no attribute 'init_dist'`; `update(..., None)` works. The idiomatic pattern (e.g. `composition.py:195`) guards None. Not hit by the standard `seq_estimate` driver (always passes prev model), but breaks any caller honoring the protocol's nullable `estimate`.
- **Verdict:** FINDING(E1)

### `estimate` M-step (`integer_bernoulli_edit.py:814-921`)
- **Computes:** `p(pres|miss)=added/s0`, `p(miss|pres)=removed/s1`, `p(pres|pres)=kept/s1`, `p(miss|miss)=(s0−added)/s0`, where `s1=removed+kept` (#prev-present), `s0=tot−s1` (#prev-missing); normalized per conditioning state.
- **Why correct:** Closed-form conditional Bernoulli MLE; both conditional pairs verified to normalize to 1; all four conditional probs recovered on 120k samples with a real init_dist (true [0.2,0.5,0.3,0.7]/[0.8,0.6,0.9,0.4] recovered to 3 decimals). n=0 → 0.5; nz0/nz1 masks guard unobserved conditioning states (p(miss|miss)=1, p(pres|pres)=1 defaults).
- **Numerical stability:** `np.errstate(divide="ignore")` around `log(0)` for exact-0 probs (intended −inf). min_prob>0 floors+renormalizes.
- **Verdict:** OK

---

## Module: pysp/stats/sets/integer_step_bernoulli_edit.py

Distribution, log_density, seq_log_density, sampler, encoder, and accumulator are
structurally identical to `integer_bernoulli_edit.py` (the estimator differs).

### log_density / seq_log_density (`integer_step_bernoulli_edit.py:140-203`)
- **Verdict:** OK (same factorization, verified by inheritance of the identical logic).

### Accumulator `seq_update` (`integer_step_bernoulli_edit.py:490-512`)
- **BUG:** Line 512 `self.init_acc.seq_update(init_enc, weights, estimate.init_dist)` — same un-guarded `estimate.init_dist` as the non-step variant. Verified crash on `seq_update(enc, w, None)`; scalar `update(None)` works (it guards, lines 462-466).
- **Verdict:** FINDING(E2)

### `__effective_step_counts` (`integer_step_bernoulli_edit.py:728-755`)
- **Computes:** removal successes/trials = (count[:,0], s1); addition successes/trials = (count[:,1], s0); with optional pseudo-count smoothing.
- **Why correct:** Matches the binomial counts for the two step-fitted probabilities (removal = miss|pres over present-prev; addition = pres|miss over missing-prev).
- **Verdict:** OK

### `__get_pqk` two-level step fit (`integer_step_bernoulli_edit.py:757-811`)
- **Computes:** Best two-level (p,q) split: sort elements by empirical rate, for each prefix k assign pooled p to top-k and pooled q to the rest, keep the split maximizing the binomial log-likelihood.
- **How:** Cumulative sums of successes/trials; per-split `v1+v2` with `sh log p + (th−sh) log1p(−p)` (guarded `if sh>0` / `if th>sh`); elements with no trials get the overall pooled rate (0.5 if none).
- **Why correct:** Standard profile-likelihood step search; `log1p(-p)` for the complement; clip via `__clip_prob` to `[min_prob, 1−min_prob]`. Verified end-to-end: a two-level generator (add 0.8/0.1, rem 0.7/0.05) recovered exactly two distinct levels matching the truth on 150k samples.
- **Numerical stability:** `log1p(-p)` complement; success/trial-edge guards prevent `0·log0`. Clip bounds enforced.
- **Verdict:** OK

### `estimate` (`integer_step_bernoulli_edit.py:813-842`)
- **Computes:** Per-element edit probs replaced by the two `__get_pqk` step fits; builds 4-col log_pmat with `log(arr)`/`log(1−arr)`.
- **Why correct:** addition arr2 → cols (pres|miss=log arr2, miss|miss=log(1−arr2)); removal arr1 → cols (miss|pres=log arr1, pres|pres=log(1−arr1)). Conditional pairs normalize by construction. Verified.
- **Numerical stability:** `np.errstate(divide="ignore")`; arr clipped away from 0/1 when min_prob>0.
- **Verdict:** OK

---

## Findings summary

- **B1** (`bernoulli_set.py:887-900`): `_beta_posterior_mode` mis-computes the Beta posterior mode for `b>=a`; `a==b` returns 1.0 (true 0.5), `b>a` returns `(a-1)/(a+b-2)` instead of `a/(a+b)`. Wrong conjugate point estimates. HIGH.
- **I1** (`integer_bernoulli_set.py:101-109`,`121-132`): degenerate `p_k=1` produces `+inf`/finite-instead-of-`-inf` densities (no `required` mechanism); reachable via estimator at `min_prob=0`. MEDIUM.
- **E1** (`integer_bernoulli_edit.py:609`): `seq_update` dereferences `estimate.init_dist` without a None guard → crash when `estimate=None`. MEDIUM.
- **E2** (`integer_step_bernoulli_edit.py:512`): same un-guarded `estimate.init_dist` in `seq_update`. MEDIUM.
