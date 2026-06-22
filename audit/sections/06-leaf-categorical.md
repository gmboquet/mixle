# Categorical / multinomial leaf distributions — computation ledger

Scope: `pysp/stats/leaf/{categorical,integer_categorical,integer_multinomial,categorical_multinomial,dirichlet_multinomial,integer_uniform_spike}.py`.
All numeric checks run with `.venv/bin/python`.

---

## Module: pysp/stats/leaf/categorical.py

### log_density / density (`categorical.py:286,300`)
- **Computes:** `density(x) = pmap.get(x, default_value) / (1 + default_value)`; `log_density = log(p) - log1p(default_value)`.
- **How:** dict lookup with `default_value` fallback; division by `1 + default_value` so the (countably many) off-support labels each carry `default_value/(1+default_value)`.
- **Why correct:** with `default_value=0` reduces to the plain categorical `p_x`; with `default_value>0` the seen-mass `sum p_i/(1+dv)` plus per-unseen `dv/(1+dv)` sums to 1 over the seen∪default universe. Verified: `{a:.2,b:.5,c:.3}` sums to 1.0; with `dv=.1` seen mass 0.9091 and each unseen 0.0909 — consistent.
- **Numerical stability:** `log(0) → -inf` guarded (`p <= 0.0` returns `-inf`); `log1p` used for the default term. `np.errstate(divide="ignore")` in `seq_log_density`.
- **Engine-swap:** `backend_seq_log_density` keeps the object→index lookup host-side and returns an engine tensor; parity with numpy verified for sibling leaves.
- **Verdict:** OK

### seq_log_density (`categorical.py:315`)
- **Computes:** vectorized `log_density` over the encoded `(xs, val_map_inv)`.
- **Why correct:** maps unique-value log-probs then gathers by `xs`. Verified equal to scalar path incl. unseen `z → -inf` and zero-prob category `a → -inf`.
- **Verdict:** OK

### CategoricalEstimator.estimate — count normalization + smoothing (`categorical.py:892`)
- **Computes:** MLE `p_k = n_k/N`; pseudo-count branch `(n_k + pc/K)/(N + pc)`; member-suff_stat branch `(n_k + member_k·pc)/(N + sum(member)·pc)`; `default_value` branch `dv=(1/N)^2`.
- **Why correct:** all three normalization branches verified to sum to 1 and match hand-computed values (e.g. pc-only `{a:3,b:1,c:0}`, pc=3 → 4/7,2/7,1/7; member branch → 4/6,2/6). `nobs==0` falls back to uniform (verified).
- **Numerical stability:** n/a.
- **Verdict:** FINDING(cat-mutate) — the pure-pseudo_count branch (lines 939–942) writes probabilities back into the *caller's* `suff_stat` dict in place (`suff_stat[k] = ...; p_map = suff_stat`). Confirmed: input `{'a':3,'b':1}` becomes `{'a':0.667,'b':0.333}` after `estimate`.

### Dirichlet MAP / conjugate path (`categorical.py:860`)
- **Computes:** `num_k = max(alpha_k - 1 + n_k, 0)`, `p = num/sum(num)`, posterior `alpha+n`; falls back to posterior mean when the MAP is degenerate (`sum(num)=0`).
- **Why correct:** standard Dirichlet-categorical MAP (mode `(alpha_k+n_k-1)/(sum-K)`) with boundary clamp; posterior-mean fallback is sensible.
- **Verdict:** OK

### Sampler / Enumerator / quantized indices (`categorical.py:516,560,484`)
- **Computes:** `rng.choice` over `(levels,probs)`; enumeration in descending prob; bit-quantized indices guard `no_default`.
- **Why correct:** standard; enumerator/quantized correctly raise `EnumerationError` for non-zero `default_value` (unbounded support).
- **Verdict:** OK  *(see LOW note: `sample(size=None)` calls `rng.choice(..., size=None)` then indexes `self.levels[idx]` with a 0-d array — works but relies on numpy scalar coercion.)*

---

## Module: pysp/stats/leaf/integer_categorical.py

### log_density / seq_log_density (`integer_categorical.py:276,295`)
- **Computes:** `log_p_vec[x-min_val]` on `[min_val,max_val]`, else `-inf`.
- **How:** integer index into precomputed `log_p_vec`; float inputs accepted only if integer-valued (`xi != x` reject; seq path uses `|x-round(x)|<1e-9`).
- **Why correct:** Verified `p_vec` from M-step sums to 1, `log_density(2.0)` works, `log_density(2.5) → -inf`, seq float path matches.
- **Numerical stability:** `np.log` of zero prob → `-inf` via `errstate(divide=ignore)` in `__init__`. Float-index guard avoids a TypeError on `log_p_vec[float]`.
- **Engine-swap:** `backend_seq_log_density` parity with numpy verified (`True`).
- **Verdict:** OK

### Estimator (M-step + pseudo_count) (`integer_categorical.py:952`)
- **Computes:** `p = count_vec/sum`; pseudo-count `(count + pc/K)/(N+pc)`; conjugate Dirichlet MAP analogous to categorical.
- **Why correct:** verified sums to 1; pseudo_count smoothing verified.
- **Verdict:** OK

### Accumulator dynamic-range growth (`integer_categorical.py:614,653`)
- **Computes:** bincount histogram with min/max bookkeeping; `seq_update_engine` reduces on the active engine.
- **Why correct:** range-extension logic copies the old `count_vec` into the correctly-offset slice. Standard.
- **Verdict:** OK

---

## Module: pysp/stats/leaf/integer_multinomial.py

### log_density (`integer_multinomial.py:226`)
- **Computes:** `sum_k cnt_k · log p_{val_k}` — **NO** multinomial coefficient and **NO** `log(n!)`/`log(x_k!)`.
- **Why "correct":** matches the seq path (`-6.50229...` verified) and the enumerator's documented semantics (the IntegerMultinomialEnumerator docstring states the support is countably infinite precisely because the coefficient is omitted). This is an intentional un-normalized scoring form used inside Multinomial/composite models.
- **Numerical stability:** FINDING(im-nan) — for an **out-of-support** value with **count 0** the scalar path computes `(-inf) * 0 = nan` (line 246). Verified: `log_density([(5,0)]) → nan` while `[(5,2)] → -inf`. The seq path is safe (masks before multiplying by `cnt`).
- **Docstring mismatch:** FINDING(im-doc) — the module header (lines 9–14) and `log_density` docstring (line 231) state the density includes `log(n!) - sum log(x_k!)`, which the code does not compute.
- **Verdict:** FINDING(im-nan), FINDING(im-doc)

### seq_log_density (`integer_multinomial.py:249`)
- **Computes:** masked `log_p_vec` gather × `cnt`, `bincount` by observation index, plus optional `len_dist` term.
- **Why correct:** out-of-range rows held at `-inf` before the `*= cnt` (so cnt=0 there stays `-inf`, not NaN). Verified vs scalar.
- **Verdict:** OK

### Estimator M-step (`integer_multinomial.py:1028`)
- **Computes:** `p = count_vec/sum`; pseudo_count variants `(count + pc/K)/(N+pc)`; all branches guard `sum==0 → uniform`.
- **Why correct:** standard count normalization; zero-mass fallback present.
- **Verdict:** OK

### Sampler / Enumerator (`integer_multinomial.py:521,468`)
- **Computes:** `rng.multinomial(n, p)` with trial count from `len_dist`; best-first length-frontier enumeration.
- **Why correct:** sampler matches the count-vector data type; enumerator correctly raises when a category has prob 1 (divergent support).
- **Verdict:** OK

---

## Module: pysp/stats/leaf/categorical_multinomial.py (MultinomialDistribution)

### log_density (`categorical_multinomial.py:161`)
- **Computes:** `sum_j n_j · dist.log_density(V_j)` (+ `len_dist.log_density(n)`); if `len_normalized`, divide the value term by `n` (geometric mean).
- **Why "correct":** matches the integer-multinomial scoring form (no multinomial coefficient). Verified `-6.50229...` against the raw sum, and `len_normalized` gives `raw/4` (verified `-1.27899...`).
- **Numerical stability:** trial count `cc` starts at integer 0 so integer-supported `len_dist` gets an int. n/a otherwise.
- **Docstring mismatch:** FINDING(mn-doc) — module header (line 15) and `log_density` docstring (line 173) claim `log(n!) - sum n_j·log(p_j) - log(n_j!)`; the code omits the coefficient AND has the sign as `+ n_j·log p_j` (the docstring's `- sum n_j log p_j` is also wrong-signed). Same defect class as integer_multinomial.
- **Verdict:** FINDING(mn-doc)

### seq_log_density (`categorical_multinomial.py:201`) + encoder (`992`)
- **Computes:** child `seq_log_density` × per-value counts, `bincount` to per-observation, optional `len_normalized` scaling by `icnt = 1/n`, plus `len_dist`.
- **Why correct:** encoder builds `rv2 = 1/n` (reciprocal trial counts) which scales the summed value term — verified equal to scalar `len_normalized` path.
- **Verdict:** OK

### to_exponential_family (`categorical_multinomial.py:90`)
- **Computes:** exp-family view only when `len_dist` is Null and not `len_normalized` and the value element is itself exp-family.
- **Why correct:** the guards exactly match the cases where the single-exp-family form holds.
- **Verdict:** OK

### Accumulator / Estimator (`categorical_multinomial.py:573,888`)
- **Computes:** delegates count-weighted updates to the value accumulator and `n`-counts to the length accumulator; `len_normalized` reweights by `1/n`.
- **Why correct:** verified the child categorical M-step recovers count-proportional probabilities from a 2-observation multinomial dataset.
- **Verdict:** OK

### MultisetProductEnumerator (`categorical_multinomial.py:405`)
- **Computes:** best-first enumeration of size-n multisets over a sorted log-prob stream.
- **Why correct:** rank-tuple successor scheme (only the right-most of equal ranks moves) reaches each multiset once with monotone non-increasing scores; `_score` recomputed to avoid float drift.
- **Verdict:** OK

---

## Module: pysp/stats/leaf/dirichlet_multinomial.py

### log_density (`dirichlet_multinomial.py:62`)
- **Computes:** `log P(x) = [log n! + log Γ(Σα) − log Γ(n+Σα)] + Σ_k [log Γ(x_k+α_k) − log Γ(α_k) − log Γ(x_k+1)]`.
- **How:** precomputed `_log_const`, per-element `gammaln`; `-inf` when any `x_k<0` or `Σx ≠ n`.
- **Why correct:** Polya/Beta-function ratio `B(α+x)/B(α)` × multinomial coefficient. Verified: enumerating all count vectors of `n=5, K=3` sums to **0.99999999...** (normalizes); `seq_log_density` matches scalar; wrong total → `-inf`.
- **Numerical stability:** all in log-space via `gammaln`; `==` on the float total `xx.sum()` is the only sharp edge (count vectors are integral in practice).
- **Engine-swap:** numpy/scipy only (`gammaln`); no backend hooks declared — consistent with a numpy-only leaf.
- **Verdict:** OK

### Accumulator — Minka cumulative-count suff stat (`dirichlet_multinomial.py:106`)
- **Computes:** `c[k,j] = Σ_i w_i · 1{x_ik > j}` for `j=0..n-1`.
- **How:** `update` does `c[k,:x_k] += w`; `seq_update` builds a reversed-cumsum tail of the per-category histogram.
- **Why correct:** Verified `c` equals the manual `Σ_i 1{x_ik > j}` over a 20k-sample draw, and `update` vs `seq_update` give identical `c` and `count`.
- **Verdict:** OK

### Estimator — Minka fixed point (`dirichlet_multinomial.py:202`)
- **Computes:** `α_k ← α_k · [Σ_j c[k,j]/(α_k+j)] / [N · Σ_j 1/(Σα+j)]`, iterated to `tol`.
- **Why correct:** this is exactly Minka's MLE fixed point expressed via the digamma-difference recurrence `ψ(α+x)-ψ(α) = Σ_{j<x} 1/(α+j)`. Verified recovery: true `α=[0.8,2,4]` → est `[0.805,2.004,4.007]` from 20k samples at `n=12`. Degenerate `count<=0` or `n==0` → `α=1` default.
- **Numerical stability:** denominators `α_k+j`, `Σα+j` are strictly positive (α>0 enforced in ctor). n/a.
- **Verdict:** OK

---

## Module: pysp/stats/leaf/integer_uniform_spike.py

### log_density / seq_log_density (`integer_uniform_spike.py:124,142`)
- **Computes:** `log p` at `x=k`; `log((1-p)/(num_vals-1))` elsewhere in `[min_val,max_val]`; `-inf` outside.
- **How:** precomputed `log_p`, `log_1p = log1p(-p) - log(num_vals-1)`.
- **Why correct:** spike-and-slab mixture; verified normalizes to 1.0 over the range, `P(k)=0.4`, `P(other)=0.15` for `p=.4,num_vals=5`. seq matches scalar incl. out-of-range `-inf`.
- **Numerical stability:** `log1p(-p)` good for `p→1`. FINDING(spike-degenerate) — `num_vals==1` makes `log(num_vals-1)=log(0)=-inf` → `log_1p=nan` (RuntimeWarning at ctor line 98 and estimator line 703). Density at `k` is still correct (1.0) since `log_1p` is unused, but the NaN/warning leaks for a degenerate-but-valid single-value range.
- **Engine-swap:** `backend_seq_log_density` parity with numpy verified (`True`).
- **Verdict:** FINDING(spike-degenerate)

### Estimator — joint (k, p) MLE (`integer_uniform_spike.py:682`)
- **Computes:** for each candidate index, `ll = n_k·log p_k + (N−n_k)·[log1p(−p_k) − log(M−1)]` with `p_k=n_k/N`, then `k = argmax`, `p = p_k`.
- **Why correct:** profile likelihood — for a spike at index `i` the MLE of `p` is `n_i/N`; `argmax` over `i` of the profiled log-lik is the joint MLE. Verified: 50k draws from `k=4,p=.5` → est `k=4, p=0.501`; manual argmax agrees.
- **Numerical stability:** `errstate(divide=ignore)` wraps it; `log(p_k)` for an unobserved index gives `-inf·n_k = 0` only when `n_k=0` (so that index just scores `(N)·log1p(0)... ` finitely) — safe. Degenerate `M=1` warns (see above) but returns `p=1`.
- **Verdict:** FINDING(spike-mutate) — all pseudo_count branches (lines 719, 739, 757) mutate the caller's `count_vec` in place (`count_vec[k] += ...` / `count_vec += pc`). Verified: input `[1,2,3]` becomes `[2,3,4]` after `estimate`.

### Sampler / Accumulator (`integer_uniform_spike.py:320,371`)
- **Computes:** Bernoulli(p) spike vs uniform over `non_k`; weighted count histogram with dynamic range.
- **Why correct:** standard mixture sampler; accumulator mirrors the integer-categorical growth logic.
- **Verdict:** OK

---

## Cross-cutting notes
- The two **un-normalized multinomial** leaves (`integer_multinomial`, `categorical_multinomial`) deliberately omit the multinomial coefficient; this is internally consistent and matched by their enumerators/seq paths, but the **docstrings are wrong** (and `categorical_multinomial`'s also has a sign error in the written formula).
- **In-place mutation of caller-supplied sufficient statistics** recurs in three estimators (categorical pseudo_count, spike pseudo_count, and implicitly safe elsewhere because `value()` returns copies). In the standard EM loop the accumulator's `value()` dict/array is freshly produced per round so the corruption is usually invisible, but it is a latent aliasing hazard for any code that retains and reuses the suff_stat object.
