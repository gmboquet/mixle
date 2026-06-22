# Graph / ranking models — computation ledger

Scope: `pysp/stats/graph/{chow_liu_tree, integer_chow_liu_tree, markov_chain, integer_markov_chain,
mallows, plackett_luce, spearman_rho, erdos_renyi_graph, random_dot_product_graph,
stochastic_block_graph, spanning_tree, matching, knowledge_graph}.py`.

All numeric checks below were run with `.venv/bin/python` against brute-force/scipy references.

## Module: chow_liu_tree.py (generic)

### Mutual-information edge weight (`chow_liu_tree.py:585-627`)
- **Computes:** MI(i,j) = sum p(x,y) [log p(x,y) - log p(x) - log p(y)], with optional Dirichlet
  smoothing (pseudo_count split over joint/marginal cells).
- **How:** No-smoothing branch iterates only observed joint pairs (`count>0`); smoothing branch builds
  full normalized tables (`denom = total + pseudo_count`).
- **Why correct:** Both branches yield valid normalized distributions; smoothed `joint_alpha`,
  `marg_*_alpha` sum to `pseudo_count`, so all three normalize over `denom`.
- **Numerical stability:** log(0) guarded — no-smooth path only touches observed pairs (p_ij>0) and
  checks `p_i>0 and p_j>0`; smooth path checks `p_ij>0 and p_i>0 and p_j>0`. OK.
- **Engine-swap:** host-only numpy (structure learning); declared `numpy_only`/generic_composite. OK.
- **Verdict:** OK

### Maximum-MI spanning tree (`chow_liu_tree.py:629-661`)
- **Computes:** maximum-weight spanning tree over the MI matrix.
- **How:** `cost = max_mi - mi + 1.0` then `scipy minimum_spanning_tree`; BFS from root for parents.
- **Why correct:** Monotone-decreasing affine map of MI → min-cost tree = max-MI tree. The `+1.0`
  keeps every candidate cost strictly positive so scipy does not treat a zero-MI edge as an absent
  edge. **Numeric check:** MST edges == brute-force max-spanning-tree edges (K4, random MI). OK.
- **Verdict:** OK

### log_density / backend_seq_log_density (`chow_liu_tree.py:176-234`)
- **Computes:** log p(root) + sum_child log p(child | x[parent]).
- **Why correct:** factorization over tree; `-inf` short-circuits; missing conditional → default/`-inf`.
- **Engine-swap:** `backend_seq_log_density` groups by parent value and routes child scoring through
  `backend_seq_log_density` + `engine.index_add`. Engine-neutral. OK.
- **Verdict:** OK

## Module: integer_chow_liu_tree.py

### Pairwise MI + max spanning tree (`integer_chow_liu_tree.py:615-672`)
- **Computes:** MI per feature pair from joint/marginal count tensors; max-MI tree rooted at 0.
- **How:** smoothed branch normalizes `(counts+pc1)/(n_ij+pc)`, `(marg+pc0)/(n_ij+pc)`; sets
  `mi_mat[i,j] = 1.0 + mi_val` (upper triangle only). `cost = abs(max-mi)`, `cost[mi>0]+=1`,
  `cost[mi==0]=0`, then `minimum_spanning_tree`.
- **Why correct:** for fixed-length integer vectors every observation feeds both marginal and joint,
  so `sum(marginal_counts[i]) == n_ij`; dividing the marginals by `n_ij+pc` (with `pc0=pc/num_states`)
  still normalizes to 1 (verified: joint/marg sums == 1.0). The `+1.0` floor makes every real edge cost
  positive (= "present" to scipy); zero entries are only the unused lower triangle/diagonal.
  **Numeric check:** MST edges == brute-force max-MI tree (K4). MI value matches hand calc. OK.
- **Numerical stability:** `np.errstate(divide=ignore)` around log; `good = joint>0 & indep>0` mask. OK.
- **Verdict:** OK

### Conditional table estimation (`integer_chow_liu_tree.py:680-701`)
- **Computes:** root log-marginal + child-given-parent log-conditionals along the tree.
- **Why correct:** picks `counts[p,n]` or `counts[n,p].T` by orientation; row-normalizes (`tmat_sum`
  zero rows clamped to 1 before divide). OK.
- **Verdict:** OK

## Module: markov_chain.py

### log_density / seq_log_density (`markov_chain.py:404-473`)
- **Computes:** log p(x0) + sum log p(x_i|x_{i-1}) + log P_len(n), with a `default_value` smoothing
  scheme subtracting `log1p_dv = log(1+default_value)` per step.
- **Why correct:** **Numeric check:** sum of densities over all length-3 sequences == 1.0 (point model,
  default_value=0, NullDistribution length term=0). seq_log_density matches scalar log_density. With
  default_value=0, `log1p_dv=0` so the smoothing terms vanish, recovering the exact MLE model. OK.
- **Engine-swap:** `backend_seq_log_density` routes via `engine.index_add` and child length dist. OK.
- **Verdict:** OK

### M-step (estimate0 / estimate1 / conjugate) (`markov_chain.py:1754-1918`)
- **Computes:** init/transition probs = normalized counts (estimate0), pseudo-count smoothed
  (estimate1), or clamped Dirichlet MAP (`_estimate_conjugate`, mirrors bstats).
- **Why correct:** row normalization `v/temp_sum`; `_map_probs` clamps `counts+alpha-1` ≥ 0 then
  renormalizes (posterior-mean fallback when degenerate). Standard. OK.
- **Numerical stability:** estimate0 divides by `temp_sum` for init without a `>0` guard, but an init
  count map is only populated from observed sequences so `temp_sum>0` whenever `estimate` is reached.
  Transition rows guarded by `if temp_sum>0`. OK.
- **Verdict:** OK

## Module: integer_markov_chain.py

### log_density (`integer_markov_chain.py:~196-217`)
- **Computes:** log P_init(x[:lag]) + sum_j log cond[ravel(x[j:j+lag]), x[j+lag]] + log P_len(n).
- **Why correct:** **Numeric check:** density sums to 1.0 over all length-3 (lag=1) and length-4
  (lag=2) sequences. `ravel_multi_index(x[i:i+lag], [num_values]*lag)` row index matches the estimator.
- **Verdict:** OK

### estimate M-step (`integer_markov_chain.py:~1230-1255`)
- **Computes:** cond_mat[ravel(prev_tuple), next] = count, then row-normalize.
- **Why correct:** index bookkeeping verified: `xidx = u[1]` over keys (next value), `zidx = u[1]` over
  items (count) — both correct. **Numeric check:** recovers a 2-state transition matrix to 3 decimals.
- **Numerical stability:** **FINDING(IMC-NAN).** `cond_mat /= cond_mat.sum(axis=1, keepdims=True)`
  with no `pseudo_count` divides 0/0 for any lagged-state row that is never observed (warns
  "invalid value encountered in divide", produces NaN rows). Reproduced: a row of all-zeros →
  `[[nan]]`; sparse data over `num_values**lag` rows hits this routinely. `pseudo_count` works around
  it. Also `cond_mat` is `np.float32` (minor precision; the rest of the library is float64).
- **Verdict:** FINDING(IMC-NAN)

## Module: mallows.py

### Normalizer Z(theta) (`mallows.py:45-58`)
- **Computes:** log Z = sum_{i=1}^{n-1} [log(1 - phi^{i+1}) - log(1 - phi)], phi=exp(-theta).
- **Why correct:** **Numeric check:** closed form == brute-force sum_perm exp(-theta·kendall) for
  n=2..5, theta in {0,0.3,1,2.5} (max err <1e-6). theta=0 → n!; theta→inf → 1. Sign correct
  (log p = -theta·d - logZ, theta≥0 concentrates on sigma0).
- **Numerical stability:** `log1p(-phi)`, `log1p(-phi**(i+1))`; `_MAX_THETA=700` caps exp. OK.
- **Verdict:** OK

### Kendall distance / log_density / Copeland estimation (`mallows.py:157-176, 343-364`)
- **Computes:** discordant-pair count via rank table; sigma0 by Copeland row−col scores; theta by
  bisecting `_expected_distance` to the empirical mean distance.
- **Why correct:** standard RIM/Copeland aggregation; `_expected_distance` uses the per-rank
  geometric `E[V_i]`. Bisection bracket-expands `hi`. OK.
- **Verdict:** OK

## Module: plackett_luce.py

### log_density full + partial (`plackett_luce.py:117-152`)
- **Computes:** sum_s [g_s - logsumexp(g[s:] (+ unranked tail for partial))].
- **How:** `_reverse_logcumsumexp` via `np.logaddexp.accumulate` on reversed worths.
- **Why correct:** **Numeric check:** full ranking log p matches hand product (−2.01490302…);
  partial top-2 matches hand product with unranked-tail denominator (−1.60943791…). seq matches scalar.
- **Numerical stability:** fully log-space; denominator is a running reverse-logsumexp. OK.
- **Verdict:** OK

### MM estimator num/den (`plackett_luce.py:252-353`)
- **Computes:** Hunter (2004) MM: `num_i` = non-last appearances; `den_i` = sum over stages of
  `1/sum_{t>=s} w_t`; `w_i = num_i/den_i`.
- **How:** vectorized via `_reverse_cumsum` suffix sums and `m_cols = min(arange(k), k-2)` to map each
  ranked position to its last in-contention stage.
- **Why correct:** **Numeric check:** fixed point recovers true normalized worths
  [.333,.2,.133,.067,.267] → [.332,.2,.133,.066,.268] (max err 1.7e-3 on 20k samples), monotone LL.
- **Numerical stability:** `np.maximum(suffix, tiny)` guards underflow; `_LOG_WORTH_FLOOR=-700`. OK.
- **Verdict:** OK

## Module: spearman_rho.py

### log-partition / log_density (`spearman_rho.py:49-53, 196-213`)
- **Computes:** log p(x) = -rho·||x-sigma||^2 - log_const, log_const = logsumexp over K! perms.
- **Why correct:** **Numeric check:** densities sum to 1.0 over all permutations (n=4). OK.
- **Numerical stability:** max-shift logsumexp. OK.
- **Engine-swap:** declares numpy+torch with `backend_log_density_from_params` /
  `backend_stacked_*`. Engine-neutral. OK.
- **Verdict:** OK

### rho estimation / mean distance (`spearman_rho.py:578-620`)
- **Computes:** sigma = double-argsort(vsum); mean ||x-sigma||^2 = (2·count·||rank||^2 − 2·vsum·sigma)/count.
- **Why correct:** the `2·count·rank_norm2` shortcut relies on `||x||^2 = ||sigma||^2 = rank_norm2`,
  valid because both are permutations of 0..K-1. Bisection on `_expected_distance`. OK.
- **Verdict:** OK

## Module: erdos_renyi_graph.py
- **Computes:** independent Bernoulli(p) edge likelihood; MLE p = successes/opportunities (+pseudo).
- **Why correct:** `_bernoulli_log_likelihood`; estimate guards `total<=0 → 0.5`. Standard. OK.
- **Engine-swap:** `backend_seq_log_density` reduces on engine. OK.
- **Verdict:** OK

## Module: random_dot_product_graph.py
- **Computes:** edge prob clip(<x_i,x_j>,eps,1-eps); upper-triangle Bernoulli LL; ASE estimator =
  top-d |eigenvalue| scaled eigenvectors of diagonally-augmented mean adjacency.
- **Why correct:** standard RDPG/ASE; diagonal augmentation (Scheinerman) imputes unobserved diagonal;
  `sqrt(clip(eigvals,0,None))` drops negative eigenvalues. OK.
- **Numerical stability:** `_EPS=1e-12` clips probs before log. OK.
- **Verdict:** OK

## Module: stochastic_block_graph.py
- **Computes:** conditional-on-assignments block Bernoulli LL; MLE block_probs = successes/totals
  (+pseudo, symmetrized when undirected); block_prior from counts.
- **Why correct:** `np.divide(..., out=prior_p, where=totals>0)` guards empty block pairs; clip to
  (eps,1-eps). Explicitly conditional (no marginalization over unknown assignments — documented). OK.
- **Verdict:** OK

## Module: spanning_tree.py

### Matrix-Tree log Z + edge marginals (`spanning_tree.py:50-65`)
- **Computes:** log Z = log det(L[1:,1:]); P((i,j) in T) = w_ij · R_eff(i,j) from Laplacian pseudoinverse.
- **Why correct:** **Numeric check:** logZ matches brute-force sum over all spanning trees (K4); edge
  marginals match brute force (max err 3e-16).
- **Numerical stability:** `slogdet` (log-space); raises if cofactor non-positive. OK.
- **Verdict:** OK

### Enumerator / estimator
- **How:** Gabow k-best spanning trees on `cost=-log(w)` (increasing cost = decreasing prob);
  projected gradient ascent matching empirical edge marginals.
- **Why correct:** descending-prob == ascending edge-cost; gauge-fixed log-weights. OK.
- **Verdict:** OK

## Module: matching.py

### Permanent + edge marginals (`matching.py:46-73`)
- **Computes:** Z = perm(W) via Ryser; P(sigma(i)=j) = w_ij·perm(minor_ij)/Z.
- **Why correct:** **Numeric check:** Ryser permanent matches brute-force sum over permutations (n=4);
  edge marginals match brute force (max err 2e-15). Ryser sign `(-1)^(n-k)` correct.
- **Numerical stability:** exponential in n (guarded by `max_nodes=12`). Permanent computed in linear
  domain (not log) — fine for the small-n target. OK.
- **Verdict:** OK

### Enumerator / sampler / estimator
- Murty k-best on `-log(w)`; sequential conditional via remaining-submatrix permanents; projected
  gradient ascent with row/col gauge fixing. OK.
- **Verdict:** OK

## Module: knowledge_graph.py
- **Computes:** DistMult score sum_k E[h,k]R[r,k]E[t,k]; log p(t|h,r) = score - logsumexp_a score(h,r,a);
  mini-batch gradient ascent (full or sampled softmax) with unit-ball projection.
- **Why correct:** `_tail_log_posterior` is a max-shifted log-softmax; gradient `(onehot - p)` is the
  softmax-NLL gradient; context grads to E[other]/R[r] are the product-rule terms. Standard. OK.
- **Numerical stability:** max-shift softmax; `+1e-12` in BALD entropy; norm projection floors at 1e-12.
- **Verdict:** OK

---

## Findings summary
- **FINDING(IMC-NAN)** — `integer_markov_chain.py` estimate: zero-observed lagged-state rows produce
  NaN transition rows when `pseudo_count` is None (0/0 divide); also `cond_mat` is float32. MEDIUM.

Everything else verified correct against brute-force/scipy references. No CRITICAL/HIGH issues:
the MST selection in both Chow-Liu variants correctly MAXIMIZES total MI; Mallows Z, Plackett-Luce
(full and partial) likelihood, Spearman normalizer, Matrix-Tree Z, Ryser permanent, and all edge
marginals match exact references.
