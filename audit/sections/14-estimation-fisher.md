# Estimation / optimization / Fisher-information — computation ledger

Scope: `pysp/utils/{estimation,fit,objectives,fisher,density_rank,priors,enumeration}.py`.
These are the cross-cutting numeric drivers (EM/MAP loops, gradient fitters, Fisher
geometry, ranking, priors, best-first enumeration). Verified the math, the
convergence tests, and the matrix algebra; finite-difference / Monte-Carlo checks
noted inline.

---

## Module: pysp/utils/estimation.py

### EM convergence test — `_em_loop` (`estimation.py:338-371`)
- **Computes:** standard EM loop. `dll = ll_new - old_ll`; accept step if finite and
  (`dll >= 0` or `delta is None` or `not monotone`); stop when `dll < delta`.
- **How:** absolute log-likelihood-gain test (`delta` default `1e-9` in `optimize`,
  `1e-6` in `fit`). Best model tracked by validation score.
- **Why correct:** EM is monotone in the data LL, so `dll >= 0` is the expected case;
  `dll < delta` with small positive `delta` correctly catches both convergence and any
  numerical decrease.
- **Numerical stability:** `ll_finite = np.isfinite(ll)` guard (estimation.py:353) keeps a
  NaN/-inf step from being accepted and from poisoning `old_ll` — good. Note: a *rejected*
  finite-but-decreasing step still sets `old_ll = ll` (line 364), so the next identical step
  yields `dll≈0 < delta` and terminates; benign.
- **Engine-swap:** neutral; engine path routed via `_em_step_fn`/`_ll_sum_fn`.
- **Verdict:** OK

### Fused EM loop — `_fused_em_loop` (`estimation.py:374-436`)
- **Computes:** reuses the E-step posterior normalizer as the data LL, lagging the
  convergence test by one iteration.
- **Why correct:** compares LL of successive accepted models → same fixed point; returned
  model is best-by-(validation/LL). Only enabled for `mle` + local list enc + no engine/strategy
  (estimation.py:628-635), which is exactly where the reused normalizer equals the data LL.
- **Numerical stability:** `np.isfinite(ll_model)` guard before advancing `prev_ll`.
- **Verdict:** OK

### `best_of` trial selection (`estimation.py:92-115`)
- **Computes:** runs `optimize` per trial, rescoring each returned model on
  `score_data` (= `enc_vdata` if present else `enc_data`) and keeping the max.
- **Why correct:** `optimize` already returns best-by-validation; re-scoring on the same
  validation set and taking the argmax across trials is consistent. `rv_ll` init `-inf`.
- **Verdict:** OK

### `fit` (MAP/VB objective) convergence (`estimation.py:726-824`)
- **Computes:** `obj = data_term + prior_term`; data term auto-switches to `seq_local_elbo`
  for variational models, else observed-data LL; prior term `estimator.model_log_density`.
- **Numerical stability:** wraps loop in `np.seterr(divide="ignore")` and restores in
  `finally` — defensive against `log(0)` in conjugate prior terms.
- **Verdict:** OK

### Objective resolution — `_resolve_objective` / `_model_objective` (`estimation.py:671-707`)
- **Computes:** `auto` → `vb` if `seq_local_elbo`, `map` if `model_log_density != 0.0`, else `mle`.
- **Numerical stability:** the `!= 0.0` test (estimation.py:705) misclassifies a MAP estimator
  whose log-prior happens to be exactly 0 at the initial model as MLE. Narrow edge case.
- **Verdict:** FINDING(E1) (LOW)

### Streaming schedules — `harmonic` / `constant` / `forgetting` (`estimation.py:827-875`)
- **Computes:** `harmonic` requires `alpha in (0.5, 1.0]` → satisfies Robbins–Monro
  (`sum rho = inf`, `sum rho^2 < inf`). `forgetting`/`constant` require `rho in (0,1]`.
- **Verdict:** OK

### `BayesianStreamingEstimator` (`estimation.py:900-1001`)
- **Computes:** recursive conjugate updating (`posterior_carry`) or power-prior `forgetting`
  (scale suff-stats and nobs by `rho`).
- **Why correct:** carries posterior forward via `model.estimator()`; forgetting scales both
  accumulator and `nobs` consistently.
- **Verdict:** OK

---

## Module: pysp/utils/fit.py  (gradient MLE/MAP)

### Constraint reparameterization (`fit.py:94-166`)
- **Computes:** raw↔canonical maps — `positive`→`exp`/`log`(clamp eps=1e-8); `unit_interval`→
  `sigmoid`/`logit`; simplex→`softmax`/`log`; ordered bounds→`anchor ± exp(log_delta)`.
- **Why correct:** standard differentiable transforms; eps-clamp on the *forward* (init) log/logit
  prevents `log(0)` at construction.
- **Numerical stability:** see E2 below for the prior side.
- **Engine-swap:** torch-only by design (autograd path); guarded by `supports_autograd`.
- **Verdict:** OK

### MAP log-prior — `_gradient_log_prior_state` (`fit.py:524-592`)
- **Computes (unnormalized):** gamma `(shape-1)log v - rate·v`; beta `(α-1)log v + (β-1)log1p(-v)`;
  dirichlet `(α-1)log v`; fallback raw-L2 ridge.
- **Why correct:** matches the log-density kernels (normalizers dropped — fine for MAP since only
  objective differences matter; consistent with priors.py docstrings).
- **Numerical stability:** the prior log-terms act on the *constrained* value. For beta on a
  saturated `sigmoid(raw)==1.0` (reachable in both float32 AND float64 for raw≳37), `log1p(-v) = -inf`;
  for dirichlet on a `softmax` tail that underflows to exactly 0.0, `log(v) = -inf` (confirmed
  numerically). The objective then becomes `-inf` and that iterate is rejected by best-tracking,
  so it does not corrupt the returned model, but it can stall Adam at a boundary and pollutes
  `final_lp`/diagnostics. Gamma/positive are safe (`v=exp(raw)>0`).
- **Verdict:** FINDING(E2) (MEDIUM)

### `_raw_l2_prior` (`fit.py:348-355`) and `_gradient_objective_norm` (`fit.py:290-308`)
- **Computes:** `-0.5·strength·||θ-θ0||²`; gradient L2 norm via backward over leaves.
- **Verdict:** OK

---

## Module: pysp/utils/objectives.py

### `UnnormalizedLogLikelihood` partition (`objectives.py:226-232`)
- **Computes:** self-normalized IS log-partition `logsumexp_j(log f(y_j) - log q(y_j)) - log M`.
- **Why correct:** standard SNIS estimator of `log Z`; uses `engine.logsumexp` (log-space, stable).
- **Verdict:** OK

### Adam/LBFGS loops + convergence (`objectives.py:285-587`)
- **Computes:** `-sign·objective` minimized; converge when
  `|cur - prev| < tol·max(1,|cur|)` (relative-with-floor) after ≥3 evals; restore best state.
- **Why correct:** relative tolerance with absolute floor is a sound stopping rule; best-state
  restore guards against a final bad step.
- **Numerical stability:** `_objective_best_entry` uses `np.nanargmax/argmin`, which raises
  `ValueError("All-NaN slice")` if *every* history entry is NaN (confirmed). Only hit when the
  objective is NaN at init and every step — pathological model, but it would crash rather than
  return. Minor.
- **Verdict:** FINDING(E3) (LOW)

### Constrained raw tensors — `_objective_raw_tensor` / `_objective_constrained_value` (`objectives.py:666-714`)
- Same transforms as fit.py; simplex init renormalizes then logs after eps-clamp. Coupled bounds
  validated by `_objective_bound_delta` (raises if init violates the bound). **Verdict:** OK

---

## Module: pysp/utils/fisher.py

### Empirical Fisher — `FisherView.fisher_information` (`fisher.py:297-307`)
- **Computes:** `I ≈ (1/n) Σ (s_i - s̄)(s_i - s̄)^T + ridge·I` (centered outer product = `E[ss^T]`).
- **Why correct:** for exponential-family / complete-data scores the score is the centered
  sufficient statistic, so the empirical covariance is the (observed) Fisher estimate.
- **Numerical stability:** divides by `n` (MLE covariance); ridge `1e-8` on the diagonal.
- **Verdict:** OK

### Whitening — `FisherView.fisher_vectors` metric='full' (`fisher.py:335-339`, also `FixedFisherView:530-534`)
- **Computes:** `X_white = X_centered · V · diag(1/sqrt(max(λ,ridge)))` from `eigh(I)`.
- **Why correct / stable:** uses symmetric eigendecomposition with eigenvalue flooring instead of
  an explicit inverse — numerically the right choice (no `linalg.inv`/`solve` anywhere; confirmed
  by grep). Floors λ at `ridge` so near-singular metrics don't blow up.
- **Verdict:** OK

### `MixtureFisherView._model_fisher` (`fisher.py:695-729`)
- **Computes:** complete-data Fisher covariance of `[z ; z_k·s_k]`: top-left `diag(w)-ww^T`,
  cross `(w_k 1[i=k] - w_i w_k)·μ_k`, diagonal blocks `w_k·I_k + w_k(1-w_k) μ_k μ_k^T`,
  off-diagonal `-w_k w_l μ_k μ_l^T`.
- **Why correct:** **Monte-Carlo verified** (4e6 draws, 2-component mixture, dims [2,1]):
  max abs diff between this formula and the empirical covariance = `8.8e-4` (pure MC noise).
- **Verdict:** OK

### `JointMixtureFisherView` (`fisher.py:784-910`)
- **Computes:** same block structure over component *pairs*, with `_pair_weights` normalized to 1
  and posterior from logsumexp-stable `_posterior_from_scores` (max-subtract). Reuses the verified
  mixture formula. **Verdict:** OK

### `CompositeFisherView` / `_PairProductFisherView` / `OptionalFisherView` block Fishers
  (`fisher.py:610-619, 769-781, 1344-1360`)
- **Computes:** block-diagonal (independent fields) or the optional-gate Bernoulli×child cross
  blocks `[diag(p,q)-outer ; ∓pq·μ ; q·I + pq·μμ^T]`.
- **Why correct:** independence → block-diagonal joint covariance; optional gate is a 2-state
  categorical coupled to the present-child stats, matching the mixture derivation with K=2.
- **Verdict:** OK

### HMM forward-backward stats — `_sequence_forward_backward` (`fisher.py:1715-1762`)
- **Computes:** scaled α/β recursion → γ (state posteriors) and ξ (transition posteriors).
- **Numerical stability:** per-step scaling (`alpha[t] /= scale[t]`), `obs = exp(log_b - safe_max)`
  with max-subtraction, NaN/inf guards on `obs` and `scale`, early-return on degenerate scale.
  Log-space-then-exp emission handling is sound.
- **Verdict:** OK

### HMM path moments — `_path_moments_for_length` (`fisher.py:1915-1956`)
- **Computes:** exact mean/second-moment of complete-data stats by a forward DP over states,
  carrying `(p_state, first, second)` and folding `2·inc·first` for the second moment.
- **Why correct:** the `next_second += a·(second_prev + 2·inc·first_prev + p_prev·inc2)` recursion is
  the correct expansion of `E[(S+inc)²]` along the chain. Diagonal moments only (used for the
  diagonal fallback metric).
- **Verdict:** OK

### Enumerated model Fisher (HMM/PCFG) — `_enumerated_model_mean_cov` (`fisher.py:1544-1596, 1999-2045`)
- **Computes:** `mean = Σ p·s`, `cov = Σ p·s s^T - mean·mean^T`, symmetrized, diagonal floored ≥0.
- **Numerical stability:** the `E[ss^T]-E[s]E[s]^T` form is a catastrophic-cancellation pattern, but
  it is mitigated — weights are normalized (`weights /= total` with a `|total-1|<=1e-8` mass check)
  and the diagonal is floored to ≥0 (`fisher.py:1587-1588, 2042-2043`). Acceptable for a finite
  enumerable support; documented as exact only there.
- **Verdict:** OK (note the cancellation form is intentional and guarded)

### `_length_support` count families (`fisher.py:988-1060`)
- **Computes:** normalized pmf over the support for Bernoulli/Binomial/Poisson/Geometric/Categorical;
  Poisson grows `hi` until ≥`1-tol` mass; `_finite_support_from_log_density` normalizes via
  `lp -= max(lp)` (stable). **Verdict:** OK

---

## Module: pysp/utils/density_rank.py

### `density_rank` head-enumeration + sampling hybrid (`density_rank.py:52-124`)
- **Computes:** exact `G(x)=Σ_{p(y)>=p(x)} p(y)` and rank from the descending enumerator until the
  stream drops below `p(x)`; else Monte-Carlo `Ĝ = mean 1[log p(Y)>=t]` with binomial stderr.
- **Why correct:** descending order makes the early-exit exact; sampling is reliable precisely in the
  tail where `G` is large. `t==-inf` short-circuits to 0 (log(0) handled).
- **Numerical stability:** stderr `sqrt(g(1-g)/n)` floored at 0. `mass` accumulated in linear space
  but only over the bounded head — fine.
- **Verdict:** OK

### `cumulative_probability` structural mass (`density_rank.py:529-555`)
- **Computes:** bulk prefix of the exact per-bucket `_mass_histogram` plus an item-by-item resolved
  smear band (true `log_density >= t-1e-9`), `min(1.0, ...)`.
- **Why correct:** mass multiplies / bits add → convolves exactly like counts; band absorbs the
  floored-bucket smear. Docstring cites 1e-16 agreement on a 12-factor product.
- **Verdict:** OK

### `count_dp_top_p` nucleus-size bracket (`density_rank.py:558-649`)
- **Computes:** `size_upper` = whole-bucket cover until cumulative mass ≥ p; `size_lower` caps each
  item in bucket b at `2^(-b·bits_per_bucket)` (max possible per-item prob) → provable floor.
- **Why correct:** upper cover is a valid covering set; lower cap over-estimates coverage so fewer
  items provably can't reach p. `log_prob_threshold = -boundary·bits·ln2` sign correct (p=0→inf).
- **Numerical stability:** `need = ceil(residual/cap_here - tol)` then `max(0,min(need,c))` — no
  negative/over-run. **Verdict:** OK

### `mixture_cross_rank` (`density_rank.py:696-730`)
- **Computes:** true-marginal rank via the joint K-dim per-component bucket histogram; representative
  marginal prob at bucket midpoints `Σ w_k 2^(-(key_k+0.5)·bits)`. Exponential in K (documented).
- **Verdict:** OK

---

## Module: pysp/utils/priors.py
- Pure serializable prior specs (NormalGamma/Dirichlet/Beta/Gamma/Composite/...); no math beyond
  `float()` coercion and `as_dict` payloads with legacy aliases. NormalGamma docstring density matches
  the standard kernel. **Verdict:** OK

---

## Module: pysp/utils/enumeration.py

### `QuantizedEnumerationIndex.bin_for_log_prob` (`enumeration.py:155-161`)
- **Computes:** `bin = floor(max(0,-log2 p)/width + 1e-12)`. `log_prob==-inf` filtered upstream;
  `p=1`→bin 0 (verified). **Verdict:** OK

### `ProductEnumerator.__next__` (`enumeration.py:546-572`)
- **Computes:** best-first k-best over a Cartesian product; canonical duplicate-free successor rule
  (advance only coords ≥ `min_coord`).
- **Numerical stability:** successor key re-based on the freshly re-summed exact `score`
  (`succ_key = score + (nxt[1]-items[k][1])`) → keys stay within 1 ULP of exact, no accumulating
  drift. **Verdict:** OK

### `best_first_union` / `_best_first_union` (`enumeration.py:691-775`)
- **Computes:** union of overlapping sorted streams, deduped via `freeze`, re-scored exactly,
  released once exact score ≥ `logsumexp(live head scores)` (mixture frontier bound).
- **Why correct:** any unseen `x` has `p_k(x) <= head_k` ∀k ⇒ `Σ w_k p_k(x) <= exp(bound)`. The
  three-way release test (cheap upper bound `max+logK`, cheap reject `< max`, exact only in the
  logK band) is a valid bracketing of the exact `logsumexp` bound.
- **Numerical stability:** all in log-space via `log_sum` (logsumexp). **Verdict:** OK

### `bounded_best_first_union_index` (+ component-aware variant) (`enumeration.py:778-1048`)
- **Computes:** bounded quantized index from the union; stops when the live frontier
  `logsumexp(live heads)` drops below the bit threshold.
- **Why correct:** frontier is an upper bound on every unseen value; below threshold all qualifying
  values are already buffered. Component-aware path tracks per-component max contributions and an
  exact `log_sum_known` over them. **Verdict:** OK

### `sound_top_k` mass certificate (`enumeration.py:1065-1135`)
- **Computes:** pull distinct `(value, log_prob)` from the count-budget index, keep best `start+k` in
  a min-heap, certify when `remaining = total_mass - accumulated < exp(kth-best) - tol`.
- **Why correct:** every unpulled item's prob ≤ `remaining`; once `remaining` < kth-best the heap is
  the exact true top-(start+k) regardless of the seek stream's tropical ordering. Budget doubles to
  `max_budget_bits` on exhaustion. Sound certificate. **Verdict:** OK

### `merge_enumerators` / `LengthFrontierMerge` / `frontier_merge` (`enumeration.py:481-688`)
- **Computes:** lazy k-way / frontier merges of sorted streams with per-stream offsets; disjoint
  supports assumed (documented). Frontier instantiates a length/key only when its `lp` can beat the
  best instantiated head — valid because `make_stream` log-probs are ≤ the key's `lp`.
- **Verdict:** OK

---

## Findings summary
- **E1 (LOW)** estimation.py:705 — MAP-vs-MLE auto-detection via `model_log_density != 0.0`.
- **E2 (MEDIUM)** fit.py:573 / fit.py:582 — beta/dirichlet MAP prior `log1p(-v)`/`log(v)` can hit
  `-inf` at saturated sigmoid/softmax boundaries.
- **E3 (LOW)** objectives.py:601-604 — `nanargmax/argmin` raises on an all-NaN history.
