# Core numeric primitives + numeric engines — computation ledger

Scope: `pysp/utils/special.py`, `pysp/utils/vector.py`, `pysp/arithmetic.py`,
`pysp/engines/{numpy_engine,base,precision,__init__}.py`.

All numeric spot-checks below were run with `.venv/bin/python` against
`scipy.special` / `mpmath` references.

---

## Module: pysp/utils/special.py

### `log_erfcx` (`special.py:28`)
- **Computes:** stable `log(erfcx(x))`, `erfcx(x)=exp(x^2)*erfc(x)`.
- **How:** three branches — moderate `|x|<25` direct `log(erfcx)`; `x>25`
  asymptotic series `erfcx ~ 1/(x√π)·(1 − 1/2x² + 3/4x⁴ − 15/8x⁶)`; `x<-25`
  `x² + log(erfc(x))` (erfc→2 finite).
- **Why correct:** standard erfcx asymptotic expansion; series coefficients
  `1, -0.5, 0.75, -1.875` match `1, -1/2, 3/4, -15/8`.
- **Numerical stability:** explicitly avoids underflow (x→+∞) and overflow
  (x→−∞); `inv2=inv*inv` avoids `x*x` overflow for astronomically large x.
- **Engine-swap:** host-only (numpy/scipy); used only by the numpy EMG leaf.
- **Verdict:** OK. Verified vs mpmath on grid x∈[-1e6,1e6]: max abs err 3.1e-11
  (worst at the +25 branch seam, x=26: -3.1e-11; well within EMG tolerance).

### `stirling2` (`special.py:89`)
- **Computes:** Stirling number of the 2nd kind S(n,k), exact integer.
- **How:** DP recurrence `S(n,k)=k·S(n-1,k)+S(n-1,k-1)` over a rolling row.
- **Why correct:** canonical recurrence; base cases S(0,0)=1, S(n,0)=0 (n>0),
  k>n→0 handled.
- **Numerical stability:** exact Python ints, no overflow.
- **Verdict:** OK. Verified S(n,k) for n,k∈[0,8) vs closed-form inclusion-exclusion.

### `logpdet` (`special.py:121`)
- **Computes:** log-pseudo-determinant = sum of logs of nonzero eigenvalues.
- **How:** `eigvalsh`, |eig|, threshold at `max·max(shape)·eps`, sum of logs;
  −inf if no surviving eigenvalues.
- **Why correct:** standard rank-cutoff pseudo-det of a symmetric matrix.
- **Numerical stability:** uses `eigvalsh` (symmetric path); relative eps cutoff.
- **Verdict:** OK. Verified diag([1,2,0,4])→log8; zeros→−inf.

### `polygamma_loc` (`special.py:140`)
- **Computes:** ψ^(n)(y) via `(-1)^(n+1)·Γ(n+1)·ζ(n+1,y)`.
- **Why correct:** identity holds for n≥1.
- **Verdict:** OK for n≥1 (verified vs `scipy.special.polygamma`, err 0 for n∈{1,2,3}).
  **FINDING(N5):** for n=0 it returns `inf` (ζ(1,·) pole) instead of digamma —
  diverges from `scipy.special.polygamma(0,·)`. Function is **unused** in the
  repo (dead code); low risk but a latent footgun if called with n=0.

### `trigamma` (`special.py:150`)
- **Computes:** ψ'(y) = ζ(2,y).
- **Verdict:** OK. Exact match vs `polygamma(1,·)` (err 0.0 on grid).

### `digammainv` (`special.py:164`)
- **Computes:** ψ⁻¹(y) via Newton iteration `x -= (ψ(x)-y)/ψ'(x)`.
- **How:** init `x = exp(y)+0.5` (y≥−2.22) else `-1/(y-ψ(1))`; 5 Newton steps.
  Array path handles +inf→+inf and reuses `out=` buffers for ψ, ζ.
- **Why correct:** standard Newton on monotone ψ; well-chosen init (Minka).
- **Numerical stability:** 5 fixed iterations converge to machine eps on the
  tested range; +inf guarded.
- **Verdict:** OK. Verified ψ(ψ⁻¹(y))≈y, max err 8.9e-16 (array and scalar) on
  y∈[-5,10]. **FINDING(N6, LOW):** `out=` parameter is accepted but silently
  ignored (docstring says "Deprecated. Kept for consistency"); harmless but
  misleading signature.

---

## Module: pysp/utils/vector.py

### `gammaln` (`vector.py:21`)
- **Computes:** log Γ(x) (wrapper over `scipy.special.gammaln`).
- **Verdict:** OK numerically. **FINDING(N7, LOW):** return-type inconsistency —
  `isinstance(x, float)` returns python `float`, but an `int` (e.g. `gammaln(5)`)
  or `np.float64` scalar falls to the `np.asarray` path and returns a **0-d
  ndarray** / float respectively. Verified: `gammaln(5)` → `np.ndarray`,
  `gammaln(5.0)` → `float`. Minor; callers that index or json-serialize a scalar
  could be surprised.

### `sorted_merge`, `sorted_dict_merge_add` (`vector.py:39,62`)
- **Computes:** merge of sorted arrays / key-count merge-add.
- **Verdict:** OK (combinatorial, not numeric).

### `make_pdf` (`vector.py:114`)
- **Computes:** normalize log-densities so `exp(rv).sum()==1`.
- **How:** subtract `log(Σ exp(x−max))+max`; all-−inf → uniform `−log(n)`.
- **Numerical stability:** standard max-shift LSE; degenerate all-−inf guarded.
- **Verdict:** OK.

### `log_sum` (`vector.py:284`)
- **Computes:** `log Σ exp(x)` (1-d), max-shifted; −inf if max is −inf.
- **Numerical stability:** classic stable LSE, in-place exp into shifted buffer.
- **Verdict:** OK.

### `weighted_log_sum` (`vector.py:302`)
- **Computes:** `log Σ exp(x_i + w_i)`.
- **How:** `y=x+w`, force −inf where either x or w is ±inf, then `log_sum(y)`.
- **Numerical stability:** stable; note the `isinf` mask zeroes any **+inf**
  term to −inf — documented as "+inf not supported (hot EM path)". Inputs are
  log-densities/log-weights ≤0 so this is fine in practice.
- **Verdict:** OK. Verified vs reference (test asserts 12 places).

### `log_posterior` / `posterior` / `log_posterior_sum` (`vector.py:323,356,406`)
- **Computes:** normalized (log-)posterior from per-component log-likelihoods.
- **How:** max-shift LSE; on inf/nan max → uniform (`-log n` / `1/n`).
- **Numerical stability:** stable; `posterior` reuses `out=` and divides by the
  exp-sum.
- **Verdict:** OK.

### `weighted_log_posterior(_sum)` (`vector.py:437,482`)
- **Computes:** weighted normalized log-posterior; python-loop variant.
- **Numerical stability:** manual max-track + LSE; uniform fallback on inf/nan.
- **Verdict:** OK (matches the vectorized `log_posterior` fallback semantics).

### `matrix_log_posteriors` (`vector.py:532`)
- **Computes:** nested row/outer posteriors + total log-evidence for a matrix of
  log-likelihoods.
- **How:** inner LSE per (i,j) slice with uniform fallback, then outer LSE over
  rows with `u[i]` log-priors.
- **Numerical stability:** stable nested LSE; impossible-slice/row → uniform,
  −inf evidence.
- **Verdict:** OK.

### `row_choice` (`vector.py:581`)
- **Computes:** vectorized categorical sampling per row of `p_mat`.
- **How:** inverse-CDF: `u=rng.rand(N)`, `bins=cumsum`, `rv=Σ(u≥bins[:,:-1])`.
- **Why correct:** standard inverse-CDF; equivalent to `np.searchsorted` per row.
- **Numerical stability:** an **all-zero / sub-unit row** silently selects the
  last index (u≥0 for every cumsum entry). For a degenerate (all-impossible) row
  this returns m−1 rather than erroring — acceptable for the EM callers (LDA),
  which never pass all-zero rows.
- **Verdict:** OK. Verified it reproduces an independent inverse-CDF reference
  (test `test_row_choice_matches_inverse_cdf`).

---

## Module: pysp/arithmetic.py

### Dispatch helpers (`arithmetic.py:118`)
- **Computes:** backend-neutral array ops; each `_dispatch(name)` routes on
  `engine_of(args)`, falling back to the active default engine.
- **Verdict:** OK. **FINDING(N8, LOW):** `sum` and `max` are dispatched
  (lines 144-145) and importable as module attributes, but are **absent from
  `__all__`** (lines 25-69). So `from pysp.arithmetic import *` does NOT export
  `sum`/`max`, while every other op is exported. Verified `ar.sum`/`ar.max` work
  via attribute access. Inconsistent surface; could cause `NameError` in a module
  relying on star-import.

### Engine constants via PEP 562 `__getattr__` (`arithmetic.py:169`)
- **Computes:** `pi/e/euler_gamma/one/zero/two/half/inf` resolved from the active
  engine; `maxint/maxrandint/eps` stay engine-independent.
- **Why correct:** lets symbolic engine keep exact constants; numeric default
  returns plain floats.
- **Verdict:** OK.

---

## Module: pysp/engines/numpy_engine.py

### `accumulator_dtype` (`numpy_engine.py:30`)
- **Computes:** always `float64` for sufficient-statistic reductions.
- **Why correct/important:** this is the precision policy — even a float32 fit
  accumulates sums in float64, guarding against variance cancellation / large-N
  drift. **Callers must opt in** by passing `dtype=accumulator_dtype` to their
  reductions; `engine.sum` itself does NOT auto-promote (see N9).
- **Verdict:** OK (policy is sound and documented).

### `asarray` (`numpy_engine.py:43`)
- **Computes:** convert to ndarray applying the float dtype policy (cast float
  inputs to engine dtype; leave int/other untouched).
- **Verdict:** OK functionally (float32 engine casts float arrays to float32,
  leaves int64 alone — verified). **FINDING(N10, LOW):** converts twice —
  `arr = np.asarray(x)` only to read `arr.dtype.kind`, then `np.asarray(x, dt)`.
  Could be `np.asarray(arr, dtype=dt)` to avoid the redundant conversion.

### `zeros`/`empty`/`arange` (`numpy_engine.py:51,55,59`)
- **Computes:** allocate honoring `self.dtype`; `arange` only applies the float
  dtype when a float arg is present (`_has_float_arg`).
- **Verdict:** OK.

### Static `staticmethod` ops (`numpy_engine.py:73-95`)
- **Computes:** `log,exp,sqrt,abs,where,maximum,clip,floor,isnan,isinf,sum,max,
  dot,matmul,cumsum,logsumexp,bincount,unique,searchsorted,gammaln,digamma,
  betaln,erf` bound to numpy/scipy.
- **Numerical stability:** `logsumexp` is `scipy.special.logsumexp` (handles
  `axis=`, max-shift). Verified `axis=1` and full-array reductions return correct
  values.
- **Engine-swap:** all 30 dispatched names are present on both NumpyEngine and
  SymbolicEngine (verified by `hasattr` sweep) — engine surface is consistent.
- **Verdict:** OK, with one precision caveat → **FINDING(N9, MEDIUM):**
  `sum = staticmethod(np.sum)` accumulates in the **input array's dtype**. On a
  float32 array `engine.sum` accumulates in float32 (verified: `sum(full(2e6,
  1.1, f32))` rel err 9.2e-8 vs float64 accumulation; numpy's pairwise summation
  keeps it modest, not catastrophic). This is safe ONLY because the precision
  policy is that sufficient-statistic reductions pass `dtype=accumulator_dtype`
  explicitly — any caller that does `engine.sum(float32_array)` for a statistic
  without that kwarg silently drifts. Policy depends on caller discipline, not
  enforced by the engine. (Audit the stats accumulators for compliance.)

### `index_add` (`numpy_engine.py:97`)
- **Computes:** scatter-add via `np.add.at`.
- **Verdict:** OK (correct unbuffered semantics for repeated indices).

---

## Module: pysp/engines/base.py

### `ComputeEngine` ABC + constants + capability flags
- **Computes:** the engine surface contract; default math constants; flags
  `supports_autograd/supports_numba/resident_estep`; `precision`/`with_precision`.
- **Verdict:** OK. Abstract methods (`asarray/zeros/empty/arange/to_numpy/stack`)
  enforced; numpy implements all. Constants default to floats (overridable by
  symbolic). Clean abstraction.

---

## Module: pysp/engines/precision.py

### `precision_name` / `normalize_numpy_dtype` / `normalize_torch_dtype`
- **Computes:** canonicalize precision specifiers to dtypes.
- **Verdict:** OK. numpy path rejects bfloat16 and non-floating dtypes; alias
  table covers common spellings.

### `auto_precision` + helpers (`precision.py:141`)
- **Computes:** recommend float32 vs float64 from data magnitude/spread and
  hardware.
- **How:** float32 only for a **GPU torch** engine AND well-conditioned data
  (amax<1e4 and amax/std<1e3); float64 otherwise.
- **Why correct:** guards float32's ~7 sig-digits in scoring; accumulation is
  already float64 (accumulator_dtype). Conservative defaults to float64.
- **Verdict:** OK.

---

## Module: pysp/engines/__init__.py

### `engine_of` / `_direct_engine` / `register_array_type` / `to_numpy`
- **Computes:** map an array/payload to its owning engine; recurse into
  list/tuple/dict children; raise on mixed engine classes.
- **Why correct:** symbolic object-arrays routed before the ndarray→numpy rule
  (line 73-74 comment); torch tensors carry device/dtype into a fresh
  TorchEngine.
- **Engine-swap:** correctly central to dispatch; mixed-engine payloads raise
  (good — silent host/device mixing is a bug).
- **Verdict:** OK.

---

## Summary of findings
- **N9 (MEDIUM):** `engine.sum`/reductions accumulate in input dtype; float64
  accumulation depends on callers passing `accumulator_dtype` — not enforced.
- **N5 (LOW/dead):** `polygamma_loc(0, ·)` returns inf (ζ pole), unlike scipy;
  function is unused.
- **N6 (LOW):** `digammainv(out=)` silently ignored.
- **N7 (LOW):** `vector.gammaln` returns float for python-float but 0-d ndarray
  for int input — type inconsistency.
- **N8 (LOW):** `sum`/`max` dispatched but missing from `arithmetic.__all__`
  (star-import gap).
- **N10 (LOW):** `numpy_engine.asarray` converts the input twice.

All core special functions (log_erfcx, digammainv, trigamma, stirling2, logpdet,
polygamma_loc n≥1) and all LSE/posterior routines verified numerically correct
against scipy/mpmath. No CRITICAL/HIGH issues found.
