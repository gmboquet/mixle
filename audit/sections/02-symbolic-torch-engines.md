# Symbolic compute engine + export + torch engine — computation ledger

Scope: `pysp/engines/torch_engine.py`, `pysp/engines/symbolic_engine.py`,
`pysp/engines/symbolic_export.py`, cross-checked against the reference
`pysp/engines/numpy_engine.py` and the `ComputeEngine` ABC in
`pysp/engines/base.py`.

All numeric parity checks were run with `.venv/bin/python` (torch + sympy
installed; sage not installed — sage handlers audited by reading).

---

## Module: pysp/engines/torch_engine.py

### Default dtype / precision policy (`torch_engine.py:39`, `:45`)
- **Computes:** float dtype for tensors; accumulator dtype for sufficient-stat reductions.
- **How:** `self.dtype = float64` when `dtype is None`; `accumulator_dtype` always `torch.float64`.
- **Why correct:** Matches numpy engine (numpy default float is float64; `accumulator_dtype` is `np.float64`). Verified: `TorchEngine().dtype == torch.float64`, accumulator float64 even for a float32 engine. No silent float32 drift.
- **Numerical stability:** Good — reductions accumulate in float64 regardless of scoring precision, same contract as numpy.
- **Engine-swap:** Neutral for the default path.
- **Verdict:** OK

### asarray (`torch_engine.py:59`)
- **Computes:** host/tensor → torch tensor on device with dtype policy.
- **How:** float kind → engine dtype; bool → `torch.bool`; everything else (int) → `torch.int64`.
- **Why correct:** int64/bool round-trip losslessly through `asarray`→`to_numpy` (verified). float upcasts to engine float64.
- **Numerical stability:** n/a.
- **Engine-swap:** **Minor parity divergence.** numpy engine has default `dtype=None`, so `numpy.asarray(float32_arr)` *preserves* float32; `torch.asarray(float32_arr)` *upcasts* to float64. Verified. Upcast is lossless and arguably more correct, but the two engines do not produce the same float dtype from the same input. Benign in practice (float64 is the intended precision); flagged as a divergence, not a bug.
- **Verdict:** FINDING(E2-1)

### to_numpy (`torch_engine.py:97`)
- **Computes:** tensor/DTensor → host numpy.
- **How:** DTensor → `full_tensor().detach().cpu().numpy()`; tensor → `detach().cpu().numpy()`.
- **Why correct:** Lossless gather/detach; dtype preserved (int64→int64, bool→bool, float64→float64 verified).
- **Engine-swap:** Neutral.
- **Verdict:** OK

### Elementwise ops: log/exp/sqrt/abs/where/maximum/clip/floor/isnan/isinf (`torch_engine.py:162-171`)
- **Computes:** standard elementwise math.
- **How:** thin lambdas over `torch.*`; `clip` → `torch.clamp(min=,max=)`.
- **Why correct:** `where` 1-arg returns indices like numpy (verified); `clamp(min=0,max=None)` matches `np.clip(...,0,None)` (verified); `floor`/`abs`/`isnan`/`isinf` direct equivalents.
- **Numerical stability:** n/a (same primitives as numpy).
- **Engine-swap:** Neutral.
- **Verdict:** OK

### sum / max / logsumexp axis↔dim shim (`torch_engine.py:173-197`)
- **Computes:** reductions, accepting numpy-style `axis=` kwarg.
- **How:** rename `axis`→`dim`; `max` unwraps the `.values` of torch's namedtuple.
- **Why correct:** `sum`/`logsumexp` accept a **tuple** dim in torch and match numpy/scipy on `(0,1)` (verified). `max` with single int axis returns `.values` correctly.
- **Numerical stability:** `logsumexp` is torch's stabilized version (matches scipy to 1e-6, verified).
- **Engine-swap:** **Parity hazard, latent.** `engine.max(x, axis=(0,1))` works on numpy but `torch.max(x, dim=(0,1))` raises `TypeError` (torch.max takes a single int dim). Verified. No current caller passes a tuple axis to `engine.max` (all `engine.max(..., axis=int)`), so unexercised — but it is an API gap that breaks on a future tuple-axis call.
- **Verdict:** FINDING(E2-2)

### cumsum (`torch_engine.py:190`)
- **Computes:** cumulative sum.
- **How:** `lambda x, *args, **kwargs: torch.cumsum(x, *args, **kwargs)`.
- **Why correct:** identical when an axis/dim is given.
- **Engine-swap:** **Parity hazard, latent.** `np.cumsum(x)` with no axis flattens and returns 1-D; `torch.cumsum(x)` with no `dim` raises `TypeError` (verified). The lambda does not default a dim, nor translate `axis`→`dim`. No `engine.cumsum` callers exist today (grep: all `cumsum` uses are direct `np.cumsum`), so unexercised, but the op is non-conforming relative to numpy.
- **Verdict:** FINDING(E2-3)

### gammaln / digamma / betaln / erf (`torch_engine.py:202-205`)
- **Computes:** log-Γ, ψ, log-B, erf.
- **How:** `torch.special.gammaln`/`digamma`/`erf`; `betaln = lgamma(x)+lgamma(y)-lgamma(x+y)`.
- **Why correct:** Verified vs scipy on `x∈{0.001,0.5,1,2.5,10}`: gammaln/erf max abs diff 1.1e-16, digamma 1.1e-13, betaln 1.8e-15. `betaln` identity is exact (torch has no native betaln; lgamma==gammaln).
- **Numerical stability:** Good (uses log-gamma directly, no Γ overflow).
- **Engine-swap:** Neutral; `torch.special.gammaln/digamma` are present in installed torch.
- **Verdict:** OK

### index_add (`torch_engine.py:207`)
- **Computes:** scatter-add `out[index] += values` with duplicate accumulation.
- **How:** coerce index to long on out.device; `out.index_add(0, index, values)`.
- **Why correct:** Accumulates duplicate indices correctly; verified return value equals `np.add.at` result `[3,3,0]`.
- **Numerical stability:** n/a.
- **Engine-swap:** **Semantic divergence (masked).** numpy's `np.add.at(out, ...)` mutates `out` **in place** and returns it; torch's `Tensor.index_add` (no trailing `_`) returns a **new** tensor and does **not** mutate `out` (verified: `out` stays zeros). All current callers use the return-value form (`rv = engine.index_add(rv, ...)`), so the mismatch is masked. A caller relying on the numpy in-place side effect (mutating a shared buffer without reassigning) would silently break under torch. Latent hazard, not a live bug.
- **Verdict:** FINDING(E2-4)

### DTensor placement (asarray/replicate/place_component_axis/_replicate_tensor) (`torch_engine.py:109-150, 214-232`)
- **Computes:** replicated / component-sharded placement on a DeviceMesh.
- **How:** `distribute_tensor` with `Replicate()` per mesh dim, `Shard(axis)` on the last mesh dim for components.
- **Why correct:** Placement-only; does not alter values. `shard` validated to `{None,'components'}`. Not numerically exercised here (no mesh in tests by default).
- **Engine-swap:** Torch-only feature; degrades to plain `asarray` when `mesh is None`.
- **Verdict:** OK

---

## Module: pysp/engines/symbolic_engine.py

### SymbolicExpression tree + operator overloads (`symbolic_engine.py:22-156`)
- **Computes:** immutable op-node tree; Python operators build nodes (`add/sub/mul/div/pow/neg`, comparisons, `and/or/invert`).
- **How:** dunder methods call `SymbolicExpression.call(op, ...)`; `__bool__` raises to prevent accidental truthiness.
- **Why correct:** Argument order preserved for non-commutative ops (`sub`,`div`,`pow`) including the `__r*__` reflected forms. Verified via export round-trip (below).
- **Engine-swap:** Symbolic-only.
- **Verdict:** OK

### Named constants pi/e/euler_gamma/inf/half (`symbolic_engine.py:168-175`)
- **Computes:** exact symbolic constants; `half = 1/2` kept exact (not float 0.5).
- **Why correct:** Lower to `sympy.pi/E/EulerGamma/oo` (verified) and `sage.pi/e/euler_gamma/oo`; `evaluate()` yields `math.pi`/`math.e`/`0.5772156649…`/`math.inf`.
- **Engine-swap:** Matches base engine's float constants when evaluated; stays exact when lowered. Correct by design.
- **Verdict:** OK

### evaluate (`symbolic_engine.py:47-54`, `_EVAL_OPS` `:502`)
- **Computes:** numeric evaluation of a traced tree against a symbol→value map.
- **How:** post-order eval dispatching on `_EVAL_OPS`.
- **Why correct:** Verified numerically against numpy/scipy for log/exp/sqrt/sum/where/gammaln/betaln/logsumexp.
- **Numerical stability:** Uses Python `math.*` scalars.
- **Engine-swap:** **Missing op — `digamma`.** The engine *emits* `digamma` nodes (`symbolic_engine.py:266`, used by LDA / labeled_lda M-steps) and lowers them correctly to `sympy.digamma`→`polygamma(0,·)` / `sage.psi` (verified), but `_EVAL_OPS` has **no `digamma` entry**, so `expr.evaluate(...)` raises `KeyError('digamma')` (verified). Any numeric reduction of a symbolic trace touching digamma fails. (`bincount/unique/searchsorted/index_add` are also absent from `_EVAL_OPS`, but those are intentionally non-symbolic data ops — see `_NON_SYMBOLIC_OPS` — and are out of scope for evaluate.)
- **Verdict:** FINDING(E2-5)

### Elementwise ops via `_elementwise_call` (`symbolic_engine.py:260-348, 415-423`)
- **Computes:** log/exp/sqrt/abs/floor/gammaln/digamma/erf/where/maximum/clip/comparisons/logical/isnan/isinf/betaln as per-element nodes with numpy broadcasting.
- **How:** `np.broadcast_arrays` then per-index `SymbolicExpression.call(op, ...)`.
- **Why correct:** broadcasting matches numpy object-array semantics; op names map 1:1 to numeric engine ops. `maximum`→`max` node, `logical_not`→`invert` node (consistent with `__invert__`).
- **Engine-swap:** Neutral within symbolic domain.
- **Verdict:** OK

### sum / max / logsumexp reductions (`symbolic_engine.py:269-277, 341, 376-412`)
- **Computes:** `sum`=Σ, `max`=pairwise max fold, `logsumexp`=log(Σ exp).
- **How:** `_reduce_symbolic` with `np.apply_along_axis` per axis; tuple axis handled by reducing high→low.
- **Why correct:** `sum`/`max` match numpy on axis=0 (verified). Argument order/fold correct.
- **Numerical stability:** **`logsumexp` is the NAIVE `log(Σ exp(x))` with no max-shift** (`_logsumexp_values`, `:394`). Verified: `evaluate()` of `logsumexp([1000,1000])` raises `OverflowError` where scipy/torch return `1000.693…`. The traced *expression* is mathematically exact, but unlike the numpy (`scipy.special.logsumexp`) and torch (`torch.logsumexp`) engines, the symbolic engine's logsumexp is not numerically stabilized — a parity divergence in the lowered graph's evaluation behavior. Acceptable for pure symbolic simplification (sympy keeps it exact), but a hazard if the symbolic trace is evaluated numerically.
- **Engine-swap:** Stability differs from numpy/torch logsumexp.
- **Verdict:** FINDING(E2-6)

### clip evaluation (`_clip_value` `:494`)
- **Computes:** clamp to `[a_min, a_max]`.
- **How:** Python `max(x,a_min)`/`min(x,a_max)` at eval time.
- **Why correct:** Verified eval: x=7→5, x=-3→0, x=2→2. Matches numpy/torch clamp.
- **Verdict:** OK

### data-dependent ops bincount/unique/searchsorted/index_add (`symbolic_engine.py:345-352`)
- **Computes:** opaque node placeholders.
- **Why correct:** Deliberately non-symbolic; export raises `NotImplementedError` (in `_NON_SYMBOLIC_OPS`). Consistent.
- **Verdict:** OK

### zeros dtype ignored (`symbolic_engine.py:196`)
- **Computes:** object array of `const(0.0)`.
- **How:** `np.full(shape, const(0.0))` — `dtype` arg accepted but ignored.
- **Why correct:** Fills float-zero constants regardless of requested dtype; harmless for symbolic tracing (the const carries a Python float). Minor: an `int` dtype request still yields `0.0` not `0`. LOW, not flagged separately (symbolic domain).
- **Verdict:** OK

---

## Module: pysp/engines/symbolic_export.py

### to_sympy / `_sympy_ops` (`symbolic_export.py:73-151`)
- **Computes:** lowering of the native tree to sympy.
- **How:** op→sympy callable map; `where`→`Piecewise((a,cond),(b,True))`; `gammaln`→`loggamma`; `digamma`→`digamma`(polygamma); `betaln`→`loggamma(a)+loggamma(b)-loggamma(a+b)`; `clip`→`Max/Min`; nullary constants stay exact.
- **Why correct:** Numerically verified that `to_sympy(expr).subs(...)` matches `expr.evaluate(...)` for sub/div/pow/log/exp/sqrt/floor/gammaln/erf/betaln/where/max/clip (all order-sensitive cases included). digamma→`polygamma(0,5)=1.50611…` matches scipy.
- **Numerical stability:** Exact symbolic; n/a.
- **Engine-swap:** Faithful to the numeric op (sign + argument order verified).
- **Verdict:** OK

### `_sympy_const` (`symbolic_export.py:154`)
- **Computes:** Python/numpy scalar → sympy literal.
- **How:** bool→true/false, int→Integer, float→Float, else sympify.
- **Why correct:** Preserves type; floats kept as Float (not rationalized).
- **Verdict:** OK

### to_sage / `_sage_ops` (`symbolic_export.py:176-259`)
- **Computes:** lowering to sage (full SageMath or passagemath-symbolics).
- **How:** op→sage callable; `gammaln`→`log_gamma`, `digamma`→`psi`, `betaln` via log_gamma identity, constants→`sage.pi/e/euler_gamma/oo`.
- **Why correct (by reading; sage not installed):** Op map mirrors the verified sympy map with the correct sage spellings. **One concern:** `_sage_where` (`:214`) encodes `where(cond,a,b)` as `a*indicator + b*(1-indicator)` where `indicator = sage.SR(cond)`. For a relational `cond` this is NOT a 0/1 indicator in sage's symbolic ring — `SR(relation)` is a symbolic equation/inequality, and multiplying it by `a` does not evaluate to a Piecewise. Unlike the sympy path (`Piecewise`), this produces an algebraically wrong expression unless `cond` already evaluates to a 0/1 SR element. Could not execute (sage absent); flagged for review since it differs structurally from the (correct) sympy `where`. The module comment itself admits sage "lacks a Piecewise-over-relations primitive."
- **Verdict:** FINDING(E2-7)

### Non-symbolic op guard (`symbolic_export.py:25, 138, 247`)
- **Computes:** raises `NotImplementedError` for bincount/unique/searchsorted/index_add.
- **Why correct:** These have no closed-form symbolic meaning; explicit failure is correct.
- **Verdict:** OK

### Array mapping `_map_array` (`symbolic_export.py:60`)
- **Computes:** elementwise lowering of an object array, shape preserved.
- **Why correct:** `np.ndindex` over shape; verified shape preservation implicitly via scalar tests.
- **Verdict:** OK

---

## Summary of parity posture
- **Special functions (gammaln/digamma/erf/betaln):** numpy↔torch↔symbolic→sympy all agree to ≤1e-13. Solid.
- **Core elementwise + reductions:** torch shims handle `axis`→`dim` and namedtuple unwrapping; numerically faithful.
- **Live hazards:** none currently exercised (all flagged items are latent API gaps or masked semantic divergences).
- **Real correctness gaps:** symbolic `evaluate` of `digamma` (E2-5, KeyError) and sage `where` (E2-7, likely wrong lowering) are the two that would produce a failure/wrong result if hit.
