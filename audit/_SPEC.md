# Shared audit spec (read by every audit agent)

You are auditing **pysparkplug** (`pysp`), a probability/statistics library with a pluggable
compute-engine layer (numpy / torch / symbolic). Working dir: `/Users/grantboquet/codex/pysparkplug`.
Python venv: `.venv/bin/python`.

## What to audit (priority order)
1. **Computational correctness** — every nontrivial formula. Verify against the canonical
   math (densities, log-densities, sufficient statistics, M-step closed forms, samplers,
   gradients, log-partitions, normalizing constants). This is the top priority.
2. **Numerical stability** — log-space vs naive products, `logsumexp` usage, catastrophic
   cancellation (e.g. `var = E[x^2] - E[x]^2`), `log(0)`/`log1p`/`expm1`, division by near-zero,
   overflow in `exp`, clipping bounds, dtype/precision (float32 drift), guards on degenerate params.
3. **Engine-swap safety** — does the code stay engine-neutral where it claims to
   (`backend_*` / declaration hooks / `exp_family_*`)? Any host-only numpy leaking into a path
   that must run on torch/symbolic? numpy-vs-torch parity hazards. Mismatch between the numpy
   accumulator `value()` and the declaration-generated (torch/stacked) sufficient statistics.
4. **API-standard conformance** — the real pysp Distribution contract (see
   `pysp/stats/compute/pdist.py` ABCs and a clean reference leaf such as
   `pysp/stats/leaf/poisson.py` or `pysp/stats/leaf/gaussian.py`): `density`, `log_density`,
   `seq_log_density`, `dist_to_encoder`/`seq_encode`, `estimator`, `accumulator_factory`,
   `sampler`, `enumerator`, declaration/capabilities. Flag deviations, dead/duplicated code,
   and non-idiomatic syntax.

## How to verify
- Read the code carefully; cite exact `file:line`.
- Where a formula is in doubt, **run a small numeric check** with `.venv/bin/python` (compare to
  `scipy.stats` if available, or finite-difference a gradient, or cross-check numpy vs torch
  accumulate). Prefer evidence over assertion.
- Do not modify any source file. You may only create your one assigned section file.

## Output — two parts

### Part 1: write your section file
Write `audit/sections/<NN-name>.md` (path given in your task) with this structure:

```
# <Domain> — computation ledger

## Module: <relative/path.py>
### <computation name> (`file:line`)
- **Computes:** <what, with the math formula>
- **How:** <implementation approach in 1-3 lines>
- **Why correct:** <derivation/justification, or reference identity>
- **Numerical stability:** <log-space? cancellation? guards? precision? — or "n/a">
- **Engine-swap:** <neutral / host-only / parity hazard — note specifics>
- **Verdict:** OK | FINDING(<id>)
... (repeat per computation)
```
Cover every meaningful computation, not just buggy ones — this doubles as a reference ledger.

### Part 2: return to me (your final message)
A compact findings list, one per line, ONLY for real issues, each as:
`SEVERITY | file:line | one-line description | suggested fix`
where SEVERITY ∈ {CRITICAL, HIGH, MEDIUM, LOW}. CRITICAL/HIGH = wrong results or NaN/overflow
in normal use; MEDIUM = stability/precision/edge-case or engine-swap break; LOW = style/API/dead code.
If you ran numeric checks, state the result. End with the section file path you wrote.
Be precise and skeptical; do not invent issues. If a computation is correct, say so rather than padding.
```
