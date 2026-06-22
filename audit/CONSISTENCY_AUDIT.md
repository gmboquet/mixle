# pysparkplug — API-Consistency & Abstraction Audit + Remediation

**Date:** 2026-06-22 · **Branch:** `consistency-fixes` (isolated worktree off `speed-lda-gamma-cap`).

Separate from `COMPUTATION_AUDIT.md` (which checked the math). This audit asked: *are the parts of
pysparkplug consistent, and are there single-use classes that should be a shared abstraction?* Five
read-only agents swept the tree against the `pdist.py` ABCs and the cleanest reference implementations.

**Answer: no, it was not uniformly consistent** — broadly contract-conformant, but rapid feature
growth left (A) drift that was silently buggy, (B) single-use classes that were special cases or
verbatim copies of an existing one, (C) pervasive `.key`/`.keys` naming drift, and (D) one
under-specified abstraction (the compute engine). **This branch remediates essentially all of it**
behind the green suite (2520 passing).

## Remediation status

### A — drift that was actually a bug → FIXED (commit `6063802`, `f7d5140`)
| # | Fix |
|---|---|
| A1 | `power_law_hawkes` encoder/accumulator now subclass their ABCs (+ `__eq__`, module-level factory) |
| A2 | `pdist.seq_ld_lambda` returns `[self.seq_log_density]` instead of `None` |
| A3,A4 | merge key threaded through `estimate()` (VMF, MVN, bernoulli-sets) — was silently dropped |
| A5,A6,A7 | `markov_chain` len-recursion / `markov_transform` dead-stub restore / `grammar` keyed merge |
| A8 | `semi_supervised_hidden_markov_model` divergence documented (genuine, not a bug) |
| A9 | DPM/HDP `scale()` overrides protect non-linear metadata |
| A10 | `random_graph` `@classmethod fit_mle` → module-level estimation functions |

### D1 — engine ABC formalized → FIXED (commit `f4b0e26`)
`ComputeEngine` now declares a 30-op `REQUIRED_OPS` contract enforced at class definition
(`__init_subclass__`); symbolic gaps filled; `engine_op_parity_test.py` added. The root cause of the
engine-parity bug class is now a compile-time guarantee.

### C — naming drift → FIXED (commits `595b35c`, `1f21968`)
- `self.nobs` → `self.count` (half_normal, gamma, inverse_gaussian).
- geometric `backend_legacy_*` → `exp_family_legacy_sufficient_statistics`.
- `select.py` dead camelCase `accumulatorFactory` fallback removed.
- **The big one:** `self.key` → `self.keys` normalized across the merge-key attribute —
  **573 occurrences in 70 files** (verified every `self.key` was the merge key). One name everywhere,
  matching the `keys=` constructor parameter.

### D (helpers) — duplication consolidated → FIXED (commit `1acd99e`)
Six stable-math helpers re-implemented module-locally now live once in `utils/special.py`:
`log1mexp`, `logsubexp`, `logsumexp`, `softmax`/`softmax_rows` (with the all-`-inf`-row guard),
`valid_integer` — all call sites repointed, drop-in verified.

### B — single-use classes / duplication → FIXED / assessed
| # | Outcome | commit |
|---|---|---|
| B9 | concrete default `key_merge`/`key_replace` on the accumulator base (was abstract stubs) | `8f4d442` |
| B1 | `GaussianMixture` subclasses `Mixture` — **−477 lines**, bit-identical | `93862fc` |
| B3 | `IntegerStepBernoulliEdit` subclasses `IntegerBernoulliEdit` — **−581 lines**, fixed wrong `__str__` | `c26bedf` |
| B5 | shared `_MeanScatterAccumulator` for Wishart/InverseWishart | `93770e8` |
| B7,B8 | combinator `SingleChildAccumulator` + `MaskedBaseEncoder` bases | `e5d366a` |
| B6 | shared `InitTransKeyedAccumulator` for markov transforms (E-step correctly kept separate) | `0ec8926` |
| B4 | shared `_lda_vi_fixed_point` + `_lda_elbo_from_gamma` (LabeledLDA ⊂ LDA loop) — −122 net | `(wave 4)` |
| B12 | `PosteriorResult` Protocol typing `RandomVariable.result` (9 result classes) | `(wave 4)` |
| B13 | `OptimizationResult` base for the BO result classes | `(wave 4)` |
| B14 | shared numpy `_kernels.py` (RBF/Matérn); torch GP justifiably left backend-specific | `(wave 4)` |
| B15 | generic `FitResult` for the models `(model, history)` dataclasses | `(wave 4)` |
| B2 | **assessed — false positive.** `IntegerHiddenAssociation` genuinely differs (dense+numba vs delegated, 5- vs 3-tuple); only redundant `density()` overrides removed | `328d6fd` |

**Honest non-findings:** B2 and the markov E-step (B6) were *not* force-deduplicated — the families
genuinely diverge, and inventing a shared base would add indirection for ~0 real savings. The torch
GP kernel stays separate (autograd backend difference).

## Deferred (intentionally not done in this branch)
- **Leaf B9 boilerplate cleanup** — the ~15 leaf accumulators still carry their own (now redundant
  with the base default) standard `key_merge`/`key_replace`. Harmless; the base default makes deleting
  them a trivial future cleanup. Skipped as low-value churn against the live concurrent session.
- **B10 / B11** (mixture-responsibility helper; moment-accumulator base) — touch central, hot code;
  deferred as higher-risk for modest savings.
- Smaller shared helpers (Ogata thinning sampler, Chow-Liu structure, Bartlett-Wishart sampler,
  Dirichlet entropy, assignment enumerator) — incremental, left for a follow-up.

## Net effect
~1,800+ lines of duplication removed, the pervasive `key`/`keys` split eliminated, the engine
op-surface made a hard contract, six stable-math helpers given one home, and 10 silent contract bugs
fixed — all behind the existing green suite plus new regression tests
(`consistency_fixes_regression_test.py`, `engine_op_parity_test.py`).
