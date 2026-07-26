# `mixle.experimental` — exploratory surfaces, not yet in the stable package

This is a holding area for mechanisms that haven't earned promotion into the stable `mixle` package yet.
Code here may change or be removed without the usual stability guarantees; import it expecting churn.

## Graduation rule

A mechanism graduates out of `experimental/` into the stable package when it:

1. **beats the E1 baseline on the E7 evaluation suite at matched FLOPs**, and
2. **has misfit/truncation receipts** — honest, measured error-characterization artifacts for its state
   structure (e.g. a sketch's collision rate, a tree's truncation error, a moment-closure residual), not
   just "it works" anecdote.

Both conditions are required. This is the same "every acceptance criterion is a receipt" ethos as the rest
of the long-context roadmap: a mechanism doesn't graduate on vibes, it graduates on artifacts checked
against a fixed baseline and a fixed compute budget.

**Status as of this scaffold: forward-looking contract, not yet enforced.** E1 (the baseline mechanism)
and E7 (the referee evaluation suite) are later items on the same roadmap track and don't exist yet. This
document states the rule those items will satisfy; nothing in `mixle.experimental` checks it automatically
today. `mixle.experimental.graduation.ExperimentalMechanism` gives later items a place to record their
receipts once E1/E7 land, and `is_eligible()` is the (already-testable) bookkeeping check for "does this
mechanism have both receipts" — it does not itself run any evaluation.

## Testing convention

Tests exercising code under `mixle/experimental/` are tagged `@pytest.mark.experimental` (registered in
`pyproject.toml`) so they can be filtered, run, and reported on distinctly from the stable-package suite —
mirroring how `torch`/`numba`/`jax`/`optional` mark backend-gated tests elsewhere in this repo.

## Current contents

`mixle/experimental/__init__.py`'s own module docstring is the maintained, detailed listing — one entry per
module, cross-referenced to its roadmap item (Track E long-context, Track P theory/verification, program.py,
graduation.py, `typed_runtime/`) and its design note under `notes/designs/` where one exists. Read it there
rather than here, so this file doesn't drift into a second, competing list.

That docstring only covers modules the package actually re-exports, though, and the directory has grown past
that: eight modules currently exist as files but aren't imported (lazily or otherwise) by `__init__.py`, so
they don't appear in its listing and are only reachable by their full dotted path. Documented here instead,
since an omission from both files at once is a real gap, not a formatting choice:

- `growth_operators.py` — function-preserving network growth (depth/width-split, structure-expansion) over a
  real transformer; the inverse of G3's coarsening, which folds capacity down instead of splitting it up.
- `kv_cache_quant.py` — two KV-cache quantization mechanisms built on already-existing machinery: quantized
  exact outliers plus parametric tails for a far-field attention bank's own outlier bookkeeping.
- `law_discovery.py` — the "propose a relationship" discovery tier: fits and selects among candidate
  functional forms for a black-box simulator's input/output behavior (distinct from `equation_discovery.py`'s
  SINDy recovery of a *known* operator, graded against ground truth).
- `moment_closure_attention.py` — E2, a far-field attention mechanism keeping bounded moment-closure summary
  statistics of evicted tokens alongside E1's exact near-field sliding window.
- `quantized_key_attention.py` — E10, product-quantized-key "cell" attention: attention weights exact by
  construction once keys are quantized onto a learned per-block codebook lattice.
- `sketch_state_attention.py` — E3, a far-field sketch-based (tensor-sketch / frequent-directions) state with
  a provable approximation guarantee, contrasted with E2's adaptive learned closure.
- `structure_edit_schedule.py` — a real architecture-edit action space (grow / prune / reshape) wired into
  `mixle.inference.conditional_jit_controller`'s previously-unimplemented `STRUCTURE_EDIT` extension point.
- `tying_discovery.py` — the R1 profile/permutation (copula-style) decomposition of a weight tensor's
  flattened values, surfacing compatible tying candidates in trained weights and evaluating each tie on an
  isolated model under an explicit output-error budget.
- `unlearning.py` — a two-phase exact-unlearning protocol for audited additive single-step estimators:
  committed sufficient statistics at ingestion, then retained-record-only deterministic certification after
  deletion against an externally anchored manifest digest.

Whether these eight should graduate into `__init__.py`'s exports, stay reachable only by full path, or get
folded elsewhere is a separate decision from documenting that they exist; this list makes that decision
visible rather than silently deferring it.
