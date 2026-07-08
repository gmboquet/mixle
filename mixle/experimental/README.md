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

- `program.py` / `graduation.py` — see `mixle/experimental/__init__.py` for what each module is.
