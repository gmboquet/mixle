"""E10 acceptance gate: the E7 long-context referee, E10 vs E1 + the three E3 sketches at matched state bytes.

Every mechanism gets the SAME shared state-byte budget; configs below are tuned (via the calibration
pass this script prints first) so each mechanism actually USES a comparable number of bytes rather than
merely fitting under a loose cap -- E1 spends its bytes on a longer exact window, the E3 sketches on
oblivious far-field summaries, E10 on a quantized-key cell store. Ranges are chosen so the smallest sits
near the byte-matched E1 window (its comfort zone) and the larger two are far beyond every window, where
only far-field state can answer.

Pre-stated read (from notes/standout-roadmap-tasks.md's E10 card): E10 must beat the sketch baselines at
equal bytes on content retrieval (needle, multi-hop); long-range COPY needs positional order in the far
field, which E10's position-free cells structurally cannot represent -- a loss there is the mechanism's
honest boundary, not a tuning failure. Kill criterion: >10% worse than the best sketch on needle/multi-hop
at matched bytes.
"""

import sys
import time

import numpy as np
import torch

from mixle.experimental.context_spine import SlidingWindowSpine
from mixle.experimental.long_context_eval import _state_bytes, comparison_table, copy_suite, evaluate
from mixle.experimental.quantized_key_attention import QuantizedKeyAttentionSpine
from mixle.experimental.sketch_state_attention import (
    FrequentDirectionsSpine,
    LinearAttentionSpine,
    TensorSketchSpine,
)

VOCAB = 16
D_MODEL, N_LAYER, N_HEAD = 16, 2, 2
WINDOW = 8  # the E10/E3 near window; E1 gets a LONGER window costing the same bytes
RANGES = (16, 64, 256)
SEED = 7
BUDGET_BYTES = 12_000


def build_mechanisms() -> dict[str, torch.nn.Module]:
    torch.manual_seed(0)
    e1 = SlidingWindowSpine(VOCAB, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, window=36)
    torch.manual_seed(0)
    e3a = LinearAttentionSpine(VOCAB, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD)
    torch.manual_seed(0)
    # ell is capped by d_row = 2*head_dim = 16, and ell near that ceiling makes the FD shrink's SVD
    # degenerate (ell=16 and ell=12 both diverged here); ell=8 is the config this repo's own E7 test
    # exercises. FD therefore runs below the shared budget at its numerical ceiling, reported, not
    # hidden -- and if it still diverges over these much longer un-detached streams (the design note's
    # documented differentiable-shrink instability), its row is excluded with the reason printed.
    e3b = FrequentDirectionsSpine(VOCAB, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, window=WINDOW, ell=8)
    torch.manual_seed(0)
    e3c = TensorSketchSpine(
        VOCAB, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, window=WINDOW, sketch_dim=72, degree=2
    )
    torch.manual_seed(0)
    e10 = QuantizedKeyAttentionSpine(
        VOCAB,
        d_model=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        window=WINDOW,
        n_blocks=2,
        codes_per_block=8,
        max_cells=32,
    )
    return {
        "E1_sliding_window": e1,
        "E3a_linear_attention": e3a,
        "E3b_frequent_directions": e3b,
        "E3c_tensor_sketch": e3c,
        "E10_quantized_key_cells": e10,
    }


def calibrate_bytes(mechanisms: dict[str, torch.nn.Module]) -> None:
    """Stream the largest range's copy suite through each mechanism and print the bytes its carried
    state actually holds -- the number `evaluate` budgets against. Run this first; tune configs until
    the arms sit within ~25% of each other, then trust the table."""
    rng = np.random.RandomState(0)
    x, y = copy_suite(rng, distance=max(RANGES), vocab=VOCAB)
    print(f"== state-byte calibration (copy suite, distance {max(RANGES)}) ==")
    for name, mech in mechanisms.items():
        state = mech.init_state(1)
        with torch.no_grad():
            for start in range(0, x.shape[1], WINDOW):
                state, _ = mech.step(state, (x[:, start : start + WINDOW], y[:, start : start + WINDOW]))
        print(f"  {name:28s} state_bytes={_state_bytes(state):6d}  (budget {BUDGET_BYTES})")


def main() -> None:
    mechanisms = build_mechanisms()
    calibrate_bytes(mechanisms)
    if "--calibrate-only" in sys.argv:
        return

    results = {}
    unavailable = {}
    for name, mech in mechanisms.items():
        t0 = time.perf_counter()
        try:
            results[name] = evaluate(
                mech,
                ranges=RANGES,
                state_budget_bytes=BUDGET_BYTES,
                seed=SEED,
                hops=2,
                n_train_steps=500,
                n_eval_trials=16,
                perplexity_steps=6,
                curriculum_rounds=8,
            )
        except (RuntimeError, torch.linalg.LinAlgError) as exc:  # FD's documented SVD-shrink divergence
            # at long un-detached streams -- record and keep the rest of the panel honest.
            unavailable[name] = f"{type(exc).__name__}: {exc}"
        print(f"[{name}] evaluated in {time.perf_counter() - t0:.1f}s", flush=True)

    print("\n" + comparison_table(results))
    print("\nneedle mean probe loss vs chance (supplementary lens -- 'solved' thresholds at 0.5*chance):")
    for name, res in results.items():
        row = "  ".join(
            f"d={d}: {res['suites'][d]['needle']['mean_probe_loss']:.2f}/{res['suites'][d]['needle']['chance_loss']:.2f}"
            for d in res["ranges"]
        )
        print(f"  {name:28s} {row}")
    for name, reason in unavailable.items():
        print(f"\n[{name} row unavailable] diverged during its own training/eval, honestly excluded: {reason}")

    e10 = mechanisms["E10_quantized_key_cells"]
    rng = np.random.RandomState(1)
    x, y = copy_suite(rng, distance=max(RANGES), vocab=VOCAB)
    state = e10.init_state(1)
    with torch.no_grad():
        for start in range(0, x.shape[1], WINDOW):
            state, _ = e10.step(state, (x[:, start : start + WINDOW], y[:, start : start + WINDOW]))
    print(f"\nE10 occupancy after the {max(RANGES)}-token stream (trained model): {e10.occupancy_receipt(state)}")


if __name__ == "__main__":
    main()
