# Group-structured quantized attention (E10 falsification)

Date: 2026-07-12 · Scripts: `train_quantized_keys.py` (torch, end-to-end), plus the numpy
exactness/scaling/LSE study earlier the same day. Kill criteria were pre-stated in the script
headers before the first run.

## The idea under test

Keys product-quantized to a `k^dim` lattice (dim blocks × k codes; per-block simplex-group ≅
additive logits). Attention weights are then cell-constant, so softmax over n tokens collapses
EXACTLY to `Σ_cells count · exp(score) · v̄` — values stay continuous and arbitrary. Complexity
per query: O(occupied cells · dim), independent of n; O(log n) for streaming windows via integer
count trees (Z^cells is a true group: eviction = exact subtraction, no float cancellation).

## Verdicts

| criterion | result |
|---|---|
| numpy exactness (cell form vs dense softmax) | max err 4e-17 (n=50k, 1496 cells) |
| numpy scaling (fixed cell pool) | 330x over dense at n=1M; per-query cost flat |
| quantized LSE via histogram + exp-LUT | 8-bit: 2e-3; 12-bit: 6e-5 (one 2^bits exp table) |
| **K1 — trainability**: STE+VQ quantized-key transformer on associative recall | **PASS: 97.7% vs dense 99.9% recall** (2 layers, 2 heads, 4 blocks × 16 codes/head) |
| **K2 — exactness on the trained model** | **PASS: cell ≡ dense-over-quantized to 1e-9** |
| **K3 — inference speed with the trained code distribution** | **PASS: 70x at 64k context** (3.2x @ 4k, 8.7x @ 16k — grows linearly with n) |

## The headline finding: trained models collapse the key space

The trained model occupies **49 cells out of 65,536 possible** — and the count stays flat at 49
from 4k to 64k context. Training did not merely tolerate quantization; it learned a ~49-symbol
discrete key language (≈ the task's semantic types), making attention over ANY context length a
49-row computation. The Zipf-occupancy premise was conservative: on a trained model, per-query
cost was O(1) in context length here.

This answers the question the June MGF falsification could not: the NATIVE path works. Where
Gaussian cluster approximations of a dense-trained cache failed (misspecification amplified by
exp), making quantization part of the model moves the approximation error to training time,
where SGD absorbs it — measured cost 2.3% recall on a pure-retrieval task.

## Honest caveats

- Tiny model/task; K3 contexts are random-token streams, not natural data; the 49-cell occupancy
  reflects the task's small symbol vocabulary. Richer tasks will occupy more cells — the
  mechanism scales O(cells) regardless, and the E7 referee bake-off is the proper next test.
- The quantized arm needed ~2x the steps to break through (VQ optimization friction; EMA
  codebooks / temperature annealing are the standard remedies, untried here).
- K2/K3 examined one head of one block; full-model long-context inference wiring is the E10
  engineering task.

## Follow-ups: all four landed (same day)

1. **E10 mechanism** — `mixle/experimental/quantized_key_attention.py`
   (`QuantizedKeyAttentionSpine`): exact RoPE'd near window + bounded far-field cell store
   (integer counts, value sums, code vectors), ONE joint softmax via `log(count)` logit offsets,
   straight-through gap bank so visibility is a pure function of (query pos, key pos) — the
   streaming-equivalence test caught the chunk-size blind spot the naive fold-at-chunk-end design
   has, and the gap bank is the fix. Collapse identity asserted to 1e-12 through the production
   functions; integer cell contents bit-identical across chunk sizes 4/8/16.
2. **Q-LSE kernel** — `mixle.engines.qlut.quantized_logsumexp` (+ `lse_error_bound`): integer
   histogram + one 2^bits exp table; the weighted form computes the cell-collapsed attention mass
   directly. Fused-kernel wiring is deliberately follow-up scope (gated by the typed runtime's
   compute-band axis).
3. **Integer count structures** — `SlidingCellWindow` (exact group eviction: counts equal a
   recount, always) and `CellCountTree` (Fenwick of sparse cell dicts; arbitrary-window counts in
   O(log n) node merges via the group inverse, receipts prove the touch count).
4. **`_bitpacked` shipped** — `.pyx` sources now included in sdists/wheels (MANIFEST.in +
   package-data; wheel inspected), `kernels` extra pulls the toolchain, and the compiled
   extension's tests unskip and pass (6/6).

## E7 referee bake-off at matched state bytes (same day)

Script: `e7_bakeoff.py`. Panel at d_model=16, n_layer=2, n_head=2, vocab=16, seed=7, shared
budget 12 kB: E10 (window 8, 2 blocks x 8 codes, 32 slots -> 9216 B) vs E1 byte-matched at
window 36 (9216 B), E3a linear attention (1152 B, structurally fixed), E3c tensor sketch
(11264 B). FD diverged inside its own SVD shrink at every ell tried (16/12/8) over these long
un-detached streams — the design note's documented differentiable-shrink instability — and is
excluded with the reason printed rather than a fabricated row.

| range | E1 needle/copy/hop | E3a | E3c | E10 | ppl E1 / E3a / E3c / **E10** |
|---|---|---|---|---|---|
| 16  | **1.000**/0.688/0.062 | 0.312/0.125/0 | 0.062/0.125/0.125 | 0/0.125/0.062 | 675.6 / 614.6 / 275.7 / **69.5** |
| 64  | 0/0/0 | 0/0.062/0 | 0/0.125/0.062 | 0/0.125/0.062 | **48.8** / 101.4 / 138.0 / 60.5 |
| 256 | 0/0/0 | 0/0/0 | 0/0/0.125 | 0/0/0.125 | 363.4 / 348.4 / 498.7 / **64.5** |

Read (n_train_steps=500, batch 1 — the referee's own budget):

- **Retrieval at referee budget: nobody crosses their window.** E1 solves needle at d=16 INSIDE
  its 36-token window and drops to 0 beyond; E3a's 0.312@16 rides its whole-stream prefix; every
  mechanism scores 0.000 needle at 64/256. E10's 0.000@16 (retrieval must cross its 8-token
  window) is the kill signal *at this budget* — see the budget probe below before reading it as
  structural.
- **Perplexity: the far field carries real long-context signal.** At d=256 (32x E10's window),
  E10's 64.5 is 5.4x better than the best alternative (348.4); it is the only mechanism whose ppl
  does NOT blow up with range.
- **Occupancy receipt:** 8/10 cells occupied (of 32 slots, 64 possible), 0 dropped tokens. The
  straight-through gap bank mattered: without the encoder gradient path, layer-1 keys collapsed
  to 2 cells (VQ collapse) and occupancy receipts caught it.

## Budget probe: retrieval through the cell store is learnable — the referee is ~20x short

`needle_budget_probe.py`: same mechanism, needle d=16 through window 8 (retrieval MUST cross the
far field), batch 8:

| examples seen | needle acc | probe loss / chance |
|---|---|---|
| 8 (step 1) | 0.000 | 9.62 / 2.77 |
| 2,400 | 0.594 | 1.39 / 2.77 |
| 9,600 | **1.000** | 0.09 / 2.77 |
| 24,000 (step 3000) | 0.938–1.000 stable | 0.04–0.17 |

Occupancy stays at 5–7 cells/layer throughout, 0 drops. This is the falsification's VQ-friction
prediction playing out quantitatively (its quantized arm needed ~2x the dense arm's budget;
~160k examples there): **the cell store supports 100% far-field retrieval — the E7 referee's
500-example default is simply ~20x below the quantized mechanism's learning budget.** Kill
criterion verdict: not killed on structure; at the referee's own budget the sketches' smoother
optimization wins needle@16 (E3a 0.312 vs 0), recorded as the honest scale caveat. Untried
remedies for the friction itself: EMA codebooks, temperature annealing.
